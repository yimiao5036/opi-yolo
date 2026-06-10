import time
import math
from enum import Enum


class MissionState(Enum):
    INIT = 0  # 初始化/等待连接
    ARMING = 1  # 解锁中
    TAKEOFF = 2  # 自动起飞中
    NAVIGATING = 3  # 航点巡航导航
    HOLD_TASK = 4  # 到达航点后的原地搜索/悬停
    VISUAL_TRACKING = 5  # 🎯 视觉追踪中（由视觉主循环接管控制权）
    RETURNING = 6  # 返航中
    LANDING = 7  # 自动降落中
    FINISHED = 8  # 任务结束/已上锁
    HOLD_FINAL = 9  # 结束巡点后原地悬停（当 return_to_home 为 False 时使用）


class FailsafeTriggered(Exception):
    """自定义安全异常：当安全员人工介入切出自控模式时抛出，用于暴力熔断主循环"""
    pass


class MissionManager:
    def __init__(self, drone, waypoints, target_altitude=1.5, arrival_radius=0.3, hold_duration=5.0,
                 return_to_home=True):
        """
        :param drone: 实例化的 DroneController 底层驱动对象
        :param waypoints: 航点列表 [{'x': 1.0, 'y': 2.0, 'z': -1.5, 'yaw': 0}, ... ]
        :param target_altitude: 默认起飞高度（米，NED中向上为负值）
        :param arrival_radius: 航点到达判定半径（米）
        :param hold_duration: 抵达巡点后，没有发现目标时的最大搜寻/悬停时间（秒）
        :param return_to_home: 航点巡完后是否返航，False则在最后一个点原地悬停
        """
        self.drone = drone
        self.waypoints = waypoints
        self.target_altitude = target_altitude if target_altitude < 0 else -target_altitude
        self.arrival_radius = arrival_radius
        self.hold_duration = hold_duration
        self.return_to_home = return_to_home

        self.state = MissionState.INIT
        self.current_wp_index = 0

        # 计时器定义
        self.hold_start_time = 0.0
        self.tracking_start_time = 0.0

        # 起始点（Home）坐标缓存
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0

    def _is_tracking_task_complete(self):
        """
        🛠️ 预留接口：判断视觉追踪任务是否完成
        你可以根据需求修改内部逻辑。例如：
        - 当前：假设持续追踪 5 秒视为完成任务。
        - 后续扩展：可以引入与目标的物理距离、传感器状态等。
        """
        tracking_duration = time.time() - self.tracking_start_time
        target_hold_time = 5.0  # 默认设定5秒

        if tracking_duration >= target_hold_time:
            return True
        return False

    def update(self, target_detected=False):
        """
        状态机高频核心业务逻辑，由外部主循环驱动
        :param target_detected: 当前帧是否发现了视觉追踪目标
        """
        # 0. 越权检查（最强人肉兜底）：只要切出自主控制模式，立刻暴力熔断
        if self.state not in [MissionState.INIT, MissionState.FINISHED]:
            if self.drone.current_mode not in ['GUIDED', 'OFFBOARD']:
                print(f"\n⚠️ [Failsafe] 检测到飞控模式被切为: {self.drone.current_mode}! 外部人工强行接管，熔断任务！")
                self.state = MissionState.FINISHED
                raise FailsafeTriggered("安全员强行接管")

        # 1. INIT: 等待飞控连接并获取当前位置作为 Home 点
        if self.state == MissionState.INIT:
            if self.drone.is_connected:
                print("[INIT] 飞控已连上。获取当前坐标为起始点...")
                self.home_x, self.home_y, self.home_z = self.drone.local_position
                print(f"[INIT] 起始点坐标: ({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f})")
                print("[INIT] 正在下发 GUIDED/OFFBOARD 模式并尝试解锁...")

                target_mode = "OFFBOARD" if "OFFBOARD" in self.drone.master.mode_mapping() else "GUIDED"
                self.drone.set_mode(target_mode)
                self.drone.arm()
                self.state = MissionState.ARMING

        # 2. ARMING: 监测解锁状态
        elif self.state == MissionState.ARMING:
            if self.drone.is_armed and self.drone.current_mode in ['GUIDED', 'OFFBOARD']:
                print("\n[解锁成功] 正在下发自动起飞指令...")
                # 下发位置到期望高度
                self.drone.send_target_position(self.home_x, self.home_y, self.target_altitude, 0)
                self.state = MissionState.TAKEOFF

        # 3. TAKEOFF: 自动起飞爬升判定
        elif self.state == MissionState.TAKEOFF:
            current_z = self.drone.local_position[2]
            # 持续发送起飞点位置，激活控制链路
            self.drone.send_target_position(self.home_x, self.home_y, self.target_altitude, 0)

            # NED 坐标系下 Z 为负数，判断是否接近目标高度
            if abs(current_z - self.target_altitude) < self.arrival_radius:
                print("\n[起飞完成] 达到预定高度。开始前往第 1 个巡航航点...")
                self.state = MissionState.NAVIGATING
                self.current_wp_index = 0

        # 4. NAVIGATING: 航点导航巡航
        elif self.state == MissionState.NAVIGATING:
            if not self.waypoints or self.current_wp_index >= len(self.waypoints):
                # 航点全部飞完
                if self.return_to_home:
                    print("\n[巡航完成] 所有航点已巡视完毕。正在返回起始点上方...")
                    self.state = MissionState.RETURNING
                else:
                    print("\n[巡航完成] 所有航点已巡视完毕。根据配置切换至末端就地悬停...")
                    self.state = MissionState.HOLD_FINAL
                return

            # 获取当前目标航点
            wp = self.waypoints[self.current_wp_index]
            self.drone.send_target_position(wp['x'], wp['y'], wp['z'], wp['yaw'])

            # 计算水平及高度距离
            dx = wp['x'] - self.drone.local_position[0]
            dy = wp['y'] - self.drone.local_position[1]
            dz = wp['z'] - self.drone.local_position[2]
            dist_xyz = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            print(
                f"\n\r[巡航中] 前往航点 [{self.current_wp_index + 1}/{len(self.waypoints)}] | 距离目标: {dist_xyz:.2f}m",
                end="", flush=True)

            if dist_xyz < self.arrival_radius:
                print(f"\n[到达航点 {self.current_wp_index + 1}] 进入原地悬停与视觉检索阶段...")
                self.state = MissionState.HOLD_TASK
                self.hold_start_time = time.time()

        # 5. HOLD_TASK: 抵达巡点后的搜索悬停阶段
        elif self.state == MissionState.HOLD_TASK:
            # 优先判定视觉：如果在巡点悬停搜寻期间，视觉发现了目标
            if target_detected:
                print("\n🎯 [目标锁定] 视觉检测到目标！立刻中断悬停，切入 [视觉追踪] 状态！")
                self.state = MissionState.VISUAL_TRACKING
                self.tracking_start_time = time.time()
                return

            # 如果没有发现目标，则保持航点位置，并检查悬停是否超时
            wp = self.waypoints[self.current_wp_index]
            self.drone.send_target_position(wp['x'], wp['y'], wp['z'], wp['yaw'])

            elapsed_time = time.time() - self.hold_start_time
            print(f"\r[检索中] 保持航点中... 剩余搜寻时间: {max(0.0, self.hold_duration - elapsed_time):.1f}s", end="",
                  flush=True)

            if elapsed_time >= self.hold_duration:
                print(f"\n[未发现目标] 搜寻超时。放弃当前点，前往下一个航点...")
                self.current_wp_index += 1
                self.state = MissionState.NAVIGATING

        # 6. VISUAL_TRACKING: 🎯 视觉闭环追踪状态（核心新增）
        elif self.state == MissionState.VISUAL_TRACKING:
            # 在此状态下，高层状态机【不主动发送】绝对位置坐标，把控制权完全让渡给主循环的 PID 速度控制
            # 我们只需要在这里进行任务结束判定
            print(f"\r[追踪中] 正在执行AI视觉闭环精准追踪... 计时: {time.time() - self.tracking_start_time:.1f}s",
                  end="", flush=True)

            if self._is_tracking_task_complete():
                print("\n✅ [任务完成] 当前目标追踪任务圆满完成！重置追踪器，继续执行航点巡航...")
                self.current_wp_index += 1  # 切换到下一个航点
                self.state = MissionState.NAVIGATING

        # 7. RETURNING: 自动返航至起始点上方
        elif self.state == MissionState.RETURNING:
            self.drone.send_target_position(self.home_x, self.home_y, self.target_altitude, 0)

            dist_to_home = math.sqrt((self.home_x - self.drone.local_position[0]) ** 2 +
                                     (self.home_y - self.drone.local_position[1]) ** 2)
            print(f"\r[返航中] 正在飞回起始点位置 | 剩余水平距离: {dist_to_home:.2f}m", end="", flush=True)

            if dist_to_home < self.arrival_radius:
                print("\n[到达] 已安全返回起始点上方。正在下发自动降落指令 (LAND)...")
                if self.drone.set_mode('LAND'):
                    self.state = MissionState.LANDING
                else:
                    print("⚠️ 降落模式切换失败，正在重新下发...")

        # 8. LANDING: 自动降落中
        elif self.state == MissionState.LANDING:
            print(f"\r[降落中] 当前高度: {self.drone.local_position[2]:+.2f}m | 解锁状态: {self.drone.is_armed}",
                  end="", flush=True)
            if not self.drone.is_armed:
                print("\n[完结] 无人机已安全着陆并自动上锁。整个任务流结束。")
                self.state = MissionState.FINISHED

        # 9. HOLD_FINAL: 末端就地悬停
        elif self.state == MissionState.HOLD_FINAL:
            final_wp = self.waypoints[-1] if self.waypoints else {'x': self.home_x, 'y': self.home_y,
                                                                  'z': self.target_altitude, 'yaw': 0}
            self.drone.send_target_position(final_wp['x'], final_wp['y'], final_wp['z'], final_wp['yaw'])
            print(
                f"\r[挂起] 无人机保持在最终航点悬停。坐标:({final_wp['x']}, {final_wp['y']}, {final_wp['z']}) | 等待人工接管中...",
                end="", flush=True)