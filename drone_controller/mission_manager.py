import time
import math
from enum import Enum


class MissionState(Enum):
    INIT = 0        # 初始化/流式发送指令准备中
    ARMING = 1      # 模式切换与解锁中
    TAKEOFF = 2     # 自动起飞中（虚拟航点导航起飞）
    NAVIGATING = 3  # 航点巡航导航
    HOLD_TASK = 4   # 到达航点后的原地任务悬停检测
    RETURNING = 5   # 任务完成，准备切换降落
    LANDING = 6     # 自动降落中
    FINISHED = 7    # 任务结束/已上锁退出


class FailsafeTriggered(Exception):
    """自定义安全异常：当安全员人工介入切出自控模式时抛出，用于暴力熔断主循环"""
    pass


class MissionManager:
    def __init__(self, drone, waypoints, target_altitude=1.5, arrival_radius=0.3, hold_duration=5.0, return_home=True):
        """
        专为 PX4 固件优化的自动巡点状态机管理器
        :param drone: 实例化的 DroneController 底层驱动对象
        :param waypoints: 航点列表，格式为 [{'x': 1.0, 'y': 2.0, 'z': -1.5, 'yaw': 0}, ...]
                          注意：NED 坐标系中，Z轴向下为负，所以起飞或目标高度输入正数，内部会自动转为NED负值
        :param target_altitude: 初始起飞高度（米，正数，内部会自动转为NED负值）
        :param arrival_radius: 判定航点到达的三维欧氏距离阈值（米）
        :param hold_duration: 在每个航点到达后，原地执行悬停/任务的停留时间（秒）
        """
        self.drone = drone
        self.waypoints = waypoints
        self.target_altitude = -abs(target_altitude)  # 强制转为 NED 坐标系的负高度
        self.arrival_radius = arrival_radius
        self.hold_duration = hold_duration
        self.return_home = return_home

        self.state = MissionState.INIT
        self.current_wp_index = 0
        self.hold_start_time = None

        # 链路活性流式计数器：确保以固定频率下发心跳期望（防PX4失控刹车）
        self.last_send_time = 0.0
        self.send_interval = 0.2  # 0.2秒对应 5Hz 下发频率

    def calculate_distance(self, target_x, target_y, target_z):
        """计算无人机当前局部局部位置（来自飞控 LOCAL_POSITION_NED 遥测）与目标的欧氏距离"""
        curr_x, curr_y, curr_z = self.drone.local_position
        dx = target_x - curr_x
        dy = target_y - curr_y
        dz = target_z - curr_z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def _stream_target_position(self, x, y, z, yaw):
        """维持 PX4 OFFBOARD 模式所需的流式位置期望下发（限制 5Hz 频率）"""
        now = time.time()
        if now - self.last_send_time >= self.send_interval:
            # 假设你的底层 base_control.py 中有 send_target_position 接口
            self.drone.send_target_position(x, y, z, yaw)
            self.last_send_time = now

    def update(self):
        """
        核心状态机轮询函数。
        高层主循环（如主循环 10Hz）必须不间断调用此函数驱动飞行任务。
        一旦检测到人肉介入，直接抛出 FailsafeTriggered 异常终止运行。
        """
        # ==================== 1. 安全人肉熔断机制 ====================
        if self.state not in [MissionState.INIT, MissionState.ARMING, MissionState.FINISHED]:
            if self.drone.current_mode != 'OFFBOARD':
                print(f"\n🚨 [熔断保护] 检测到飞控退出自控模式！当前模式为: {self.drone.current_mode}")
                print("🚨 任务管理器瞬间拉断，上位机停止发送所有控制流，完全交由安全员接管！")
                raise FailsafeTriggered("安全员强行越权接管，自控流程熔断")

        # ==================== 2. 状态机业务分支 ====================

        # --- STATE: INIT ---
        if self.state == MissionState.INIT:
            if self.drone.is_connected:
                # PX4 要求：在切入 OFFBOARD 模式之前，上位机必须已经开始流式下发位置/速度期望包，
                # 否则飞控会拒绝切换模式。此处我们在原位流式灌入当前局部坐标作为活性维持。
                cx, cy, cz = self.drone.local_position
                self._stream_target_position(cx, cy, cz, self.drone.yaw)

                print("\n[状态] 飞控通信正常，正在向 PX4 流式下发原位期望以激活控制链路...")
                print("[动作] 正在向飞控请求切换至 OFFBOARD 模式并解锁...")

                if self.drone.set_mode('OFFBOARD'):
                    # 假设底层接口 drone.arm() 会发送解锁包并等待确认
                    if self.drone.arm():
                        self.state = MissionState.ARMING
            else:
                print("\r[状态] 等待飞控 MAVLink 串口通信建立...", end="", flush=True)

        # --- STATE: ARMING ---
        elif self.state == MissionState.ARMING:
            # 持续流式下发当前点，防模式闪退
            cx, cy, cz = self.drone.local_position
            self._stream_target_position(cx, cy, cz, self.drone.yaw)

            if self.drone.is_armed and self.drone.current_mode == 'OFFBOARD':
                print("\n[成功] 飞控已成功解锁，且处于 OFFBOARD 模式。开始执行平滑虚拟起飞流程...")
                self.state = MissionState.TAKEOFF

        # --- STATE: TAKEOFF (虚拟航点平滑起飞) ---
        elif self.state == MissionState.TAKEOFF:
            # PX4 泛用性最强的起飞：发送原地 (0, 0) 坐标，配合目标负高度值
            self._stream_target_position(0.0, 0.0, self.target_altitude, 0.0)

            # 计算距离起飞航点的距离
            dist = self.calculate_distance(0.0, 0.0, self.target_altitude)
            print(f"\r[起飞中] 目标高度: {-self.target_altitude:.2f}m | 当前距离起飞点: {dist:.2f}m", end="",
                  flush=True)

            if dist <= self.arrival_radius:
                print(f"\n[完成] 虚拟起飞安全到达目标高度！开始按顺序巡航。总计航点数: {len(self.waypoints)}")
                self.current_wp_index = 0
                self.state = MissionState.NAVIGATING

        # --- STATE: NAVIGATING (航点巡航导航) ---
        elif self.state == MissionState.NAVIGATING:
            wp = self.waypoints[self.current_wp_index]
            # 流式下发当前期望的绝对 NED 坐标与偏航角（激活PX4控制环并防止失控刹车）
            self._stream_target_position(wp['x'], wp['y'], wp['z'], wp['yaw'])

            dist = self.calculate_distance(wp['x'], wp['y'], wp['z'])
            print(
                f"\r[巡航中] 正在前往航点 [{self.current_wp_index + 1}/{len(self.waypoints)}] | 剩余距离: {dist:.2f}m",
                end="", flush=True)

            if dist <= self.arrival_radius:
                print(f"\n[到达] 成功进入航点 [{self.current_wp_index + 1}] 容差圈！开始执行就地悬停任务...")
                self.hold_start_time = time.time()
                self.state = MissionState.HOLD_TASK

        # --- STATE: HOLD_TASK (到达航点后的原地任务悬停/拍照) ---
        elif self.state == MissionState.HOLD_TASK:
            wp = self.waypoints[self.current_wp_index]
            # 即使在原地停留/拍照，上位机也绝对不能“闭嘴”，必须以 5Hz 持续锁定该点坐标
            self._stream_target_position(wp['x'], wp['y'], wp['z'], wp['yaw'])

            elapsed = time.time() - self.hold_start_time
            print(
                f"\r[任务中] 航点 [{self.current_wp_index + 1}] 原地执行中... 已耗时: {elapsed:.1f}s / {self.hold_duration}s",
                end="", flush=True)

            if elapsed >= self.hold_duration:
                print(f"\n[完成] 航点 [{self.current_wp_index + 1}] 停留时间结束。")
                self.current_wp_index += 1

                if self.current_wp_index < len(self.waypoints):
                    self.state = MissionState.NAVIGATING
                else:
                    print("\n[巡视完毕] 所有计划航点均已完成！准备进入返航/降落切换阶段。")
                    if self.return_home:
                        self.state = MissionState.RETURNING
                    self.state = MissionState.RETURNING

        # --- STATE: RETURNING ---
        elif self.state == MissionState.RETURNING:
            # 持续发送当前点维持链路
            cx, cy, cz = self.drone.local_position
            self._stream_target_position(cx, cy, cz, self.drone.yaw)

            print("\n[降落中] 正在向 PX4 下发自动降落指令 (AUTO.LAND)...")
            # 提示：PX4 固件的标准降落模式名称通常为 'AUTO.LAND' 或 'LAND'，依底层驱动映射而定
            if self.drone.set_mode('AUTO.LAND') or self.drone.set_mode('LAND'):
                self.state = MissionState.LANDING
            else:
                print("⚠️ 切换降落模式失败，正在高频重试...")

        # --- STATE: LANDING ---
        elif self.state == MissionState.LANDING:
            # 进入飞控原生 LAND 模式后，控制权已交还给飞控底层，此时上位机可以安全停止控制包输出了。
            print("\r[降落中] 正在等待飞控触地并自动上锁...", end="", flush=True)
            if not self.drone.is_armed:
                print("\n[完成] 无人机已安全着陆并自动锁桨！任务圆满结束。")
                self.state = MissionState.FINISHED

        # --- STATE: FINISHED ---
        elif self.state == MissionState.FINISHED:
            return True

        return False