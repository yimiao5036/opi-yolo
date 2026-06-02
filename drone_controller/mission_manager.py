import time
import math
from enum import Enum


class MissionState(Enum):
    INIT = 0        # 初始化/等待连接
    ARMING = 1      # 解锁中
    TAKEOFF = 2     # 自动起飞中
    NAVIGATING = 3  # 航点巡航导航
    HOLD_TASK = 4   # 到达航点后的原地任务悬停
    RETURNING = 5   # 返航或直接准备降落
    LANDING = 6     # 自动降落中
    FINISHED = 7    # 任务结束/已上锁


class MissionManager:
    def __init__(self, drone, waypoints, target_altitude=1.5, arrival_radius=0.3, hold_duration=5.0):
        """
        :param drone: 实例化的 DroneController 底层驱动对象
        :param waypoints: 航点列表，格式为 [{'x': 1.0, 'y': 2.0, 'z': -1.5, 'yaw': 0}, ...]
                          注意：NED 坐标系中，Z轴向上为负，所以起飞或目标高度应为负值
        :param target_altitude: 初始起飞高度（米，正数，内部会自动转为NED负值）
        :param arrival_radius: 判定航点到达的三维欧氏距离阈值（米）
        :param hold_duration: 在每个航点到达后，执行任务（或悬停）的停留时间（秒）
        """
        self.drone = drone
        self.waypoints = waypoints
        self.target_altitude = -abs(target_altitude)  # 强制转为 NED 坐标系的负高度
        self.arrival_radius = arrival_radius
        self.hold_duration = hold_duration

        self.state = MissionState.INIT
        self.current_wp_index = 0
        self.hold_start_time = None

    def calculate_distance(self, target_x, target_y, target_z):
        """计算无人机当前局部位置与目标点之间的三维欧氏距离"""
        dx = target_x - self.drone.local_x
        dy = target_y - self.drone.local_y
        dz = target_z - self.drone.local_z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def update(self):
        """
        核心状态机轮询函数。
        高层主循环高频调用此函数，根据当前状态驱动飞行任务。
        """
        # 安全人肉兜底：如果安全员从遥控器切出 GUIDED 模式，状态机挂起，不执行任何行为
        if self.state not in [MissionState.INIT, MissionState.ARMING]:
            if self.drone.current_mode not in ['GUIDED', 'OFFBOARD']:
                print(f"\r\n⚠️  [警告] 安全员切出控制模式！当前模式为: {self.drone.current_mode}, 任务管理程序挂起...", end="")
                return False

        # ----------------- 状态机逻辑 -----------------
        if self.state == MissionState.INIT:
            if self.drone.is_connected:
                print("\n[状态] 飞控通信已建立，准备切换至 GUIDED 模式并解锁...")
                if self.drone.set_mode('GUIDED'):   #若为 PX4 则是OFFBOARD
                    self.state = MissionState.ARMING
            else:
                print("\r[状态] 等待飞控心跳包连接...", end="", flush=True)

        elif self.state == MissionState.ARMING:
            print("[状态] 正在发送解锁指令...")
            if self.drone.arm():
                print("[成功] 飞控解锁成功，正在下发自动起飞指令...")
                # 转换到起飞高度，注意发送的是相对高度或绝对位置
                if self.drone.takeoff(abs(self.target_altitude)):
                    self.state = MissionState.TAKEOFF
            else:
                print("[重试] 解锁未被飞控确认，1秒后重新尝试解锁...")
                time.sleep(1)

        elif self.state == MissionState.TAKEOFF:
            # 判定起飞是否到达指定高度（NED 坐标系下，比较 z 值）
            dist_to_takeoff = abs(self.drone.local_z - self.target_altitude)
            print(
                f"\r[起飞中] 当前高度: {-self.drone.local_z:.2f}M -> 目标: {abs(self.target_altitude)}M | 误差: {dist_to_takeoff:.2f}M",
                end="", flush=True)

            if dist_to_takeoff <= self.arrival_radius:
                print(f"\n[成功] 已到达起飞高度！开始执行巡点导航，共有 {len(self.waypoints)} 个航点。")
                if len(self.waypoints) > 0:
                    self.current_wp_index = 0
                    self.state = MissionState.NAVIGATING
                else:
                    self.state = MissionState.RETURNING

        elif self.state == MissionState.NAVIGATING:
            wp = self.waypoints[self.current_wp_index]
            # 持续高频向飞控下发当前期望航点
            self.drone.send_target_position(wp['x'], wp['y'], wp['z'], wp['yaw'])

            # 计算距离
            dist = self.calculate_distance(wp['x'], wp['y'], wp['z'])
            print(
                f"\r[巡航中] 正在飞往航点 [{self.current_wp_index + 1}]: 目标({wp['x']}, {wp['y']}, {wp['z']}) | 剩余距离: {dist:.2f}米",
                end="", flush=True)

            # 到达判定
            if dist <= self.arrival_radius:
                print(f"\n[到达] 已成功到达航点 [{self.current_wp_index + 1}]！开始悬停执行特定任务...")
                self.hold_start_time = time.time()
                self.state = MissionState.HOLD_TASK

        elif self.state == MissionState.HOLD_TASK:
            wp = self.waypoints[self.current_wp_index]
            # 悬停期间依然要持续发点，保持飞机锁死在原位
            self.drone.send_target_position(wp['x'], wp['y'], wp['z'], wp['yaw'])

            elapsed = time.time() - self.hold_start_time
            print(
                f"\r[任务中] 航点 [{self.current_wp_index + 1}] 原地悬停检测中... 已耗时: {elapsed:.1f}s / {self.hold_duration}s",
                end="", flush=True)

            if elapsed >= self.hold_duration:
                print(f"\n[完成] 航点 [{self.current_wp_index + 1}] 任务执行完毕。")
                # 切换下一个点
                self.current_wp_index += 1
                if self.current_wp_index < len(self.waypoints):
                    self.state = MissionState.NAVIGATING
                else:
                    print("\n[完成] 所有航点均已巡视完毕！准备返航/降落。")
                    self.state = MissionState.RETURNING

        elif self.state == MissionState.RETURNING:
            print("\n[降落中] 正在下发自动降落指令 (LAND)...")
            if self.drone.set_mode('LAND'):
                self.state = MissionState.LANDING
            else:
                print("⚠️ 降落模式切换失败，正在重新下发...")

        elif self.state == MissionState.LANDING:
            # 判断是否已经安全着陆并自动上锁
            if not self.drone.is_armed:
                print("\n[成功] 检测到无人机已安全着陆并自动上锁。任务圆满结束！")
                self.state = MissionState.FINISHED
            else:
                print(f"\r[降落中] 正在着陆... 当前高度: {-self.drone.local_z:.2f}M", end="", flush=True)

        elif self.state == MissionState.FINISHED:
            return True  # 任务全部结束

        return False