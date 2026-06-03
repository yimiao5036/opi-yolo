import time
import math
from enum import Enum


class MissionState(Enum):
    INIT = 0  # 初始化/等待连接
    ARMING = 1  # 解锁中
    TAKEOFF = 2  # 自动起飞中
    NAVIGATING = 3  # 航点巡航导航
    HOLD_TASK = 4  # 到达航点后的原地任务悬停
    RETURNING = 5  # 返航或直接准备降落
    LANDING = 6  # 自动降落中
    FINISHED = 7  # 任务结束/已上锁
    HOLD_FINAL = 8  # 结束巡点后原地悬停（当 return_to_home 为 False 时使用）

class FailsafeTriggered(Exception):
    """自定义安全异常：当安全员人工介入切出自控模式时抛出，用于暴力熔断主循环"""
    pass

class MissionManager:
    def __init__(self, drone, waypoints, target_altitude=1.5, arrival_radius=0.3, hold_duration=5.0,
                 return_to_home=True):
        """
        :param drone: 实例化的 DroneController 底层驱动对象
        :param waypoints: 航点列表，格式为 [{'x': 1.0, 'y': 2.0, 'z': -1.5, 'yaw': 0}, ... ]
                          注意：NED 坐标系中，Z轴向上为负，所以起飞或目标高度应为负值
        :param target_altitude: 初始起飞高度（米，正数，内部会自动转为NED负值）
        :param arrival_radius: 判定航点到达的三维欧氏距离阈值（米）
        :param hold_duration: 在每个航点到达后，执行任务（或悬停）的停留时间（秒）
        :param return_to_home: 巡点结束后是否返回起始点降落。如果为 False，则在最后一个航点原地悬停。
        """
        self.drone = drone
        self.waypoints = waypoints
        self.target_altitude = -abs(target_altitude)  # 强制转为 NED 负值
        self.arrival_radius = arrival_radius
        self.hold_duration = hold_duration
        self.return_to_home = return_to_home

        self.state = MissionState.INIT
        self.current_wp_index = 0
        self.hold_start_time = 0.0

        # 记录起始点坐标，用于返航
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = self.target_altitude

    def get_distance(self, target_wp):
        """计算当前无人机局部 NED 坐标到目标航点的三维欧氏距离"""
        curr_x, curr_y, curr_z = self.drone.local_position
        dx = target_wp['x'] - curr_x
        dy = target_wp['y'] - curr_y
        dz = target_wp['z'] - curr_z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def update(self):
        """
        状态机核心主循环，需由上位机主程序以 5Hz~10Hz 的高频流式调用。
        包含链路维持、安全熔断以及新增的原地悬停/返航分支。
        """
        # 1. 专属安全熔断机制：一旦检测到场外安全员切回手动模式，立即熔断
        if self.drone.current_mode not in ['GUIDED', 'OFFBOARD'] and self.state != MissionState.INIT:
            print(f"\n⚠️ [安全熔断] 检测到飞控退出自动控制模式(当前模式: {self.drone.current_mode})！上位机状态机退出。")
            self.state = MissionState.FINISHED
            return False

        # 2. 状态机逻辑分支
        if self.state == MissionState.INIT:
            if self.drone.is_connected:
                print("\n[通信成功] 无人机已连接，准备记录起始点并解锁...")
                # 记录解锁时的局部位置作为 Home 点
                hx, hy, _ = self.drone.local_position
                self.home_x = hx
                self.home_y = hy
                self.state = MissionState.ARMING
            else:
                print("\r[等待中] 等待飞控心跳连接...", end="", flush=True)

        elif self.state == MissionState.ARMING:
            print("\r[指令下发] 正在尝试解锁电机...", end="", flush=True)
            if self.drone.is_armed:
                print("\n[成功] 电机已解锁！准备切入控制模式并起飞。")
                # 引导无人机切入 GUIDED 或 OFFBOARD
                target_mode = 'OFFBOARD' if 'PX4' in getattr(self.drone, 'firmware_type', 'PX4') else 'GUIDED'
                self.drone.set_mode(target_mode)
                self.state = MissionState.TAKEOFF
            else:
                self.drone.arm()
                time.sleep(0.5)

        elif self.state == MissionState.TAKEOFF:
            # 虚拟/流式起飞：高频下发起飞期望高度，维持链路活性
            self.drone.send_target_position(self.home_x, self.home_y, self.target_altitude, 0)

            # 判定是否到达起飞高度
            _, _, curr_z = self.drone.local_position
            if abs(curr_z - self.target_altitude) < self.arrival_radius:
                print(f"\n[完成] 已到达安全起飞高度: {abs(curr_z):.2f}m。开始巡航任务...")
                if len(self.waypoints) > 0:
                    self.current_wp_index = 0
                    self.state = MissionState.NAVIGATING
                else:
                    print("⚠️ 警告: 航点列表为空！")
                    if self.return_to_home:
                        self.state = MissionState.RETURNING
                    else:
                        self.state = MissionState.HOLD_FINAL
            else:
                print(f"\r[起飞中] 当前高度: {abs(curr_z):.2f}m / 目标: {abs(self.target_altitude):.2f}m", end="",
                      flush=True)

        elif self.state == MissionState.NAVIGATING:
            wp = self.waypoints[self.current_wp_index]
            # “零期望命令”防失控刹车：流式高频持续推流当前目标点
            self.drone.send_target_position(wp['x'], wp['y'], wp['z'], wp['yaw'])

            dist = self.get_distance(wp)
            print(f"\r[任务中] 正在飞往航点 [{self.current_wp_index + 1}] | 剩余距离: {dist:.2f}m", end="", flush=True)

            if dist < self.arrival_radius:
                print(f"\n[到达] 已精准触达航点 [{self.current_wp_index + 1}]，开启原地任务阶段。")
                self.hold_start_time = time.time()
                self.state = MissionState.HOLD_TASK

        elif self.state == MissionState.HOLD_TASK:
            wp = self.waypoints[self.current_wp_index]
            # 悬停期间同样高频重发当前坐标，让飞控稳定在原地
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
                    # 【核心修改点】所有航点巡视完毕，根据参数决定去向
                    if self.return_to_home:
                        print("\n[完成] 所有航点均已巡视完毕！准备返航起始点。")
                        self.state = MissionState.RETURNING
                    else:
                        print("\n[完成] 所有航点均已巡视完毕！[设置] 不执行返航，就地保持悬停。")
                        self.state = MissionState.HOLD_FINAL

        elif self.state == MissionState.RETURNING:
            # 飞回记录的 Home 起始点高度位置
            self.drone.send_target_position(self.home_x, self.home_y, self.home_z, 0)

            dist_to_home = math.sqrt((self.home_x - self.drone.local_position[0]) ** 2 +
                                     (self.home_y - self.drone.local_position[1]) ** 2)
            print(f"\r[返航中] 正在飞回起始点位置 | 剩余水平距离: {dist_to_home:.2f}m", end="", flush=True)

            if dist_to_home < self.arrival_radius:
                print("\n[到达] 已安全返回起始点上方。正在下发自动降落指令 (LAND)...")
                if self.drone.set_mode('LAND'):
                    self.state = MissionState.LANDING
                else:
                    print("⚠️ 降落模式切换失败，正在重新下发...")

        elif self.state == MissionState.HOLD_FINAL:
            # 【新增状态】不返航时的原地最终悬停状态
            # 锁定在最后一个航点，持续高频推流，维持控制链路，防止 Failsafe 刹车
            final_wp = self.waypoints[-1] if self.waypoints else {'x': self.home_x, 'y': self.home_y,
                                                                  'z': self.target_altitude, 'yaw': 0}
            self.drone.send_target_position(final_wp['x'], final_wp['y'], final_wp['z'], final_wp['yaw'])
            print(
                f"\r[挂起] 无人机保持在最终航点悬停。坐标:({final_wp['x']}, {final_wp['y']}, {final_wp['z']}) | 等待人工接管中...",
                end="", flush=True)

        elif self.state == MissionState.LANDING:
            # 判断是否已经安全着陆并自动上锁
            if not self.drone.is_armed:
                print("\n[完成] 无人机已安全着陆并自动上锁，任务圆满结束！")
                self.state = MissionState.FINISHED
            else:
                print("\r[降落中] 正在下降着陆中...", end="", flush=True)

        elif self.state == MissionState.FINISHED:
            return False

        return True