import time
import math
import threading
from pymavlink import mavutil


class DroneController:
    def __init__(self, connection_string='udpout:127.0.0.1:14550', baud=115200):
        self.connection_string = connection_string
        self.baud = baud
        self.master = None
        self.is_connected = False

        # 线程锁：保护多线程共享的遥测数据
        self.data_lock = threading.Lock()

        # 飞控实时状态变量（受锁保护）
        self._current_mode = None
        self._is_armed = False
        self._local_x = 0.0
        self._local_y = 0.0
        self._local_z = 0.0
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        # 无人机电量信息
        self._battery_voltage = 0.0
        self._battery_remaining = -1

        # 看门狗与异步事件
        self.last_heartbeat_time = 0.0
        self.arm_ack_event = threading.Event()
        self.arm_ack_result = -1

        # 线程控制开关
        self.running = False
        self.telemetry_thread = None

    # --- 线程安全的数据访问属性 (Properties) ---
    @property
    def current_mode(self):
        with self.data_lock: return self._current_mode

    @property
    def is_armed(self):
        with self.data_lock: return self._is_armed

    @property
    def local_position(self):
        with self.data_lock: return self._local_x, self._local_y, self._local_z

    @property
    def yaw(self):
        with self.data_lock: return self._yaw

    def connect(self):
        """初始化 MAVLink 串口连接，并等待首个心跳包"""
        try:
            print(f"正在尝试连接飞控 ({self.connection_string})...")
            self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud)
            self.master.wait_heartbeat(timeout=5)
            self.is_connected = True
            self.last_heartbeat_time = time.time()
            print("飞控连接成功，已接收到首个心跳包！")

            # 开启后台线程
            self.running = True
            self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self.telemetry_thread.start()
            return True
        except Exception as e:
            print(f"连接飞控失败: {e}")
            self.is_connected = False
            return False

    def _telemetry_loop(self):
        """后台线程：高频接收飞控状态反馈，并包含看门狗与ACK劫持"""
        while self.running:
            if not self.is_connected:
                # 触发断线自动重连机制
                print("⚠️ 检测到连接断开，尝试重新连接...")
                if self.connect():
                    time.sleep(1.0)
                else:
                    time.sleep(2.0)
                continue

            try:
                # 阻塞式接收
                msg = self.master.recv_match(blocking=True, timeout=0.1)
                if not msg:
                    # 心跳看门狗：超过5秒没收到任何包，判定掉线
                    if time.time() - self.last_heartbeat_time > 5.0:
                        print("\n🚨 [警告] 飞控心跳超时！数传或串口可能已断开！")
                        self.is_connected = False
                    continue

                msg_type = msg.get_type()

                # 1. 解析心跳包
                if msg_type == 'HEARTBEAT':
                    self.last_heartbeat_time = time.time()
                    mode_id = msg.custom_mode
                    is_armed_status = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

                    # 反向查找模式名称
                    matched_mode = None
                    for name, num in self.master.mode_mapping().items():
                        if num == mode_id:
                            matched_mode = name
                            break

                    with self.data_lock:
                        self._current_mode = matched_mode
                        self._is_armed = is_armed_status

                # 2. 解析局部 NED 位置坐标
                elif msg_type == 'LOCAL_POSITION_NED':
                    with self.data_lock:
                        self._local_x = msg.x  # 北 (North)
                        self._local_y = msg.y  # 东 (East)
                        self._local_z = msg.z  # 地 (Down, 负数代表无人机在空中)

                # 3. 解析姿态角
                elif msg_type == 'ATTITUDE':
                    with self.data_lock:
                        self._roll = math.degrees(msg.roll)
                        self._pitch = math.degrees(msg.pitch)
                        self._yaw = math.degrees(msg.yaw)

                # 4. 劫持命令响应 ACK (解决原代码抢不到包的 Bug)
                elif msg_type == 'COMMAND_ACK':
                    if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                        self.arm_ack_result = msg.result
                        self.arm_ack_event.set()  # 通知主线程解锁结果已到

                elif msg_type == 'BATTERY_STATUS':
                    with self.data_lock:
                        self._battery_voltage = msg.voltages[0] / 1000.0    # 毫伏转伏特
                        self._battery_remaining = msg.battery_remaining     # 剩余百分比 %

            except Exception as e:
                print(f"\n遥测线程捕获到异常: {e}")
                time.sleep(0.1)

    def set_mode(self, mode_name):
        """切换飞控模式 (例如 GUIDED 或 OFFBOARD)"""
        if not self.is_connected: return False

        if mode_name not in self.master.mode_mapping():
            print(f"错误: 飞控不支持 {mode_name} 模式")
            return False

        mode_id = self.master.mode_mapping()[mode_name]

        # 连续发送3次，确保机载端切换命令的送达率
        for _ in range(3):
            self.master.set_mode(mode_id)
            time.sleep(0.05)
            if self.current_mode == mode_name:
                print(f" ✅ 成功切换到模式: {mode_name}")
                return True

        print(f" 已发送模式切换指令 -> {mode_name} (等待飞控确认中)")
        return True

    def arm(self):
        """发送解锁指令（采用线程安全的事件驱动等待机制）"""
        if not self.is_connected: return False
        print("正在发送解锁指令...")

        self.arm_ack_event.clear()  # 重置事件信号
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

        # 等待后台遥测线程触发通知，超时 3.0 秒
        if self.arm_ack_event.wait(timeout=3.0):
            if self.arm_ack_result == 0:
                print(" ✅ [飞控通信] 飞控解锁成功 (Armed Success)！")
                return True
            else:
                print(f" ❌ [飞控通信] 飞控拒绝解锁，错误码 (Result): {self.arm_ack_result}")
        else:
            print(" ⚠️ [飞控通信] 飞控未响应解锁确认，请检查安全开关或地面站报错。")
        return False

    def disarm(self):
        """发送上锁指令"""
        if not self.is_connected: return False
        print("正在发送强制上锁指令...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        return True

    def send_body_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """
        机体坐标系（Body Frame）下的速度控制
        注意：NED坐标系中，vz正数代表下降！为了符合直觉，此处将输入的vz取反（即正数代表上升）。
        """
        if not self.is_connected: return
        if self.current_mode not in ['GUIDED', 'OFFBOARD']: return

        coordinate_frame = mavutil.mavlink.MAV_FRAME_BODY_NED
        type_mask = 0b0000111111000111  # 仅开放速度和偏航率

        # vz 取反：转换为符合直觉的（+为上升，-为下降）
        mavlink_vz = -vz

        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            coordinate_frame, type_mask,
            0, 0, 0,  # 位置 (屏蔽)
            vx, vy, mavlink_vz,  # 速度 (前后、左右、上下)
            0, 0, 0,  # 加速度 (屏蔽)
            0, yaw_rate  # 偏航，偏航率 (rad/s)
        )

    def takeoff(self, altitude):
        """发送起飞指令(自动帮用户先切入控制模式)"""
        if not self.is_connected: return False

        # 自动兜底：强制切入自主模式，否则飞控不响应起飞指令
        if self.current_mode not in ['GUIDED', 'OFFBOARD']:
            # 优先尝试 GUIDED (ArduPilot)，不行则 OFFBOARD (PX4)
            target_m = 'GUIDED' if 'GUIDED' in self.master.mode_mapping() else 'OFFBOARD'
            self.set_mode(target_m)
            time.sleep(0.2)

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )
        print(f"已下发自动起飞指令 -> 目标高度: {altitude} 米")
        return True

    def send_target_position(self, x, y, z, yaw):
        """发送绝对局部位置 (NED 坐标系)"""
        if not self.is_connected: return
        if self.current_mode not in ['GUIDED', 'OFFBOARD']: return

        coordinate_frame = mavutil.mavlink.MAV_FRAME_NED
        type_mask = 0b0000101111111000  # 只保留位置 (x, y, z) 和偏航角 (yaw)

        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            coordinate_frame, type_mask,
            x, y, z,
            0, 0, 0,
            0, 0, 0,
            math.radians(yaw), 0
        )

    def send_relative_position(self, delta_x, delta_y, delta_z, yaw_offset=0.0):
        """
        增量式（相对当前位置）位置控制
        delta_x: 前向偏移量 (米)
        delta_y: 右向偏移量 (米)
        delta_z: 下向偏移量 (米，注意：若要相对当前高度爬升，需传负数)
        """
        if not self.is_connected or self.current_mode not in ['GUIDED', 'OFFBOARD']: return

        coordinate_frame = mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED
        type_mask = 0b0000101111111000  # 只开放位置和偏航角

        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            coordinate_frame, type_mask,
            delta_x, delta_y, delta_z,
            0, 0, 0,
            0, 0, 0,
            math.radians(yaw_offset), 0
        )

    def condition_yaw(self, target_angle, speed=20, direction=1, relative=0):
        """
        控制无人机偏航角 (Yaw)
        target_angle: 目标角度 (0-360 度)
        speed: 旋转速度 (度/秒)
        direction: 1 为顺时针，-1 为逆时针
        relative: 0 树立为绝对航向（0度为正北），1 树立为相对当前航向的偏移
        """
        if not self.is_connected: return False

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            target_angle,  # param 1: 目标角度
            speed,  # param 2: 旋转速度
            direction,  # param 3: 方向
            relative,  # param 4: 相对还是绝对
            0, 0, 0  # param 5-7: 未使用
        )
        return True

    def print_status(self):
        """利用 \r 实现终端单行丝滑刷新"""
        x, y, z = self.local_position
        print(
            f"\r[Status] Mode: {self.current_mode} | Armed: {self.is_armed} | "
            f"Pos(N,E,D): ({x:+.2f}, {y:+.2f}, {z:+.2f}) | "
            f"Yaw: {self.yaw:+.1f}°\033[K |"
            f"Batt: {self._battery_voltage}({self._battery_remaining})",
            end="", flush=True
        )

    def return_to_launch(self):
        """触发飞控自主返航 (RTL / Return to Launch)"""
        print("[特殊情况] 触发自主返航流程！")
        return self.set_mode('RTL')

    def land(self):
        """原地垂直降落"""
        print("[特殊情况] 触发原地降落！")
        return self.set_mode('LAND')

    def close(self):
        """释放线程，关闭串口连接"""
        self.running = False
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=1.0)
        print("\n底层控制接口已安全断开。")