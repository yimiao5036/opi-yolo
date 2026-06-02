import time
import math
import threading
from pymavlink import mavutil


class DroneController:
    def __init__(self, connection_string='/dev/ttyUSB0', baud=115200):
        self.connection_string = connection_string
        self.baud = baud
        self.master = None
        self.is_connected = False

        # 飞控实时状态变量（多线程共享）
        self.current_mode = None
        self.is_armed = False
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # 线程控制开关
        self.running = False
        self.telemetry_thread = None

    def connect(self):
        """初始化 MAVLink 串口连接，并等待首个心跳包"""
        try:
            print(f"正在尝试连接飞控 ({self.connection_string})...")
            self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud)
            self.master.wait_heartbeat(timeout=5)
            self.is_connected = True
            print("飞控连接成功，已接收到首个心跳包！")

            # 开启后台线程，负责高频监听状态
            self.running = True
            self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self.telemetry_thread.start()
            return True
        except Exception as e:
            print(f"连接飞控失败: {e}")
            self.is_connected = False
            return False

    def _telemetry_loop(self):
        """后台线程：高频高吞吐量接收飞控状态反馈，刷新状态变量"""
        while self.running and self.is_connected:
            try:
                # 阻塞式接收，超时 0.1 秒
                msg = self.master.recv_match(blocking=True, timeout=0.1)
                if not msg:
                    continue

                msg_type = msg.get_type()

                # 1. 解析心跳包获取模式与解锁状态
                if msg_type == 'HEARTBEAT':
                    mode_id = msg.custom_mode
                    self.is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

                    # 尝试反向查找模式名称
                    if self.master:
                        # 寻找数值对应的模式字符串
                        for name, num in self.master.mode_mapping().items():
                            if num == mode_id:
                                self.current_mode = name
                                break

                # 2. 解析局部 NED 位置坐标
                elif msg_type == 'LOCAL_POSITION_NED':
                    self.local_x = msg.x  # 北 (North)
                    self.local_y = msg.y  # 东 (East)
                    self.local_z = msg.z  # 地 (Down)

                # 3. 解析姿态角 (弧度转角度)
                elif msg_type == 'ATTITUDE':
                    self.roll = math.degrees(msg.roll)
                    self.pitch = math.degrees(msg.pitch)
                    self.yaw = math.degrees(msg.yaw)

            except Exception as e:
                print(f"\n遥测线程捕获到异常: {e}")
                time.sleep(0.1)

    def set_mode(self, mode_name):
        """切换飞控模式 (例如 GUIDED 或 OFFBOARD)"""
        if not self.is_connected: return False
        print(f"正在尝试切换到模式: {mode_name}...")

        # 将模式名称转换为飞控可识别的内部数值
        if mode_name not in self.master.mode_mapping():
            print(f"错误,飞控不支持{mode_name}模式")
            return False

        mode_id = self.master.mode_mapping()[mode_name]
        self.master.set_mode(mode_id)
        print(f"已发送模式切换指令 -> {mode_name}")
        return True

    def arm(self):
        """发送解锁指令"""
        if not self.is_connected: return False
        print("正在发送解锁指令...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

        # 等待并捕获飞控的 ACK 响应，超时时间设为 3 秒
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3.0)
        if ack and ack.command == mavutil.mavlink.MAV_COMMAND_ACK:
            if ack.result == 0:
                print(" ✅ [飞控通信] 飞控解锁成功 (Armed Success)！")
            else:
                print(f" ❌ [飞控通信] 飞控拒绝了解锁请求，拒绝代码 (Result): {ack.result}")
        else:
            print(" ⚠️ [飞控通信] 飞控未响应解锁确认，请检查飞控状态或数传指示灯。")
        return True

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
        核心控制接口：发送机体坐标系（Body Frame）下的速度指令
        注意：在下发控制前，如果安全员人为切出了自主模式，则该函数拒绝发送，确保人肉兜底。
        yaw_rate:单位（弧度/秒，rad/s）,后续使用得转换为角度
        """
        if not self.is_connected: return

        if self.current_mode not in ['GUIDED', 'OFFBOARD']:
            return

        # 坐标系掩码定义：使用机体坐标系 (MAV_FRAME_BODY_NED)
        # 掩码屏蔽：只保留 vx, vy, vz 和偏航速率 yaw_rate
        coordinate_frame = mavutil.mavlink.MAV_FRAME_BODY_NED
        type_mask = 0b0000111111000111  # 仅开放速度和偏航率

        self.master.mav.set_position_target_local_ned_send(
            0,  # 系统时间戳（可传0）
            self.master.target_system,
            self.master.target_component,
            coordinate_frame,
            type_mask,
            0, 0, 0,  # 位置 (屏蔽)
            vx, vy, vz,  # 速度 (机体前后、机体左右、机体上下)
            0, 0, 0,  # 加速度 (屏蔽)
            0, yaw_rate  # 偏航，偏航率
        )

    def print_status(self):
        """利用 \r 实现终端单行丝滑刷新，动态打印 PID 命令与真实状态对比"""
        print(
            f"\r[Status] Armed: {self.is_armed} | "
            f"Pos(N,E,D): ({self.local_x:+.2f}, {self.local_y:+.2f}, {self.local_z:+.2f}) | "
            f"Yaw: {self.yaw:+.1f}°\033[K",
            end="", flush=True
        )

    def takeoff(self, altitude):
        """发送起飞指令(仅在 GUIDED 或 OFFBOARD 模式下有效)"""
        if not self.is_connected: return False
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )
        print(f"已下发自动起飞指令 -> 目标高度: {altitude} 米")
        return True

    def send_target_position(self, x, y, z, yaw):
        """
        发送绝对局部位置( NED 坐标系)
        x:北向目标位置
        y:东向目标位置
        z:垂直目标位置
        """
        if not self.is_connected: return
        # 当人为切出切换模式时，香橙派不再发送命令
        if self.current_mode not in ['GUIDED', 'OFFBOARD']: return

        # 使用局部 NED 坐标系
        coordinate_frame = mavutil.mavlink.MAV_FRAME_NED

        # 掩码屏蔽:只保留位置 (x, y, z) 和偏航角 (yaw)
        type_mask = 0b0000101111111000

        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            coordinate_frame,
            type_mask,
            x, y, z,
            0, 0, 0,
            0, 0, 0,
            math.radians(yaw),
            0
        )

    def close(self):
        """释放线程，关闭串口连接"""
        self.running = False
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=1.0)
        print("\n底层控制接口已安全断开。")