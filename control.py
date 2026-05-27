import time
from pymavlink import mavutil

class UAVController:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        """初始化与飞控的 MAVLink 串口连接"""
        try:
            self.master = mavutil.mavlink_connection(port, baud=baud)
            print(f" [飞控通信] 已连接串口{port}, 请等待心跳包...")
            self.master.wait_heartbeat()
            print(" [飞控通信] 收到飞控心跳包, 链接正常")
        except Exception as e:
            print(f" [飞控通信] 连接失败:{e}, 请检查")

        # =================== PID 参数 ===================
        # 此时参数设定，err_x: 无人机的左右平移, err_y: 无人机的上下平移 (后续还要修改)
        # ===============================================
        self.kp = 1.2   # 响应强度
        self.ki = 0.02  # 消除风偏
        self.kd = 0.3   # 刹车减震

        # 内部缓存
        self.last_err_x = 0.0
        self.last_err_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_time = time.time()

    def update_and_send_command(self, err_x, err_y):
        """通过输入图像误差, 计算 PID 并通过 MAVLink 发送指令"""
        if self.master is None:
            return

        current_time = time.time()
        dt = current_time - self.last_time
        if dt <=0: dt = 0.02 #防止除以0

        # ------------ PID 计算 -------------
        # 1. 比例项 (p)
        p_out_x = self.kp * err_x
        p_out_y = self.ki * err_y

        # 2. 积分项 (I)
        self.integral_x = max(-0.3, min(self.integral_x + err_x * dt, 0.3))
        self.integral_y = max(-0.3, min(self.integral_y + err_y * dt, 0.3))
        i_out_x = self.ki * self.integral_x
        i_out_y = self.kd * self.integral_y

        # 3. 微分项 (D)
        d_out_x = self.kp * (err_x - self.last_err_x) / dt
        d_out_y = self.kp * (err_y - self.last_err_y) / dt

        # 4. 总输出 (映射为物理速度: m/s)
        vel_x_cmd = p_out_x + i_out_x + d_out_x
        vel_y_cmd = p_out_y + i_out_y + d_out_y

        # ------------- 安全限速 --------------
        # 限制下速度，怕测试时飞太快
        MAX_VEL = 1.0
        vel_x_cmd = max(min(MAX_VEL, vel_x_cmd), -MAX_VEL)
        vel_y_cmd = max(min(MAX_VEL, vel_y_cmd), -MAX_VEL)

        # 更新缓存
        self.last_err_x = err_x
        self.last_err_y = err_y
        self.last_time = current_time

        # ------------- 坐标系映射 -------------
        # 图像 X 轴正方向对应无人机机体坐标系的右侧平移速度 (vy)
        # 图像 Y 轴正方向对应无人机(此时对应前视相机)下降速度 (vz)
        vy = vel_x_cmd      # 图像右侧，往右飞
        vz = vel_y_cmd * 1.0# 图像下方，往下飞, (乘1.0表示向下为正)
        vx = 0.5            # 无人机以 0.5 m/s 持续前进

        # ---------- 发送 MAVLink 指令 --------
        # 飞控处于 OFFBOARD (PX4) 模式才能响应此指令
        self.master.mav.set_position_target_local_ned_send(
            0,                                          # 系统启动时间戳（填 0 即可）
            self.master.target_system,                  # 目标系统 ID（自动获取）
            self.master.target_component,               # 目标组件 ID（自动获取）
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # 选择“机体坐标系”（以无人机当前朝向为基准）
            0b0000111111000111,                         # 位掩码控制：二进制代表“忽略位置和加速度，只控制速度”
            0, 0, 0,                                    # 位置 X, Y, Z (被掩码忽略)
            vx, vy, vz,                                 # 实际控制物理速度 (单位: 米/秒): 前进, 右移, 下降
            0, 0, 0,                                    # 加速度 (被掩码忽略)
            0, 0                                        # 偏航角, 偏航角速度 (被掩码忽略)
        )
        print(f" MAVLink -> 前进:{vx:.2f} m/s, 右移:{vy:.2f} m/s, 下降:{vz:.2f} m/s")

    def send_hover_cmd(self):
        """ 目标丢失保护: 发送指令, 让无人机原地悬停"""
        if self.master is None: return
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            0b0000111111000111,
            0, 0, 0,
            0.0, 0.0, 0.0,
            0, 0, 0,
            0, 0
        )
        print("[安全兜底] 目标跟丢或未准许控制！已向飞控发送原地刹车悬停指令。")
