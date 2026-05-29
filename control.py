import time
from pymavlink import mavutil
from pymavlink.mavextra import integral


# 自定义 PID 部分
class PID:
    """ 将 PID 部分分离出来为一个类"""

    def __init__(self, kp, ki, kd, max_out, min_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.min_out = min_out

        self.last_err = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.033  # 30Hz

        p_out = self.kp * error

        self.integral = error * dt
        i_out = self.ki * self.integral
        i_out = max(min(i_out, self.max_out * 0.2), self.min_out * 0.2)

        d_out = self.kd * (self.last_err - error) / dt

        output = p_out + i_out + d_out
        output = max(min(output, self.max_out), self.min_out)

        self.last_err = error
        self.last_time = current_time

        return output

    def reset(self):
        self.last_err = 0.0
        self.integral = 0.0


class UAVController:
    def __init__(self, port='/dev/ttyUSB0', baud=57600):
        """
        初始化与飞控的 MAVLink 串口连接
            port:串口地址
            baud:波特率
        """
        try:
            print(f"正在尝试通过 MAVLink 连接飞控:{port}...")
            self.master = mavutil.mavlink_connection(port, baud=baud)
            print(f" [飞控通信] 已连接串口{port}, 请等待心跳包...")
            self.master.wait_heartbeat()
            print(" [飞控通信] 收到飞控心跳包, 链接正常")

            # 🌟 核心改动：连接成功并确认收到心跳包后，立即开始解锁
            self.arm_drone()

        except Exception as e:
            print(f" [飞控通信] 连接失败:{e}, 请检查")
            self.master = None

        # 初始化Y轴 (左右方向) 和Z轴 (上下方向) 的 PID 调节器
        self.vy = PID(kp=1.0, ki=0.02, kd=0.3, max_out=0.8, min_out=-0.8)
        self.vz = PID(kp=1.0, ki=0.02, kd=0.3, max_out=0.4, min_out=-0.4)

        # 视觉丢失保护
        self.lost_counter = 0
        self.LOST_THRESHOLD = 30  # 假设摄像头是30帧就对应1秒

    def arm_drone(self):
        """
        🌟 核心新增：向飞控发送解锁 (ARM) 指令
        已确认：无机翼且机身已固定，台架安全测试。
        """
        if self.master is None:
            print(" [飞控通信] 错误：飞控未连接，无法发送解锁指令。")
            return

        print(" [⚠️安全提示] 正在向飞控下发解锁命令（确认无桨叶且已固定）...")

        # 使用 command_long_send 发送 MAV_CMD_COMPONENT_ARM_DISARM 消息
        # param1: 1 代表解锁 (ARM), 0 代表上锁 (DISARM)
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # 确认号 (Confirmation)
            1,  # param1: 1 = ARM (解锁)
            0,  # param2: 强制解锁参数（默认0，部分固件若强制防呆需填21196）
            0, 0, 0, 0, 0  # param3 ~ param7 (未定义，填0)
        )

        # 等待并捕获飞控的 ACK 响应，超时时间设为 3 秒
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3.0)
        if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if ack.result == 0:
                print(" ✅ [飞控通信] 飞控解锁成功 (Armed Success)！")
            else:
                print(f" ❌ [飞控通信] 飞控拒绝了解锁请求，拒绝代码 (Result): {ack.result}")
                print(
                    " 👉 提示：可能触发了飞控内部的解锁安全检查（如GPS、遥控器信号未就绪），室内台架测试请在地面站将 'ARMING_CHECK' 临时设为 0。")
        else:
            print(" ⚠️ [飞控通信] 飞控未响应解锁确认，请检查飞控状态或数传指示灯。")

    def send_body_velocity_cmd(self, vx, vy, vz):
        """
        使用 MAVLink 标准消息向飞控发送【机体坐标系下】的三个轴向移动速度。
        👉 坐标系规定 (MAV_FRAME_BODY_OFFSET_NED):
           vx (+) 前飞 / (-) 后退
           vy (+) 右移 / (-) 左移
           vz (+) 下降 / (-) 上升  (注意：NED 坐标系下 Z 轴向下为正)
        """
        if self.master is None: return

        # 核心：定义掩码（Type Mask）
        # 16 位整型，每一位用来“忽略”某些参数。
        # 我们只想要控制 vx, vy, vz 速度，因此需要将其余的 位置(Position)、加速度(Acceleration) 和 偏航(Yaw) 标志位掩蔽屏蔽掉。
        # 二进制：0b0000111111000111 -> 十进制: 4039
        type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )

        self.master.mav.set_position_target_local_ned_send(
            0,  # 系统时间戳（填0即可）
            self.master.target_system,  # 目标系统 ID
            self.master.target_component,  # 目标组件 ID
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # 🌟 关键：指定控制基于“机体本地坐标系”
            type_mask,  # 掩码配置
            0, 0, 0,  # x, y, z 位置 (被忽略)
            vx, vy, vz,  # 🌟 实际生效的机体期望物理速度 (m/s)
            0, 0, 0,  # ax, ay, az 加速度 (被忽略)
            0, 0  # yaw, yaw_rate (被忽略)
        )

    def update_control(self, has_target, err_x=0.0, err_y=0.0):
        """
        接收检测结果并更新控制输出
        """

        if has_target:
            self.lost_counter = 0

            # 1. 计算机体坐标系的各速度分量
            # 气球偏右 -> err_x > 0 -> 无人机右移对准 -> vy 需要正数
            vy_cmd = self.vy.update(err_x)
            # 若气球偏下 -> err_y > 0 -> 画面中视野偏低表示无人机相对较高，需向下对齐气球。
            # 根据 MAVLink 机体 NED 坐标系：Z轴向下为正，故向上爬升（UP）vz 必须是【负数】。
            raw_vz = self.vz.update(err_y)
            vz_cmd = raw_vz * 1.0
            # 初始追踪先原地锁定悬停对准，前飞速度 vx_cmd 设为 0。调试稳定后再缓慢给 0.1~0.2
            vx_cmd = 0.0

            print(
                f"\r[PID 控制速度] 🎯目标:已锁定 | 误差(X:{err_x:+.2f}, Y:{err_y:+.2f}) | 期望速度 -> Vx(前): {vx_cmd:+.2f} m/s, Vy(右): {vy_cmd:+.2f} m/s, Vz(下): {vz_cmd:+.2f} m/s",
                end="", flush=True)
            self.send_body_velocity_cmd(vx_cmd, vy_cmd, vz_cmd)
        else:
            self.lost_counter += 1
            if self.lost_counter >= self.LOST_THRESHOLD:
                # 🚨 安全锁：长时间丢失目标，清除 PID 内部积分，强制发送 0 速度原地刹车悬停
                self.vy.reset()
                self.vz.reset()
                print(f"🚨 警告: 目标丢失超过 {self.LOST_THRESHOLD} 帧！MAVLink 发送安全原地锁死悬停指令 🔒")
                self.send_body_velocity_cmd(0.0, 0.0, 0.0)