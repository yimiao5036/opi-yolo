import time
from pymavlink import mavutil

# 【核心配置】连接飞控
# 这里的路径取决于香橙派和飞控是怎么连的。
# 如果是通过 USB 转 TTL 模块插在香橙派上，通常是 '/dev/ttyUSB0'
# 如果是直接连在香橙派的板载 GPIO 串口上，可能是 '/dev/ttyAMA0' 或 '/dev/ttyS5'
# 波特率通常飞控默认是 115200 或 57600
FC_CONNECTION = '/dev/ttyUSB0'
BAUD_RATE = 57600

print(f"正在尝试连接飞控 ({FC_CONNECTION})...")
master = mavutil.mavlink_connection(FC_CONNECTION, baud=BAUD_RATE)

# 等待飞控的心跳信号（确认物理链路已通）
print("正在等待飞控心跳信号 (Heartbeat)...")
master.wait_heartbeat()
print("飞控连接成功！开始获取姿态数据...\n")

try:
    while True:
        # 阻塞或等待接收 MAVLink 消息
        # ATTITUDE 消息包含了无人机的姿态（滚转、俯仰、偏航）
        msg = master.recv_match(type='ATTITUDE', blocking=True, timeout=1.0)

        if msg:
            # 飞控传出来的弧度（radians），需要转换成角度（degrees）
            import math

            roll = math.degrees(msg.roll)
            pitch = math.degrees(msg.pitch)
            yaw = math.degrees(msg.yaw)

            # 如果偏航角是负数（-180到180），可以转换成 0-360 度标准航向
            if yaw < 0:
                yaw += 360

            # 刷新打印到你的 PC SSH 终端屏幕上
            print(f"【飞控姿态】 偏航角(Yaw): {yaw:.2f}° | 俯仰角(Pitch): {pitch:.2f}° | 滚转角(Roll): {roll:.2f}°",
                  end='\r')
        else:
            print("未收到 ATTITUDE 消息，正在等待...", end='\r')

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n用户终止，退出监控。")