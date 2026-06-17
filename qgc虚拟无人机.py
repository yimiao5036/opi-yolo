import time
from pymavlink import mavutil

# 1. 连接到本地的 QGC
# QGC 启动后会自动监听 UDP 14550 端口，这里该端口发送数据
# source_system=1 表示伪装成系统 ID 为 1 的飞机
gc_connection = mavutil.mavlink_connection('udpout:localhost:14550', source_system=1, source_component=1)

# 2. 定义你想在地图上呈现的点的坐标（这里以北京天安门为例）
latitude = 39.9042        # 纬度
longitude = 116.4074      # 经度
altitude = 50.0           # 高度 (米)

print(f"虚拟无人机已启动，正在向 QGC 发送坐标: {latitude}, {longitude}")
print("请打开 QGC 地图查看...")

try:
    while True:
        # 3. 发送心跳包（必须发送，否则 QGC 不会理会后续数据）
        # MAV_TYPE_QUADROTOR 表示伪装成一架四轴飞行器
        # MAV_AUTOPILOT_ARDUPILOT 表示伪装成 ArduPilot 飞控
        gc_connection.mav.heartbeat_send(
            2,
            3,
            0, 0, 0
        )

        # 4. 发送 GPS 原始数据（GLOBAL_POSITION_INT）
        # MAVLink 协议要求经纬度乘以 1e7（10000000）转换为整数
        gc_connection.mav.global_position_int_send(
            time_boot_ms=int(time.time() * 1000) & 0xFFFFFFFF,
            lat=int(latitude * 1e7),
            lon=int(longitude * 1e7),
            alt=int(altitude * 1000),           # 毫米
            relative_alt=int(altitude * 1000),  # 相对高度
            vx=0, vy=0, vz=0,                   # 速度为 0
            hdg=0                               # 朝向正北
        )

        # 每秒发送一次
        time.sleep(1)

except KeyboardInterrupt:
    print("\n已停止发送。")