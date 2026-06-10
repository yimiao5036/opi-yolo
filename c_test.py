import time
from pymavlink import mavutil

# 根据你的实际 UDP 配置修改
# 如果飞控是作为 UDP 服务端（监听），则 Python 应作为客户端连接：
# connection_string = "udpout:飞控IP:14550"
# 如果飞控是作为 UDP 客户端（主动连接 Python），则 Python 应作为服务端监听：
connection_string = "udpin:127.0.0.1:14550"

master = mavutil.mavlink_connection(connection_string)

print("等待飞控心跳包...")
try:
    master.wait_heartbeat(timeout=5)
    print(f"收到心跳，系统 ID: {master.target_system}, 组件 ID: {master.target_component}")
except:
    print("超时：未收到飞控心跳包，请检查 IP 和端口配置以及飞控端是否在发送心跳")