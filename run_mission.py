import time
from drone_controller.base_control import DroneController
from drone_controller.mission_manager import MissionManager

def main():
    # 1. 实例化飞控底层驱动 SDK
    # 注意：根据现场情况修改你的串口和波特率
    drone = DroneController(connection_string='/dev/ttyUSB0', baud=115200)
    
    try:
        # 启动底层数据链连接与异步状态遥测线程
        drone.connect()
        
        # 2. 规划室内巡点航点列表 (NED 相对局部坐标系)
        # 规则说明：X-北为正, Y-东为正, Z-向下为正（所以想要在 1.5 米高度飞，Z 必须为 -1.5）
        # yaw：偏航角角度(0-360度)，0代表正北
        my_waypoints = [
            {'x': 1.0,  'y': 0.0,  'z': -1.5, 'yaw': 0},   # 点1：向正前方推进1米
            {'x': 1.0,  'y': 1.0,  'z': -1.5, 'yaw': 90},  # 点2：保持前推，向右平移1米，机头偏向正东
            {'x': 0.0,  'y': 1.0,  'z': -1.5, 'yaw': 180}, # 点3：倒飞回起点水平线，机头朝向正南
            {'x': 0.0,  'y': 0.0,  'z': -1.5, 'yaw': 270}  # 点4：返回起飞原点，机头朝向正西
        ]
        
        # 3. 初始化任务管理器模块
        mission = MissionManager(
            drone=drone,
            waypoints=my_waypoints,
            target_altitude=1.5,    # 自动起飞高度 1.5 米
            arrival_radius=0.25,    # 判定到点的欧氏距离误差阈值锁定在 25 厘米以内
            hold_duration=4.0       # 每个点到达后悬停停留 4 秒
        )
        
        print("\n================ 自动巡点系统就绪 ================")
        print("提示: 现场安全员请时刻保持遥控器手柄在握，一旦发生异常，切出控制模式即可随时越权人工接管。")
        print("================================================\n")
        
        # 4. 高频业务轮询主循环
        while True:
            # 驱动任务状态机
            is_mission_finished = mission.update()
            
            if is_mission_finished:
                print("\n[通知] 主程序检测到巡航任务模块已正常运行结束，退出程序。")
                break
                
            # 20Hz 的控制频率（50ms一拍），既能保证位置控制实时下发，又不会拖垮处理器
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n🛑 用户手动中止程序 (Ctrl+C)！正在触发紧急安全退出。")
    finally:
        # 5. 清理与释放资源
        print("[清理] 正在关闭后台遥测数据链线程...")
        drone.disconnect()
        print("[清理] 程序已安全解脱。")

if __name__ == '__main__':
    main()