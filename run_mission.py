import time
import json
from drone_controller.base_control import DroneController
from drone_controller.mission_manager import MissionManager, FailsafeTriggered

# 1. 定义室内台架或室内试验场的相对局部航点 (X=北, Y=东, Z=地, yaw=偏航角)
# 注意：NED 坐标系下，Z轴向上为负数！
flight_waypoints = [
    {'x': 1.0, 'y': 0.0, 'z': -1.5, 'yaw': 0},  # 航点 1：向前 1 米，维持高度 1.5 米
    {'x': 1.0, 'y': 1.0, 'z': -1.5, 'yaw': 90},  # 航点 2：向右平移 1 米，机头转向正东
    {'x': 0.0, 'y': 0.0, 'z': -1.2, 'yaw': 0},  # 航点 3：回到原点上方，高度降到 1.2 米
]

def load_config(config_path="./config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 加载配置
    config = load_config()
    cfg = config["flight_control"]
    # 2. 初始化底层 SDK 驱动
    uav = DroneController(connection_string=cfg.get("connection_string"), baud=cfg.get("baud_rate"))
    if not uav.connect():
        print("物理串口打开失败，请检查数传连接或端口权限。")
        return

    # 3. 实例化任务管理器 (设置到达半径为0.3米，每个点悬停3秒)
    manager = MissionManager(drone=uav, waypoints=flight_waypoints, target_altitude=1.5, arrival_radius=0.3,
                             hold_duration=3.0)

    print("=================== 巡航控制系统启动 ===================")
    try:
        # 高频业务控制主循环 (10Hz)
        while True:
            # 轮询状态机
            is_finished = manager.update()
            if is_finished:
                print("=================== 整个巡航任务已安全退出 ===================")
                break

            time.sleep(0.1)  # 严格的 10Hz 轮询，配合管理器内部 5Hz 下发，完美防失控

    except FailsafeTriggered as e:
        # 4. 最强安全保护：一旦遥控器切出模式，代码在这里捕获，直接结束整个 Python 进程
        print(f"\n🚨 [主循环紧急熔断] 系统安全退出: {e}。香橙派已处于安全挂起状态，不占用串口输出。")
    except KeyboardInterrupt:
        print("\n用户手动通过 Ctrl+C 终止了上位机脚本。")
    finally:
        uav.running = False  # 销毁并关闭底层遥测监听守护线程


if __name__ == '__main__':
    main()