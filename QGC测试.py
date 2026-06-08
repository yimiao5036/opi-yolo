import time
import math
from drone_controller.qgc_router import QGCMissionReceiver

def gps_to_ned(lon, lat, alt, home_lon, home_lat):
    """
    仅基于经纬度计算东向(E)和北向(N)，忽略高度。
    home_lon, home_lat: 原点经度、纬度
    lon, lat: 目标点经度、纬度
    返回 (E, N) 单位：米
    """
    from pyproj import Transformer
    # 1. 创建 WGS84 经纬度 -> ECEF 的转换器（高度固定为0）
    transformer = Transformer.from_crs(
        {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        always_xy=True
    )

    # 2. 目标点和 home 点的 ECEF 坐标（alt=0）
    x_t, y_t, z_t = transformer.transform(lon, lat, 0)
    x_h, y_h, z_h = transformer.transform(home_lon, home_lat, 0)

    # 3. 差值
    dx, dy, dz = x_t - x_h, y_t - y_h, z_t - z_h

    # 4. 旋转到 ENU（只取 E,N）
    phi = math.radians(home_lat)
    lam = math.radians(home_lon)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    E = -sin_lam * dx + cos_lam * dy
    N = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    U = -alt

    return E, N, U

def main():
    HOME_LAT = 34
    HOME_LONG = 113

    # 创建一个用于存放本地局部坐标的列表
    points = []

    qgc_receiver = QGCMissionReceiver(connection_str="udpin:0.0.0.0:14550", home_lat=HOME_LAT, home_lon=HOME_LONG)
    qgc_receiver.start()

    print("\n[MainLoop] 核心控制线程（线程2）启动，等待地面站下发任务...")

    try:
        while True:
            current_mission = qgc_receiver.get_waypoints()

            if not current_mission:
                # 列表为空，原地挂起
                print('[MainLoop] 当前无航点任务...', end='\r')
                time.sleep(1.0)
                continue

            print(f"\n[MianLoop] 检测到新航线任务! 包含{len(current_mission)} 个航点, 准备进行香橙派自动巡点飞行控制。。。")

            for wp in current_mission:
                y, x, z = gps_to_ned(*wp,home_lon=HOME_LONG, home_lat=HOME_LAT)
                points.append({'x': x, 'y': y, 'z': z, 'yaw':0})
                print(f"--> 状态机驱动：开始前往目标航点 #{wp['seq']} -> 本地坐标 NED: X={x:.2f}米, Y={y:.2f}米, Z={z:.2f}米")

                time.sleep(3.0)
                print(f"--> 航点 #{wp['seq']} 任务处理完毕！")

            print("[MainLoop] 航线列表中所有航点全部执行闭环,任务完毕,开始清空队列...")
            print(points)
            qgc_receiver.clear_waypoints()
            points.clear()
    
    except KeyboardInterrupt:
        qgc_receiver.stop()
        
if __name__ == '__main__':
    main()