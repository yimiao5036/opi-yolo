import time
import threading
from pymavlink import mavutil


class QGCMissionProxy:
    def __init__(self, qgc_link):
        """
        :param qgc_link: 专门连接地面站数传电台的 mavutil 链接对象
        """
        self.qgc = qgc_link
        self.translated_waypoints = []  # 翻译出来供你本地 mission_manager 使用的航点列表
        self.is_mission_ready = False
        self.running = True

        # 锁机制防止数据脏读
        self.lock = threading.Lock()

        # 启动后台监听线程
        self.proxy_thread = threading.Thread(target=self._listen_qgc_loop, daemon=True)
        self.proxy_thread.start()

    def _listen_qgc_loop(self):
        """高频监听地面站发来的 MAVLink 消息"""
        print("[QGC Proxy] 拦截网关已启动，等待地面站上传航线...")

        while self.running:
            try:
                # 阻塞读取来自 QGC 的消息，超时时间为0.5秒
                msg = self.qgc.recv_match(blocking=True, timeout=0.5)
                if not msg:
                    continue

                msg_type = msg.get_type()

                # 1. 拦截：收到地面站宣告的航点总数
                if msg_type == 'MISSION_COUNT':
                    self._handle_mission_count(msg)

                # 2. 拦截：收到地面站发来的开始执行命令
                elif msg_type == 'COMMAND_LONG':
                    if msg.command == mavutil.mavlink.MAV_CMD_MISSION_START:
                        self._handle_mission_start()
                    elif msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                        print(f"[QGC Proxy] 拦截到地面站解锁指令 -> 参数: {msg.param1}")

            except Exception as e:
                print(f"[QGC Proxy] 接收异常: {e}")
                time.sleep(0.1)

    def _handle_mission_count(self, msg):
        """处理航点上传握手协议"""
        target_system = msg.target_system
        target_component = msg.target_component
        wp_count = msg.count

        print(f"[QGC Proxy] 检测到 QGC 正在上传航线，包含航点数量: {wp_count}")

        temp_waypoints = []
        handshake_success = True

        # 依次向 QGC 请求每一个航点的具体坐标
        for i in range(wp_count):
            # 向 QGC 请求第 i 个航点 (使用 MISSION_REQUEST_INT 保证高精度)
            self.qgc.mav.mission_request_int_send(
                target_system,
                target_component,
                i,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION
            )

            # 等待 QGC 回应具体的航点条目
            item_msg = self.qgc.recv_match(type='MISSION_ITEM_INT', blocking=True, timeout=2.0)

            if item_msg is None:
                print(f"[QGC Proxy] 错误: 请求航点 {i} 超时，中止握手。")
                handshake_success = False
                break

            # 解析地理/局部坐标
            # frame 1: MAV_FRAME_GLOBAL_INT (室外经纬度)
            # frame 12: MAV_FRAME_LOCAL_NED (室内/本地局部坐标)
            frame = item_msg.frame

            # 坐标转换逻辑（核心翻译部分）
            if frame == mavutil.mavlink.MAV_FRAME_LOCAL_NED:
                # 如果同伴在 QGC 规划的是室内本地 NED 坐标，直接除以 1000 转换为米
                x = item_msg.x / 1000.0
                y = item_msg.y / 1000.0
                z = item_msg.z / 1000.0  # 注意：飞控 NED 的 Z 轴向下为正
            elif frame in [mavutil.mavlink.MAV_FRAME_GLOBAL_INT, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT]:
                # 如果是室外 GPS 航点
                # x 乘以 1e-7 得到 纬度，y 乘以 1e-7 得到 经度
                lat = item_msg.x / 1e7
                lon = item_msg.y / 1e7
                alt = item_msg.z  # 高度（米）

                # 【关键生产力转换】：在此处可以调用你在本地计算的投影函数，将经纬度转换为相对起始点的 (x, y) 米
                x, y = self._gps_to_local_meters(lat, lon)
                z = -alt  # 转换为你本地代码习惯的 Z 轴定义（若你的代码向上为正，则取负值；若严格遵循 NED 则保持向下为正）
            else:
                x, y, z = 0.0, 0.0, 0.0

            # 提取航点附加动作参数（如偏航角、悬停时间）
            param1_hold_time = item_msg.param1  # 悬停时间（秒）
            param4_yaw = item_msg.param4  # 期望偏航角（度）

            # 翻译并打包成符合你本地脚本格式的字典对象
            wp_dict = {
                "index": i,
                "command": item_msg.command,  # 如 MAV_CMD_NAV_WAYPOINT
                "x": x,
                "y": y,
                "z": z,
                "hold_time": param1_hold_time if param1_hold_time > 0 else 3.0,  # 缺省悬停3秒
                "yaw": param4_yaw
            }
            temp_waypoints.append(wp_dict)
            print(f" -> 成功拦截并翻译航点 [{i}]: X={x:.2f}, Y={y:.2f}, Z={z:.2f}, 偏航={param4_yaw}°")

        if handshake_success:
            # 向 QGC 发送 MISSION_ACK，宣告上传成功，让地面站界面显示绿色的“保存成功”
            self.qgc.mav.mission_ack_send(
                target_system,
                target_component,
                mavutil.mavlink.MAV_MISSION_RESULT_ACCEPTED,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION
            )
            with self.lock:
                self.translated_waypoints = temp_waypoints
                self.is_mission_ready = True
            print("[QGC Proxy] 航线拦截握手完美结束！影子航线已在香橙派内存中就绪。")
        else:
            # 告诉 QGC 上传失败
            self.qgc.mav.mission_ack_send(
                target_system,
                target_component,
                mavutil.mavlink.MAV_MISSION_RESULT_ERROR,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION
            )

    def _gps_to_local_meters(self, lat, lon):
        """
        一个简易的平地经纬度转局部米制坐标函数。
        实际使用中，可以保存起飞点的经纬度作为参考原点（Home点）。
        """
        # 伪代码演示，你可以替换为标准的 GeographicLib 或 pyproj 投影库
        # 这里假设以固定参考点进行转换
        home_lat, home_lon = 30.000000, 120.000000
        delta_lat = lat - home_lat
        delta_lon = lon - home_lon

        # 地球每度纬度约为 111320 米
        x_meters = delta_lat * 111320.0
        # 经度长度随纬度余弦收缩
        y_meters = delta_lon * 111320.0 * 0.866
        return x_meters, y_meters

    def _handle_mission_start(self):
        """当地面站点击『开始任务』时触发该本地回调"""
        print("[QGC Proxy] 地面站发出了【START MISSION】指令！")
        # 这里用来激活你在香橙派上写的状态机
        # 例如将全局事件设置为 True：event_start_local_mission.set()

    def get_local_waypoints(self):
        """提供给你本地外部业务循环读取的接口"""
        with self.lock:
            if self.is_mission_ready:
                return self.translated_waypoints
            return None

    def close(self):
        self.running = False