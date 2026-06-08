import threading
import time
from pymavlink import mavutil


class QGCMissionReceiver:
    def __init__(self, connection_str='udpin:0.0.0.0:14550', sys_id=1, comp_id=1, home_lat=34.0, home_lon=113.0):
        """
        QGC 航线拦截接收器 (模拟飞控节点)
        :param connection_str: pymavlink连接字符串。udpin:0.0.0.0:14550 表示作为本地服务器等待QGC主动连接
        :param sys_id: 伪装的系统ID (通常飞控为1)
        :param comp_id: 伪装的组件ID (通常飞控为1)
        :param home_lat: 模拟的起始点纬度（用于在没有真实GPS的室内环境让QGC地图定点）
        :param home_lon: 模拟的起始点经度
        """
        self.connection_str = connection_str
        self.sys_id = sys_id
        self.comp_id = comp_id
        self.home_lat = home_lat
        self.home_lon = home_lon

        # 初始化 MAVLink 链路
        self.master = mavutil.mavlink_connection(
            self.connection_str,
            source_system=self.sys_id,
            source_component=self.comp_id
        )

        # 线程安全与航点存储
        self.lock = threading.Lock()
        self.waypoints_list = []  # 正式供外部控制脚本读取的航点列表
        self._temp_waypoints = []  # 协议传输过程中的临时缓存缓存

        # 协议状态控制变量
        self.is_running = False
        self.expected_count = 0
        self.current_seq = 0

    def start(self):
        """启动后台模拟线程"""
        self.is_running = True

        # 线程 1-A: 维持心跳与状态遥测广播 (5Hz)
        self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self.telemetry_thread.start()

        # 线程 1-B: 专门阻塞监听并应答 QGC 的航线握手协议
        self.protocol_thread = threading.Thread(target=self._protocol_loop, daemon=True)
        self.protocol_thread.start()

        print(f"[QGCMissionProxy] 虚拟飞控线程已启动，正在监听 UDP 端口: {self.connection_str}")

    def stop(self):
        """停止接收器"""
        self.is_running = False
        if self.master:
            self.master.close()
        print("[QGCMissionProxy] 虚拟飞控线程已关闭")

    def _telemetry_loop(self):
        """高频向QGC广播心跳和定位，激活QGC的航线规划和上传按钮"""
        boot_time = time.time()
        while self.is_running:
            try:
                time_ms = int((time.time() - boot_time) * 1000)

                # 1. 发送 HEARTBEAT (伪装成运行中的 ArduPilot 多旋翼，且处于 GUIDED 模式解锁状态)
                self.master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_QUADROTOR,  # 四轴/多旋翼
                    mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOT,  # 固件类型
                    mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED | mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
                    0,  # custom_mode
                    mavutil.mavlink.MAV_STATE_ACTIVE  # 活动状态
                )

                # 2. 发送 GLOBAL_POSITION_INT (不发这个QGC地图无法显示飞机，也无法规划航线)
                self.master.mav.global_position_int_send(
                    time_ms,
                    int(self.home_lat * 1e7),  # 纬度 (degrees * 1E7)
                    int(self.home_lon * 1e7),  # 经度 (degrees * 1E7)
                    10000,  # 绝对高度 (毫米, 10米)
                    0,  # 相对高度 (毫米, 0米)
                    0, 0, 0,  # vx, vy, vz 速度 (cm/s)
                    9000  # hdg 偏航角 (cdeg, 90度朝东)
                )

                # 3. 发送系统状态 (维持QGC右上角图标正常显示，可选)
                self.master.mav.sys_status_send(0, 0, 0, 500, 11100, 100, 80, 0, 0, 0, 0, 0, 0)

            except Exception as e:
                print(f"[QGCMissionProxy] 遥测发送异常: {e}")
            time.sleep(0.2)  # 5Hz 频率广播

    def _protocol_loop(self):
        """标准 MAVLink 航线协议状态机逻辑"""
        while self.is_running:
            try:
                # 阻塞读取消息
                msg = self.master.recv_match(blocking=True, timeout=0.5)
                if not msg:
                    continue

                msg_type = msg.get_type()
                src_sys = msg.get_srcSystem()
                src_comp = msg.get_srcComponent()

                # ---- 阶段 A: QGC点击了“发送航线”，通知飞机即将上传的总数 ----
                if msg_type == 'MISSION_COUNT':
                    self.expected_count = msg.count
                    self.current_seq = 0
                    self._temp_waypoints = []
                    print(f"\n[QGCMissionProxy] 拦截到QGC航线上传请求！预计航点总数: {self.expected_count}")

                    if self.expected_count > 0:
                        # 核心回复：主动向地面站索要第 0 个航点数据
                        self.master.mav.mission_request_int_send(
                            src_sys, src_comp, self.current_seq, msg.mission_type
                        )
                    else:
                        # 总数为0则直接发送成功确认
                        self.master.mav.mission_ack_send(
                            src_sys, src_comp, mavutil.mavlink.MAV_MISSION_ACCEPTED, msg.mission_type
                        )

                # ---- 阶段 B: 接收具体的航点细节 (新版QGC通常使用 MISSION_ITEM_INT 整型坐标协议) ----
                elif msg_type == 'MISSION_ITEM_INT':
                    if msg.seq == self.current_seq:
                        # 解析QGC发来的单条数据
                        wp = {
                            'seq': msg.seq,
                            'frame': msg.frame,
                            'command': msg.command,
                            'lat': msg.x / 1e7,  # 还原浮点经纬度
                            'lon': msg.y / 1e7,
                            'alt': msg.z,  # 高度
                            'param1': msg.param1,  # 悬停时间等控制参数
                            'param2': msg.param2,
                            'param3': msg.param3,
                            'param4': msg.param4
                        }
                        self._temp_waypoints.append(wp)
                        print(
                            f"[QGCMissionProxy] 成功接收并缓存航点 [{msg.seq + 1}/{self.expected_count}]: Lat={wp['lat']:.6f}, Lon={wp['lon']:.6f}, Alt={wp['alt']:.2f}m")

                        self.current_seq += 1
                        if self.current_seq < self.expected_count:
                            # 继续索要下一个航点
                            self.master.mav.mission_request_int_send(
                                src_sys, src_comp, self.current_seq, msg.mission_type
                            )
                        else:
                            # ---- 阶段 C: 全部接收完毕，原子化存入正式列表，并向QGC回复“握手成功” ----
                            with self.lock:
                                self.waypoints_list = list(self._temp_waypoints)
                            print(f"🎉 [QGCMissionProxy] 航线完美拦截并填充！列表当前长度: {len(self.waypoints_list)}")

                            self.master.mav.mission_ack_send(
                                src_sys, src_comp, mavutil.mavlink.MAV_MISSION_ACCEPTED, msg.mission_type
                            )

                # ---- 兼容机制: 兼容老版本QGC使用的浮点数协议形式 ----
                elif msg_type == 'MISSION_ITEM':
                    if msg.seq == self.current_seq:
                        wp = {
                            'seq': msg.seq, 'frame': msg.frame, 'command': msg.command,
                            'lat': msg.x, 'lon': msg.y, 'alt': msg.z,
                            'param1': msg.param1, 'param2': msg.param2, 'param3': msg.param3, 'param4': msg.param4
                        }
                        self._temp_waypoints.append(wp)
                        self.current_seq += 1
                        if self.current_seq < self.expected_count:
                            self.master.mav.mission_request_send(src_sys, src_comp, self.current_seq, msg.mission_type)
                        else:
                            with self.lock:
                                self.waypoints_list = list(self._temp_waypoints)
                            self.master.mav.mission_ack_send(src_sys, src_comp, mavutil.mavlink.MAV_MISSION_ACCEPTED,
                                                             msg.mission_type)

                # ---- 处理地面站点击“清除航线”的请求 ----
                elif msg_type == 'MISSION_CLEAR_ALL':
                    print("[QGCMissionProxy] 拦截到QGC清除航线指令")
                    with self.lock:
                        self.waypoints_list = []
                    self.master.mav.mission_ack_send(src_sys, src_comp, mavutil.mavlink.MAV_MISSION_ACCEPTED,
                                                     msg.mission_type)

            except Exception as e:
                print(f"[QGCMissionProxy] 协议状态机运行异常: {e}")
                time.sleep(0.1)

    def get_waypoints(self):
        """提供给控制主线程调用的线程安全方法：获取当前最新的航点列表"""
        with self.lock:
            return list(self.waypoints_list)

    def clear_waypoints(self):
        """从香橙派端手动清空列表"""
        with self.lock:
            self.waypoints_list = []