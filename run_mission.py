"""
run_mission.py — 基于 ZMQ Router 协议的主任务入口

【架构概览】
  本脚本是整个无人机任务系统的总装粘合剂。采用领域驱动模块化设计，
  将各子系统按职责独立实例化，在主循环中进行控制权协调。

    ┌─ MissionManager ────────────┐
    │  RouterProxy (own)          │  状态机管理 + 航点巡航
    │  POSITION SETPOINT (非追踪) │  控制权：NAVIGATING/HOLD_TASK/...
    └─────────────────────────────┘
              ↕ 控制权协调
    ┌─ UAVControlLoop ────────────┐
    │  RouterProxy (own)          │  20Hz VELOCITY 闭环追踪
    │  TargetTracker + PID        │  控制权：仅 VISUAL_TRACKING
    └─────────────────────────────┘
    ┌─ 感知模块 ───────────────────┐
    │  VideoStreaming             │  摄像头/RTSP 采集
    │  YOLO26UAVInfer             │  昇腾 NPU 推理
    └─────────────────────────────┘

【控制权协调规则】
  同一时刻只有一个人持有「SETPOINT 发令权」：
    - VISUAL_TRACKING 之外 → MissionManager 发 POSITION SETPOINT
    - VISUAL_TRACKING 之中 → UAVControlLoop 线程发 VELOCITY SETPOINT
    - 切换由主循环检测 MissionManager.state 变化时触发

【安全设计】
    - try / except FailsafeTriggered / finally 全面资源释放
    - 两个独立 RouterProxy，避免 REQ socket 线程竞争
    - 控制线程启停由状态机状态变化驱动，逻辑可预测

【可扩展的图像输入接口】
    如需替换视频源（如模拟器/录播文件），替换 VideoStreaming 实例
    即可，保持 read_frame() → (ret, frame) 的协议不变。
"""

import json
import logging
import os
import sys
import threading
import time
import socket
import cv2

from drone_controller.mission_manager import (
    MissionManager,
    MissionState,
    FailsafeTriggered,
)
# ---- ZMQ Router 代理与状态机 ----
from drone_controller.router_proxy import RouterProxy
# ---- 视觉闭环控制（复用 infer_camera_modular 现有实现） ----
# 注意：此处不修改 UAVControlLoop 任何内部逻辑。
# run_mission.py 在外部协调其控制线程的启停。
from infer_camera_modular import UAVControlLoop, VideoStreaming, YOLO26UAVInfer
from utils.coord import latlon_to_ned

logger = logging.getLogger("RunMission")


class MissionOrchestrator:
    """
    🎯 全任务编排器

    职责：
      1. 创建并初始化所有子系统
      2. 在主循环中协调 MissionManager 状态机与 UAVControlLoop 的控制权
      3. 处理 Failsafe 熔断、键盘中断等异常退出
      4. 提供标准视频源接口（可替换）
    """

    def __init__(self,waypoints,config_path="./config.json"):
        self.logger = logging.getLogger("Orchestrator")
        self.logger.info("正在加载配置: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.cfg = cfg

        # ---- 感知子系统 ----
        self.streamer = VideoStreaming(cfg["video"], cfg["network"])
        self.detector = YOLO26UAVInfer(cfg["model"])

        # ---- 航线定义（NED 坐标系，Z 向上为负） ----
        self.waypoints = waypoints

        # ================================================================
        #  🚁 MissionManager（状态机 + 航点巡航）
        # ================================================================
        # 拥有独立的 RouterProxy，用于读取 STATE 和发送 COMMAND /
        # WAYPOINT / POSITION SETPOINT。
        fc = cfg["flight_control"]
        self.mission_proxy = RouterProxy(
            req_endpoint=fc.get("req_endpoint", "tcp://127.0.0.1:5555"),
            sub_endpoint=fc.get("sub_endpoint", "tcp://127.0.0.1:5556"),
        )

        self.mission_proxy.set_waypoints_callback(self._on_qgc_waypoints)

        self.mission = MissionManager(
            proxy=self.mission_proxy,
            waypoints=self.waypoints,
            target_altitude=1.5,
            arrival_radius=0.3,
            hold_duration=8.0,
            return_to_home=False,
        )

        # ================================================================
        #  🛸 UAVControlLoop（20Hz VELOCITY 闭环追踪）
        # ================================================================
        # 拥有独立的 RouterProxy。其 20Hz 控制线程仅在 VISUAL_TRACKING
        # 期间运行，其余时间处于停止状态，避免与 MissionManager 冲突。
        self.controller = UAVControlLoop(
            cfg["flight_control"], cfg["pid_y"], cfg["pid_z"],
        )
        self._control_active = False   # 控制线程运行标志

        # ---- 主循环参数 ----
        self.running = False
        self.loop_interval = 0.05      # 20Hz
        # ---- TCP 命令通道（急停 / 后续扩展） ----
        self.tcp_port = cfg.get("command_channel", {}).get("tcp_port", 9999)
        self.tcp_server_socket = None
        self.tcp_server_running = False
        self.tcp_server_thread = None
        self.tcp_clients = []  # 保存已连接的客户端 socket（用于广播）

        self.logger.info("MissionOrchestrator 初始化完成")

    # ================================================================
    # 📥 QGC 航点回调（由 RouterProxy 的 SUB 线程触发）
    # ================================================================

    def _on_qgc_waypoints(self,data: dict):
        """
        收到 QGC 下发的航点列表 -> 转换坐标系 -> 注入 MissionManager
        """
        try:
            raw_wps = data.get("waypoints", [])
            if not raw_wps:
                self.logger.warning("QGC 航点列表为空，请检查是否提交或路由问题")
                return

            self.logger.info("📥 收到 %d 个航点，开始坐标转换...", len(raw_wps))

            # ---- ① 获取参考点（起飞点） ----
            state = self.mission_proxy.get_latest_state()
            if state is None:
                self.logger.error("无法获取 STATE，缺少参考经纬度，航点转换失败")
                return

            home = state.get("home", {})
            ref_lat = home.get("lat", 0.0)
            ref_lon = home.get("lon", 0.0)
            ref_alt = home.get("alt", 0.0)

            # ---- 容错：如果 HOME 未初始化，用第一个航点作为参考（纯相对飞行） ----
            if ref_lat == 0.0 and ref_lon == 0.0:
                self.logger.warning("STATE.home 未初始化，使用第一个航点作为参考原点")
                ref_lat = raw_wps[0].get("lat", 0.0)
                ref_lon = raw_wps[0].get("lon", 0.0)
                ref_alt = raw_wps[0].get("z", 0.0)

            self.logger.info("参考原点 (起飞点): lat=%.6f, lon=%.6f, alt=%.1fm",
                             ref_lat, ref_lon, ref_alt)

            # ---- ② 逐点转换 ----
            converted = []
            for idx, wp in enumerate(raw_wps):
                lat = wp.get("lat", 0.0)
                lon = wp.get("lon", 0.0)
                alt = wp.get("z", 0.0)          # QGC 中 z 为海拔
                cmd = wp.get("command", 16)     # 16=WAYPOINT, 22=TAKEOFF, 21=LAND
                yaw = wp.get("param4", 0.0)     # param4 通常存偏航角
                hold_time = wp.get("param1", 8.0)   # 悬停时间（如有）

                x, y, z = latlon_to_ned(lat, lon, alt, ref_lat, ref_lon, ref_alt)
                print(f"航点信息；{x}{y}{z}")

                # 根据 MAV_CMD 类型构建内部航点
                if cmd == 22:  # TAKEOFF
                    # 起飞点强制归零水平位置，高度取绝对海拔差
                    converted.append({
                        "x": 0.0,
                        "y": 0.0,
                        "z": z,  # 已按 NED 转换（向上为负）
                        "yaw": yaw,
                        "command": "TAKEOFF",
                        "hold_duration": hold_time,
                    })
                    self.logger.info("航点 %d [TAKEOFF]: 高度 %.1fm", idx, -z)

                elif cmd == 21:  # LAND
                    converted.append({
                        "x": x,
                        "y": y,
                        "z": 0.0,  # 着陆点 z=0（地面）
                        "yaw": yaw,
                        "command": "LAND",
                    })
                    self.logger.info("航点 %d [LAND]: NED(%.1f, %.1f)", idx, x, y)

                else:  # 普通航点 (command=16 或未识别)
                    converted.append({
                        "x": x,
                        "y": y,
                        "z": z,  # NED 高度
                        "yaw": yaw,
                        "command": "WAYPOINT",
                        "hold_duration": hold_time,
                    })
                    self.logger.debug("航点 %d [WAYPOINT]: NED(%.1f, %.1f, %.1f)",
                                      idx, x, y, z)

            self.mission.set_waypoints(converted)

            # 如果当前处于等待航点状态，状态机会在下一帧自动响应
            self.logger.info("✅ 航点转换完成，共 %d 个航点已加载", len(converted))

        except Exception as e:
            self.logger.error("航点处理异常: %s",e)

    # ================================================================
    # 🖥️ TCP Server
    # ================================================================
    # run_mission.py — MissionOrchestrator 类中新增

    def _start_tcp_command_server(self):
        """启动 TCP 命令服务器（独立线程）"""
        if self.tcp_server_thread and self.tcp_server_thread.is_alive():
            return

        self.tcp_server_running = True
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server_socket.bind(("0.0.0.0", self.tcp_port))
        self.tcp_server_socket.listen(5)
        self.tcp_server_socket.settimeout(1.0)  # 用于检查 self.tcp_server_running

        self.tcp_server_thread = threading.Thread(
            target=self._tcp_command_server_worker,
            daemon=True,
            name="tcp-command-server"
        )
        self.tcp_server_thread.start()
        self.logger.info(f"🔌 TCP 命令通道已启动: port={self.tcp_port} (指令: STOP)")

    def _tcp_command_server_worker(self):
        """TCP 服务器主循环：接受客户端，分发处理"""
        while self.tcp_server_running:
            try:
                client_sock, addr = self.tcp_server_socket.accept()
                self.logger.info(f"📡 地面站已连接: {addr[0]}:{addr[1]}")
                # 为每个客户端创建独立处理线程（支持多地面站同时连接）
                client_thread = threading.Thread(
                    target=self._handle_tcp_client,
                    args=(client_sock, addr),
                    daemon=True
                )
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.tcp_server_running:
                    self.logger.error(f"TCP Server 异常: {e}")
                break

        # 清理
        if self.tcp_server_socket:
            try:
                self.tcp_server_socket.close()
            except Exception:
                pass
            self.tcp_server_socket = None
        self.logger.info("🔌 TCP 命令通道已关闭")

    def _handle_tcp_client(self, client_sock, addr):
        """处理单个 TCP 客户端连接（长连接）"""
        client_sock.settimeout(1.0)  # 读超时，便于检测连接断开
        try:
            while self.tcp_server_running:
                try:
                    data = client_sock.recv(1024).decode().strip()
                    if not data:
                        # 客户端主动断开
                        self.logger.info(f"📡 地面站断开: {addr[0]}:{addr[1]}")
                        break

                    msg = data.upper()
                    self.logger.info(f"📨 收到指令: {msg} (来自 {addr[0]}:{addr[1]})")

                    if msg == "STOP":
                        self.logger.warning(f"🚨 触发紧急降落 (指令来自 {addr[0]}:{addr[1]})")
                        self.mission.emergency_stop()
                        self.running = False
                        client_sock.send(b"ACK: EMERGENCY_STOP_TRIGGERED\n")
                        # 触发后不立即断开，让客户端收到 ACK
                        # 但主循环即将退出，服务器也会关闭

                    elif msg == "PING":
                        # 心跳 / 保活响应
                        client_sock.send(b"PONG\n")

                    elif msg == "STATUS":
                        # 返回当前状态（便于调试）
                        status = f"STATE:{self.mission.state.name}, WP:{self.mission.current_wp_index}/{len(self.mission.waypoints)}\n"
                        client_sock.send(status.encode())

                    else:
                        client_sock.send(f"UNKNOWN_CMD: {msg}\n".encode())

                except socket.timeout:
                    # 读超时，继续循环检查 self.tcp_server_running
                    continue
                except BrokenPipeError:
                    self.logger.warning(f"📡 地面站连接异常断开: {addr[0]}:{addr[1]}")
                    break
                except Exception as e:
                    self.logger.error(f"TCP 客户端处理异常: {e}")
                    break
        finally:
            try:
                client_sock.close()
            except Exception:
                pass

    # ================================================================
    #  🔌 启动
    # ================================================================

    def start(self):
        """启动所有子系统"""
        self.logger.info("正在启动所有子系统...")

        # 1. 启动 MissionManager 的 ZMQ 代理（SUB 监听 + REQ 就绪）
        self.mission_proxy.start()
        self.logger.info("✔ MissionManager Proxy 已启动")

        # 2. 启动 UAVControlLoop 的 ZMQ 代理
        #    start_uav() 会执行完整的起飞序列，我们不需要 ——
        #    MissionManager 负责起飞。这里只启动 SUB 线程收 STATE。
        self.controller.proxy.start()
        self.logger.info("✔ UAVControlLoop Proxy 已启动")

        # 3. 等待收到第一条 Router STATE
        self._wait_for_initial_state()

        # ---- 启动 TCP 命令通道 ----
        self._start_tcp_command_server()

        self.running = True
        self.logger.info("🚀 所有子系统就绪，主循环启动")

    def _wait_for_initial_state(self, timeout=5.0):
        """等待 Router 推送第一条 STATE"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.mission_proxy.get_latest_state()
            if state is not None:
                drone = state.get("drone", {})
                self.logger.info(
                    "✔ 收到 Router 状态: mode=%s armed=%s alt=%.1fm",
                    drone.get("mode"), drone.get("armed"),
                    drone.get("alt_rel", 0),
                )
                return True
            time.sleep(0.1)
        self.logger.warning("⚠ 未收到 Router 状态 (%ds 超时)，继续启动...", timeout)
        return False

    # ================================================================
    #  🔁 主循环
    # ================================================================

    def run(self):
        """主业务循环（20Hz）"""
        self.logger.info("主循环开始 (20Hz)")

        prev_state = MissionState.INIT
        loop_count = 0

        try:
            while self.running:
                loop_start = time.time()

                # ---- ① 读取视频帧 ----
                ret, frame = self.streamer.read_frame()
                if not ret:
                    # 无新帧时短暂等待，不浪费 CPU
                    time.sleep(0.005)
                    continue

                loop_count += 1
                orig_shape = frame.shape[:2]

                # ---- ② NPU 推理 + 后处理 ----
                input_tensor, ratio, dwdh = self.detector.preprocess(frame)
                outputs = self.detector.session.infer([input_tensor])
                detections = self.detector.postprocess(
                    outputs, ratio, dwdh, orig_shape,
                )
                target_detected = len(detections) > 0

                # ---- ③ 控制权协调 ----
                # 根据当前状态机 state 决定 UAVControlLoop 控制线程启停
                current_state = self.mission.state
                self._coordinate_control(current_state, prev_state,
                                         detections, orig_shape)
                prev_state = current_state

                # ---- ④ 状态机更新 ----
                self.mission.update(target_detected=target_detected)

                # ---- ⑤ HUD 渲染 + 图传 ----
                frame = self._render_hud(frame, loop_count, detections)
                self.streamer.push_to_stream(frame)

                # ---- 频率控制 ----
                elapsed = time.time() - loop_start
                sleep_time = self.loop_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except FailsafeTriggered as e:
            # Failsafe 熔断：安全员切出手动模式 → 程序安全退出
            self.logger.warning("🚨 [Failsafe] %s", e)
        except KeyboardInterrupt:
            self.logger.info("⌨ 用户键盘中断，正在安全退出...")
        except Exception as e:
            self.logger.exception("💥 主循环异常: %s", e)
        finally:
            self.shutdown()

    # ================================================================
    #  控制权协调
    # ================================================================

    def _coordinate_control(self, current_state, prev_state,
                            detections, orig_shape):
        """
        控制权协调：基于状态机状态启停 UAVControlLoop 线程

        ┌──────────────┬─────────────────────┬──────────────────────┐
        │ 状态机状态    │ MissionManager      │ UAVControlLoop       │
        ├──────────────┼─────────────────────┼──────────────────────┤
        │ 非 TRACKING  │ proxy → POSITION    │ 控制线程已停止        │
        │              │   SETPOINT          │                      │
        ├──────────────┼─────────────────────┼──────────────────────┤
        │ TRACKING     │ 不发 SETPOINT       │ 20Hz VELOCITY 线程   │
        │              │ (控制权已让渡)       │ TargetTracker + PID  │
        └──────────────┴─────────────────────┴──────────────────────┘
        """
        is_tracking = current_state == MissionState.VISUAL_TRACKING
        was_tracking = prev_state == MissionState.VISUAL_TRACKING

        if is_tracking:
            # ---- 进入 VISUAL_TRACKING → 启动/保持控制线程 ----
            if not self._control_active:
                self._start_control_thread()

            # 不断喂入检测数据（TargetTracker 在主线程更新）
            self.controller.update_detections(detections, orig_shape)

        else:
            # ---- 非 VISUAL_TRACKING → 停止控制线程 ----
            if self._control_active:
                self._stop_control_thread()

            # 非追踪期间也喂入空检测，保持数据结构新鲜
            self.controller.update_detections([], orig_shape)

    def _start_control_thread(self):
        """启动 UAVControlLoop 的 20Hz VELOCITY 控制线程"""
        self.logger.info("▶ [控制权] 启动 UAVControlLoop VELOCITY 线程")

        # 记录进入追踪时的飞机状态
        state = self.mission_proxy.get_latest_state()
        if state:
            d = state.get("drone", {})
            self.logger.info("追踪初始状态: mode=%s alt=%.1fm armed=%s",
                             d.get("mode"), d.get("alt_rel", 0),
                             d.get("armed"))

        # 直接复用 UAVControlLoop 已有的内部 _control_loop 方法
        # （不修改任何内部逻辑，仅从外部启动其线程）
        self.controller.running = True
        self.controller.control_thread = threading.Thread(
            target=self.controller._control_loop,
            daemon=True,
            name="uav-vel-control",
        )
        self.controller.control_thread.start()
        self._control_active = True

    def _stop_control_thread(self):
        """安全停止 UAVControlLoop 的控制线程"""
        if not self._control_active:
            return

        self.logger.info("⏹ [控制权] 停止 UAVControlLoop 线程")
        self.controller.running = False

        if self.controller.control_thread:
            self.controller.control_thread.join(timeout=2.0)
            self.controller.control_thread = None

        # 复位 PID 积分项和跟踪器，避免下次启动时积分暴冲
        self.controller.pid_y.reset()
        self.controller.pid_z.reset()
        self.controller.tracker.reset()
        self._control_active = False
        self.logger.info("✔ 控制线程已停止，PID/跟踪器已重置")

    # ================================================================
    #  渲染
    # ================================================================

    def _render_hud(self, frame, loop_count, detections):
        """
        在视频帧上叠加 HUD 信息

        包含：状态机阶段、航点索引、控制权归属、检测结果
        """
        # ---- 检测框 ----
        frame = self.detector.draw_boxes(frame, detections)

        # ---- 状态面板 ----
        state_name = self.mission.state.name
        wp_text = f"WP: {self.mission.current_wp_index + 1}/{len(self.waypoints)}"
        ctrl_text = "CTRL:VELOCITY" if self._control_active else "CTRL:POSITION"

        cv2.putText(frame, f"State: {state_name} | {wp_text}",
                    (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.putText(frame, ctrl_text,
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if self._control_active else (255, 255, 0), 2)

        # ---- 每 100 帧打印详细状态日志 ----
        if loop_count % 100 == 0:
            state = self.mission_proxy.get_latest_state()
            if state:
                d = state.get("drone", {})
                self.logger.info(
                    "📡 mode=%s armed=%s alt=%.1fm spd=%.1f bat=%.0f%% "
                    "%s %s detect=%d",
                    d.get("mode"), d.get("armed"),
                    d.get("alt_rel", 0), d.get("ground_speed", 0),
                    d.get("battery", 0),
                    state_name, ctrl_text,
                    len(detections),
                )

        return frame

    # ================================================================
    #  🧹 资源释放
    # ================================================================

    def shutdown(self):
        """
        安全关闭所有子系统

        释放顺序（逆初始化顺序）：
          1. 停止控制线程
          2. 关闭 MissionManager 的 ZMQ 代理
          3. 关闭 UAVControlLoop 的 ZMQ 代理
          4. 释放视频流资源
        """
        self.logger.info("====== 安全关闭所有子系统 ======")
        self.running = False

        # 1. 停止控制线程
        self._stop_control_thread()

        # 2. 关闭 MissionManager 的 Proxy
        try:
            self.mission_proxy.close()
            self.logger.info("✔ MissionManager Proxy 已关闭")
        except Exception as e:
            self.logger.warning("mission_proxy close 异常: %s", e)

        # 3. 关闭 UAVControlLoop 的 Proxy（直接访问其内部 proxy）
        try:
            self.controller.proxy.close()
            self.logger.info("✔ UAVControlLoop Proxy 已关闭")
        except Exception as e:
            self.logger.warning("control_proxy close 异常: %s", e)

        # 4. 释放视频资源
        try:
            self.streamer.release()
            self.logger.info("✔ VideoStreamer 已释放")
        except Exception as e:
            self.logger.warning("streamer release 异常: %s", e)

        # ---- 停止 TCP 命令通道 ----
        self.tcp_server_running = False
        if self.tcp_server_socket:
            try:
                self.tcp_server_socket.close()
            except Exception:
                pass
            self.tcp_server_socket = None
        if self.tcp_server_thread and self.tcp_server_thread.is_alive():
            self.tcp_server_thread.join(timeout=2.0)
            self.logger.info("✔ TCP 命令通道已关闭")

        self.logger.info("🏁 所有资源已安全释放，程序退出")


# ================================================================
#  入口
# ================================================================

def main():
    """主入口：日志配置 → 编排器创建 → 启动 → 运行"""
    # 创建日志目录
    os.makedirs("./log", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("./log/mission_run.log", mode="a"),
        ],
    )

    # 允许命令行指定配置文件路径
    config_file = sys.argv[1] if len(sys.argv) > 1 else "./config.json"
    # 也可以传入固定点
    # waypoints = [
    #         {'x': 1.0, 'y': 0.0, 'z': -1.5, 'yaw': 0},
    #         {'x': 1.0, 'y': 1.0, 'z': -1.5, 'yaw': 90},
    #         {'x': 0.0, 'y': 0.0, 'z': -1.2, 'yaw': 0},
    #     ]
    #  orchestrator = MissionOrchestrator(config_path=config_file,waypoints=waypoints)

    orchestrator = MissionOrchestrator(config_path=config_file,waypoints=[])
    orchestrator.start()
    orchestrator.run()


if __name__ == "__main__":
    main()
