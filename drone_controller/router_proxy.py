"""
router_proxy.py — ZeroMQ 通信代理

【职责】
  封装与 Router 的 REQ/SUB 通信，遵循 python_protocol.md 规范：
  - REQ (tcp://127.0.0.1:5555) — 发送 SETPOINT / COMMAND / QUERY / WAYPOINT，接收 ACK
  - SUB (tcp://127.0.0.1:5556) — 订阅 Router 推送的 STATE / ALERT / PX4_ACK

【使用】
    proxy = RouterProxy()
    proxy.start()


    # 发送指令
    ok, ack = proxy.send_command("ARM")
    ok, ack = proxy.send_setpoint(vx=1.0, vy=0.0, vz=0.0, control_mode="VELOCITY")
    ok, ack = proxy.send_query("HOME_POSITION")

    # 读取最新状态（线程安全）
    state = proxy.get_latest_state()
    if proxy.is_state_fresh(state):
        ...

    proxy.close()

【设计要点】
  - REQ_RELAXED + REQ_CORRELATE 防止单次 recv 超时导致 socket 卡死
  - SUB 采用专用后台线程 + zmq.Poller 非阻塞接收
  - 所有公开接口线程安全
  - 状态缓存原子读写，外部获取为深拷贝，无共享引用风险

【作者】根据 python_protocol.md 重构
"""

import zmq
import json
import time
import threading
import copy
import logging

logger = logging.getLogger(__name__)


class RouterProxy:
    """ZeroMQ 通信代理：封装与 Router 的 REQ/SUB 通信"""

    # ---- 协议常量 ----
    STALE_THRESHOLD_US = 500_000          # 500 ms，STATE 新鲜度阈值（协议 §3.8.1）
    STALE_THRESHOLD_WARN_US = 2_000_000   # 2 s，数据稍旧警告阈值

    def __init__(self, req_endpoint="tcp://127.0.0.1:5555",
                 sub_endpoint="tcp://127.0.0.1:5556",
                 recv_timeout_ms=500):
        """
        初始化 ZMQ 代理

        Args:
            req_endpoint:   REQ socket 地址（发给 Router）
            sub_endpoint:   SUB socket 地址（收 Router 推送）
            recv_timeout_ms:REQ recv 超时毫秒数
        """
        self.req_endpoint = req_endpoint
        self.sub_endpoint = sub_endpoint
        self.recv_timeout_ms = recv_timeout_ms

        # ---- ZMQ 上下文 ----
        self.ctx = zmq.Context()

        # ---- REQ socket: 发指令，收 ACK ----
        self.req = self.ctx.socket(zmq.REQ)
        self.req.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        # REQ_RELAXED + REQ_CORRELATE: 即使某次 recv 超时，
        # 下次 send 仍能正常发送，旧响应会被自动丢弃
        self.req.setsockopt(zmq.REQ_RELAXED, 1)
        self.req.setsockopt(zmq.REQ_CORRELATE, 1)
        self.req.connect(req_endpoint)

        # ---- SUB socket: 收 Router 推送 ----
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(sub_endpoint)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")   # 订阅所有消息

        # ---- 消息序号（线程安全） ----
        self._seq = 0
        self._seq_lock = threading.Lock()

        # ---- 消息ID （区分标识） ----
        self.id = 0
        self.id_lock = threading.Lock()

        # ---- 最新 STATE 缓存 ----
        self._latest_state = None
        self._state_lock = threading.Lock()

        # ---- 首次 STATE 标记 ----
        self._first_state_received = False

        # ---- SUB 监听线程 ----
        self._running = False
        self._sub_thread = None

        # ---- 外部回调钩子 ----
        self._state_callback = None
        self._alert_callback = None

    # ================================================================
    #  内部工具
    # ================================================================

    def _next_id(self) -> int:
        """递增消息ID （区分标识）"""
        with self.id_lock:
            self.id += 1
            return self.id

    def _next_seq(self) -> int:
        """递增消息序号（线程安全）"""
        with self._seq_lock:
            current_seq = self._seq
            self._seq += 1
            return current_seq


    def _build_send_summary(self, msg: dict) -> str:
        """构建发送消息的内容概要，用于日志"""
        msg_type = msg.get("type", "?")
        if msg_type == "SETPOINT":
            sp = msg.get("setpoint", {})
            return ", ".join(f"{k}={v}" for k, v in sp.items())
        elif msg_type == "COMMAND":
            return f"command={msg.get('command')}"
        elif msg_type == "QUERY":
            return f"query={msg.get('query')}"
        elif msg_type == "WAYPOINT":
            wp = msg.get("waypoint", {})
            return ", ".join(f"{k}={v}" for k, v in wp.items())
        return str(msg)

    def _recover_req(self):
        """
        REQ 超时/异常后重建 socket

        虽然设置了 REQ_RELAXED+CORRELATE，但连续超时或 ZMQ 内部状态异常时
        重建是最安全的恢复手段。
        """
        try:
            self.req.close(linger=0)
        except Exception:
            pass
        self.req = self.ctx.socket(zmq.REQ)
        self.req.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self.req.setsockopt(zmq.REQ_RELAXED, 1)
        self.req.setsockopt(zmq.REQ_CORRELATE, 1)
        self.req.connect(self.req_endpoint)
        logger.warning("REQ socket 已重建")

    # ================================================================
    #  发送接口（线程安全）
    # ================================================================

    def send_setpoint(self, control_mode="VELOCITY", **kwargs):
        """
        发送 SETPOINT 指令

        位置控制:
            proxy.send_setpoint(control_mode="POSITION",
                                x=10.0, y=0.0, z=-8.0, yaw=0.0)
        速度控制:
            proxy.send_setpoint(control_mode="VELOCITY",
                                vx=1.0, vy=0.0, vz=0.0, yaw_rate=0.0)

        Args:
            control_mode: "POSITION" | "VELOCITY"
            **kwargs:     按 control_mode 传入对应字段

        Returns:
            (成功: bool, ACK 字典: dict)
        """
        msg_id = self._next_id()
        msg = {
            "type": "SETPOINT",
            "id": msg_id,
            "timestamp_us": int(time.time() * 1_000_000),
            "setpoint": {"control_mode": control_mode, **kwargs}
        }
        return self._send_req(msg)

    def send_command(self, command):
        """
        发送 COMMAND 指令

        Args:
            command: "ARM" | "OFFBOARD" | "LAND" | "RTL" | "RESUME"
        """
        msg_id = self._next_id()
        msg = {
            "type": "COMMAND",
            "id": msg_id,
            "timestamp_us": int(time.time() * 1_000_000),
            "command": command
        }
        return self._send_req(msg)

    def send_query(self, query="HOME_POSITION"):
        """
        发送 QUERY 查询

        Args:
            query: "HOME_POSITION"（目前仅支持此查询）
        """
        msg_id = self._next_id()
        msg = {
            "type": "QUERY",
            "id": msg_id,
            "timestamp_us": int(time.time() * 1_000_000),
            "query": query
        }
        return self._send_req(msg)

    def send_waypoint(self, **wp_kwargs):
        """
        发送 WAYPOINT 航点

        Args:
            **wp_kwargs: 航点字段（action, alt, lat, lon, speed, ...）

        示例:
            proxy.send_waypoint(action="TAKEOFF", alt=15.0, alt_frame="RELATIVE")
            proxy.send_waypoint(action="HOVER", lat=22.55, lon=113.88,
                                alt=100.0, hover_time=3.0)
        """
        msg_id = self._next_id()
        current_seq = self._next_seq()
        msg = {
            "type": "WAYPOINT",
            "id": msg_id,
            "seq": current_seq,
            "timestamp_us": int(time.time() * 1_000_000),
            "waypoint": wp_kwargs
        }
        return self._send_req(msg)

    # ================================================================
    #  底层 REQ-REP 通信
    # ================================================================

    def _send_req(self, msg):
        """
        发送 REQ 并等待 ACK 回复

        ZeroMQ REQ-REP 是严格的交替模式：send → recv → send → recv
        本方法保证每次调用完成一次完整的 send-recv 周期。

        Returns:
            (ok: bool, ack_dict: dict)
            ack_dict 示例：{"type": "ACK", "ref_id": 1, "status": "OK", "message": ""}
        """
        msg_type = msg.get("type", "?")
        msg_id = msg.get("id", "?")
        summary = self._build_send_summary(msg)
        logger.debug("Sending %s id=%s: %s", msg_type, msg_id, summary)
        logger.info("🚀 发送原始 JSON 指令: %s", json.dumps(msg))
        try:
            self.req.send_string(json.dumps(msg))
            ack_str = self.req.recv_string()
            ack = json.loads(ack_str)
            ok = ack.get("status") == "OK"
            logger.info("ACK received for id=%s, status=%s",
                        ack.get("ref_id"), ack.get("status"))
            if not ok:
                logger.warning("ACK 非 OK: type=%s id=%s ack=%s",
                               msg_type, msg_id, ack)
            return ok, ack
        except zmq.Again:
            logger.error("REQ recv 超时 (type=%s id=%s)", msg_type, msg_id)
            self._recover_req()
            return False, {"status": "TIMEOUT", "message": "recv timeout"}
        except Exception as e:
            logger.error("REQ 异常 (type=%s id=%s): %s", msg_type, msg_id, e)
            self._recover_req()
            return False, {"status": "FAIL", "message": str(e)}

    # ================================================================
    #  状态获取
    # ================================================================

    def get_latest_state(self):
        """
        获取最新缓存的 STATE（线程安全，深拷贝）

        Returns:
            state dict（格式见 python_protocol.md §3.8），
            未收到过 STATE 时返回 None
        """
        with self._state_lock:
            return copy.deepcopy(self._latest_state)

    def is_state_fresh(self, state, stale_threshold_us=None):
        """
        检查 STATE 数据是否新鲜（协议 §3.8.1）

        Args:
            state:                get_latest_state() 的返回值
            stale_threshold_us:   新鲜度阈值（默认 500ms）

        Returns:
            True  数据新鲜，可正常使用
            False 数据过期，应保守飞行或触发保护
        """
        if state is None:
            return False
        if stale_threshold_us is None:
            stale_threshold_us = self.STALE_THRESHOLD_US
        now_us = int(time.time() * 1_000_000)
        try:
            age_us = now_us - state["drone"]["last_update_us"]
        except (KeyError, TypeError):
            return False
        if age_us >= stale_threshold_us:
            if age_us >= self.STALE_THRESHOLD_WARN_US:
                logger.warning("STATE 严重过期: age=%.2fs（阈值=%.0fms）",
                               age_us / 1e6, stale_threshold_us / 1000)
            else:
                logger.warning("STATE 过期: age=%.1fms（阈值=%.0fms）",
                               age_us / 1000, stale_threshold_us / 1000)
        return age_us < stale_threshold_us

    def get_state_freshness(self, state):
        """
        返回 STATE 新鲜度详细信息（用于日志/调试）

        Returns:
            (age_us: int, is_fresh: bool) 或 (None, False)
        """
        if state is None:
            return None, False
        try:
            now_us = int(time.time() * 1_000_000)
            age_us = now_us - state["drone"]["last_update_us"]
            return age_us, age_us < self.STALE_THRESHOLD_US
        except (KeyError, TypeError):
            return None, False

    # ---- 回调注册 ----

    def set_state_callback(self, callback):
        """
        注册 STATE 回调（在 SUB 线程中调用，不应阻塞）
        callback 签名: callback(state_dict)
        """
        self._state_callback = callback

    def set_alert_callback(self, callback):
        """
        注册 ALERT 回调（在 SUB 线程中调用，不应阻塞）
        callback 签名: callback(alert_dict)
        """
        self._alert_callback = callback

    # ================================================================
    #  生命周期管理
    # ================================================================

    def start(self):
        """启动 SUB 监听后台线程"""
        if self._running:
            return
        self._running = True
        self._sub_thread = threading.Thread(
            target=self._sub_worker,
            name="router-proxy-sub",
            daemon=True
        )
        self._sub_thread.start()
        logger.info("RouterProxy 已启动，SUB 监听线程已开启（REQ=%s SUB=%s）",
                    self.req_endpoint, self.sub_endpoint)

    def _sub_worker(self):
        """后台线程：通过 zmq.Poller 非阻塞接收 SUB 消息"""
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        while self._running:
            try:
                socks = dict(poller.poll(timeout=100))  # 100 ms 超时轮询
                if self.sub not in socks:
                    continue

                msg_str = self.sub.recv_string()
                data = json.loads(msg_str)
                msg_type = data.get("type")

                if msg_type == "STATE":
                    with self._state_lock:
                        self._latest_state = data
                        if not self._first_state_received:
                            self._first_state_received = True
                            logger.info("首次收到 Router 状态推送")

                    # DEBUG 输出关键字段 + 新鲜度
                    try:
                        drone = data.get("drone", {})
                        now_us = int(time.time() * 1_000_000)
                        last_up = drone.get("last_update_us", now_us)
                        age_us = now_us - last_up
                        logger.debug("STATE: mode=%s armed=%s "
                                     "alt_rel=%.1f batt=%.0f%% age=%.0fms",
                                     drone.get("mode"), drone.get("armed"),
                                     drone.get("alt_rel", 0),
                                     drone.get("battery", 0),
                                     age_us / 1000)
                    except Exception:
                        pass

                    if self._state_callback:
                        try:
                            self._state_callback(data)
                        except Exception as e:
                            logger.error("STATE 回调异常: %s", e)

                elif msg_type == "ALERT":
                    logger.warning("⚠️ ALERT: %s — %s",
                                   data.get("alert"), data.get("message"))
                    if self._alert_callback:
                        try:
                            self._alert_callback(data)
                        except Exception as e:
                            logger.error("ALERT 回调异常: %s", e)

                elif msg_type == "PX4_ACK":
                    logger.info("PX4_ACK: cmd=%s result=%s",
                                data.get("ref_cmd"), data.get("result"))

                elif msg_type == "QUERY_REPLY":
                    # QUERY_REPLY 由 send_query 的同步 recv 处理，
                    # SUB 通道上的 QUERY_REPLY 暂不处理
                    pass

                else:
                    logger.warning("收到未知消息类型: %s", msg_type)

            except zmq.Again:
                continue
            except json.JSONDecodeError as e:
                logger.warning("SUB 收到非 JSON 消息: %s", e)
                continue
            except Exception as e:
                logger.error("SUB 线程异常: %s", e)
                time.sleep(0.1)

    def close(self):
        """关闭代理，释放所有 ZMQ 资源"""
        self._running = False
        if self._sub_thread and self._sub_thread.is_alive():
            self._sub_thread.join(timeout=2.0)

        for name, sock in [("REQ", self.req), ("SUB", self.sub)]:
            try:
                sock.close(linger=0)
            except Exception:
                pass

        try:
            self.ctx.term()
        except Exception:
            pass

        logger.info("RouterProxy 已关闭")
