"""
mission_manager.py — 基于 ZeroMQ RouterProxy 的航点巡航状态机

【协议变更 v2 — 2026-07】
  航点存储格式使用经纬度 (lat, lon, alt)。发送 POSITION SETPOINT
  时在本模块内转换为相对当前位置的 LOCAL_OFFSET_NED 偏移
  (x=北向, y=东向, z=向下)。

【核心改动】
  1. self.drone → self.proxy（注入的 RouterProxy 实例）
  2. 所有状态读取：proxy.get_latest_state()["drone"][...]
  3. 所有指令发送：proxy.send_command() / send_waypoint() / send_setpoint()
  4. 新增 STATE 新鲜度校验（协议 §3.8.1），过期数据保守处理
  5. 新增 Failsafe 模式断流保护，退出 OFFBOARD 即熔断
  6. VISUAL_TRACKING 状态下完全让渡控制权，不发送任何 SETPOINT

【位置判定】
  航点存储为经纬高 (lat, lon, alt)，使用 Haversine 公式计算
  球面距离进行到达判定，不再依赖速度积分推算 NED 位置。

【控制权协调】
  MissionManager 通过自有 proxy 发送指令。连续 20Hz SETPOINT
  流由 UAVControlLoop 的独立线程负责。run_mission.py 负责协调：
  在 VISUAL_TRACKING 外停止 UAVControlLoop 线程，避免两套
  SETPOINT 流同时激活产生冲突。
"""

import time
import logging
from enum import Enum

from utils.coord import haversine, dist_3d_latlon, latlon_alt_to_local_offset

logger = logging.getLogger(__name__)


class MissionState(Enum):
    """无人机任务状态枚举"""
    WAITING_WAYPOINTS = -1  # 等待地面站下发航点
    INIT = 0                # 初始化 / 等待连接
    ARMING = 1              # 解锁中
    TAKEOFF = 2             # 自动起飞中
    NAVIGATING = 3          # 航点巡航导航
    HOLD_TASK = 4           # 到达航点后的原地搜索 / 悬停
    VISUAL_TRACKING = 5     # 🎯 视觉追踪中（控制权让渡给 UAVControlLoop）
    RETURNING = 6           # 返航中
    LANDING = 7             # 自动降落中
    FINISHED = 8            # 任务结束 / 已上锁
    HOLD_FINAL = 9          # 结束巡点后原地悬停（return_to_home=False）


class FailsafeTriggered(Exception):
    """
    自定义安全异常：当安全员人工介入切出自控模式时抛出，
    用于暴力熔断主循环。
    """
    pass


# ================================================================
#  航点巡航管理器
# ================================================================

class MissionManager:
    """
    🛸 ZMQ 航点巡航状态机管理器（经纬度航点 + 相对位置 SETPOINT）

    职责：
      - 封装完整的无人机任务生命周期（初始化 → 解锁 → 起飞 →
        航点巡航 → 视觉追踪 → 返航 → 降落）
      - 所有底层通信通过注入的 RouterProxy 完成
      - 在 VISUAL_TRACKING 状态下让渡控制权给 UAVControlLoop
      - Failsafe 熔断保护

    航点格式（经纬度协议 v2）：
        {
            "lat": 37.7749,        # 纬度（度）
            "lon": -122.4194,      # 经度（度）
            "alt": 1.5,            # 相对起飞点高度（米，正值向上）
            "yaw": 0.0,            # 偏航角（度）
            "command": "WAYPOINT", # TAKEOFF / LAND / WAYPOINT
            "hold_duration": 8.0,  # 悬停搜索等待时长（秒）
        }

    使用示例：
        proxy = RouterProxy()
        proxy.start()
        manager = MissionManager(proxy, waypoints=[...], target_altitude=1.5)

        while True:
            manager.update(target_detected=some_bool)
            time.sleep(0.05)
    """

    # ---- Failsafe：自主飞行的合法模式白名单 ----
    AUTONOMOUS_MODES = frozenset({
        "OFFBOARD", "GUIDED", "AUTO", "HOLD", "LAND", "RTL",
    })

    # ---- 解锁/起飞阶段允许的过渡模式 ----
    TRANSITION_MODES = frozenset({
        "MANUAL", "ALTCTL", "POSCTL", "AUTO", "HOLD",
        "AUTO.READY", "AUTO.TAKEOFF", "AUTO.LOITER",
    })

    # ---- 起飞完成后可接受的悬停/保持模式 ----
    TAKEOFF_HOLD_MODES = frozenset({
        "HOLD", "AUTO", "AUTO.LOITER", "AUTO.TAKEOFF",
    })

    # ---- 目标丢失超时（秒） ----
    DEFAULT_TARGET_LOST_TIMEOUT = 3.0

    # ---- 各阶段超时 ----
    TAKEOFF_TIMEOUT = 15.0
    OFFBOARD_TIMEOUT = 5.0
    ARM_TIMEOUT = 5.0

    def __init__(self, proxy, waypoints, target_altitude=1.5,
                 arrival_radius=0.3, hold_duration=5.0,
                 return_to_home=True, target_lost_timeout=None):
        """
        Args:
            proxy:               RouterProxy 实例（调用方负责 start/close）
            waypoints:           航点列表（经纬度格式 v2）
            target_altitude:     默认起飞高度（米，正值向上）
            arrival_radius:      航点到达判定半径（米）
            hold_duration:       巡点无目标的最大悬停搜索时间（秒）
            return_to_home:      航点巡完后是否返航，False 则终点悬停
            target_lost_timeout: VISUAL_TRACKING 目标丢失超时（秒）
        """
        # ---- 依赖注入 ----
        self.proxy = proxy

        # ---- 任务参数 ----
        # 航点列表（经纬度协议 v2：[{lat, lon, alt, yaw, command}, ...]）
        self.waypoints = list(waypoints) if waypoints else []
        # 根据是否有航点决定初始状态
        if self.waypoints:
            self.state = MissionState.INIT
            self._waypoints_ready = True
            logger.info("MissionManager 初始化完成，已有 %d 个航点", len(self.waypoints))
        else:
            self.state = MissionState.WAITING_WAYPOINTS
            self._waypoints_ready = False
            logger.info("MissionManager 初始化完成，等待 QGC 下发航点...")
        # v2: 目标高度正值向上（原 NED 负值协议已废除）
        self.target_altitude = abs(target_altitude)
        self.arrival_radius = arrival_radius
        self.hold_duration = hold_duration
        self.return_to_home = return_to_home
        self.target_lost_timeout = (
            target_lost_timeout if target_lost_timeout is not None
            else self.DEFAULT_TARGET_LOST_TIMEOUT
        )

        # ---- 状态机 ----
        self.current_wp_index = 0

        # ---- 当前目标航点（共享给外部主循环 / UAVControlLoop） ----
        self.current_target = {
            "lat": 0.0, "lon": 0.0,
            "alt": self.target_altitude, "yaw": 0.0,
        }

        # ---- 计时器 ----
        self.hold_start_time = 0.0
        self.tracking_start_time = 0.0
        self._target_lost_start = None
        self._state_start_time = time.time()

        # ================================================================
        #  追踪完成信号（外部设置）
        # ================================================================
        self.tracking_completed = False
        self.tracking_max_duration = 10.0

        logger.info("MissionManager 初始化: %d 个航点, 目标高度=%.1fm, "
                     "到达半径=%.1fm, 搜寻时长=%.1fs, 返航=%s",
                     len(self.waypoints), self.target_altitude,
                     self.arrival_radius, self.hold_duration,
                     self.return_to_home)

        # ---- 日志节流控制 ----
        self._last_throttle_time = {}
        self._last_throttle_state = {}

    # ================================================================
    #  主入口
    # ================================================================

    def update(self, target_detected=False):
        """
        状态机高频核心业务逻辑，由外部主循环驱动（建议 ≥20Hz）

        Args:
            target_detected: 当前帧视觉检测是否有目标

        Raises:
            FailsafeTriggered: 检测到非法模式切换时抛出
        """
        # ---------------------------------------------------------------
        #  步骤 0：获取最新 STATE + 新鲜度校验 + Failsafe
        # ---------------------------------------------------------------
        state = self.proxy.get_latest_state()

        # ---- 新鲜度校验（协议 §3.8.1） ----
        if self.state not in (MissionState.INIT, MissionState.FINISHED):
            if state is not None:
                if not self.proxy.is_state_fresh(state):
                    age_us, _ = self.proxy.get_state_freshness(state)
                    age_s = (age_us or 0) / 1_000_000
                    logger.warning("⚠️ STATE 严重过期 (age=%.2fs)，"
                                   "本轮保守跳过", age_s)
                    return
            else:
                logger.warning("⚠️ 尚未收到 STATE，跳过本轮处理")
                return

        # ---- 特殊状态，等待航点 ----
        if self.state == MissionState.WAITING_WAYPOINTS:
            if self._waypoints_ready:
                logger.info("✅ 航点已就绪，转入 INIT 状态")
                self.state = MissionState.INIT
                self._state_start_time = time.time()
            else:
                return

        # ---- Failsafe 越权检查 ----
        if self.state not in (MissionState.INIT, MissionState.FINISHED):
            if state is not None:
                mode = state["drone"].get("mode", "")
                if not self._is_mode_allowed_for_state(mode, self.state):
                    logger.warning(
                        "\n⚠️ [Failsafe] 飞控模式被切为: %s！"
                        "人工接管，熔断任务！", mode
                    )
                    self.state = MissionState.FINISHED
                    raise FailsafeTriggered(
                        f"安全员接管，当前模式: {mode}"
                    )

        # ---- 无 STATE 时仅 INIT 可容忍 ----
        if state is None:
            if self.state == MissionState.INIT:
                logger.info("[INIT] 等待 Router 状态推送...")
            return

        # ---------------------------------------------------------------
        #  状态机分支（位置估计已移除 v2 — 使用经纬度距离计算）
        # ---------------------------------------------------------------
        try:
            if self.state == MissionState.INIT:
                self._handle_init(state)
            elif self.state == MissionState.ARMING:
                self._handle_arming(state)
            elif self.state == MissionState.TAKEOFF:
                self._handle_takeoff(state)
            elif self.state == MissionState.NAVIGATING:
                self._handle_navigating(state)
            elif self.state == MissionState.HOLD_TASK:
                self._handle_hold_task(state, target_detected)
            elif self.state == MissionState.VISUAL_TRACKING:
                self._handle_visual_tracking(state, target_detected)
            elif self.state == MissionState.RETURNING:
                self._handle_returning(state)
            elif self.state == MissionState.LANDING:
                self._handle_landing(state)
            elif self.state == MissionState.HOLD_FINAL:
                self._handle_hold_final(state)
        except FailsafeTriggered:
            raise
        except Exception as e:
            logger.error("状态机异常 (state=%s): %s", self.state.name, e,
                         exc_info=True)

    # ================================================================
    #  辅助方法
    # ================================================================

    def _is_autonomous_mode(self, mode):
        """判断是否属于自主飞行模式，兼容 PX4 AUTO 子模式。"""
        if mode in self.AUTONOMOUS_MODES:
            return True
        return isinstance(mode, str) and mode.startswith("AUTO.")

    def _is_transition_mode(self, mode):
        """解锁/起飞阶段允许的 PX4 过渡模式。"""
        if mode in self.TRANSITION_MODES:
            return True
        return isinstance(mode, str) and mode.startswith("AUTO.")

    def _is_mode_allowed_for_state(self, mode, state):
        """按任务阶段判断当前飞控模式是否允许。"""
        if state in (MissionState.ARMING, MissionState.TAKEOFF):
            return self._is_transition_mode(mode)
        return self._is_autonomous_mode(mode)

    def _is_takeoff_hold_mode(self, mode):
        """起飞高度到位后可接受的保持/悬停状态。"""
        return mode in self.TAKEOFF_HOLD_MODES

    def _get_current_alt(self, state):
        """从 STATE 获取当前相对高度（正值向上）"""
        try:
            return state["drone"].get("alt_rel", 0.0)
        except (KeyError, TypeError):
            return self.target_altitude

    # ---- 日志节流辅助 ----

    def _throttle_log(self, key, interval, msg, *args, level=logging.INFO):
        """节流日志：同一 key 在 interval 秒内只输出一次"""
        now = time.time()
        last = self._last_throttle_time.get(key, 0)
        if now - last >= interval:
            self._last_throttle_time[key] = now
            logger.log(level, msg, *args)

    def _throttle_log_on_change(self, key, current_value, msg, *args,
                                 level=logging.INFO):
        """状态变化日志：仅当 current_value 与上次记录不同时输出一次"""
        last = self._last_throttle_state.get(key)
        if last != current_value:
            self._last_throttle_state[key] = current_value
            logger.log(level, msg, *args)

    # ================================================================
    #  状态处理器
    # ================================================================

    # ---- 初始化 ----

    def _handle_init(self, state):
        """
        [INIT] 等待飞控连接，判断是否需要执行完整起飞序列

        三种分支：
          A) 已解锁 + OFFBOARD → 直接跳 NAVIGATING
          B) 未解锁 → 发 ARM 指令
          C) 已解锁但非 OFFBOARD → 进入 ARMING 等待后续 TAKEOFF 序列
        """
        try:
            drone = state["drone"]
            armed = drone.get("armed", False)
            mode = drone.get("mode", "")
            alt_rel = drone.get("alt_rel", 0.0)
            home = state.get("home", {})

            self._throttle_log("init_state", 5.0, "[INIT] mode=%s armed=%s alt_rel=%.1fm",
                               mode, armed, alt_rel)

            if not home.get("valid", False):
                self._throttle_log("home_valid_init", 5.0,
                                   "[INIT] home.valid=false，等待起飞点有效...")
                return

            # A) 已在 OFFBOARD + 已解锁 → 直接巡航
            if armed and mode == "OFFBOARD":
                logger.info("[INIT] 已在 OFFBOARD，直接进入航点巡航")
                self.state = MissionState.NAVIGATING
                self.current_wp_index = 0
                self._set_current_target_from_wp(0)
                return

            # B) 未解锁
            if not armed:
                logger.info("[INIT] 执行解锁...")
                ok, ack = self.proxy.send_command("ARM")
                if ok:
                    self.state = MissionState.ARMING
                    self._state_start_time = time.time()
                else:
                    logger.error("[INIT] ARM 失败: %s", ack)
                return

            # C) 已解锁但非 OFFBOARD
            if armed and mode != "OFFBOARD":
                logger.info("[INIT] 已解锁但 mode=%s，进入起飞序列", mode)
                self.state = MissionState.ARMING
                self._state_start_time = time.time()

        except (KeyError, TypeError) as e:
            logger.warning("[INIT] 解析异常: %s", e)

    # ---- 解锁 ----

    def _handle_arming(self, state):
        """
        [ARMING] 等待解锁确认，成功后进入 TAKEOFF
        """
        try:
            armed = state["drone"].get("armed", False)
            home = state.get("home", {})

            if not armed:
                elapsed = time.time() - self._state_start_time
                if elapsed > self.ARM_TIMEOUT:
                    logger.warning("[ARMING] 超时 (%.1fs)，重试 ARM", elapsed)
                    self.proxy.send_command("ARM")
                    self._state_start_time = time.time()
                else:
                    self._throttle_log("arming_wait", 3.0,
                                       "[ARMING] 等待解锁... (%.1fs)", elapsed)
                return

            if not home.get("valid", False):
                self._throttle_log("home_valid_arming", 5.0,
                                   "[ARMING] home.valid=false，等待起飞点有效...")
                return

            # ---- 解锁成功 → 起飞 ----
            logger.info("\n[解锁成功] 发送起飞指令 (alt=%.1fm)", self.target_altitude)

            ok, ack = self.proxy.send_waypoint(
                lat=state["home"]["lat"],
                lon=state["home"]["lon"],
                alt=self.target_altitude,
                alt_frame="RELATIVE",
                speed=3.0, action="TAKEOFF",
                yaw=state["drone"]["yaw"],
                acceptance_radius=0.5,
                is_last=True
            )
            if ok:
                self.state = MissionState.TAKEOFF
                self._state_start_time = time.time()
            else:
                logger.warning("[TAKEOFF] WAYPOINT 失败: %s，尝试 SETPOINT 爬升", ack)
                drone = state.get("drone", {})
                ok2, _ = self._send_relative_position_setpoint(
                    state,
                    drone.get("lat", 0.0), drone.get("lon", 0.0),
                    self.target_altitude, yaw=0.0,
                )
                if ok2:
                    self.state = MissionState.TAKEOFF
                    self._state_start_time = time.time()
                else:
                    logger.error("[TAKEOFF] 降级爬升也失败")

        except (KeyError, TypeError) as e:
            logger.warning("[ARMING] 异常: %s", e)

    # ---- 起飞 ----

    def _handle_takeoff(self, state):
        """
        [TAKEOFF] 等待 WAYPOINT(TAKEOFF) 完成，然后预热并切换 OFFBOARD

        流程（协议 §3.3.1）：
          ① 等待高度到达目标 90% 且 mode == "HOLD"
          ② 发送预热 SETPOINT（相对当前位置悬停）
          ③ time.sleep(0.2) 等待流建立
          ④ 发送 COMMAND("OFFBOARD")
          ⑤ 轮询等待 mode == "OFFBOARD"
          ⑥ 跳转 NAVIGATING
        """
        try:
            drone = state["drone"]
            home = state.get("home", {})
            alt_rel = drone.get("alt_rel", 0.0)
            mode = drone.get("mode", "")

            # ---- ① 等待 TAKEOFF 完成 ----
            reached_alt = alt_rel >= self.target_altitude * 0.9
            in_hold = self._is_takeoff_hold_mode(mode)

            if not (reached_alt and in_hold):
                self._throttle_log("takeoff_wait", 3.0,
                                   "[起飞中] alt=%.1f/%.1f mode=%s 耗时=%.1fs",
                                   alt_rel, self.target_altitude, mode,
                                   time.time() - self._state_start_time)
                return

            # ---- ② 高度到位 + HOLD 模式 → 预热 + 切 OFFBOARD ----
            logger.info("[TAKEOFF] 高度 %.1fm，模式 %s，准备切换 OFFBOARD...",
                        alt_rel, mode)

            # 发送预热 SETPOINT（相对当前位置悬停）
            self._send_relative_position_setpoint(
                state,
                drone.get("lat", 0.0), drone.get("lon", 0.0),
                alt_rel, yaw=0.0,
            )
            time.sleep(0.2)

            # 发送 OFFBOARD 指令
            ok, ack = self.proxy.send_command("OFFBOARD")
            if not ok:
                logger.error("[TAKEOFF] OFFBOARD 指令失败: %s", ack)
                if time.time() - self._state_start_time > self.TAKEOFF_TIMEOUT:
                    logger.warning("[TAKEOFF] 超时，强行进入巡航")
                    self.state = MissionState.NAVIGATING
                    self.current_wp_index = 0
                    self._set_current_target_from_wp(0)
                return

            # 等待模式切换
            for _ in range(int(self.OFFBOARD_TIMEOUT / 0.1)):
                st = self.proxy.get_latest_state()
                if st and st["drone"]["mode"] == "OFFBOARD":
                    logger.info("[TAKEOFF] 已进入 OFFBOARD")
                    self.state = MissionState.NAVIGATING
                    self.current_wp_index = 0
                    self._set_current_target_from_wp(0)
                    return
                time.sleep(0.1)

            logger.warning("[TAKEOFF] OFFBOARD 切换超时，强行进入巡航")
            self.state = MissionState.NAVIGATING
            self.current_wp_index = 0
            self._set_current_target_from_wp(0)

        except (KeyError, TypeError) as e:
            logger.warning("[TAKEOFF] 异常: %s", e)

    # ---- 航点导航 ----

    def _handle_navigating(self, state):
        """
        [NAVIGATING] 每帧发送 POSITION SETPOINT 前往当前航点

        POSITION SETPOINT 发送 LOCAL_OFFSET_NED 相对偏移；到达判定使用 Haversine 距离。
        到达判定后推进索引；所有航点飞完后根据 return_to_home 决定返航或悬停。
        """
        if not self.waypoints or self.current_wp_index >= len(self.waypoints):
            self._finish_navigation(state)
            return

        wp = self.waypoints[self.current_wp_index]
        cmd = wp.get("command", "WAYPOINT")

        # ---- 起飞航点 ----
        if cmd == "TAKEOFF":
            # 起飞已完成，直接推进到下一个航点
            self.current_wp_index += 1
            return

        # ---- 降落航点 ----
        if cmd == "LAND":
            logger.info("🛬 执行降落")
            self.proxy.send_command("LAND")
            self.state = MissionState.LANDING
            return

        # ---- 普通航点 ----
        wp_lat = wp["lat"]
        wp_lon = wp["lon"]
        wp_alt = wp.get("alt", self.target_altitude)
        wp_yaw = wp.get("yaw", 0.0)

        # 发送 POSITION SETPOINT（相对当前位置 LOCAL_OFFSET_NED）
        self._send_relative_position_setpoint(
            state, wp_lat, wp_lon, wp_alt, yaw=wp_yaw,
        )

        # ---- 共享目标 ----
        self.current_target = {"lat": wp_lat, "lon": wp_lon,
                               "alt": wp_alt, "yaw": wp_yaw}

        # ---- 到达判定（Haversine 球面距离） ----
        drone = state["drone"]
        drone_lat = drone.get("lat", 0.0)
        drone_lon = drone.get("lon", 0.0)
        current_alt = self._get_current_alt(state)
        dist = dist_3d_latlon(drone_lat, drone_lon, current_alt,
                              wp_lat, wp_lon, wp_alt)

        if dist < self.arrival_radius:
            logger.info("\n✅ [到达航点 %d/%d] dist=%.2fm",
                        self.current_wp_index + 1, len(self.waypoints), dist)
            self.state = MissionState.HOLD_TASK
            self.hold_start_time = time.time()
        else:
            self._throttle_log("navigating", 5.0,
                               "[巡航中] 航点 [%d/%d] 距离=%.2fm",
                               self.current_wp_index + 1,
                               len(self.waypoints), dist)

    def _finish_navigation(self, state):
        """所有航点飞完后的跳转"""
        home = state.get("home", {}) if state else {}
        home_lat = home.get("lat", 0.0)
        home_lon = home.get("lon", 0.0)

        if self.return_to_home:
            logger.info("\n[巡航完成] 返回起始点 (lat=%.6f lon=%.6f)...",
                        home_lat, home_lon)
            self.state = MissionState.RETURNING
            self.current_target = {
                "lat": home_lat, "lon": home_lon,
                "alt": self.target_altitude, "yaw": 0,
            }
        else:
            logger.info("\n[巡航完成] 末端悬停...")
            self.state = MissionState.HOLD_FINAL
        self._state_start_time = time.time()

    # ---- 悬停搜索 ----

    def _handle_hold_task(self, state, target_detected):
        """
        [HOLD_TASK] 抵达航点后的悬停搜索

        策略：
          - 检测到目标 → VISUAL_TRACKING
          - 超时无目标 → 下一个航点
          - 持续发送 POSITION SETPOINT 悬停
        """
        # ---- 发现目标 ----
        if target_detected:
            logger.info("\n🎯 [目标锁定] 切入视觉追踪状态！")
            self.state = MissionState.VISUAL_TRACKING
            self.tracking_start_time = time.time()
            self._target_lost_start = None
            self.tracking_completed = False
            return

        # ---- 保持当前位置悬停（相对当前位置 LOCAL_OFFSET_NED） ----
        wp = self.waypoints[self.current_wp_index]
        self._send_relative_position_setpoint(
            state,
            wp["lat"], wp["lon"], wp.get("alt", self.target_altitude),
            yaw=wp.get("yaw", 0),
        )
        self.current_target = {
            "lat": wp["lat"], "lon": wp["lon"],
            "alt": wp.get("alt", self.target_altitude),
            "yaw": wp.get("yaw", 0),
        }

        # ---- 超时判定 ----
        elapsed = time.time() - self.hold_start_time
        if elapsed >= self.hold_duration:
            logger.info("\n[未发现目标] 搜寻超时 (%.1fs)，前往下一航点",
                        self.hold_duration)
            self.current_wp_index += 1
            self.state = MissionState.NAVIGATING
        else:
            self._throttle_log("hold_task", 3.0,
                               "[检索中] 航点 %d 剩余 %.1fs",
                               self.current_wp_index + 1,
                               self.hold_duration - elapsed)

    # ---- 视觉追踪 ----

    def _handle_visual_tracking(self, state, target_detected):
        """
        [VISUAL_TRACKING] 🎯 完全让渡控制权给 UAVControlLoop

        MissionManager 在此状态下 **不发送任何 SETPOINT**。
        UAVControlLoop (20Hz 独立线程) 通过其自有 proxy 发送
        VELOCITY SETPOINT 进行视觉闭环追踪。

        退出条件：
          1. self.tracking_completed == True（外部模块设置）
          2. 目标丢失超过 target_lost_timeout 秒
          3. 追踪总时长超过 tracking_max_duration（兜底）
        """
        tracking_elapsed = time.time() - self.tracking_start_time

        # ---- 兜底超时 ----
        if tracking_elapsed > self.tracking_max_duration:
            logger.info("\n⏰ [追踪超时] %.1fs ≥ %.1fs",
                        tracking_elapsed, self.tracking_max_duration)
            self._exit_tracking()
            return

        # ---- 外部完成信号 ----
        if self.tracking_completed:
            logger.info("\n✅ [追踪完成] 外部模块报告完成，返回巡航")
            self.tracking_completed = False
            self._exit_tracking()
            return

        # ---- 目标丢失超时 ----
        if not target_detected:
            if self._target_lost_start is None:
                self._target_lost_start = time.time()
                lost_elapsed = 0.0
            else:
                lost_elapsed = time.time() - self._target_lost_start
                if lost_elapsed >= self.target_lost_timeout:
                    logger.info("\n⏰ [丢失超时] %.1fs，放弃追踪",
                                lost_elapsed)
                    self._exit_tracking()
                    return
            self._throttle_log("track_lost", 2.0,
                               "[追踪中] 丢失 %.1fs/%.1fs",
                               lost_elapsed if self._target_lost_start
                               else 0.0,
                               self.target_lost_timeout)
        else:
            self._throttle_log_on_change("track_lost_clear",
                                         self._target_lost_start is not None,
                                         "[追踪中] 目标恢复")
            self._target_lost_start = None

        self._throttle_log("track_heartbeat", 5.0,
                           "[追踪中] 进行中... %.1fs", tracking_elapsed)

    def _exit_tracking(self):
        """退出 VISUAL_TRACKING 的统一出口"""
        self.current_wp_index += 1
        self.state = MissionState.NAVIGATING
        self._target_lost_start = None
        logger.info("[状态切换] VISUAL_TRACKING → NAVIGATING，"
                    "前往航点 %d", self.current_wp_index + 1)

    # ---- 返航 ----

    def _handle_returning(self, state):
        """
        [RETURNING] 发送 POSITION SETPOINT 到 HOME 点

        POSITION SETPOINT 发送到 HOME 的 LOCAL_OFFSET_NED 相对偏移；
        到达判定使用 Haversine 球面距离。
        """
        home = state.get("home", {})
        drone = state.get("drone", {})
        home_lat = home.get("lat", 0.0)
        home_lon = home.get("lon", 0.0)

        # 发送 POSITION SETPOINT（相对当前位置 LOCAL_OFFSET_NED）
        self._send_relative_position_setpoint(
            state, home_lat, home_lon, self.target_altitude, yaw=0,
        )
        self.current_target = {
            "lat": home_lat, "lon": home_lon,
            "alt": self.target_altitude, "yaw": 0,
        }

        # ---- Haversine 距离到达判定 ----
        current_alt = self._get_current_alt(state)
        drone_lat = drone.get("lat", 0.0)
        drone_lon = drone.get("lon", 0.0)
        dist = dist_3d_latlon(drone_lat, drone_lon, current_alt,
                              home_lat, home_lon, self.target_altitude)

        self._throttle_log("returning", 3.0,
                           "[返航中] 距离=%.2fm", dist)

        if dist < self.arrival_radius:
            logger.info("\n[到达] 已返回起始点，发送 LAND 指令...")
            ok, ack = self.proxy.send_command("LAND")
            if ok:
                self.state = MissionState.LANDING
            else:
                logger.warning("[返航] LAND 失败: %s", ack)

    # ---- 降落 ----

    def _handle_landing(self, state):
        """[LANDING] 监控 disarm 状态判断着陆"""
        try:
            armed = state["drone"].get("armed", False)
            alt_rel = state["drone"].get("alt_rel", 0.0)
            self._throttle_log("landing", 3.0,
                               "[降落中] 高度=%.1fm 已解锁=%s",
                               alt_rel, not armed)

            if not armed:
                logger.info("\n[完结] 已着陆上锁，任务结束")
                self.state = MissionState.FINISHED
        except (KeyError, TypeError) as e:
            logger.warning("[LANDING] 异常: %s", e)

    # ---- 末端悬停 ----

    def _handle_hold_final(self, state):
        """
        [HOLD_FINAL] 末端悬停，保持最终航点位置

        POSITION SETPOINT 发送 LOCAL_OFFSET_NED 相对偏移。
        """
        if self.waypoints:
            final_wp = self.waypoints[-1]
        else:
            final_wp = {"lat": 0.0, "lon": 0.0,
                        "alt": self.target_altitude, "yaw": 0}

        self._send_relative_position_setpoint(
            state,
            final_wp["lat"], final_wp["lon"],
            final_wp.get("alt", self.target_altitude),
            yaw=final_wp.get("yaw", 0),
        )
        self.current_target = {
            "lat": final_wp["lat"], "lon": final_wp["lon"],
            "alt": final_wp.get("alt", self.target_altitude),
            "yaw": final_wp.get("yaw", 0),
        }
        self._throttle_log("hold_final", 10.0,
                           "[挂起] 保持 (%.6f, %.6f, %.1f) 等待接管",
                           final_wp["lat"], final_wp["lon"],
                           final_wp.get("alt", self.target_altitude))

    # ================================================================
    #  辅助
    # ================================================================

    def _send_relative_position_setpoint(self, state, target_lat, target_lon,
                                         target_alt, yaw=0.0):
        """按相对当前位置的 LOCAL_OFFSET_NED 发送 POSITION SETPOINT。"""
        drone = state.get("drone", {}) if state else {}
        current_lat = drone.get("lat", 0.0)
        current_lon = drone.get("lon", 0.0)
        current_alt = self._get_current_alt(state)

        x, y, z = latlon_alt_to_local_offset(
            target_lat, target_lon, target_alt,
            current_lat, current_lon, current_alt,
        )

        return self.proxy.send_setpoint(
            x=x, y=y, z=z, yaw=yaw,
            frame="LOCAL_OFFSET_NED",
            control_mode="POSITION",
        )

    def _set_current_target_from_wp(self, index):
        """从航点列表更新 current_target"""
        if self.waypoints and index < len(self.waypoints):
            wp = self.waypoints[index]
            self.current_target = {
                "lat": wp["lat"], "lon": wp["lon"],
                "alt": wp.get("alt", self.target_altitude),
                "yaw": wp.get("yaw", 0),
            }

    def set_waypoints(self, waypoints):
        """由外部调用，动态注入/更新航点列表（经纬度格式 v2）"""
        self.waypoints = list(waypoints)
        self.current_wp_index = 0
        self._waypoints_ready = True
        logger.info("MissionManager 航点已更新，共 %d 个", len(waypoints))

    # ================================================================
    #  🚨 紧急停靠（外部调用）
    # ================================================================

    def emergency_stop(self):
        """外部触发的紧急降落入口（可在任何状态下调用）"""
        logger.warning("🚨 [emergency_stop] 紧急降落触发！")

        ok, ack = self.proxy.send_command("LAND")
        if ok:
            logger.info("✅ LAND 指令已发送")
        else:
            logger.error(f"❌ LAND 指令发送失败: {ack}")

        self.state = MissionState.LANDING
        self._state_start_time = time.time()
