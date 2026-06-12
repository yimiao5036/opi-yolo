"""
mission_manager.py — 基于 ZeroMQ RouterProxy 的航点巡航状态机

【重构背景】
  系统底层架构已升级为 ZMQ Router 代理通信协议。本模块彻底摒弃
  旧版 DroneController 直连方式，全面迁移至 RouterProxy。

【核心改动】
  1. self.drone → self.proxy（注入的 RouterProxy 实例）
  2. 所有状态读取：proxy.get_latest_state()["drone"][...]
  3. 所有指令发送：proxy.send_command() / send_waypoint() / send_setpoint()
  4. 新增 STATE 新鲜度校验（协议 §3.8.1），过期数据保守处理
  5. 新增 Failsafe 模式断流保护，退出 OFFBOARD 即熔断
  6. VISUAL_TRACKING 状态下完全让渡控制权，不发送任何 SETPOINT

【位置估计】
  Router 协议未暴露局部 NED 坐标，本模块使用速度积分
  （vx × dt 累加）推算相对起飞点的水平位置，用于航点到
  达判定。垂直轴直接使用 state["drone"]["alt_rel"]。

【控制权协调】
  MissionManager 通过自有 proxy 发送指令。连续 20Hz SETPOINT
  流由 UAVControlLoop 的独立线程负责。run_mission.py 负责协调：
  在 VISUAL_TRACKING 外停止 UAVControlLoop 线程，避免两套
  SETPOINT 流同时激活产生冲突。
"""

import time
import math
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MissionState(Enum):
    """无人机任务状态枚举"""
    INIT = 0              # 初始化 / 等待连接
    ARMING = 1            # 解锁中
    TAKEOFF = 2           # 自动起飞中
    NAVIGATING = 3        # 航点巡航导航
    HOLD_TASK = 4         # 到达航点后的原地搜索 / 悬停
    VISUAL_TRACKING = 5   # 🎯 视觉追踪中（控制权让渡给 UAVControlLoop）
    RETURNING = 6         # 返航中
    LANDING = 7           # 自动降落中
    FINISHED = 8          # 任务结束 / 已上锁
    HOLD_FINAL = 9        # 结束巡点后原地悬停（return_to_home=False）


class FailsafeTriggered(Exception):
    """
    自定义安全异常：当安全员人工介入切出自控模式时抛出，
    用于暴力熔断主循环。
    """
    pass


# ================================================================
#  内部工具
# ================================================================

def _dist_3d(x1, y1, z1, x2, y2, z2):
    """三维欧氏距离"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


# ================================================================
#  航点巡航管理器
# ================================================================

class MissionManager:
    """
    🛸 ZMQ 航点巡航状态机管理器

    职责：
      - 封装完整的无人机任务生命周期（初始化 → 解锁 → 起飞 →
        航点巡航 → 视觉追踪 → 返航 → 降落）
      - 所有底层通信通过注入的 RouterProxy 完成
      - 在 VISUAL_TRACKING 状态下让渡控制权给 UAVControlLoop
      - Failsafe 熔断保护

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

    # ---- 位置估计：速度积分有效时间窗口 ----
    MAX_POSITION_DT = 1.0

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
            waypoints:           航点列表 [{'x':1,'y':2,'z':-1.5,'yaw':0}, ...]
            target_altitude:     默认起飞高度（米，正值=向上）
            arrival_radius:      航点到达判定半径（米）
            hold_duration:       巡点无目标的最大悬停搜索时间（秒）
            return_to_home:      航点巡完后是否返航，False 则终点悬停
            target_lost_timeout: VISUAL_TRACKING 目标丢失超时（秒）
        """
        # ---- 依赖注入 ----
        self.proxy = proxy

        # ---- 任务参数 ----
        # 航点列表（外部队列已经使用 NED 坐标系，Z 负值向上）
        self.waypoints = list(waypoints) if waypoints else []
        self.target_altitude = -abs(target_altitude)   # NED Z（负值向上）
        self.arrival_radius = arrival_radius
        self.hold_duration = hold_duration
        self.return_to_home = return_to_home
        self.target_lost_timeout = (
            target_lost_timeout if target_lost_timeout is not None
            else self.DEFAULT_TARGET_LOST_TIMEOUT
        )

        # ---- 状态机 ----
        self.state = MissionState.INIT
        self.current_wp_index = 0

        # ---- 当前目标航点（共享给外部主循环 / UAVControlLoop） ----
        self.current_target = {
            "x": 0.0, "y": 0.0,
            "z": self.target_altitude, "yaw": 0.0,
        }

        # ================================================================
        #  本地位置估计（速度积分法）
        # ================================================================
        self._est_x = 0.0            # NED 北向（相对起飞点）
        self._est_y = 0.0            # NED 东向
        self._pos_time = 0.0         # 上次位置更新时间戳
        self._home_set = False       # 首次 STATE 标记

        # ---- 计时器 ----
        self.hold_start_time = 0.0
        self.tracking_start_time = 0.0
        self._target_lost_start = None
        self._state_start_time = time.time()

        # ================================================================
        #  追踪完成信号（外部设置）
        # ================================================================
        # UAVControlLoop 或 run_mission.py 在判定视觉追踪完成时置 True
        self.tracking_completed = False
        # 追踪最大时长兜底
        self.tracking_max_duration = 30.0

        logger.info("MissionManager 初始化: %d 个航点, 目标高度=%.1fm, "
                     "到达半径=%.1fm, 搜寻时长=%.1fs, 返航=%s",
                     len(self.waypoints), -self.target_altitude,
                     self.arrival_radius, self.hold_duration,
                     self.return_to_home)

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

        # ---- Failsafe 越权检查 ----
        if self.state not in (MissionState.INIT, MissionState.FINISHED):
            if state is not None:
                mode = state["drone"].get("mode", "")
                if mode not in self.AUTONOMOUS_MODES:
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
        #  步骤 1：更新本地位置估计
        # ---------------------------------------------------------------
        self._update_local_position(state)

        # ---------------------------------------------------------------
        #  步骤 2：状态机分支
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
    #  位置估计
    # ================================================================

    def _update_local_position(self, state):
        """通过速度积分更新本地 NED 位置估计"""
        try:
            drone = state["drone"]
            now = time.time()

            if not self._home_set:
                self._est_x = 0.0
                self._est_y = 0.0
                self._pos_time = now
                self._home_set = True
                logger.info("[位置] 初始位置记录完成")

            dt = now - self._pos_time
            if 0 < dt < self.MAX_POSITION_DT:
                vx = drone.get("vx", 0.0)
                vy = drone.get("vy", 0.0)
                self._est_x += vx * dt
                self._est_y += vy * dt
            # dt ≥ MAX_POSITION_DT 时跳过积分防跳变

            self._pos_time = now
        except (KeyError, TypeError) as e:
            logger.warning("[位置] 更新异常: %s", e)

    def _get_current_z(self, state):
        """从 STATE 获取当前 NED Z（负值向上）"""
        try:
            return -state["drone"].get("alt_rel", 0.0)
        except (KeyError, TypeError):
            return self.target_altitude

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

            logger.info("[INIT] mode=%s armed=%s alt_rel=%.1fm",
                        mode, armed, alt_rel)

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

            if not armed:
                elapsed = time.time() - self._state_start_time
                if elapsed > self.ARM_TIMEOUT:
                    logger.warning("[ARMING] 超时 (%.1fs)，重试 ARM", elapsed)
                    self.proxy.send_command("ARM")
                    self._state_start_time = time.time()
                else:
                    logger.info("[ARMING] 等待解锁... (%.1fs)", elapsed)
                return

            # ---- 解锁成功 → 起飞 ----
            takeoff_alt = -self.target_altitude   # 正值（协议 altitude 为正）
            logger.info("\n[解锁成功] 发送起飞指令 (alt=%.1fm)", takeoff_alt)

            ok, ack = self.proxy.send_waypoint(
                action="TAKEOFF", alt=takeoff_alt,
                alt_frame="RELATIVE", speed=2.0,
            )
            if ok:
                self.state = MissionState.TAKEOFF
                self._state_start_time = time.time()
            else:
                logger.warning("[TAKEOFF] WAYPOINT 失败: %s，尝试 SETPOINT 爬升", ack)
                ok2, _ = self.proxy.send_setpoint(
                    x=0, y=0, z=self.target_altitude, yaw=0.0,
                    control_mode="POSITION",
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
        [TAKEOFF] 持续发送 POSITION SETPOINT 爬升，到达后进入 OFFBOARD

        流程：
          ① 每帧发送 POSITION 悬停/爬升 SETPOINT（预热 + 保持流）
          ② 高度到达目标 90% → 发送 OFFBOARD 指令
          ③ 模式切换确认 → 跳转到 NAVIGATING
        """
        try:
            drone = state["drone"]
            alt_rel = drone.get("alt_rel", 0.0)
            mode = drone.get("mode", "")
            target_up = -self.target_altitude   # 正值

            # ---- ① 持续发送 POSITION SETPOINT（预热 OFFBOARD 流） ----
            # 通过 MissionManager 自有 proxy 发送，此阶段 UAVControlLoop
            # 的 20Hz 线程会被 run_mission.py 关闭，不会冲突。
            # NED 坐标系：x=0,y=0,z=target_altitude 表示飞到正上方目标高度
            self.proxy.send_setpoint(
                x=0, y=0, z=self.target_altitude, yaw=0.0,
                control_mode="POSITION",
            )

            # ---- ② 到达目标高度 → 进入 OFFBOARD ----
            reached_alt = alt_rel >= target_up * 0.9

            if reached_alt and mode != "OFFBOARD":
                logger.info("[TAKEOFF] 高度 %.1f/%.1f，切换 OFFBOARD...",
                            alt_rel, target_up)
                # 先预热悬停（协议 §3.3.1 ③→④）
                self.proxy.send_setpoint(
                    x=0, y=0, z=-alt_rel, yaw=0.0,
                    control_mode="POSITION",
                )
                time.sleep(0.2)

                ok, ack = self.proxy.send_command("OFFBOARD")
                if ok:
                    # 等待模式切换
                    for _ in range(int(self.OFFBOARD_TIMEOUT / 0.1)):
                        st = self.proxy.get_latest_state()
                        if st and st["drone"]["mode"] == "OFFBOARD":
                            logger.info("[TAKEOFF] 已进入 OFFBOARD")
                            break
                        time.sleep(0.1)
                    else:
                        logger.warning("[TAKEOFF] OFFBOARD 切换超时")
                else:
                    logger.error("[TAKEOFF] OFFBOARD 指令失败: %s", ack)

            # ---- ③ 高度到位 + OFFBOARD 模式 → 跳转巡航 ----
            if reached_alt and mode == "OFFBOARD":
                logger.info("\n[起飞完成] 高度 %.1fm，进入航点巡航", alt_rel)
                self.state = MissionState.NAVIGATING
                self.current_wp_index = 0
                self._set_current_target_from_wp(0)
                return

            # ---- 爬升超时保护 ----
            elapsed = time.time() - self._state_start_time
            if elapsed > self.TAKEOFF_TIMEOUT:
                logger.warning("[TAKEOFF] 超时 (%.1fs)，强行进入巡航", elapsed)
                self.state = MissionState.NAVIGATING
                self.current_wp_index = 0
                self._set_current_target_from_wp(0)
                return

            logger.info("[起飞中] alt=%.1f/%.1f mode=%s 耗时=%.1fs",
                        alt_rel, target_up, mode, elapsed)

        except (KeyError, TypeError) as e:
            logger.warning("[TAKEOFF] 异常: %s", e)

    # ---- 航点导航 ----

    def _handle_navigating(self, state):
        """
        [NAVIGATING] 每帧发送 POSITION SETPOINT 前往当前航点

        到达判定后推进索引；所有航点飞完后根据 return_to_home
        决定返航或悬停。
        """
        if not self.waypoints or self.current_wp_index >= len(self.waypoints):
            self._finish_navigation()
            return

        wp = self.waypoints[self.current_wp_index]
        wp_x = wp["x"]
        wp_y = wp["y"]
        wp_z = wp.get("z", self.target_altitude)
        wp_yaw = wp.get("yaw", 0.0)

        # ---- 发送 POSITION SETPOINT（通过自有 proxy） ----
        # 此 SETPOINT 让 PX4 飞向目标航点
        self.proxy.send_setpoint(
            x=wp_x, y=wp_y, z=wp_z, yaw=wp_yaw,
            control_mode="POSITION",
        )

        # ---- 共享目标 ----
        self.current_target = {"x": wp_x, "y": wp_y, "z": wp_z, "yaw": wp_yaw}

        # ---- 到达判定 ----
        current_z = self._get_current_z(state)
        dist = _dist_3d(self._est_x, self._est_y, current_z,
                        wp_x, wp_y, wp_z)

        if dist < self.arrival_radius:
            logger.info("\n✅ [到达航点 %d/%d] dist=%.2fm",
                        self.current_wp_index + 1, len(self.waypoints), dist)
            self.state = MissionState.HOLD_TASK
            self.hold_start_time = time.time()
        else:
            logger.info("[巡航中] 航点 [%d/%d] 距离=%.2fm",
                        self.current_wp_index + 1, len(self.waypoints), dist)

    def _finish_navigation(self):
        """所有航点飞完后的跳转"""
        if self.return_to_home:
            logger.info("\n[巡航完成] 返回起始点...")
            self.state = MissionState.RETURNING
            self.current_target = {"x": 0, "y": 0,
                                   "z": self.target_altitude, "yaw": 0}
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

        # ---- 保持当前位置悬停 ----
        wp = self.waypoints[self.current_wp_index]
        self.proxy.send_setpoint(
            x=wp["x"], y=wp["y"],
            z=wp.get("z", self.target_altitude),
            yaw=wp.get("yaw", 0),
            control_mode="POSITION",
        )
        self.current_target = {
            "x": wp["x"], "y": wp["y"],
            "z": wp.get("z", self.target_altitude),
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
            logger.info("[检索中] 航点 %d 剩余 %.1fs",
                        self.current_wp_index + 1,
                        self.hold_duration - elapsed)

    # ---- 视觉追踪 ----

    def _handle_visual_tracking(self, state, target_detected):
        """
        [VISUAL_TRACKING] 🎯 完全让渡控制权给 UAVControlLoop

        【核心设计】
          MissionManager 在此状态下 **不发送任何 SETPOINT**。
          UAVControlLoop (20Hz 独立线程) 通过其自有 proxy 发送
          VELOCITY SETPOINT 进行视觉闭环追踪。

          本模块仅负责追踪退出条件的判定和状态切换。

        【退出条件】
          1. self.tracking_completed == True（外部模块设置）
          2. 目标丢失超过 target_lost_timeout 秒
          3. 追踪总时长超过 tracking_max_duration（兜底）
        """
        # ---- 不发送任何 SETPOINT — 控制权已让渡 ----

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
            else:
                lost_elapsed = time.time() - self._target_lost_start
                if lost_elapsed >= self.target_lost_timeout:
                    logger.info("\n⏰ [丢失超时] %.1fs，放弃追踪",
                                lost_elapsed)
                    self._exit_tracking()
                    return
                logger.info("[追踪中] 丢失 %.1fs/%.1fs",
                            lost_elapsed, self.target_lost_timeout)
        else:
            self._target_lost_start = None

        logger.info("[追踪中] 进行中... %.1fs", tracking_elapsed)

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
        [RETURNING] 发送 POSITION SETPOINT 到起始点 (0,0,z)
        """
        self.proxy.send_setpoint(
            x=0, y=0, z=self.target_altitude, yaw=0,
            control_mode="POSITION",
        )
        self.current_target = {"x": 0, "y": 0,
                               "z": self.target_altitude, "yaw": 0}

        current_z = self._get_current_z(state)
        dist = _dist_3d(self._est_x, self._est_y, current_z,
                        0.0, 0.0, self.target_altitude)
        logger.info("[返航中] 距离=%.2fm", dist)

        if dist < self.arrival_radius:
            logger.info("\n[到达] 已返回起始点，发送 LAND 指令...")
            ok, ack = self.proxy.send_command("LAND")
            if ok:
                self.state = MissionState.LANDING
            else:
                logger.warning("[返航] LAND 失败: %s", ack)

    # ---- 降落 ----

    def _handle_landing(self, state):
        """
        [LANDING] 监控 disarm 状态判断着陆
        """
        try:
            armed = state["drone"].get("armed", False)
            alt_rel = state["drone"].get("alt_rel", 0.0)
            logger.info("[降落中] 高度=%.1fm 已解锁=%s",
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
        """
        if self.waypoints:
            final_wp = self.waypoints[-1]
        else:
            final_wp = {"x": 0, "y": 0,
                        "z": self.target_altitude, "yaw": 0}

        self.proxy.send_setpoint(
            x=final_wp["x"], y=final_wp["y"],
            z=final_wp.get("z", self.target_altitude),
            yaw=final_wp.get("yaw", 0),
            control_mode="POSITION",
        )
        self.current_target = {
            "x": final_wp["x"], "y": final_wp["y"],
            "z": final_wp.get("z", self.target_altitude),
            "yaw": final_wp.get("yaw", 0),
        }
        logger.info("[挂起] 保持 (%.1f, %.1f, %.1f) 等待接管",
                    final_wp["x"], final_wp["y"],
                    final_wp.get("z", self.target_altitude))

    # ================================================================
    #  辅助
    # ================================================================

    def _set_current_target_from_wp(self, index):
        """从航点列表更新 current_target"""
        if self.waypoints and index < len(self.waypoints):
            wp = self.waypoints[index]
            self.current_target = {
                "x": wp["x"], "y": wp["y"],
                "z": wp.get("z", self.target_altitude),
                "yaw": wp.get("yaw", 0),
            }
