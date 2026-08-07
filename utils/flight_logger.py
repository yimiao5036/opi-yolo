"""
flight_logger.py — 异步飞行日志记录器

将飞行状态数据异步写入 CSV 文件，不影响主循环性能。

用法：
    logger = FlightLogger(log_config)

    # 在主循环中（按间隔调用）
    logger.collect(state, mission_state_name, wp_index, detect_count, tracking_active)

    # 退出时
    logger.stop()
"""

import os
import time
import csv
import queue
import threading
import logging

logger = logging.getLogger(__name__)


class FlightLogger:
    """
    异步飞行日志记录器

    后台线程负责批量写入 CSV，主线程仅做轻量级入队操作。
    """

    def __init__(self, log_config=None):
        """
        Args:
            log_config: 日志配置字典，支持字段：
                enable_flight_log (bool):   是否启用（默认 True）
                flight_log_interval (int):  写入间隔（帧数，默认 20）
                log_dir (str):              日志目录（默认 "./log"）
        """
        if log_config is None:
            log_config = {}

        self.enabled = log_config.get("enable_flight_log", True)
        self.log_interval = log_config.get("flight_log_interval", 20)
        self.log_dir = log_config.get("log_dir", "./log")

        self._queue = queue.Queue(maxsize=500)
        self._thread = None
        self._running = False
        self._file_path = None
        self._started = False
        self._frame_counter = 0

        if self.enabled:
            self._start()

    # ================================================================
    #  内部方法
    # ================================================================

    def _start(self):
        """启动后台日志线程"""
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._file_path = os.path.join(self.log_dir, f"flight_{timestamp}.csv")

        # 写入 CSV 表头
        with open(self._file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_unix',
                'alt_rel_m',
                'ground_speed_mps',
                'vx', 'vy', 'vz',
                'battery_percent',
                'state',
                'wp_index',
                'detect_count',
                'tracking_active',
                'yaw',          # 姿态偏航角 (rad)
                'heading'       # GPS 航向角 (deg)
            ])

        self._running = True
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="flight-log-writer"
        )
        self._thread.start()
        self._started = True
        logger.info("📝 异步飞行日志已启动 -> %s (降采样: 每%d帧写1次)",
                     self._file_path, self.log_interval)

    def _worker(self):
        """后台线程：从队列取数据，批量写入 CSV"""
        buffer = []
        BATCH_SIZE = 10  # 攒够 10 条一次性写入，减少 I/O

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:  # 停止信号
                    break
                buffer.append(item)

                if len(buffer) >= BATCH_SIZE:
                    self._flush(buffer)
                    buffer = []

            except queue.Empty:
                # 超时无数据，检查是否还有残留
                if not self._running and not self._queue.empty():
                    while not self._queue.empty():
                        try:
                            buffer.append(self._queue.get_nowait())
                        except queue.Empty:
                            break
                    if buffer:
                        self._flush(buffer)
                continue
            except Exception as e:
                logger.error("飞行日志线程异常: %s", e)
                time.sleep(0.1)

        # 最终收尾：清空缓冲区
        if buffer:
            self._flush(buffer)
        logger.info("📝 飞行日志线程已停止")

    def _flush(self, buffer):
        """将缓冲区数据写入 CSV 文件"""
        if not buffer or not self._file_path:
            return
        try:
            with open(self._file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(buffer)
        except Exception as e:
            logger.error("写入飞行日志失败: %s", e)

    # ================================================================
    #  公开接口
    # ================================================================

    def should_collect(self):
        """
        判断当前帧是否需要采集（基于帧数间隔降采样）

        Returns:
            bool: 当前帧是否应采集

        用法：
            if flight_logger.should_collect():
                flight_logger.collect(...)
        """
        if not self._started:
            return False
        self._frame_counter += 1
        return self._frame_counter % self.log_interval == 0

    def collect(self, state, mission_state_name, wp_index,
                detect_count, tracking_active):
        """
        采集当前飞行状态，推入写入队列（主线程调用，极轻量）

        Args:
            state:               proxy.get_latest_state() 的返回值
            mission_state_name:  状态机当前状态名 (str)，如 "NAVIGATING"
            wp_index:            当前航点索引
            detect_count:        当前帧检测到的目标数
            tracking_active:     追踪线程是否激活（1 激活 / 0 未激活）
        """
        if not self._started or state is None:
            return

        try:
            drone = state.get("drone", {})
            row = (
                time.time(),                       # timestamp_unix
                drone.get("alt_rel", 0.0),         # alt_rel_m
                drone.get("ground_speed", 0.0),    # ground_speed_mps
                drone.get("vx", 0.0),              # vx
                drone.get("vy", 0.0),              # vy
                drone.get("vz", 0.0),              # vz
                drone.get("battery", 0.0),         # battery_percent
                mission_state_name,                # state
                wp_index,                          # wp_index
                detect_count,                      # detect_count
                tracking_active,                   # tracking_active
                drone.get("yaw", 0.0),             # yaw: 姿态偏航角 (rad)
                drone.get("heading", 0.0),         # heading: GPS 航向角 (deg)
            )

            # 非阻塞入队，满则丢弃
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                pass

        except Exception:
            pass  # 日志采集失败不影响主循环

    def stop(self):
        """停止后台日志线程"""
        if not self._started:
            return

        self._running = False
        # 发送停止信号
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            logger.info("✔ 飞行日志线程已停止")
