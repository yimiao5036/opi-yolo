#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光流法障碍检测 - 网格点追踪版（复位点不参与统计）
特性：
1. 全图均匀网格点，覆盖无盲区
2. 点漂移后硬复位（当前帧和上一帧均置为初始位置）
3. 复位点在该帧不参与任何统计（散度、平均光流等）
4. 分区（左/中/右）风险统计，仅基于有效运动点
5. 与 PID 和 MAVLink 无缝集成
"""

import cv2
import numpy as np
import time
from collections import deque
from typing import Optional, Dict, Any

# ==================== PID 部分 =====================
from dataclasses import dataclass


@dataclass
class PIDConfig:
    """PID 参数配置"""
    kp: float
    ki: float
    kd: float
    max_out: float
    min_out: float
    dt_default: float = 0.033
    alpha: float = 0.2  # 微分滤波系数 (0 < alpha <= 1)


class ImprovedPID:
    def __init__(self, config: PIDConfig):
        # 参数校验
        if config.max_out <= config.min_out:
            raise ValueError("max_out must be greater than min_out")
        if not (0 < config.alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")

        self.cfg = config

        self.last_err: float = 0.0
        self.last_fb: float = 0.0
        self.integral: float = 0.0
        self.last_d: float = 0.0
        self.last_time: float = time.perf_counter()
        self.first_run: bool = True   # 标记首次调用

    def update(self, error: float, feedback: Optional[float] = None, dt: Optional[float] = None) -> float:
        current_time = time.perf_counter()

        # --- 首次调用处理：初始化状态，避免微分冲击 ---
        if self.first_run:
            self.first_run = False
            self.last_err = error
            if feedback is not None:
                self.last_fb = feedback
            self.last_time = current_time
            # 首次仅返回比例项，微分和积分暂不参与（或也可返回0，视应用而定）
            # 通常直接返回 P 项能让系统平滑起步
            return self.cfg.kp * error

        # 1. 计算或使用传入的 dt
        if dt is None:
            dt = current_time - self.last_time
            if dt <= 0:
                dt = self.cfg.dt_default

        # 2. 比例项 (P)
        p_out = self.cfg.kp * error

        # 3. 微分项 (D) —— 微分先行 + 低通滤波
        if feedback is not None:
            # 对反馈微分，设定值变化不引入冲击
            d_raw = (self.last_fb - feedback) / dt
            self.last_fb = feedback
        else:
            d_raw = (error - self.last_err) / dt

        self.last_d = self.cfg.alpha * d_raw + (1.0 - self.cfg.alpha) * self.last_d
        d_out = self.cfg.kd * self.last_d

        # 4. 积分项 (I) —— 先计算潜在值
        potential_integral = self.integral + error * dt
        i_out = self.cfg.ki * potential_integral

        # 5. 总输出与抗积分饱和（基于积分方向判断）
        output = p_out + i_out + d_out

        if output > self.cfg.max_out:
            output = self.cfg.max_out
            # 如果积分项仍在推高输出（i_out > 0），则冻结积分
            if i_out > 0:
                pass   # 不更新 self.integral
            else:
                self.integral = potential_integral
        elif output < self.cfg.min_out:
            output = self.cfg.min_out
            # 如果积分项仍在拉低输出（i_out < 0），则冻结积分
            if i_out < 0:
                pass
            else:
                self.integral = potential_integral
        else:
            # 未饱和，正常更新积分
            self.integral = potential_integral

        # 6. 更新状态变量
        self.last_err = error
        self.last_time = current_time

        return output

    def reset(self) -> None:
        """重置所有状态，恢复到首次运行前的状态"""
        self.last_err = 0.0
        self.last_fb = 0.0
        self.integral = 0.0
        self.last_d = 0.0
        self.last_time = time.perf_counter()
        self.first_run = True

# =================================================


class OpticalFlowObstacleDetector:
    """网格点光流检测器，复位点不参与统计"""
    def __init__(self, grid_step: int = 25, reset_threshold: float = 30.0):
        """
        Args:
            grid_step: 网格点间距（像素），值越小点越多
            reset_threshold: 点偏离初始位置超过此值时复位（像素）
        """
        self.grid_step = grid_step
        self.reset_threshold = reset_threshold

        # 光流跟踪参数
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # 状态变量
        self.old_gray = None
        self.p0 = None               # 当前点坐标 (N,1,2)
        self.grid_initial = None     # 初始网格坐标 (N,2)
        self.frame_count = 0

        # 分区统计历史（用于兜底）
        self.last_zone_stats = None
        self.flow_history = deque(maxlen=10)

    def _generate_grid(self, img_w: int, img_h: int) -> np.ndarray:
        """生成网格点，返回 (N,2) 坐标数组"""
        x = np.arange(self.grid_step // 2, img_w, self.grid_step, dtype=np.float32)
        y = np.arange(self.grid_step // 2, img_h, self.grid_step, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        points = np.stack((xv.ravel(), yv.ravel()), axis=1)
        return points

    def process_frame(self, frame):
        """
        处理单帧，返回 (is_collision, flow_data)
        更新 self.last_zone_stats（仅包含有效点）
        """
        self.frame_count += 1
        if frame is None:
            return False, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # --- 1. 初始化第一帧 ---
        if self.old_gray is None:
            self.old_gray = gray
            grid_pts = self._generate_grid(w, h)
            self.grid_initial = grid_pts.copy()
            self.p0 = grid_pts.reshape(-1, 1, 2).astype(np.float32)
            self.last_zone_stats = None
            return False, None

        # --- 2. 光流跟踪 ---
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, gray, self.p0, None, **self.lk_params
        )

        if p1 is None:
            p1 = self.p0.copy()
            st = np.ones(len(self.p0), dtype=np.uint8)

        # 将跟踪结果展开为 (N,2)
        p1_flat = p1.reshape(-1, 2)
        p0_flat = self.p0.reshape(-1, 2)
        st_flat = st.flatten()

        # 初始化新的点数组（默认保持旧位置）
        new_points = p0_flat.copy()
        old_points = p0_flat.copy()   # 上一帧位置，用于计算光流

        # 成功跟踪的点，先更新位置
        success_indices = np.where(st_flat == 1)[0]
        if len(success_indices) > 0:
            new_points[success_indices] = p1_flat[success_indices]
            old_points[success_indices] = p0_flat[success_indices]

        # --- 3. 复位逻辑与有效点掩码生成 ---
        # 复位掩码：True 表示该点需要复位（跟踪失败或漂移过大）
        reset_mask = np.zeros(len(new_points), dtype=bool)

        # 3.1 跟踪失败的点直接复位
        failed_indices = np.where(st_flat == 0)[0]
        reset_mask[failed_indices] = True

        # 3.2 检查成功点是否漂移过大
        for idx in success_indices:
            if np.linalg.norm(new_points[idx] - self.grid_initial[idx]) > self.reset_threshold:
                reset_mask[idx] = True

        # 执行复位（将当前和上一帧位置均设为初始位置）
        if np.any(reset_mask):
            new_points[reset_mask] = self.grid_initial[reset_mask]
            old_points[reset_mask] = self.grid_initial[reset_mask]

        # --- 4. 生成有效点掩码（未复位且跟踪成功） ---
        # 有效点：跟踪成功且未复位
        valid_mask = np.ones(len(new_points), dtype=bool)
        valid_mask[reset_mask] = False

        # --- 5. 更新状态 ---
        self.p0 = new_points.reshape(-1, 1, 2).astype(np.float32)
        self.old_gray = gray

        # 如果有效点太少，不进行统计（但保留状态）
        if np.sum(valid_mask) < 5:
            # 保留上一帧统计值，或返回空
            return False, None

        # --- 6. 计算光流向量（仅针对有效点） ---
        valid_old = old_points[valid_mask]
        valid_new = new_points[valid_mask]
        flow_vectors = valid_new - valid_old   # (M,2)
        mean_mag = np.mean(np.linalg.norm(flow_vectors, axis=1))

        # --- 7. 分区统计（仅使用有效点） ---
        zone_stats = self._compute_zone_stats(valid_old, flow_vectors, w, h)
        self.last_zone_stats = zone_stats

        # 整体散度（用于碰撞警告）
        divergence = self._compute_divergence(valid_old, flow_vectors, w, h)

        # 碰撞判断（保留阈值）
        is_collision = False
        if divergence > 0.8 and mean_mag > 1.0:
            is_collision = True
        elif mean_mag > 3.0 and len(valid_old) > 50:
            if self._check_roi_expansion(valid_old, flow_vectors):
                is_collision = True

        # 历史记录
        self.flow_history.append({
            'mean_flow': np.mean(flow_vectors, axis=0),
            'mean_mag': mean_mag,
            'divergence': divergence,
            'points': np.sum(valid_mask)
        })

        return is_collision, {
            'points': np.sum(valid_mask),
            'mean_mag': mean_mag,
            'divergence': divergence,
        }

    # ---------- 辅助方法（与之前一致，但传入的点已过滤） ----------
    def _compute_divergence(self, old_points, flow_vectors, img_w, img_h):
        if len(old_points) < 5:
            return 0.0
        center = np.array([img_w / 2, img_h / 2])
        dir_to_center = center - old_points
        dir_to_center /= (np.linalg.norm(dir_to_center, axis=1, keepdims=True) + 1e-6)
        radial_flow = np.sum(flow_vectors * dir_to_center, axis=1)
        return np.mean(radial_flow)

    def _compute_zone_stats(self, old_pts, flow_vecs, img_w, img_h):
        left_bound = img_w / 3
        right_bound = 2 * img_w / 3
        zones = {'left': [], 'center': [], 'right': []}
        for pt, vec in zip(old_pts, flow_vecs):
            x = pt[0]
            if x < left_bound:
                zones['left'].append((pt, vec))
            elif x < right_bound:
                zones['center'].append((pt, vec))
            else:
                zones['right'].append((pt, vec))

        stats = {}
        for name, pts_vecs in zones.items():
            if len(pts_vecs) < 3:  # 阈值降低，因为点可能较少
                stats[name] = {'mean_vec': np.zeros(2), 'divergence': 0.0, 'count': 0, 'risk': 0.0}
                continue
            old_pts_zone = np.array([p for p, v in pts_vecs])
            flow_zone = np.array([v for p, v in pts_vecs])
            div = self._compute_divergence(old_pts_zone, flow_zone, img_w, img_h)
            stats[name] = {
                'mean_vec': np.mean(flow_zone, axis=0),
                'divergence': div,
                'count': len(pts_vecs),
                'risk': div if div > 0 else 0.0
            }
        return stats

    def _check_roi_expansion(self, old_points, flow_vectors):
        h, w = 480, 640
        roi_center = np.array([w / 2, h / 2])
        roi_mask = (np.abs(old_points[:, 0] - w / 2) < w / 4) & (np.abs(old_points[:, 1] - h / 2) < h / 4)
        roi_points = old_points[roi_mask]
        roi_flow = flow_vectors[roi_mask]
        if len(roi_points) < 5:
            return False
        dir_vec = roi_points - roi_center
        dir_unit = dir_vec / (np.linalg.norm(dir_vec, axis=1, keepdims=True) + 1e-6)
        radial = np.sum(roi_flow * dir_unit, axis=1)
        return np.mean(radial) > 0.5

    def get_zone_decision(self, zone_stats=None, threshold_div=0.8):
        """根据分区风险生成控制建议（返回 yaw_rate, speed_scale）"""
        if zone_stats is None:
            zone_stats = self.last_zone_stats
        if zone_stats is None:
            return 0.0, 1.0

        left = zone_stats['left']['risk']
        center = zone_stats['center']['risk']
        right = zone_stats['right']['risk']
        max_r = max(left, center, right)

        if max_r < 0.3:
            return 0.0, 1.0

        if center >= left and center >= right and center > threshold_div:
            if max(left, right) > threshold_div * 0.6:
                return 0.0, 0.0
            else:
                return 0.0, 0.3

        if left > right + 0.2:
            yaw = min(0.8, 0.3 + 0.5 * (left - right))
            return yaw, 0.7
        elif right > left + 0.2:
            yaw = -min(0.8, 0.3 + 0.5 * (right - left))
            return yaw, 0.7
        else:
            return 0.0, 0.5


# ==================== 控制器（与之前相同） ====================
class ObstacleAvoidanceController:
    """避障控制器，集成网格点光流检测器、PID 和 MAVLink 发送"""
    def __init__(self,
                 camera_id: int = 0,
                 frame_width: int = 640,
                 frame_height: int = 480,
                 cruise_speed: float = 2.0,
                 grid_step: int = 25,
                 reset_threshold: float = 30.0,
                 yaw_pid_cfg: Optional[PIDConfig] = None,
                 speed_pid_cfg: Optional[PIDConfig] = None):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        self.detector = OpticalFlowObstacleDetector(grid_step=grid_step, reset_threshold=reset_threshold)
        self.cruise_speed = cruise_speed

        if yaw_pid_cfg is None:
            yaw_pid_cfg = PIDConfig(kp=1.2, ki=0.1, kd=0.05, max_out=0.8, min_out=-0.8)
        if speed_pid_cfg is None:
            speed_pid_cfg = PIDConfig(kp=0.8, ki=0.05, kd=0.0, max_out=1.0, min_out=0.0)

        self.yaw_pid = ImprovedPID(yaw_pid_cfg)
        self.speed_pid = ImprovedPID(speed_pid_cfg)

        self.current_yaw_rate = 0.0
        self.current_speed = 0.0
        self.running = False

    def update_feedback(self, yaw_rate: float, speed: float):
        self.current_yaw_rate = yaw_rate
        self.current_speed = speed

    def run_iteration(self, mav_proxy, enable_pid: bool = True) -> Dict[str, Any]:
        ret, frame = self.cap.read()
        if not ret:
            return {'success': False, 'error': '摄像头读取失败'}

        is_collision, flow_data = self.detector.process_frame(frame)
        yaw_setpoint, speed_scale = self.detector.get_zone_decision()
        speed_setpoint = speed_scale * self.cruise_speed

        if enable_pid:
            yaw_error = yaw_setpoint - self.current_yaw_rate
            speed_error = speed_setpoint - self.current_speed
            yaw_output = self.yaw_pid.update(yaw_error, feedback=self.current_yaw_rate)
            speed_output = self.speed_pid.update(speed_error, feedback=self.current_speed)
        else:
            yaw_output = yaw_setpoint
            speed_output = speed_setpoint

        # 调用您的 MAVLink 代理的 send_setpoint 方法
        success, ack = mav_proxy.send_setpoint(
            control_mode="VELOCITY",
            vx=speed_output,
            vy=0.0,
            vz=0.0,
            yaw_rate=yaw_output
        )

        return {
            'success': success,
            'ack': ack,
            'is_collision': is_collision,
            'yaw_setpoint': yaw_setpoint,
            'speed_setpoint': speed_setpoint,
            'yaw_output': yaw_output,
            'speed_output': speed_output,
            'points': flow_data['points'] if flow_data else 0,
        }

    def run_forever(self, mav_proxy, enable_pid: bool = True):
        self.running = True
        print("🔄 网格点光流避障（复位点不统计）已启动，按 'q' 退出")
        while self.running:
            result = self.run_iteration(mav_proxy, enable_pid)
            if not result.get('success', False):
                print("⚠️ 迭代失败:", result.get('error', ''))
                continue
            if result['is_collision']:
                print("🚨 碰撞警告!")

            # 显示画面（方便调试）
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('Grid Optical Flow', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


# ==================== 测试入口 ====================
if __name__ == "__main__":
    # 模拟 MAVLink 代理（实际应替换为真实对象）
    class MockMavProxy:
        def send_setpoint(self, control_mode="VELOCITY", **kwargs):
            print(f"📤 发送指令: {control_mode} {kwargs}")
            return True, {"result": "ok"}

    mav = MockMavProxy()
    controller = ObstacleAvoidanceController(
        camera_id=0,
        grid_step=25,
        reset_threshold=30.0,
        cruise_speed=2.0
    )
    controller.update_feedback(0.0, 0.0)  # 模拟初始反馈

    try:
        controller.run_forever(mav, enable_pid=True)
    except KeyboardInterrupt:
        controller.stop()
        print("🛑 已退出")