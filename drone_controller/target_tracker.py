"""
target_tracker.py — 轻量级多目标追踪过滤器

【解决的问题】
  1. 目标短暂丢帧（2-5帧）导致无人机抽搐 → 卡尔曼滤波 coast/predict 平滑过渡
  2. 多目标 ID 飘移导致甩头 → 中心距离 + IoU 加权关联锁定 ID 不变

【核心设计】
  - 恒定速度卡尔曼滤波器（Kalman Filter × 8状态：cx, cy, w, h, vx, vy, vw, vh）
  - 最邻近贪婪关联（中心距离 + IoU 加权代价），提高交叉/靠近时的 ID 稳定性
  - 死区预测（Coast/Predict）：连续丢失 ≤ max_lost_frames 帧时用滤波器预测值
  - 超过 max_lost_frames → 释放轨道，对外判丢失

【典型用法】
    tracker = TargetTracker(max_lost_frames=8)

    # 每帧将 YOLO 原始检测喂入
    result = tracker.update(yolo_detections, frame_shape=(480, 640))

    if result["tracked"]:
        cx, cy = result["center"]          # 平滑后的中心点（像素坐标）
        pid.update(...)                     # 正常 PID 控制
    else:
        pid.reset(); send_zero_velocity()   # 判丢 → 刹车

【依赖】
  - numpy（与推理模块共用，无新增重量级依赖）
  - 纯 Python，无 OpenCV / scipy 依赖
"""

import numpy as np
import time
import logging
import copy

logger = logging.getLogger(__name__)

# ================================================================
#  单目标卡尔曼滤波器
# ================================================================

class _KalmanBoxFilter:
    """
    单目标恒定速度卡尔曼滤波器

    状态向量 (8) : [cx, cy, w, h, vx, vy, vw, vh]
      位置 + 尺寸 + 各自速度

    观测向量 (4) : [cx, cy, w, h]

    所有坐标使用原始像素值（而非归一化），
    与 YOLO postprocess 输出的坐标空间一致。
    """

    # ---- 噪声权重（经验调优值，适用于 640×480 画面） ----
    _STD_POS = 1.0 / 50     # 位置过程噪声
    _STD_VEL = 1.0 / 200    # 速度过程噪声
    _INI_POS = 2.0          # 初始位置协方差系数
    _INI_VEL = 100.0        # 初始速度协方差系数

    def __init__(self):
        dt = 1.0  # 帧间步长（帧率变化时仍用 dt=1，Q 已含尺度因子）

        # ---- 状态转移矩阵 F ----
        self.F = np.eye(8, dtype=float)
        for i in range(4):
            self.F[i, i + 4] = dt  # 位置 += 速度 × dt

        # ---- 观测矩阵 H ----
        self.H = np.zeros((4, 8), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # ---- 协方差初始值 ----
        self.P = np.eye(8, dtype=float) * self._INI_POS
        for i in range(4, 8):
            self.P[i, i] = self._INI_VEL

        # ---- 状态向量 ----
        self.x = np.zeros(8, dtype=float)
        self._initialized = False

    # ------------------------------------------------------------------
    #  公开接口
    # ------------------------------------------------------------------

    def initiate(self, measurement):
        """
        用首次观测初始化滤波器

        Args:
            measurement: [cx, cy, w, h] — 像素坐标
        """
        cx, cy, w, h = measurement
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        # 初始协方差：位置置信度中等，速度置信度低
        self.P = np.eye(8, dtype=float) * self._INI_POS
        for i in range(4, 8):
            self.P[i, i] = self._INI_VEL
        self._initialized = True

    def predict(self):
        """
        预测步骤

        Returns:
            [cx, cy, w, h] — 预测的像素坐标，未初始化时返回 None
        """
        if not self._initialized:
            return None

        # ---- 动态过程噪声 Q（与当前目标尺寸相关） ----
        std_pos = [self._STD_POS * self.x[2],   # w 为宽度
                   self._STD_POS * self.x[3],   # h 为高度
                   self._STD_POS * self.x[2],
                   self._STD_POS * self.x[3]]
        std_vel = [self._STD_VEL * self.x[2],
                   self._STD_VEL * self.x[3],
                   self._STD_VEL * self.x[2],
                   self._STD_VEL * self.x[3]]

        q_diag = [s * s for s in std_pos] + [s * s for s in std_vel]
        Q = np.diag(q_diag)

        # ---- 预测 ----
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        return self.x[:4].copy()

    def update(self, measurement):
        """
        更新步骤

        Args:
            measurement: [cx, cy, w, h] — 观测到的像素坐标
        """
        if not self._initialized:
            return

        z = np.array(measurement, dtype=float)

        # ---- 动态观测噪声 R（与目标尺寸相关） ----
        std_pos = [self._STD_POS * abs(self.x[2]),
                   self._STD_POS * abs(self.x[3]),
                   self._STD_POS * abs(self.x[2]),
                   self._STD_POS * abs(self.x[3])]
        R = np.diag([s * s * 2.0 for s in std_pos])  # ×2 增强平滑

        # ---- 卡尔曼增益 ----
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # ---- 后验更新 ----
        y = z - self.H @ self.x  # 残差（innovation）
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def get_prediction(self):
        """返回预测的观测值 [cx, cy, w, h]（未初始化时返回 None）"""
        if not self._initialized:
            return None
        return self.x[:4].copy()

    def get_state(self):
        """返回完整状态向量（未初始化时返回 None）"""
        if not self._initialized:
            return None
        return self.x.copy()

    @property
    def initialized(self):
        return self._initialized


# ================================================================
#  跟踪轨道（Tracklet）
# ================================================================

class _Tracklet:
    """
    一条跟踪轨道，包装 _KalmanBoxFilter + 辅助状态

    生命周期：
      创建（首次检测到）→ 跟踪中（可能 coast）→ 超过丢失上限 → 回收
    """

    __slots__ = (
        "track_id", "kf", "lost_count", "max_lost",
        "confidence", "cls_id",
        "age", "hits", "last_center", "last_box",
    )

    def __init__(self, track_id, measurement, confidence, cls_id, max_lost=8):
        """
        Args:
            track_id:      全局唯一轨道ID
            measurement:   [cx, cy, w, h]
            confidence:    检测置信度
            cls_id:        类别ID
            max_lost:      最大连续丢失帧数
        """
        self.track_id = track_id
        self.kf = _KalmanBoxFilter()
        self.kf.initiate(measurement)

        self.lost_count = 0
        self.max_lost = max_lost
        self.confidence = confidence
        self.cls_id = cls_id
        self.age = 1          # 已存在帧数
        self.hits = 1         # 成功匹配次数

        cx, cy, w, h = measurement
        self.last_center = (float(cx), float(cy))
        self.last_box = None  # 首次调用 predict/update 后设置

    def predict(self):
        """预测 -> 更新 last_center / last_box"""
        pred = self.kf.predict()
        if pred is not None:
            cx, cy, w, h = pred
            self.last_center = (float(cx), float(cy))
            self.last_box = self._center_to_box(cx, cy, w, h)
            self.age += 1
        return self.last_center, self.last_box

    def update(self, detected_box, confidence, cls_id):
        """
        用检测结果更新轨道

        Args:
            detected_box: [x1, y1, x2, y2]
            confidence:   检测置信度
            cls_id:       类别ID
        """
        cx = (detected_box[0] + detected_box[2]) / 2.0
        cy = (detected_box[1] + detected_box[3]) / 2.0
        w  = detected_box[2] - detected_box[0]
        h  = detected_box[3] - detected_box[1]

        self.kf.update([cx, cy, w, h])
        self.last_center = (float(cx), float(cy))
        self.last_box = detected_box

        self.confidence = confidence
        self.cls_id = cls_id
        self.lost_count = 0
        self.hits += 1
        self.age += 1

    def mark_miss(self):
        """标记当前帧未匹配到检测"""
        self.lost_count += 1
        self.age += 1

    @property
    def is_dead(self):
        """超过最大丢失帧数时标记为死亡，可回收"""
        return self.lost_count > self.max_lost

    @property
    def is_coasting(self):
        """当前帧是否处在预测（无实测）状态"""
        return self.lost_count > 0

    @staticmethod
    def _center_to_box(cx, cy, w, h):
        """[cx, cy, w, h] → [x1, y1, x2, y2]"""
        half_w = w / 2.0
        half_h = h / 2.0
        return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

    def __repr__(self):
        return (f"_Tracklet(id={self.track_id}, "
                f"lost={self.lost_count}/{self.max_lost}, "
                f"age={self.age}, hits={self.hits})")


# ================================================================
#  多目标追踪器
# ================================================================

class TargetTracker:
    """
    轻量级多目标追踪器

    职责：
      - 接收每帧原始 YOLO 检测列表
      - 通过卡尔曼滤波 + 最邻近关联维持稳定目标ID
      - 提供平滑后的目标框 / 中心点供 PID 控制使用
      - 自动回收超过丢失上限的旧轨道

    典型配置（建议在实例化时按需调整）：
      | 参数 | 默认值 | 说明 |
      |------|--------|------|
      | max_lost_frames | 8 | 目标连续消失帧数上限（~250ms @ 30fps） |
      | max_association_dist | 200 | 中心距离软阈值（像素），用于归一化代价 |
      | min_hits | 3 | 新轨道确认前需要连续匹配帧数 |
      | dist_weight | 0.5 | 关联代价中中心距离的权重 |
      | iou_weight | 0.5 | 关联代价中 (1-IoU) 的权重 |

    update() 返回值:
      {
        "tracked":      bool    — 是否存在有效主目标
        "primary_id":   int     — 主目标 ID（None 如果无目标）
        "center":       (cx,cy) — 平滑后中心像素坐标
        "box":  [x1,y1,x2,y2]   — 平滑后边界框
        "confidence":   float   — 当前置信度（coast 期间从最近测量继承）
        "is_predicted": bool    — True = 当前帧 coasting（无实测）
        "lost_frames":  int     — 连续未命中帧数
        "n_active":     int     — 当前活跃轨道总数
        "raw": [...]            — 关联到的原始检测（coast 时为 None）
      }
      或丢失时:
      {
        "tracked": False,
        "primary_id": None,
        ...
        "lost_frames": > max_lost_frames
      }
    """

    def __init__(self, max_lost_frames=8, max_association_dist=200.0,
                 min_hits=3, dist_weight=0.5, iou_weight=0.5):
        """
        Args:
            max_lost_frames:      最大连续丢失帧数（默认 8 ≈ 250-300ms）
            max_association_dist: 中心距离软阈值（像素，默认 200px），用于归一化
            min_hits:             新轨道确认前最少连续匹配帧数（默认 3）
            dist_weight:          关联代价中中心距离项的权重（默认 0.5）
            iou_weight:           关联代价中 (1 - IoU) 项的权重（默认 0.5）
        """
        self.max_lost_frames = max_lost_frames
        self.max_association_dist = max_association_dist
        self.min_hits = min_hits
        self.dist_weight = dist_weight
        self.iou_weight = iou_weight

        self._next_id = 0
        self._tracks = {}         # {track_id: _Tracklet}
        self._frame_count = 0

        # 缓存上一帧的 primary 结果（锁外创建副本供控制线程读取）
        self._last_result = {
            "tracked": False,
            "primary_id": None,
            "center": None,
            "box": None,
            "confidence": 0.0,
            "is_predicted": False,
            "lost_frames": 0,
            "n_active": 0,
            "raw": None,
        }

    # ------------------------------------------------------------------
    #  主接口
    # ------------------------------------------------------------------

    def update(self, detections, frame_shape=None):
        """
        每帧调用一次，更新追踪状态

        Args:
            detections:   YOLO postprocess 输出的检测列表
                          [[x1, y1, x2, y2, conf, cls_id], ...]
                          允许为空列表 []（表示无检测）
            frame_shape:  (height, width) 用于边界钳制，可选

        Returns:
            dict — 主目标跟踪结果（格式见类文档）
        """
        self._frame_count += 1

        # ---- 步骤 1：所有活跃轨道先 predict ----
        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            track.predict()

        # ---- 步骤 2：关联 — 提取检测中心 ----
        if detections:
            # 提取检测中心 + 尺寸
            det_centers = []  # [(idx, cx, cy, w, h, conf, cls_id)]
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = det
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w  = x2 - x1
                h  = y2 - y1
                det_centers.append((i, cx, cy, w, h, conf, cls_id))

            # 关联
            matches, unmatched_tids, unmatched_dids = self._associate(det_centers)
        else:
            matches = []
            unmatched_tids = list(self._tracks.keys())
            unmatched_dids = []

        # ---- 步骤 3：更新匹配的轨道 ----
        for tid, did in matches:
            _, cx, cy, w, h, conf, cls_id = det_centers[did]
            raw_box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            self._tracks[tid].update(raw_box, conf, cls_id)

        # ---- 步骤 4：未匹配的轨道 → mark_miss / 回收 ----
        for tid in unmatched_tids:
            if tid in self._tracks:
                self._tracks[tid].mark_miss()
                if self._tracks[tid].is_dead:
                    logger.debug("轨道 %d 已死亡 (lost=%d), 回收",
                                 tid, self._tracks[tid].lost_count)
                    del self._tracks[tid]

        # ---- 步骤 5：未匹配的检测 → 创建新轨道 ----
        for did in unmatched_dids:
            _, cx, cy, w, h, conf, cls_id = det_centers[did]
            measurement = [cx, cy, w, h]
            new_track = _Tracklet(
                track_id=self._next_id,
                measurement=measurement,
                confidence=conf,
                cls_id=cls_id,
                max_lost=self.max_lost_frames,
            )
            # 新轨道直接使用测量值作为 last_box
            new_track.last_box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # ---- 步骤 6：选取主目标（优先级：置信度 > 命中数 > 贴近画面中心） ----
        result = self._select_primary(frame_shape)
        self._last_result = result
        return result

    def get_last_result(self):
        """获取上一帧的跟踪结果（线程安全，外部可直接读取）"""
        return self._last_result

    def reset(self):
        """重置所有轨道（清空追踪状态）"""
        self._tracks.clear()
        self._next_id = 0
        self._frame_count = 0
        self._last_result = {
            "tracked": False,
            "primary_id": None,
            "center": None,
            "box": None,
            "confidence": 0.0,
            "is_predicted": False,
            "lost_frames": 0,
            "n_active": 0,
            "raw": None,
        }
        logger.info("TargetTracker 已重置")

    # ------------------------------------------------------------------
    #  属性
    # ------------------------------------------------------------------

    @property
    def active_tracks(self):
        """返回当前所有活跃轨道（dict: {track_id: _Tracklet} 的浅复制）"""
        return dict(self._tracks)

    @property
    def active_count(self):
        """当前活跃轨道数"""
        return len(self._tracks)

    # ------------------------------------------------------------------
    #  内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(box_a, box_b):
        """
        计算两个轴对齐框的 IoU

        Args:
            box_a, box_b: [x1, y1, x2, y2]

        Returns:
            float — IoU ∈ [0, 1]
        """
        if box_a is None or box_b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0.0:
            return 0.0

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter_area

        if union <= 0.0:
            return 0.0
        return inter_area / union

    def _associate(self, det_centers):
        """
        中心距离 + IoU 加权的贪婪关联

        代价公式：
            cost = dist_weight * (center_dist / max_association_dist)
                 + iou_weight  * (1 - IoU)

        仅当 cost < 1.0 时才视为有效匹配。

        Args:
            det_centers: [(idx, cx, cy, w, h, conf, cls_id), ...]

        Returns:
            (matches, unmatched_tids, unmatched_dids)
        """
        if not self._tracks or not det_centers:
            return [], list(self._tracks.keys()), list(range(len(det_centers)))

        # 预先计算每个检测的框
        det_boxes = []
        for _, cx, cy, w, h, _, _ in det_centers:
            det_boxes.append([cx - w / 2.0, cy - h / 2.0,
                              cx + w / 2.0, cy + h / 2.0])

        matches = []
        assigned_dets = set()
        assigned_tracks = set()

        # 置信度高的轨道优先匹配
        sorted_tracks = sorted(
            self._tracks.items(),
            key=lambda kv: kv[1].confidence,
            reverse=True
        )

        max_dist = max(self.max_association_dist, 1e-3)

        for tid, track in sorted_tracks:
            if tid in assigned_tracks:
                continue

            pred_cx, pred_cy = track.last_center
            pred_box = track.last_box

            best_cost = 1.0
            best_did = None

            for did, _, d_cx, d_cy, _, _, _ in det_centers:
                if did in assigned_dets:
                    continue

                dist = np.hypot(pred_cx - d_cx, pred_cy - d_cy)
                norm_dist = dist / max_dist
                iou_val = self._iou(pred_box, det_boxes[did])

                cost = (self.dist_weight * norm_dist +
                        self.iou_weight * (1.0 - iou_val))

                if cost < best_cost:
                    best_cost = cost
                    best_did = did

            if best_did is not None:
                matches.append((tid, best_did))
                assigned_dets.add(best_did)
                assigned_tracks.add(tid)

        unmatched_tids = [tid for tid in self._tracks if tid not in assigned_tracks]
        unmatched_dids = [did for did in range(len(det_centers)) if did not in assigned_dets]

        return matches, unmatched_tids, unmatched_dids

    def _select_primary(self, frame_shape=None):
        """
        从活跃轨道中选出主目标

        优先级：
          1. 含有实测（非 coasting）且确认（hits ≥ min_hits）
          2. 实测轨道中置信度最高
          3. 若无实测轨道，取 coast 中最接近画面中心的

        Returns:
            dict — 与 update() 返回值格式一致
        """
        if not self._tracks:
            return {
                "tracked": False,
                "primary_id": None,
                "center": None,
                "box": None,
                "confidence": 0.0,
                "is_predicted": False,
                "lost_frames": 0,
                "n_active": 0,
                "raw": None,
            }

        # 分组：实测轨道 vs coasting 轨道
        measured = []
        coasting = []

        for tid, track in self._tracks.items():
            entry = (tid, track)
            if track.lost_count == 0 and track.hits >= self.min_hits:
                measured.append(entry)
            elif track.lost_count == 0:
                # hits < min_hits 尚未确认，视为暂态
                measured.append(entry)
            else:
                coasting.append(entry)

        # 优先从实测轨道选
        chosen = None
        is_predicted = False

        if measured:
            # 置信度最高的实测轨道
            measured.sort(key=lambda kv: kv[1].confidence, reverse=True)
            chosen = measured[0]
        elif coasting:
            # 无实测时选最靠近画面中心的
            cx_frame = frame_shape[1] / 2.0 if frame_shape else 320.0
            cy_frame = frame_shape[0] / 2.0 if frame_shape else 240.0
            coasting.sort(key=lambda kv: (
                (kv[1].last_center[0] - cx_frame) ** 2 +
                (kv[1].last_center[1] - cy_frame) ** 2
            ))
            chosen = coasting[0]
            is_predicted = True

        if chosen is None:
            return {
                "tracked": False,
                "primary_id": None,
                "center": None,
                "box": None,
                "confidence": 0.0,
                "is_predicted": False,
                "lost_frames": 0,
                "n_active": len(self._tracks),
                "raw": None,
            }

        tid, track = chosen
        cx, cy = track.last_center
        box = track.last_box
        conf = track.confidence
        lost = track.lost_count

        return {
            "tracked": True,
            "primary_id": tid,
            "center": (float(cx), float(cy)),
            "box": box,
            "confidence": float(conf),
            "is_predicted": is_predicted,
            "lost_frames": lost,
            "n_active": len(self._tracks),
            "raw": (None if is_predicted else box),
        }

    def __repr__(self):
        tracks_info = ", ".join(str(t) for t in self._tracks.values())
        return (f"TargetTracker(frame={self._frame_count}, "
                f"active={len(self._tracks)}, "
                f"tracks=[{tracks_info}])")
