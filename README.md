# 🚁 OPI-YOLO — 昇腾 NPU 无人机目标检测与闭环控制系统
### *这是一个小白的个人项目，慎用*

> **硬件平台：** 香橙派 AI Pro（Orange Pi AI Pro）｜**加速单元：** 华为昇腾 Ascend NPU  
> **通信协议：** ZeroMQ Router/Dealer 架构 ｜ **飞控协议：** MAVLink（PX4 / ArduPilot）  
> **检测模型：** YOLOv26n ｜ **部署方式：** ais_bench (昇腾推理 SDK)

---

> [!IMPORTANT]
> **⚡ AI 开发者注意（AI Orientation）**  
> 本项目已建立现成的 `.codegraph` 代码图谱索引（`.codegraph/codegraph.db`）。后续任何 AI 辅助工具接入时，请**优先使用 `mcp__codegraph` 系列工具**读取该数据库以获得全局代码感知能力（调用链追踪、依赖分析、符号定位），**切勿重复生成独立索引**。索引覆盖 25 个文件、329+ 个符号节点、458+ 条依赖边。

---

## 📋 目录

- [项目简介](#-项目简介)
- [系统架构](#-系统架构)
- [模块设计](#-模块设计)
  - [VideoStreaming — 视频流采集与图传](#1-videostreaming--视频流采集与图传)
  - [YOLO26UAVInfer — NPU 推理核心](#2-yolo26uavinfer--npu-推理核心)
  - [TargetTracker — 目标追踪与卡尔曼滤波](#3-targettracker--目标追踪与卡尔曼滤波)
  - [UAVControlLoop — 飞控闭环控制](#4-uavcontrolloop--飞控闭环控制)
  - [RouterProxy — ZeroMQ 通信代理](#5-routerproxy--zeromq-通信代理)
  - [MissionManager — 任务状态机](#6-missionmanager--任务状态机)
- [环境要求与依赖](#-环境要求与依赖)
- [配置文件](#-配置文件)
- [目录结构](#-目录结构)
- [快速开始](#-快速开始)
- [免责声明](#-免责声明)

---

## 🎯 项目简介

本项目实现了一套**基于机载 NPU 硬件加速的无人机实时目标检测与视觉闭环控制系统**，在香橙派 AI Pro（昇腾平台）上全速运行。

**核心工作流：**

```
摄像头/RTSP 视频流
      │
      ▼
 ┌────────────────┐      ┌──────────────────┐
 │ VideoStreaming │ ──►  │ YOLO26UAVInfer   │
 │  帧采集 + 图传  │      │  NPU 推理 + 后处理 │
 └────────────────┘      └────────┬─────────┘
                                  │ 检测结果 (bbox + score)
                                  ▼
 ┌────────────────┐      ┌──────────────────┐
 │ MissionManager │ ◄──  │  TargetTracker   │
 │ 任务状态机      │      │  卡尔曼滤波+Coast │
 └───────┬────────┘      └────────┬─────────┘
         │                       │ 追踪坐标
         ▼                       ▼
 ┌─────────────────────────────────────────┐
 │          UAVControlLoop (20Hz)          │
 │  PID 计算机体速度 → RouterProxy → Router │
 └─────────────────────────────────────────┘
                │
                ▼
          PX4 / ArduPilot 飞控
```

**关键设计理念：**

- **解耦通信层**：通过 ZeroMQ Router 代理隔离 Python 控制层与 MAVLink 底层，REQ/SUB 双通道实现指令-推送分离
- **异步流水线**：视频采集、NPU 推理、UDP 图传、飞控控制分属独立线程，互不阻塞
- **安全优先**：防断流超时刹车（500ms）、卡尔曼 Coast 预测、PID 积分自动清空、Failsafe 人工接管熔断

---

## 🏗 系统架构

### 通信架构

```
┌─────────────────────────────────────────────────────────┐
│                   香橙派 AI Pro                          │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Python    │──► │  ZeroMQ REQ │──► │              │  │
│  │ 控制层    │    │  :5555       │    │   Router     │  │
│  │           │◄── │  ACK 回执    │◄── │   代理       │  │
│  │           │    └──────────────┘    │              │  │
│  │           │    ┌──────────────┐    │  (C++/Rust)  │──► PX4
│  │           │◄── │  ZeroMQ SUB  │◄── │              │  │
│  │           │    │  :5556       │    │              │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│                                                         │
│  ┌──────────┐    ┌──────────────┐                       │
│  │  RTSP    │──► │  VideoStream │──► UDP :9999 → 地面站  │
│  │  摄像头   │    │  + 推理      │                       │
│  └──────────┘    └──────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### 消息协议

| type | 方向 | 通道 | 频率 | 说明 |
|------|------|------|------|------|
| `SETPOINT` | Python → Router | REQ | 20Hz | 实时 Offboard 控制（NED 位置/速度） |
| `WAYPOINT` | Python → Router | REQ | 按需 | 航点飞行（GPS + 高度 + 速度 + 动作） |
| `COMMAND` | Python → Router | REQ | 按需 | ARM / OFFBOARD / LAND / RTL / RESUME |
| `QUERY` | Python → Router | REQ | 按需 | 查询 HOME_POSITION 等 |
| `STATE` | Router → Python | PUB | 10Hz | 飞机状态（模式、位置、电量等） |
| `ALERT` | Router → Python | PUB | 按需 | 紧急告警（低电量、GPS 丢失等） |
| `PX4_ACK` | Router → Python | PUB | 按需 | PX4 COMMAND_ACK 转发 |

> 详细协议规范见 [`python_protocol.md`](./python_protocol.md)

---

## 🔧 模块设计

### 1. VideoStreaming — 视频流采集与图传

**文件：** `infer_camera_modular.py:31` ｜ `class VideoStreaming`

负责视频帧的**双线程异步采集**与**UDP 图传**。

| 特性 | 说明 |
|------|------|
| **独立采集线程** | 后台 `_capture_worker` 线程持续从摄像头/RTSP 读取帧，永不阻塞主循环 |
| **单帧缓冲区** | `queue.Queue(maxsize=1)` 仅保留最新帧，积压帧自动丢弃（避免延迟累积） |
| **RTSP 低延迟** | `CAP_PROP_BUFFERSIZE=1` + 自动追加 `?tcp` 传输模式 |
| **自动重连** | 读取失败 → 释放资源 → 1 秒后重新连接循环，断流后自主恢复 |
| **UDP 图传** | 独立 `sender_thread` 线程 JPEG 压缩后以 UDP 推流至地面站（默认 `:9999`），超 65KB 自动跳过 |

### 2. YOLO26UAVInfer — NPU 推理核心

**文件：** `infer_camera_modular.py:187` ｜ `class YOLO26UAVInfer`

基于华为昇腾 `ais_bench` SDK（`InferSession`）的推理封装。

| 阶段 | 说明 |
|------|------|
| **图像预处理** | `letterbox()` — 保持宽高比的缩放 + 灰边填充（640×640），适配 NPU 固定输入 |
| **张量转换** | `preprocess()` — HWC → CHW、归一化到 `[0,1]`、`float32`、添加 batch 维度 |
| **NPU 推理** | `InferSession(device_id, model_path).infer()` — 异步硬件推理 |
| **坐标回映射** | `postprocess()` — 置信度阈值过滤 → Letterbox 逆变换 → 原图坐标裁剪 |
| **可视化** | `draw_boxes()` — 在帧上渲染检测框 + 置信度标签 |

### 3. TargetTracker — 目标追踪与卡尔曼滤波

**文件：** `drone_controller/target_tracker.py:42` ｜ `class _KalmanBoxFilter` / `class TargetTracker`

基于恒定速度卡尔曼滤波实现单目标追踪，解决 YOLO 帧间抖动和短时遮挡问题。

| 组件 | 说明 |
|------|------|
| **卡尔曼滤波器** | 8 维状态 `[cx, cy, w, h, vx, vy, vw, vh]`，过程/观测噪声与目标尺寸动态关联 |
| **Tracklet 轨道** | 记录 `track_id`、连续丢失计数、匹配命中次数、生存帧数 |
| **数据关联** | 基于 IOU 的匈牙利匹配（`linear_assignment`），支持 `max_lost_frames` 帧内 coast |
| **Coast 预测** | 目标短暂丢失时靠卡尔曼预测持续输出位置，超阈值后判定目标死亡 |
| **命中验证** | `min_hits` 击发机制：新目标需连续匹配 N 帧才确认为有效轨道 |

### 4. UAVControlLoop — 飞控闭环控制

**文件：** `infer_camera_modular.py:266` ｜ `class UAVControlLoop`

**20Hz 高频控制线程**，通过 ZeroMQ RouterProxy 间接操控飞控。

```
┌─ 主循环 ─────────────────┐
│ update_detections()       │──→ 写入共享检测结果（锁保护）
└───────────────────────────┘
           │
┌─ 控制线程 (20Hz) ─────────┐
│ _compute_velocity()        │──→ PID 计算机体速度 (vy, vz)
│ proxy.send_setpoint()      │──→ ZMQ REQ → Router → PX4
└───────────────────────────┘
```

**安全设计：**

| 机制 | 说明 |
|------|------|
| **20Hz 保活心跳** | 控制线程以固定 20Hz 循环发送 SETPOINT，满足 Offboard 协议 ≥20Hz 要求 |
| **防断流刹车** | `_last_update_time` 超过 **500ms** 未更新 → `_brake_requested=True` → 强制零速刹车 |
| **PID 积分清空** | 目标丢失时调用 `pid_y.reset()` / `pid_z.reset()`，防止积分 windup 导致失控 |
| **起飞序列自动化** | `start_uav()`：启动 ZMQ → 等待 STATE → ARM → TAKEOFF → OFFBOARD 预热 → 启动控制线程 |

### 5. RouterProxy — ZeroMQ 通信代理

**文件：** `drone_controller/router_proxy.py:49` ｜ `class RouterProxy`

Python 控制层与外部 Router 进程之间的通信桥梁。

| 功能 | 说明 |
|------|------|
| **REQ 通道** | `send_setpoint()` / `send_command()` / `send_query()` / `send_waypoint()` 四种指令接口 |
| **SUB 通道** | 后台 `_sub_worker` 持续监听 STATE / ALERT / PX4_ACK 推送 |
| **STATE 缓存** | `_latest_state` 受锁保护的线程安全缓存 + 深度拷贝读取 |
| **新鲜度检测** | `is_state_fresh()` — 500ms 新鲜度阈值 + 2s 警告阈值（协议 §3.8.1） |
| **异常重建** | `_recover_req()` — REQ 超时或异常时自动重建 ZMQ socket（含 REQ_RELAXED+CORRELATE 兼容） |
| **回调钩子** | `set_state_callback()` / `set_alert_callback()` — 可注册外部回调处理推送消息 |

### 6. MissionManager — 任务状态机

**文件：** `drone_controller/mission_manager.py:24` ｜ `class MissionManager`

面向航点巡航 + 视觉追踪融合任务的复合状态机。

```
INIT → ARMING → TAKEOFF → NAVIGATING → HOLD_TASK ──→ VISUAL_TRACKING ──→ (继续巡航)
                                  │                      │                      │
                                  └── (无目标超时) ────────┘                      │
                                                                                ▼
                                                          RETURNING → LANDING → FINISHED
                                                                                    或
                                                          HOLD_FINAL (不回航时) ─── FINISHED
```

| 特性 | 说明 |
|------|------|
| **Failsafe 熔断** | 检测到飞控模式被切出 GUIDED/OFFBOARD 时立即抛出 `FailsafeTriggered` 异常，暴力熔断主循环 |
| **视觉追踪超时** | 追踪持续 5 秒视为任务完成（`_is_tracking_task_complete`，可自定义） |
| **悬停检索** | 到达航点后进入 HOLD_TASK，在规定时间内等待视觉发现目标 |
| **返航策略** | 支持 `return_to_home=True/False`，可选择返航降落或末端悬停 |

---

## 📦 环境要求与依赖

### 硬件平台

| 组件 | 规格 |
|------|------|
| **主控** | 香橙派 AI Pro（Orange Pi AI Pro）或兼容昇腾平台 |
| **NPU** | 华为昇腾 Ascend 310B / 同等算力单元 |
| **摄像头** | USB 摄像头 或 RTSP 网络摄像头 |
| **飞控** | PX4 / ArduPilot 固件兼容飞控（串口连接） |

### 软件依赖

| 依赖 | 用途 | 最低版本 |
|------|------|----------|
| `ais_bench` | 昇腾 NPU 推理 SDK | 最新稳定版 |
| `opencv-python` | 视频采集、图像预处理、可视化渲染 | ≥4.5 |
| `pyzmq` | ZeroMQ 通信（REQ/SUB 协议） | ≥22 |
| `numpy` | 张量操作、卡尔曼滤波矩阵运算 | ≥1.21 |
| `pymavlink` | MAVLink 协议（DroneController 模式） | 最新版 |

### 昇腾环境变量

```bash
# 香橙派 AI Pro 需加载（脚本已自动处理）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

## ⚙ 配置文件

`config.json` 集中管理系统配置：

```json
{
    "model": {
        "model_path": "./om/yolo26n-balloon2.om",   // OM 模型路径
        "conf_threshold": 0.25,                       // 置信度阈值
        "device_id": 0                                // NPU 设备 ID
    },
    "video": {
        "video_source": "rtsp://192.168.144.25:8554/main.264",  // 视频源（0=USB 摄像头）
        "jpeg_quality": 75                                      // 图传 JPEG 质量
    },
    "network": {
        "ground_station_ip": "192.168.31.221",  // 地面站 IP
        "udp_port": 9999                         // 图传 UDP 端口
    },
    "flight_control": {
        "req_endpoint": "tcp://127.0.0.1:5555",  // ZMQ REQ 地址
        "sub_endpoint": "tcp://127.0.0.1:5556",  // ZMQ SUB 地址
        "takeoff_alt": 5.0,                       // 起飞目标高度 (m)
        "control_hz": 20,                         // 控制频率 (Hz)
        "target_tracker": {
            "max_lost_frames": 8,                 // 最大连续丢失帧数
            "max_asscociation_dist": 200.0,       // 最大关联距离 (px)
            "min_hits": 3                         // 最小命中确认数
        }
    },
    "pid_y": { "kp": 1.0, "ki": 0.02, "kd": 0.3, "max_out": 0.8, "min_out": -0.8 },
    "pid_z": { "kp": 1.0, "ki": 0.02, "kd": 0.3, "max_out": 0.4, "min_out": -0.4 }
}
```

---

## 📁 目录结构

```
opi-yolo/
├── infer_camera_modular.py    # 🚀 主入口：模块化视觉追踪（推荐运行入口）
├── infer_camera_direct.py     # 直接调用控制（旧版，走 MAVLink 直连）
├── gent_mission.py            # 航点巡航 + 视觉追踪融合版
├── run_mission.py             # 自动巡点飞行业务测试
├── control.py                 # PID + MAVLink 指令（旧版，直连飞控模式）
├── config.json                # 项目全局配置文件
├── python_protocol.md         # Python ↔ Router 通信协议规范
├── start_npu_infer.sh         # 🔁 NPU 推理开机自启脚本（崩溃自动重启）
├── videotest.py               # 视频源连通性测试
├── README.md                  # 本文件
│
├── drone_controller/
│   ├── router_proxy.py        # ZMQ REQ/SUB 通信代理（当前主力通信方式）
│   ├── target_tracker.py      # 卡尔曼滤波目标追踪器
│   ├── mission_manager.py     # 航点任务状态机
│   ├── base_control.py        # MAVLink 底层驱动（旧版，已弃置）
│   └── pid_counter.py         # PID 计算器（抽离版）
│
├── base_test/
│   ├── check.py               # 视频帧率测试
│   ├── fc_test.py             # 飞控通信测试
│   ├── infer.py               # 图片推理测试
│   └── receiver.py            # UDP 图传接收模拟
│
├── om/                        # 昇腾 OM 模型文件存放目录
│   └── yolo26n-balloon2.om
│
└── asset/                     # 基础测试资源文件
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 加载昇腾环境（香橙派 AI Pro 必须）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 激活 Python 环境
conda activate base

# 安装依赖
pip install opencv-python pyzmq numpy pymavlink ais_bench
```

### 2. 配置

编辑 `config.json`，根据实际部署修改：
- `video.video_source`：摄像头设备号（`0`）或 RTSP 地址
- `flight_control.req/sub_endpoint`：ZeroMQ 端点地址
- `network.ground_station_ip`：地面站 IP（图传接收端）

### 3. 运行

```bash
# 模块化视觉追踪（推荐）
python infer_camera_modular.py

# 或使用自启脚本（崩溃自动重启）
chmod +x start_npu_infer.sh
./start_npu_infer.sh

# 或航点巡航 + 视觉追踪融合
python gent_mission.py
```

### 4. 地面站接收图传

在 `ground_station_ip` 对应的机器上运行 UDP 接收端（参考 `base_test/receiver.py`）查看实时图传画面。

---

## ⚠️ 免责声明

> **安全警告**：本系统涉及无人机自主飞行控制。在实际部署前，请务必：
> 1. 在仿真环境（如 Gazebo / SITL）中充分测试全部功能
> 2. 在固定台架上验证 PID 参数和控制逻辑
> 3. 确保遥控器手动接管通道可用且优先级高于自主控制
> 4. 准备物理急停开关
> 5. 遵守当地无人机飞行法规
>
> 本项目作者不对因使用本软件造成的任何财产损失或人身伤害承担责任。

---

> [!IMPORTANT]
> **⚡ AI 开发者注意（AI Orientation）**  
> 本项目已建立现成的 `.codegraph` 代码图谱索引（`.codegraph/codegraph.db`）。后续任何 AI 辅助工具接入时，请**优先使用 `mcp__codegraph` 系列工具**读取该数据库以获得全局代码感知能力（调用链追踪、依赖分析、符号定位），**切勿重复生成独立索引**。索引覆盖 25 个文件、329+ 个符号节点、458+ 条依赖边。
