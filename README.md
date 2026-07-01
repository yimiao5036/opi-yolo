# 🚁 OPI-YOLO — 昇腾 NPU 无人机目标检测与闭环控制系统

> **硬件平台：** 香橙派 AI Pro（Orange Pi AI Pro）｜**加速单元：** 华为昇腾 Ascend NPU  
> **通信协议：** ZeroMQ Router/Dealer 架构 ｜ **飞控协议：** MAVLink（PX4 / ArduPilot）  
> **检测模型：** YOLOv26n ｜ **部署方式：** ais_bench (昇腾推理 SDK)

---

> [!IMPORTANT]
> **⚡ AI 开发者注意**  
> 本项目已建立现成的 `.codegraph` 代码图谱索引。后续任何 AI 辅助工具接入时，请**优先使用 `codegraph_explore` 等工具**读取该数据库以获得全局代码感知能力，切勿重复生成独立索引。索引覆盖 28 个文件、452 个符号节点、708 条依赖边。

---

## 📋 目录

- [项目简介](#-项目简介)
- [系统架构](#-系统架构)
- [模块说明](#-模块说明)
- [状态机流程](#-状态机流程)
- [快速开始](#-快速开始)
- [环境要求与依赖](#-环境要求与依赖)
- [目录结构](#-目录结构)
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
                                  │ 检测结果
                                  ▼
 ┌────────────────┐      ┌──────────────────┐
 │ MissionManager │ ◄──  │  TargetTracker   │
 │ 任务状态机      │      │  卡尔曼滤波+Coast │
 └───────┬────────┘      └────────┬─────────┘
         │                       │ 追踪坐标
         ▼                       ▼
 ┌─────────────────────────────────────────┐
 │          UAVControlLoop (20Hz)          │
 │  PID → VELOCITY SETPOINT → RouterProxy  │
 └─────────────────────────────────────────┘
                │
                ▼
          PX4 / ArduPilot 飞控
```

**关键设计理念：**

- **解耦通信层**：通过 ZeroMQ Router 代理隔离 Python 控制层与 MAVLink 底层
- **异步流水线**：视频采集、NPU 推理、UDP 图传、飞控控制分属独立线程
- **安全优先**：防断流刹车（500ms）、卡尔曼 Coast 预测、PID 积分清空、Failsafe 熔断

---

## 🏗 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    MissionOrchestrator                    │
│    run_mission.py — 全任务编排器                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌───────────────────┐    ┌──────────────────────────┐   │
│  │   MissionManager   │    │      UAVControlLoop       │   │
│  │   航点巡航状态机    │    │    20Hz VELOCITY 闭环      │   │
│  │   POSITION SETPOINT│    │    PID + TargetTracker    │   │
│  │  (非追踪期间)       │    │  (VISUAL_TRACKING 期间)    │   │
│  └────────┬──────────┘    └───────────┬──────────────┘   │
│           │                            │                   │
│           └──────────┬─────────────────┘                   │
│                      ▼                                     │
│           ┌──────────────────────┐                        │
│           │   RouterProxy (ZMQ)   │  共享 ZeroMQ 代理       │
│           │   REQ + SUB 双通道   │  与 PX4 Router 通信     │
│           └──────────┬───────────┘                        │
│                      │                                     │
│           ┌──────────▼───────────┐                        │
│           │  ZMQ Router (机载)    │  协议转换层              │
│           │  ←→ PX4 MAVLink      │                        │
│           └──────────────────────┘                        │
│                                                          │
│  ┌───────────────────┐    ┌──────────────────────────┐   │
│  │   VideoStreaming   │    │      YOLO26UAVInfer      │   │
│  │   摄像头/RTSP 采集   │    │   昇腾 NPU 硬件推理       │   │
│  │   UDP 图传发送      │    │   letterbox + 后处理     │   │
│  └───────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

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
| `SETPOINT` | Python → Router | REQ | 20Hz | 实时 Offboard 控制 |
| `WAYPOINT` | Python → Router | REQ | 按需 | 航点飞行 |
| `COMMAND` | Python → Router | REQ | 按需 | ARM / OFFBOARD / LAND / RTL |
| `QUERY` | Python → Router | REQ | 按需 | 查询 HOME_POSITION |
| `STATE` | Router → Python | SUB | 10Hz | 飞机状态推送 |
| `ALERT` | Router → Python | SUB | 按需 | 紧急告警 |
| `PX4_ACK` | Router → Python | SUB | 按需 | COMMAND_ACK 转发 |

> 详细协议规范见 [`python_protocol.md`](./python_protocol.md)

### 控制权协调规则

同一时刻只有一方持有 **SETPOINT 发令权**：

| 状态机状态 | 控制者 | 控制模式 |
|-----------|--------|---------|
| 非 `VISUAL_TRACKING` | MissionManager | POSITION（经纬度 lat/lon + 高度） |
| `VISUAL_TRACKING` | UAVControlLoop | VELOCITY（机体速度 m/s） |

---

## 🔧 模块说明

### 核心入口

| 文件 | 说明 |
|------|------|
| `run_mission.py` | **主任务入口**。实例化所有子系统，20Hz 主循环协调 MissionManager 状态机与 UAVControlLoop 控制权，含 HUD 渲染、CSV 飞行日志、TCP 远程急停 |
| `infer_camera_modular.py` | **模块化推理入口**。独立运行的 NPU 推理+控制程序，含 USB/RTSP 自动重连 |
| `infer_camera.py` | 推理程序（pymavlink 直连旧版） |
| `infer_camera_direct.py` | 推理程序（DroneController 直连旧版） |
| `infer_video_direct.py` | 离线视频推理程序 |
| `gent_mission.py` | 航点巡航+视觉追踪融合版（直连 pymavlink 旧版） |

### drone_controller/ — 核心业务层

| 文件 | 类/功能 | 说明 |
|------|---------|------|
| `mission_manager.py` | `MissionManager`, `MissionState`, `FailsafeTriggered` | **航点巡航状态机** — 封装完整任务生命周期（INIT→ARMING→TAKEOFF→NAVIGATING→HOLD_TASK→VISUAL_TRACKING→RETURNING→LANDING→FINISHED），所有 SETPOINT 通过 RouterProxy 发送 |
| `router_proxy.py` | `RouterProxy` | **ZMQ 通信代理** — 封装 REQ/SUB 协议，提供 `send_setpoint()`、`send_command()`、`send_waypoint()`、`get_latest_state()` 等线程安全接口，含 STATE 新鲜度检测 |
| `target_tracker.py` | `TargetTracker`, `_KalmanBoxFilter` | **目标追踪过滤器** — 恒定速度卡尔曼滤波，支持 IOU 关联、coast 预测、ID 稳定 |
| `pid_counter.py` | `PID` | **PID 控制器** — 标准比例-积分-微分调节器，支持积分限幅和手动复位 |
| `base_control.py` | `DroneController` | **底层 MAVLink 驱动** — 直连串口/UDP 的 PX4/ArduPilot 驱动（旧版，现推荐使用 RouterProxy） |

### utils/ — 工具模块

| 文件 | 功能 |
|------|------|
| `coord.py` | `haversine()` 球面距离、`dist_3d_latlon()` 3D 距离、`latlon_to_ned()` NED 坐标转换 |

### 其他

| 文件 | 说明 |
|------|------|
| `control.py` | 简单 pymavlink 飞行控制脚本 |
| `tcp_client.py` | TCP 命令通道测试客户端 |
| `transform_app.py` | 地面站数据中继/转发服务 |
| `videotest.py` | 摄像头打开测试 |
| `base_test/` | 基础测试（NPU 推理、飞控通信、UDP 接收等） |

### 模块详解

#### VideoStreaming — 视频流采集与图传

**文件：** `infer_camera_modular.py:34` ｜ `class VideoStreaming`

独立线程从摄像头/RTSP 采集帧，单帧缓冲区 `Queue(maxsize=1)` 避免延迟累积，通过独立线程 UDP 推流至地面站。

**自动重连机制：**
- 每轮 5 次快速重试，间隔 1 秒
- 5 次全部失败 → 30 秒冷却期
- 分段睡眠确保线程停止信号秒级响应

#### YOLO26UAVInfer — NPU 推理核心

**文件：** `infer_camera_modular.py:190` ｜ `class YOLO26UAVInfer`

基于华为昇腾 `ais_bench` SDK（`InferSession`）的推理封装。`letterbox()` 等比例缩放填充 + `postprocess()` 坐标回映射。

#### UAVControlLoop — 飞控闭环控制

**文件：** `infer_camera_modular.py:269` ｜ `class UAVControlLoop`

20Hz 独立线程：`TargetTracker` 平滑中心点 → PID 计算机体速度 `(vy, vz)` → `send_setpoint(VELOCITY)`。

**安全设计：** 防断流刹车（500ms）+ Coast 预测期 PID 积分清空

#### MissionManager — 任务状态机

**文件：** `drone_controller/mission_manager.py:64` ｜ `class MissionManager`

面向航点巡航 + 视觉追踪融合任务的复合状态机：

- **Failsafe 熔断**：检测到飞控模式被切出白名单时立即抛出异常
- **Haversine 距离判定**：使用经纬度球面距离进行到达判定
- **事件驱动日志**：状态切换单次输出，持续状态 3~10 秒节流

---

## 🔄 状态机流程

```
WAITING_WAYPOINTS
       │  收到航点
       ▼
    INIT ───────────────────────────────┐
       │  ARM OK                        │ 已OFFBOARD
       ▼                                │
    ARMING                              │
       │  解锁成功                       │
       ▼                                │
    TAKEOFF                             │
       │  到达高度 + OFFBOARD            │
       ▼                                │
    NAVIGATING ──────── 所有航点完成 ────┤
       │                                 │
       │ 到达航点                         │
       ▼                                 │
    HOLD_TASK                            │
       │  ├─ 发现目标 → VISUAL_TRACKING   │
       │  └─ 超时     → 下一航点          │
       ▼                                 │
    VISUAL_TRACKING                      │
       │  ├─ 追踪完成 → NAVIGATING        │
       │  ├─ 丢失超时 → NAVIGATING        │
       │  └─ 总超时   → NAVIGATING        │
       ▼                                 │
    RETURNING ←──────────────────────────┘
       │  到达返航点
       ▼
    LANDING
       │  着陆 + 上锁
       ▼
    FINISHED
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 香橙派 AI Pro 加载昇腾环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. 配置

编辑 `config.json`，根据实际部署修改关键参数：
- `video.video_source` — 摄像头设备号（`0`）或 RTSP 地址
- `flight_control.req_endpoint` / `sub_endpoint` — ZeroMQ Router 端点
- `network.ground_station_ip` — 地面站 IP（图传接收）

### 3. 运行

```bash
# 完整任务模式（航点巡航 + 视觉追踪）
python run_mission.py config.json

# 独立模块化视觉追踪
python infer_camera_modular.py config.json

# 离线视频推理
python infer_video_direct.py
```

### 4. TCP 远程控制

```bash
echo "STOP" | nc <设备IP> 9999     # 紧急降落
echo "STATUS" | nc <设备IP> 9999   # 查询状态
echo "PING" | nc <设备IP> 9999     # 心跳测试
```

### 5. 地面站接收图传

参考 `base_test/receiver.py` 在 `ground_station_ip` 对应机器上运行 UDP 接收端。

---

## 📦 环境要求与依赖

### 硬件平台

| 组件 | 规格 |
|------|------|
| **主控** | 香橙派 AI Pro（Orange Pi AI Pro） |
| **NPU** | 华为昇腾 Ascend 310B |
| **摄像头** | USB 摄像头 或 RTSP 网络摄像头 |
| **飞控** | PX4 / ArduPilot（串口或 UDP 连接 Router） |

### Python 依赖

完整依赖见 [`requirements.txt`](./requirements.txt)

| 依赖 | 用途 | 备注 |
|------|------|------|
| `opencv-python` | 视频采集、图像预处理、渲染 | ≥4.5 |
| `pyzmq` | ZeroMQ REQ/SUB 通信 | ≥22 |
| `numpy` | 张量操作、矩阵运算 | ≥1.21 |
| `pymavlink` | MAVLink 协议（旧版直连模式） | 最新版 |
| `ais_bench` | 昇腾 NPU 推理 SDK | 需从华为昇腾官方安装 |

---

## 📁 目录结构

```
opi-yolo/
├── infer_camera_modular.py     # 🚀 模块化视觉追踪（推荐入口）
├── infer_camera_direct.py      # 直连 MAVLink 旧版
├── infer_camera.py             # 直连 pymavlink 旧版
├── infer_video_direct.py       # 离线视频推理
├── run_mission.py              # 🚀 全任务入口（航点+追踪）
├── gent_mission.py             # 航点巡航+追踪融合（旧版）
├── control.py                  # 简单飞控控制脚本
├── config.json                 # 全局配置文件
├── python_protocol.md          # 通信协议规范
├── tcp_client.py               # TCP 命令通道客户端
├── transform_app.py            # 地面站数据中继
├── videotest.py                # 摄像头测试
├── README.md                   # 本文档
├── requirements.txt            # Python 依赖清单
│
├── drone_controller/
│   ├── mission_manager.py      # 航点巡航状态机
│   ├── router_proxy.py         # ZMQ 通信代理
│   ├── target_tracker.py       # 卡尔曼目标追踪
│   ├── pid_counter.py          # PID 控制器
│   └── base_control.py         # MAVLink 底层驱动（旧版）
│
├── utils/
│   └── coord.py                # 坐标转换工具
│
├── base_test/
│   ├── check.py                # 视频帧率测试
│   ├── fc_test.py              # 飞控通信测试
│   ├── infer.py                # NPU 推理测试
│   └── receiver.py             # UDP 图传接收器
│
├── om/                         # OM 模型文件
│   └── yolo26n-balloon2.om
│
└── asset/                      # 资源文件
```

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
