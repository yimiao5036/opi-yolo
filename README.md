# 这是一个***小白瞎搞***的东西,仅供参考(我甚至觉得不能作为参考)

这些会部署在香橙派AI Pro上，模型为*YOLOv26n*

## 说明下各文件的作用
#### *python部分*
#### **零散脚本**
- check.py:确认输入视频帧率，用于性能测试的;
- control.py:PID计算和MAVLink指令发送方法的定义(已废弃);
- fc_test:获取飞控信息的测试脚本;
- infer_camera_direct.py:通过摄像头检测目标并调用方法控制无人机;
- infer.py:使用图片检测目标，仅用于测试环境;
- infer_camera_modular.py:模块化处理过的摄像头检测与控制(以确认链路基本通顺，但是无人机追踪效果如何尚未测试)
- infer_camera_video.py:使用视频检测目标并调用方法控制无人机(用于测试,未调参)
- gent_mission.py:尝试将视觉追踪与自动巡点飞行相结合(未测试，这应该是本项目一阶段的最终预期)
- run_mission.py:自动巡点飞行的业务测试(未测试)
#### drone_controller:控制模块
- base_control.py:定义了部分基本动作的方法(最新规划已改用路由，已弃置)
- pid_counter.py:抽离出来的PID计算器
- mission_manager.py:自动巡点飞行的控制模块(最新规划已改用路由，已弃置)
- router_proxy.py:基于新路由的路由规则编写的相关代码
- target_tracker.py:通过math库和numpy编写的简单目标跟踪
#### 其他目录以及文件：
- om:推理使用的.om文件
- asset:用于基础测试的文件
- config.json:配置文件
- start_npu_infer:一个.sh文件,计划后续将整个脚本作为一个服务