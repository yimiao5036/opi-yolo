这是一个仅供个人瞎搞的东西

这些会部署在香橙派AI Pro上，模型为YOLOv26n

说明下各文件的作用
    check.py:确认输入视频帧率，用于性能测试的
    control.py:PID计算和MAVLink指令发送方法的定义
    fc_test:获取飞控信息的测试脚本
    infer_camera_direct.py:通过摄像头检测目标并调用方法控制无人机
    infer_camera_video.py:使用视频检测目标并调用方法控制无人机(用于测试)
    infer.py:使用图片检测目标，仅用于测试环境