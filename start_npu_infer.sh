#!/bin/bash
### 香橙派 NPU 推理 + 图传 自启脚本（支持 Ctrl+C 停止）
### 使用方法：chmod +x start_uav.sh

# ==================== 【核心注入 1：引入华为昇腾环境变量】 ====================
# 香橙派 AI Pro 默认的华为昇腾工具链环境变量脚本路径如下，必须优先加载它
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# ==================== 【核心注入 2：引入 Conda 环境】 ====================
if [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/usr/local/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/HwHiAiUser/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/HwHiAiUser/miniconda3/etc/profile.d/conda.sh"
fi

# 激活你的 (base) 虚拟环境
conda activate base
# ====================================================================

WORK_DIR="/home/HwHiAiUser/opi-yolo"
PYTHON_SCRIPT="infer_camera_modular.py"
LOG_FILE="/tmp/npu_infer.log"

PYTHON_BIN="python3"
STOP=0

# 捕获 Ctrl+C 等终止信号
trap 'STOP=1; echo "收到终止信号，正在停止..." >> "$LOG_FILE"' INT TERM

cd "$WORK_DIR" || exit 1

echo "$(date): 启动 NPU 推理程序 (按 Ctrl+C 停止)" >> "$LOG_FILE"

while [ $STOP -eq 0 ]; do
    # 启动子进程，并保存 PID
    $PYTHON_BIN "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1 &
    PYTHON_PID=$!

    # 等待子进程结束或收到信号
    wait $PYTHON_PID
    EXIT_CODE=$?

    if [ $STOP -eq 0 ]; then
        # 非人为终止时（崩溃），自动重启
        echo "$(date): 程序异常退出 (代码 $EXIT_CODE)，5 秒后重启..." >> "$LOG_FILE"
        sleep 5
    else
        echo "$(date): 用户主动终止，不再重启" >> "$LOG_FILE"
        break
    fi
done