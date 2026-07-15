import time

class PID:
    """
    通用 PID 控制器类
    封装了比例、积分、微分计算以及输出限幅、抗积分饱和功能
    """
    def __init__(self, kp, ki, kd, max_out, min_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.min_out = min_out

        self.last_err = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        # 积分冻结标志位，默认为不冻结（False）
        self.integral_frozen = False

    def update(self, error):
        """
        根据当前误差计算 PID 输出
        """
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.033  # 默认 30Hz 的更新步长

        # 1. 比例项
        p_out = self.kp * error

        # 2. 积分项
        # 增加策略：只有在未冻结时，才进行积分累加
        if not self.integral_frozen:
            self.integral += error * dt
            
        i_out = self.ki * self.integral
        # 将积分上限限制在总输出幅值的 20%
        i_out = max(min(i_out, self.max_out * 0.2), self.min_out * 0.2)

        # 3. 微分项
        d_out = self.kd * (error - self.last_err) / dt

        # 4. 总输出合成
        output = p_out + i_out + d_out

        # 5. 总输出限幅
        output = max(min(output, self.max_out), self.min_out)

        # 保存状态供下次使用
        self.last_err = error
        self.last_time = current_time

        return output

    def reset(self):
        """
        清空历史误差和积分，防止无人机在丢失目标重获后发生指令暴冲
        """
        self.last_err = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.integral_frozen = False  # 重置时默认恢复积分功能

    def reset_integral(self):
        """
        仅清空积分项（I term），保留上一帧误差以维持微分（D term）连续性
            用于卡尔曼 coast（滑行预测）期间：
            只靠比例 (P) + 微分 (D) 稳住，积分项清零防止过冲。
        """
        self.integral = 0.0

    # ================== 新增方法 ==================

    def freeze_integral(self):
        """
        冻结积分项：
        保持当前的积分值（保持之前的成果），但不再随时间累加新的误差。
        
        适用场景：执行机构饱和（如马达已满载）、或者进入特定暂态过程。
        """
        self.integral_frozen = True

    def unfreeze_integral(self, mode="maintain"):
        """
        配套解冻策略：
        
        :param mode: 解冻模式选择
            - "maintain": 保持冻结期间的值，直接从该点继续累加（最常用，平滑过渡）。
            - "clear": 解冻的同时清空历史积分，完全重新开始。
            - "fade": 仅保持原有积分的 50%（衰减），防止突变过冲。
        """
        self.integral_frozen = False
        
        if mode == "clear":
            self.integral = 0.0
        elif mode == "fade":
            self.integral *= 0.5
        elif mode == "maintain":
            pass # 保持原样，直接继续累加