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

        # 2. 积分项（增加累加计算，并对积分做出局部限幅防止初段积分饱和）
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

    def reset_integral(self):
        """
        仅清空积分项（I term），保留上一帧误差以维持微分（D term）连续性

        用于卡尔曼 coast（滑行预测）期间：
          只靠比例 (P) + 微分 (D) 稳住，积分项清零防止过冲。
        """
        self.integral = 0.0