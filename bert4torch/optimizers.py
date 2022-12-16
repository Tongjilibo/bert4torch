from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """带warmup的schedule, 源自transformers包optimization.py中
    
    :param num_warmup_steps: 需要warmup的步数, 一般为 num_training_steps * warmup_proportion(warmup的比例, 建议0.05-0.15)
    :param num_training_steps: 总的训练步数, 一般为 train_batches * num_epoch
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def extend_with_exponential_moving_average(model, decay=0.999):
    ''' 模型权重的指数滑动平均, 不参加梯度更新，只是记录滑动平均的参数，给预测使用
        注意区别于类似adam一类的自适应学习率优化器, 针对一阶二阶梯度的指数滑动平均, 两者完全不同

        Example:
            >>> # 初始化
            >>> ema = ExponentialMovingAverage(model, 0.999)

            >>> # 训练过程中, 更新完参数后, 同步update ema_weights weights
            >>> def train():
            >>>     optimizer.step()
            >>>     ema.step()

            >>> # eval前, 调用apply_ema_weights(); eval之后, restore_raw_weights()恢复原来模型的参数
            >>> def evaluate():
            >>>     ema.apply_ema_weights()
            >>>     # evaluate
            >>>     # 如果想保存ema后的模型, 请在restore方法之前调用torch.save()
            >>>     ema.restore_raw_weights()
    '''
    class ExponentialMovingAverage():
        def __init__(self, model, decay):
            self.model = model
            self.decay = decay
            # 保存ema权重（当前step的每一层的滑动平均权重）
            self.ema_weights = {}
            # 在进行evaluate的时候, 保存原始的模型权重, 当执行完evaluate后, 从ema权重恢复到原始权重
            self.model_weights = {}

            # 初始化ema_weights为model_weights
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_weights[name] = param.data.clone()

        def step(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.ema_weights
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.ema_weights[name]
                    self.ema_weights[name] = new_average.clone()
        
        def apply_ema_weights(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.ema_weights
                    self.model_weights[name] = param.data
                    param.data = self.ema_weights[name]
        
        def restore_raw_weights(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.model_weights
                    param.data = self.model_weights[name]
            self.model_weights = {}
    return ExponentialMovingAverage(model, decay)