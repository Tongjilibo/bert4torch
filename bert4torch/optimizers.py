'''Optimizer
'''

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
import torch
import math
from functools import partial


def get_linear_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch:int = -1):
    """带warmup的schedule, 源自transformers包optimization.py中
    
    :param num_warmup_steps: 需要warmup的步数, 一般为 num_training_steps * warmup_proportion(warmup的比例, 建议0.05-0.15)
    :param num_training_steps: 总的训练步数, 一般为 train_batches * num_epoch
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def extend_with_exponential_moving_average(model, decay=0.999):
    ''' 模型权重的指数滑动平均, 不参加梯度更新，只是记录滑动平均的参数，给预测使用
        注意区别于类似adam一类的自适应学习率优化器, 针对一阶二阶梯度的指数滑动平均, 两者完全不同

        Examples:
        ```python
        >>> # 初始化
        >>> ema = ExponentialMovingAverage(model, 0.999)

        >>> # 训练过程中, 更新完参数后, 同步update ema_weights weights
        >>> def train():
        ...     optimizer.step()
        ...     ema.step()

        >>> # eval前, 调用apply_ema_weights(); eval之后, restore_raw_weights()恢复原来模型的参数
        >>> def evaluate():
        ...     ema.apply_ema_weights()
        ...     # evaluate
        ...     # 如果想保存ema后的模型, 请在restore方法之前调用torch.save()
        ...     ema.restore_raw_weights()
        ```
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


class Lion(Optimizer):
  """Implements Lion algorithm.
  Straight copy from: https://github.com/google/automl/blob/master/lion/lion_pytorch.py
  """

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        p.add_(torch.sign(update), alpha=-group['lr'])
        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss


class Tiger(Optimizer):
  """
  苏神Tiger的pytorch实现
  https://github.com/bojone/tiger
  """

  def __init__(self, params, lr=1e-4, beta=0.965, weight_decay=0.01):
    """
    Attributes:
    ----------
    lr: float, learning rate, between 0 and 1

    beta: flaot, coefficients used for computing running averages of gradient and its square (default: 0.965)

    weight_decay: float, weight decay coefficient (default: 0.01)

    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= beta < 1.0:
      raise ValueError('Invalid beta parameter: {}'.format(beta))
    defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta = group['beta']

        # Weight update
        update = exp_avg * beta + grad * (1 - beta)
        p.add_(torch.sign(update), alpha=-group['lr'])

    return loss