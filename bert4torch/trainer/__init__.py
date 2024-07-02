'''Trainer
通用的Trainer定义在torch4keras中，这里定义NLP中独有的Trainer
'''

from torch4keras.trainer import (
    AutoTrainer, 
    Trainer, 
    AccelerateTrainer,
    DPTrainer,
    DDPTrainer,
    DeepSpeedArgs,
    DeepSpeedTrainer,
    add_trainer,
    add_module
)  # torch4keras>=0.1.2.post2
from .ppo_trainer import PPOTrainer
from .dpo_trainer import DPOTrainer, DPOModel
from .ptuningv2_trainer import PtuningV2Trainer, PtuningV2Model
from .sequence_classification_trainer import SequenceClassificationTrainer, SequenceClassificationModel