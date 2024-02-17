'''Trainer
通用的Trainer定义在torch4keras中，这里定义NLP中独有的Trainer
'''

from torch4keras.trainer import *  # torch4keras>=0.1.2.post2
from .ppo_trainer import PPOTrainer
from .dpo_trainer import DPOTrainer
from .ptuningv2_trainer import PtuningV2Trainer
from .sequence_classification_trainer import SequenceClassificationTrainer