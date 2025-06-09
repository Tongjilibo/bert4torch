'''修改trl包的PPOTrainer, 用于支持bert4torch框架
'''
import torch
from torch import nn
from torch4keras.trainer import Trainer
from bert4torch.models import BaseModel
from bert4torch.generation import SeqGeneration, Seq2SeqGeneration
from bert4torch.snippets import DottableDict, is_trl_available


try:
    import trl
    trl.trainer.ppo_trainer.SUPPORTED_ARCHITECTURES = (BaseModel, )
    from trl.trainer import PPOTrainer as PPOTrainerTrl
except:
    class PPOTrainerTrl:
        pass


class PPOTrainer(PPOTrainerTrl, Trainer):
    '''对trl中PPOTrainer的二次封装，使得可以使用model.complie(), model.fit()的方式进行模型训练

    :param generation_kwargs: generation使用的genration_kwargs
    :param reward_model: 奖励模型，可以用bert4torch构建的，也可以用transformers格式的
    :param reward_tokenizer: 奖励模型的tokenizer
    '''
    def __init__(self, *args, generation_kwargs=None, reward_model=None, reward_tokenizer=None, **kwargs):
        if not is_trl_available():
            raise ImportError('Please install trl by running `pip install trl`')
        Trainer.__init__(self)
        PPOTrainerTrl.__init__(self, *args, **kwargs)
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.generation_kwargs = generation_kwargs or {}
        self.reward_baseline = kwargs.pop('reward_baseline', 0)
        self.grad_accumulation_steps = self.config.gradient_accumulation_steps
        if getattr(self.model.module, 'is_decoder') is True:
            self.generation = SeqGeneration(self.model.module, **generation_kwargs)
        elif getattr(self.model.module, 'is_encoder_decoder') is True:
            self.generation = Seq2SeqGeneration(self.model.module, **generation_kwargs)
        else:
            raise ValueError('self.model.module is not a decoder/encoder-decoder model')
        self.generation.tokenizer = None  # 训练的时候已经传入的是token_ids，因此这里不tokenize了
        self.loss2metrics = False

    @staticmethod
    def get_actor_model(model):
        '''获取ActorModel'''
        from trl.models.modeling_value_head import ValueHead, AutoModelForCausalLMWithValueHead

        class ActorModel(BaseModel):
            def __init__(self, model, *arg, value_head_config=None, **kwargs):
                super().__init__(*arg, **kwargs)
                self.module = model
                if value_head_config is None:
                    value_head_config = DottableDict({'summary_dropout_prob': 0.1, 'hidden_size': self.module.config['hidden_size']})
                self.v_head = ValueHead(value_head_config, **kwargs)
                self._init_weights = AutoModelForCausalLMWithValueHead._init_weights
                self._init_weights(self, **kwargs)
            
            def forward(self, *args, **kwargs):
                self.module.with_lm = False
                hidden_states = self.module(kwargs['input_ids'])
                logits = self.module.lm_head(hidden_states)
                value = self.v_head(hidden_states).squeeze(-1)
                return logits, None, value
        return ActorModel(model)
    
    def train_step(self, train_X, _):
        if isinstance(train_X, (tuple, list)):
            question_tensors, query = train_X[0], train_X[1]
        elif isinstance(train_X, dict):
            question_tensors, query = train_X["input_ids"], train_X["query"]
        else:
            raise ValueError('Args `train_X` format illegel')
        
        # actor生成得到推理结果
        responses = []
        self.generation.decoder.with_lm = True  # 输出logits
        response_tensors = self.generation.generate(question_tensors, **self.generation_kwargs)
        self.generation.decoder.with_lm = False
        for response_tensor in response_tensors:
            r = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
            responses.append(r)

        # Compute reward score
        score_outputs = [self.get_reward_model_output(q, r) for q, r in zip(query, responses)]
        rewards = self.calculate_rewards(score_outputs, self.reward_baseline)

        # Run PPO step
        self.question_tensors = [i for i in question_tensors]
        self.response_tensors = response_tensors
        self.rewards = rewards
        self.ppo_step = True
        stats = self.step()
        
        batch = {'query': query, 'response': responses}
        self.log_stats(stats, batch, rewards)
        loss = torch.tensor(stats['ppo/loss/total'])
        loss_detail = {k:v for k,v in stats.items() if isinstance(v, (int, float))}  # 更新到logs中
        # 按照output, loss, loss_detail个顺序返回
        return None, loss, loss_detail

    def step(self):
        if self.ppo_step:
            self.ppo_step = False
            return PPOTrainerTrl.step(self, self.question_tensors, self.response_tensors, self.rewards)
        
    @staticmethod
    def calculate_rewards(reward_score_outputs, reward_baseline=0):
        """
        Calculate the reward for a given score output.
        :param reward_score_outputs: 
        :param reward_baseline: 
        :return: 
        """
        rewards = []
        for score in reward_score_outputs:
            if isinstance(score, torch.Tensor) and score.numel() == 1:
                reward_value = score.item() - reward_baseline
                rewards.append(torch.tensor(reward_value))
            else:
                # Use the average of the tensor elements as `score` is multiple elements
                reward_value = torch.mean(score).item() - reward_baseline
                rewards.append(torch.tensor(reward_value))
        return rewards
    
    def get_reward_model_output(self, question, answer):
        """
        Get the reward score for a given question and answer pair.
        """
        # 这里reward_tokenizer是transformer格式的
        inputs = self.reward_tokenizer(question, answer, return_tensors='pt').to(self.reward_model.device)
        if isinstance(self.reward_model, BaseModel):
            # bert4torch格式
            score = self.reward_model(inputs['input_ids'])
            if isinstance(score, torch.Tensor):
                score = score.cpu().detach()
            elif isinstance(score, (list, tuple)):
                score = score[0].cpu().detach()
            else:
                raise ValueError('Output `score` format illegal')
        else:
            # transformer格式
            score = self.reward_model(**inputs).logits[0].cpu().detach()
        return score

    def unwrap_model(self):
        '''返回nn.Module模块'''
        unwrap_model = self.accelerator.unwrap_model(self.model)
        if isinstance(unwrap_model, nn.Module): return unwrap_model
        return unwrap_model.module if hasattr(unwrap_model, 'module') else unwrap_model
