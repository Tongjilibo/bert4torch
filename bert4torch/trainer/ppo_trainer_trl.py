'''修改trl包的PPOTrainer, 正在修改中
'''
import torch
from torch4keras.trainer import Trainer
from torch4keras.snippets import log_warn
from bert4torch.generation import SeqGeneration
try:
    from trl.trainer import PPOTrainer
except:
    PPOTrainer = object


class PPOTrainerTrl(Trainer, PPOTrainer):
    def __init__(self, *args, generation=None, generation_kwargs=None, **kwargs):
        if PPOTrainer == object:
            raise ValueError('Please install trl by running `pip install trl`')
        Trainer.__init__(self)
        PPOTrainer.__init__(self, *args, **kwargs)
        self.generation =generation
        self.generation_kwargs = generation_kwargs or {}

    def train_step(self, batch, *args):
        device = self.model.device
        question_tensors = batch["input_ids"]
        question_tensors = [torch.LongTensor(i).to(device).squeeze(0) for i in question_tensors]
        responses = []
        response_tensors = []
        for q_tensor in question_tensors:
            response_tensor = self.generation(q_tensor, return_prompt=False, **self.generation_kwargs)
            r = self.tokenizer.batch_decode(response_tensor, skip_special_tokens=True)[0]
            responses.append(r)
            response_tensors.append(response_tensor.squeeze(0))

        # Compute reward score
        score_outputs = [
            self.get_reward_model_output(self.reward_model, self.reward_tokenizer, q, r, device) for q, r in
            zip(batch["query"], responses)
        ]
        rewards = self.calculate_rewards(score_outputs, args.reward_baseline)

        # Run PPO step
        try:
            stats = self.step(question_tensors, response_tensors, rewards)
        except ValueError as e:
            log_warn(f"Failed to log stats because of {e}")
        return None, None, dict()

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
    
    @staticmethod
    def get_reward_model_output(reward_model, reward_tokenizer, question, answer, device):
        """
        Get the reward score for a given question and answer pair.
        """
        inputs = reward_tokenizer(question, answer, return_tensors='pt').to(device)
        score = reward_model(**inputs).logits[0].cpu().detach()

        return score
