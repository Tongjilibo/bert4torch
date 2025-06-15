''' Callbacks
通用的callback定义在torch4keras中，这里是定义NLP中使用的callback
'''

import torch
import torch.nn.functional as F
from torch4keras.snippets import search_layer
from torch4keras.callbacks import *


class FGM():
    '''FGM对抗训练'''
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0 and not torch.isnan(norm):  # nan是为了apex混合精度时:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    '''PGD对抗训练'''
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False, **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):  # nan是为了apex混合精度时
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            # 修复如pooling层参与foward，但是不参与backward过程时grad为空的问题
            if param.requires_grad and (param.grad is not None):
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None):
                param.grad = self.grad_backup[name]


class VAT():
    '''虚拟对抗训练 https://github.com/namisan/mt-dnn/blob/v0.2/alum/adv_masked_lm.py'''
    def __init__(self, model, emb_name='word_embeddings', noise_var=1e-5, noise_gamma=1e-6, adv_step_size=1e-3, 
                 adv_alpha=1, norm_type='l2', **kwargs):
        self.model = model
        self.noise_var = noise_var  # 噪声的方差
        self.noise_gamma = noise_gamma # eps
        self.adv_step_size = adv_step_size  # 学习率
        self.adv_alpha = adv_alpha  # 对抗loss的权重
        self.norm_type = norm_type  # 归一化方式
        self.embed = None
        for (name, module) in self.model.named_modules():
            if emb_name in name:
                module.register_forward_hook(hook=self.hook)

    def hook(self, module, fea_in, fea_out):
        self.embed = fea_out
        return None
    
    def forward_(self, train_X, new_embed):
        # 把原来的train_X中的token_ids换成embedding形式
        if isinstance(train_X, (tuple, list)):
            new_train_X = [new_embed] + train_X[1:]  # 抛弃原有的train_X[0]，即token_ids
            adv_output = self.model.forward(*new_train_X)
        elif isinstance(train_X, torch.Tensor):
            adv_output = self.model.forward(new_embed)
        return adv_output

    def virtual_adversarial_training(self, train_X, logits):
        # 初始扰动 r
        noise = self.embed.data.new(self.embed.size()).normal_(0, 1) * self.noise_var
        noise.requires_grad_()
        # x + r
        new_embed = self.embed.data.detach() + noise
        adv_output = self.forward_(train_X, new_embed)  # forward第一次
        adv_logits = adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        adv_loss = self.kl(adv_logits, logits.detach(), reduction="batchmean")
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()
        # 梯度消失，退出
        if torch.isnan(norm) or torch.isinf(norm):
            return None
        # inner sum
        noise = noise + delta_grad * self.adv_step_size
        # projection
        noise = self.adv_project(noise, norm_type=self.norm_type, eps=self.noise_gamma)
        new_embed = self.embed.data.detach() + noise
        new_embed = new_embed.detach()
        # 在进行一次训练
        adv_output = self.forward_(train_X, new_embed)  # forward第二次
        adv_logits = adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        adv_loss_f = self.kl(adv_logits, logits.detach())
        adv_loss_b = self.kl(logits, adv_logits.detach())
        # 在预训练时设置为10，下游任务设置为1
        adv_loss = (adv_loss_f + adv_loss_b) * self.adv_alpha
        return adv_loss
    
    @staticmethod
    def kl(inputs, targets, reduction="sum"):
        """计算kl散度
        
        :param inputs：tensor，logits
        :param targets：tensor，logits
        """
        loss = F.kl_div(F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction=reduction)
        return loss

    @staticmethod
    def adv_project(grad, norm_type='inf', eps=1e-6):
        """L0,L1,L2正则，对于扰动计算"""
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction


class AdversarialTraining(Callback):
    """对抗训练Callback

    :param mode: str, 对抗训练的模式，可选{'fgm', 'pgd', 'vat', 'gradient_penalty'}
    :param adversarial: dict, 对抗训练的参数配置，不同模式所需参数不同
    """
    def __init__(self, mode, adversarial={}, **kwargs):
        super(AdversarialTraining, self).__init__(**kwargs)
        assert mode in {'', 'fgm', 'pgd', 'vat', 'gradient_penalty'}, 'adversarial_train support fgm, pgd, vat and gradient_penalty mode'
        self.mode = mode
        adversarial['epsilon'] = adversarial.get('epsilon', 1.0)
        adversarial['emb_name'] = adversarial.get('emb_name', 'word_embeddings')

        if mode == 'pgd':
            adversarial['K'] = adversarial.get('K', 3)  # 步数
            adversarial['alpha'] = adversarial.get('alpha', 0.3)  # 学习率
        elif mode == 'vat':
            adversarial['K'] = adversarial.get('K', 3)
            adversarial['noise_var'] = adversarial.get('noise_var', 1e-5)  # 噪声的方差
            adversarial['noise_gamma'] = adversarial.get('noise_gamma', 1e-6) # eps
            adversarial['adv_step_size'] = adversarial.get('adv_step_size', 1e-3)  # 学习率
            adversarial['adv_alpha'] = adversarial.get('adv_alpha', 1)  # 对抗loss的权重
            adversarial['norm_type'] = adversarial.get('norm_type', 'l2')  # 归一化方式
            adversarial['rank'] = adversarial.get('rank', 0)  # forward返回多个时指定使用的logit
        self.adversarial = adversarial

    def on_train_begin(self, logs=None):
        if self.mode in {'gradient_penalty', 'vat'}:
            self.trainer.retain_graph = True
        if self.mode == 'fgm':
            self.ad_train = FGM(self.model)
        elif self.mode == 'pgd':
            self.ad_train = PGD(self.model)
        elif self.mode == 'vat':
            self.ad_train = VAT(self.model, **self.adversarial)
    
        self.trainer.old_train_step = self.trainer.train_step
        self.trainer.train_step = self.train_step

    def train_step(self, train_X, train_y):
        output, loss, loss_detail = self.trainer.old_train_step(train_X, train_y)

        # 对抗训练执行逻辑
        if self.mode == 'fgm':
            self.ad_train.attack(**self.adversarial) # embedding被修改了
            output, loss, loss_detail = self.trainer.old_train_step(train_X, train_y)
            # loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            # 恢复Embedding的参数, 因为要在正常的embedding上更新参数，而不是增加了对抗扰动后的embedding上更新参数~
            self.ad_train.restore(**self.adversarial)
        elif self.mode == 'pgd':
            self.ad_train.backup_grad()  # 备份梯度
            for t in range(self.adversarial['K']):
                # 在embedding上添加对抗扰动, first attack时备份param.data
                self.ad_train.attack(**self.adversarial, is_first_attack=(t==0))
                if t != self.adversarial['K']-1:
                    self.optimizer.zero_grad()  # 为了累积扰动而不是梯度
                else:
                    self.ad_train.restore_grad() # 恢复正常的grad
                output, loss, loss_detail = self.trainer.old_train_step(train_X, train_y)
                # loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            self.ad_train.restore(**self.adversarial) # 恢复embedding参数
        # 梯度惩罚
        elif self.mode == 'gradient_penalty':
            para = search_layer(self.model, self.adversarial['emb_name'], retrun_first=True)
            gp = (para.grad ** 2).sum()
            loss += 0.5 * gp * self.adversarial['epsilon']
            loss.backward()
        # 虚拟对抗训练
        elif self.mode == 'vat':
            logit = output[self.adversarial['rank']] if isinstance(output, (tuple, list)) else output
            adv_loss = self.ad_train.virtual_adversarial_training(train_X, logit)
            loss += (adv_loss if adv_loss else 0)
            loss.backward()
            loss_detail.update({'loss_sup': loss.item(), 'loss_unsup': adv_loss.item()})
        return output, loss, loss_detail