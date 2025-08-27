'''Loss
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Union, Tuple, Literal
import warnings


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        :param input: torch.Tensor, shape=[N, C]
        :param target: torch.Tensor, shape=[N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        :param output: torch.Tensor, shape=[N, C]
        :param target: torch.Tensor, shape=[N, ]
        """
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵；
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        """
        :param y_true: torch.Tensor, [..., num_classes]
        :param y_pred: torch.Tensor: [..., num_classes]
        """
        y_pred = (1-2*y_true) * y_pred
        y_pred_pos = y_pred - (1-y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵；
       请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax，预测阶段则输出y_pred大于0的类；
       详情请看：https://kexue.fm/archives/7359，https://kexue.fm/archives/8888
    """
    def __init__(self, mask_zero=False, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        '''
        :param y_true: shape=[..., num_positive]
        :param y_pred: shape=[..., num_classes]
        '''
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if self.mask_zero:
            infs = zeros + float('inf')
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)  # [..., num_positive]
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)  # [..., num_positive+1]
        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)  # a
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss  # b-a
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), self.epsilon, 1)  # 1-exp(b-a)
        neg_loss = all_loss + torch.log(aux_loss)  # a + log[1-exp(b-a)]
        return pos_loss + neg_loss


class ContrastiveLoss(nn.Module):
    """对比损失：减小正例之间的距离，增大正例和反例之间的距离
    公式：labels * distance_matrix.pow(2) + (1-labels)*F.relu(margin-distance_matrix).pow(2)
    https://www.sbert.net/docs/package_reference/losses.html

    :param margin: float, 距离参数，distance>margin时候不参加梯度回传，默认为0.5
    :param size_average: bool, 是否对loss在样本维度上求均值，默认为True
    :param online: bool, 是否使用OnlineContrastiveLoss, 即仅计算困难样本的loss, 默认为False
    """
    def __init__(self, margin=0.5, size_average=True, online=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.online = online

    def forward(self, distances, labels, pos_id=1, neg_id=0):
        """
        :param distances: torch.Tensor, 样本对之间的预测距离, shape=[btz]
        :param labels: torch.Tensor, 样本对之间的真实距离, shape=[btz]
        :param pos_id: int, 正样本的label
        :param neg_id: int, 负样本的label
        """
        if not self.online:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            return losses.mean() if self.size_average else losses.sum()
        else:
            negs = distances[labels == neg_id]
            poss = distances[labels == pos_id]

            # select hard positive and hard negative pairs
            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]
            
            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
            return positive_loss + negative_loss


class RDropLoss(nn.Module):
    '''R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop

    :param alpha: float, 控制rdrop的loss的比例
    :param rank: str, 指示y_pred的排列方式, 支持['adjacent', 'updown']
    '''
    def __init__(self, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha
        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, *args):
        '''支持两种方式: 一种是y_pred, y_true, 另一种是y_pred1, y_pred2, y_true

        :param y_pred: torch.Tensor, 第一种方式的样本预测值, shape=[btz*2, num_labels]
        :param y_true: torch.Tensor, 样本真实值, 第一种方式shape=[btz*2,], 第二种方式shape=[btz,]
        :param y_pred1: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        :param y_pred2: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        '''
        assert len(args) in {2, 3}, 'RDropLoss only support 2 or 3 input args'
        # y_pred是1个Tensor
        if len(args) == 2:
            y_pred, y_true = args
            loss_sup = self.loss_sup(y_pred, y_true)  # 两个都算

            if self.rank == 'adjacent':
                y_pred1 = y_pred[1::2]
                y_pred2 = y_pred[::2]
            elif self.rank == 'updown':
                half_btz = y_true.shape[0] // 2
                y_pred1 = y_pred[:half_btz]
                y_pred2 = y_pred[half_btz:]
        # y_pred是两个tensor
        else:
            y_pred1, y_pred2, y_true = args
            loss_sup = (self.loss_sup(y_pred1, y_true) + self.loss_sup(y_pred2, y_true)) / 2

        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))
        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha


class UDALoss(nn.Module):
    '''UDALoss，使用时候需要继承一下，因为forward需要使用到global_step和total_steps
    https://arxiv.org/abs/1904.12848

    :param tsa_schedule: str, tsa策略，可选['linear_schedule', 'exp_schedule', 'log_schedule']
    :param start_p: float, tsa生效概率下限, 默认为0
    :param end_p: float, tsa生效概率上限, 默认为1
    :param return_all_loss: bool, 是否返回所有的loss，默认为True
    :return: loss
    '''
    def __init__(self, tsa_schedule=None, start_p=0, end_p=1, return_all_loss=True):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_unsup = nn.KLDivLoss(reduction='batchmean')
        self.tsa_schedule = tsa_schedule
        self.start = start_p
        self.end = end_p
        if self.tsa_schedule:
            assert self.tsa_schedule in {'linear_schedule', 'exp_schedule', 'log_schedule'}, 'tsa_schedule config illegal'
        self.return_all_loss = return_all_loss

    def forward(self, y_pred, y_true_sup, global_step, total_steps):
        ''' y_pred由[pred_sup, true_unsup, pred_unsup]三部分组成
        
        :param y_pred: torch.Tensor, 样本预测值, shape=[btz_sup+btz_unsup*2, num_labels]
        :param y_true_sup: torch.Tensor, 样本真实值, shape=[btz_sup,]
        :param global_step: int, 当前训练步数
        :param total_steps: int, 训练总步数
        '''
        sup_size = y_true_sup.size(0)
        unsup_size = (y_pred.size(0) - sup_size) // 2

        # 有监督部分, 用交叉熵损失
        y_pred_sup = y_pred[:sup_size]
        if self.tsa_schedule is None:
            loss_sup = self.loss_sup(y_pred_sup, y_true_sup)
        else:  # 使用tsa来去掉预测概率较高的有监督样本
            threshold = self.get_tsa_threshold(self.tsa_schedule, global_step, total_steps, self.start, self.end)
            true_prob = torch.gather(F.softmax(y_pred_sup, dim=-1), dim=1, index=y_true_sup[:, None])
            sel_rows = true_prob.lt(threshold).sum(dim=-1).gt(0)  # 仅保留小于阈值的样本
            loss_sup = self.loss_sup(y_pred_sup[sel_rows], y_true_sup[sel_rows]) if sel_rows.sum() > 0 else 0

        # 无监督部分，这里用KL散度，也可以用交叉熵
        y_true_unsup = y_pred[sup_size:sup_size+unsup_size]
        y_true_unsup = F.softmax(y_true_unsup.detach(), dim=-1)
        y_pred_unsup = F.log_softmax(y_pred[sup_size+unsup_size:], dim=-1)
        loss_unsup = self.loss_unsup(y_pred_unsup, y_true_unsup)
        if self.return_all_loss:
            return loss_sup + loss_unsup, loss_sup, loss_unsup
        else:
            return loss_sup + loss_unsup

    @ staticmethod
    def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
        training_progress = global_step / num_train_steps
        if schedule == "linear_schedule":
            threshold = training_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == "log_schedule":
            scale = 5
            threshold = 1 - math.exp((-training_progress) * scale)
        return threshold * (end - start) + start


class TemporalEnsemblingLoss(nn.Module):
    '''TemporalEnsembling的实现，思路是在监督loss的基础上，增加一个mse的一致性损失loss

       - 官方项目：https://github.com/s-laine/tempens
       - pytorch第三方实现：https://github.com/ferretj/temporal-ensembling
       - 使用的时候，train_dataloader的shffle必须未False
    '''
    def __init__(self, epochs, max_val=10.0, ramp_up_mult=-5.0, alpha=0.5, max_batch_num=100, hist_device='cpu'):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.max_epochs = epochs
        self.max_val = max_val
        self.ramp_up_mult = ramp_up_mult
        self.alpha = alpha
        self.max_batch_num = max_batch_num  # 设置未None表示记录全部数据历史，数据量大时耗资源
        self.hist_unsup = []  # 历史无监督logit
        self.hist_sup = []  # 历史监督信息logit
        self.hist_device = hist_device
        self.hist_input_y = []  # 历史监督标签y
        assert (self.alpha >= 0) & (self.alpha < 1)  # 等于1的时候upata写分母为0

    def forward(self, y_pred_sup, y_pred_unsup, y_true_sup, epoch, bti):
        """
        :param y_pred_sup: torch.Tensor, 监督学习样本预测值, shape=[btz, num_labels]
        :param y_pred_unsup: torch.Tensor, 无监督学习样本预测值, shape=[btz, num_labels]
        :param y_true_sup: int, 监督学习样本真实值, shape=[btz,]
        :param epoch: int, 当前epoch
        :param bti: int, 当前batch的序号
        """
        self.same_batch_check(y_pred_sup, y_pred_unsup, y_true_sup, bti)
        
        if (self.max_batch_num is None) or (bti < self.max_batch_num):
            self.init_hist(bti, y_pred_sup, y_pred_unsup)  # 初始化历史
            sup_ratio = float(len(y_pred_sup)) / (len(y_pred_sup) + len(y_pred_unsup))  # 监督样本的比例
            w = self.weight_schedule(epoch, sup_ratio)
            sup_loss, unsup_loss = self.temporal_loss(y_pred_sup, y_pred_unsup, y_true_sup, bti)

            # 更新
            self.hist_unsup[bti] = self.update(self.hist_unsup[bti], y_pred_unsup.detach(), epoch)
            self.hist_sup[bti] = self.update(self.hist_sup[bti], y_pred_sup.detach(), epoch)
            # if bti == 0:  $ 用于检查每个epoch数据顺序是否一致
            #     print(w, sup_loss.item(), w * unsup_loss.item())
            #     print(y_true_sup)
            return sup_loss + w * unsup_loss, sup_loss, w * unsup_loss
        else:
            return self.loss_sup(y_pred_sup, y_true_sup)

    def same_batch_check(self, y_pred_sup, y_pred_unsup, y_true_sup, bti):
        '''检测数据的前几个batch必须是一致的, 这里写死是10个'''
        if bti >= 10:
            return
        if bti >= len(self.hist_input_y):
            self.hist_input_y.append(y_true_sup.to(self.hist_device))
        else:  # 检测
            err_msg = 'TemporalEnsemblingLoss requests the same sort dataloader, you may need to set train_dataloader shuffle=False'
            assert self.hist_input_y[bti].equal(y_true_sup.to(self.hist_device)), err_msg
        
    def update(self, hist, y_pred, epoch):
        '''更新历史logit，利用alpha门控来控制比例
        '''
        Z = self.alpha * hist.to(y_pred) + (1. -self.alpha) * y_pred
        output = Z * (1. / (1. - self.alpha ** (epoch + 1)))
        return output.to(self.hist_device)

    def weight_schedule(self, epoch, sup_ratio):
        max_val = self.max_val * sup_ratio
        if epoch == 0:
            return 0.
        elif epoch >= self.max_epochs:
            return max_val
        return max_val * np.exp(self.ramp_up_mult * (1. - float(epoch) / self.max_epochs) ** 2)

    def temporal_loss(self, y_pred_sup, y_pred_unsup, y_true_sup, bti):
        # MSE between current and temporal outputs
        def mse_loss(out1, out2):
            quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
            return quad_diff / out1.data.nelement()
        
        sup_loss = self.loss_sup(y_pred_sup, y_true_sup)
        # 原来实现是sup和unsup作为一个tensor，整体计算的，这里由于是拆分成两个tensor，因此分开算
        unsup_loss = mse_loss(y_pred_unsup, self.hist_unsup[bti].to(y_pred_unsup))
        unsup_loss += mse_loss(y_pred_sup, self.hist_sup[bti].to(y_pred_sup))
        return sup_loss, unsup_loss
    
    def init_hist(self, bti, y_pred_sup, y_pred_unsup):
        if bti >= len(self.hist_sup):
            self.hist_sup.append(torch.zeros_like(y_pred_sup).to(self.hist_device))
            self.hist_unsup.append(torch.zeros_like(y_pred_unsup).to(self.hist_device))


class DPOLoss:
    ''' DPO算法的loss计算
    :param label_smoothing: 标签平滑
    :param loss_type: loss类型
    :param pad_token_id: pad的token_id, 用于计算mask
    :param beta: float, dpo中beta参数
    :param reference_free: bool, 默认为False
    :param prefix: 进度条展示指标的前缀
    :param offset: 是否offset, 若input_ids末尾为<eos>则一般需要offset=True

    主要思路: 优化方向: 以下值往Max方向
    (policy_chosen_logps [↑] - reference_chosen_logps [→]) - (policy_rejected_logps [↓] - reference_rejected_logps [→])
    左半部分和右半部分的margin越大越好, 左半部分的含义是chosen response相较于没训练之前的累积概率差值, 右半部分代表rejected response相较于没训练之前的累计概率差值
    1) 左边变大, 右边变小, 理想情况, chosen response概率提升, rejected response概率下降
    2) 左边变小, 右边更小, chosen response概率下降, 但是rejected response概率下降的更多, 生成的时候还是倾向于good response
    3) 左边变的更大, 右边只大了一点点, 和2)同理

    Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
    '''
    def __init__(self, 
                 label_smoothing: float = 0,
                 loss_type: Literal["sigmoid", "robust", "exo_pair", "hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "aot_pair", "aot", "apo_zero", "apo_down"] = "sigmoid",
                 pad_token_id:int=-100, 
                 beta:float=0.1, 
                 reference_free=False, 
                 prefix='', 
                 offset=True,
                 f_divergence_type: Literal["reverse_kl", "js_divergence", "alpha_divergence"] = "reverse_kl",
                 f_alpha_divergence_coef:float = 1.0
                 ) -> None:
        if (loss_type in ["hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "apo_zero", "apo_down"] and label_smoothing > 0):
            warnings.warn("You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter.")
        self.pad_token_id = pad_token_id
        self.beta = beta
        self.reference_free = reference_free
        self.prefix = prefix
        self.offset = offset
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.f_divergence_type = f_divergence_type
        self.f_divergence_params = {"alpha_divergence_coef": f_alpha_divergence_coef}
    
    @staticmethod
    def cap_exp(value, cap=-1, decimal=4):
        # Cap the exponent value below the upper-bound to avoid overflow, before calling torch.exp
        if cap < 0:
            vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
            vdtype_log_max = torch.log(vdtype_max).to(value.device)
            cap = torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max
        return torch.exp(torch.clamp(value, max=cap))

    def __call__(self, policy_reference_logits:Union[List[torch.Tensor], Tuple[torch.Tensor]], labels:torch.Tensor):
        '''
        :param logit: tuple/list, 分别表示policy_logits, reference_logits，tensor中前一半为chosen，后一半为rejected
        :param labels: 真实标签
        '''
        policy_logits, reference_logits = policy_reference_logits  # 均为[btz*2, seq_len, vocab_size]

        # 计算真实标签labels对应token位置的log prob，均为[btz]
        policy_chosen_logps, policy_rejected_logps = self.get_batch_logps(policy_logits, labels, average_log_prob=self.loss_type == "ipo")
        reference_chosen_logps, reference_rejected_logps = self.get_batch_logps(reference_logits, labels, average_log_prob=self.loss_type == "ipo")

        chosen_logratios = policy_chosen_logps - (not self.reference_free) * reference_chosen_logps
        rejected_logratios = policy_rejected_logps - (not self.reference_free) * reference_rejected_logps

        if self.f_divergence_type == "alpha_divergence":
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = 1.0
            if self.f_divergence_params and "alpha_divergence_coef" in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params["alpha_divergence_coef"])
            logits = (self.cap_exp(rejected_logratios * -alpha_coef) - self.cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef

        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps

            if self.reference_free:
                ref_logratios = 0
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            logits = pi_logratios - ref_logratios
        
            if self.f_divergence_type == "js_divergence":
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # 计算loss
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        # elif self.loss_type == "bco_pair":
        #     chosen_logratios = policy_chosen_logps - reference_chosen_logps
        #     rejected_logratios = policy_rejected_logps - reference_rejected_logps

        #     chosen_rewards = self.beta * chosen_logratios
        #     rejected_rewards = self.beta * rejected_logratios
        #     rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        #     self.running.update(rewards)
        #     delta = self.running.mean

        #     losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
        #         -(self.beta * rejected_logratios - delta)
        #     )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. 
            # The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, 
            # where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)

            delta = chosen_logratios_sorted - rejected_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios_sorted, _ = torch.sort(pi_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)

            delta = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output

            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood

            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output

            losses_chosen = F.sigmoid(self.beta * chosen_logratios)  # Decrease chosen likelihood
            losses_rejected = 1 - F.sigmoid(
                self.beta * (chosen_logratios - rejected_logratios)
            )  # Decrease rejected likelihood more

            losses = losses_chosen + losses_rejected

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        loss_detail = {'loss': losses.mean()}
        loss_detail[f"{self.prefix}chosen"] = chosen_rewards.cpu().numpy().mean()
        loss_detail[f"{self.prefix}rejected"] = rejected_rewards.cpu().numpy().mean()
        loss_detail[f"{self.prefix}accuracies"] = reward_accuracies.cpu().numpy().mean()
        loss_detail[f"{self.prefix}margins"] = (chosen_rewards - rejected_rewards).cpu().numpy().mean()
        return loss_detail

    def get_batch_logps(self, logits:torch.FloatTensor, labels:torch.LongTensor, average_log_prob:bool=False):
        """计算真实标签labels对应token位置的log prob
        :param logits: [btz*2, seq_len, vocab_size]
        :param labels: [btz*2, seq_len]
        :param average_log_prob: bool, 是否对log_prob取均值, 默认为False, 取均值可以避免样本中chosen比reject长带来的导致预测偏长
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if self.offset:
            labels = labels[:, 1:].clone()  # labels取从1到n
            logits = logits[:, :-1, :]  # logits去从0到n-1
        else:
            labels = labels.clone()
        loss_mask = labels != self.pad_token_id  # 仅计算非padding部分

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.pad_token_id] = 0

        # 取真实label对应token位置的概率值logps, [btz*2, seq_len]
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # 聚合
        if average_log_prob:
            all_logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            all_logps = (per_token_logps * loss_mask).sum(-1)

        # 要求传进来的tensor前一半是chosen的，后一半是rejected的
        split = labels.shape[0] // 2
        chosen_logps = all_logps[:split]
        rejected_logps = all_logps[split:]
        return chosen_logps, rejected_logps
    

class CausalLMLoss(nn.CrossEntropyLoss):
    '''Causal语言模型的Loss

    :param offset: 是否对logit和labels做错位处理, 取决于在做数据时候是否已经offset过
    :param logits_index: 如果model.forward()返回了多个, 则logits对应的index
    '''
    def __init__(self, *args, offset=False, logits_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self.logits_index = logits_index

    def forward(self, logits:Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]], 
                labels:Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) -> torch.Tensor:
        """
        logits: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]], 形状为[btz, seq_len, vocab_size]
        labels: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            1) token_ids: [btz, seq_len]
            2) token_ids: [btz, seq_len]  + mask: [btz, seq_len]
        """
        if isinstance(logits, (List, Tuple)):
            # 如果model.forward()返回了多个参数，则决定其中某项作为logits
            logits = logits[self.logits_index]
        assert len(logits.shape) == 3, 'Args `logits` size should be [btz, seq_len, vocab_size]'
    
        raw_dtyps = logits.dtype
        logits = logits.to(torch.float32)

        mask = None
        if isinstance(labels, (List, Tuple)):
            for item in labels[1:]:
                mask = item if mask is None else mask * item
            labels = labels[0]

        if self.offset:
            logits = logits[:, :-1, :].contiguous()  # 预测序列，错开一位
            labels = labels[:, 1:].contiguous() # 目标token_ids
            if mask is not None:
                mask = mask[:, 1:].contiguous() # 目标token_ids

        logits = logits.reshape(-1, logits.shape[-1])
        if mask is not None:
            labels = labels * mask        
        labels = labels.flatten()
        loss = super().forward(logits, labels)

        return loss.to(raw_dtyps)


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss