import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
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
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        """[summary]

        Args:
            y_true ([Tensor]): [btz, ner_vocab_size, seq_len, seq_len]
            y_pred ([Tensor]): [btz, ner_vocab_size, seq_len, seq_len]
        """
        y_true = y_true.view(y_true.shape[0]*y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0]*y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]

        y_pred = (1-2*y_true) * y_pred
        y_pred_pos = y_pred - (1-y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros(y_pred_pos.shape[0], 1, device=y_pred_pos.device)], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros(y_pred_neg.shape[0], 1, device=y_pred_neg.device)], dim=-1)
        loss = torch.sum(torch.logsumexp(y_pred_pos, 1) + torch.logsumexp(y_pred_neg, 1)) / y_pred_neg.shape[0]
        return loss


class ContrastiveLoss(nn.Module):
    """对比损失：减小正例之间的距离，增大正例和反例之间的距离，labels * distance_matrix.pow(2) + (1-labels)*F.relu(margin-distance_matrix).pow(2)
    https://www.sbert.net/docs/package_reference/losses.html
    """
    def __init__(self, margin=0.5, size_average=True, online=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.online = online

    def forward(self, distances, labels, pos_id=1, neg_id=0):
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
    '''
    def __init__(self, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, y_pred, y_true):
        '''支持两种方式，一种是奇偶相邻排列，一种是上下排列
        '''
        loss_sup = self.loss_sup(y_pred, y_true)

        if self.rank == 'adjacent':
            y_pred1 = y_pred[1::2]
            y_pred2 = y_pred[::2]
        elif self.rank == 'updown':
            half_btz = y_true.shape[0] // 2
            y_pred1 = y_pred[:half_btz]
            y_pred2 = y_pred[half_btz:]
        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))
        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha
