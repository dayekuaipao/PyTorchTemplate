import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class FocalLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction)
        return self.alpha * ((1 - torch.exp(-ce)) ** self.gamma) * ce
