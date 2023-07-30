import torch
from torch import nn
class CrossEntropy(nn.Module):
    def forward(self, input, target):
        eps = 1e-5  # 避免log(0)出现
        scores = torch.sigmoid(input)
        scores = torch.clamp(scores, eps, 1 - eps)  # 将scores限制在[eps, 1-eps]范围内
        target_active = (target == 1).float()
        loss_terms = -(target_active * torch.log(scores) + (1 - target_active) * torch.log(1 - scores))
        if torch.isnan(loss_terms).any():
            loss_terms[torch.isnan(loss_terms)] = 0.0
        b = loss_terms.sum()/len(loss_terms)
        if torch.isnan(b):
            b = 0.0
        return b




LOSS_FUNCTIONS={
    'CrossEntropy': CrossEntropy()
}