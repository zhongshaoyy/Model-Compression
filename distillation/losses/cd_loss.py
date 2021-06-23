import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import entropy

class CDLoss(nn.Module):
    """Channel Distillation Loss"""



    def __init__(self):
        super().__init__()

    def entropy(x, n=10):
        x = x.reshape(-1)
        scale = (x.max() - x.min()) / n
        entropy = 0
        for i in range(n):
            p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
            if p != 0:
                entropy -= p * torch.log(p)
        return float(entropy)

    def forward(self, stu_features: list, tea_features: list):
        multiplier= 2.0
        loss = 0.
        for s, t in zip(stu_features, tea_features):
            s = s.mean(dim=(1), keepdim=False)

            # a = 5
            # b = s.shape[1]
            # s = torch.tensor([entropy(s[i,j,:,:]) for i in range(a) for j in range(b)])
            # s = s.view(a, -1).float()

        
            
            t = t.mean(dim=(1), keepdim=False)
            # c = 5
            # d = t.shape[1]
            # t = torch.tensor([entropy(t[i,j,:,:]) for i in range(c) for j in range(d)])
            # t = t.view(c, -1).float()

            #loss += torch.mean(torch.pow(s - t, 2))


            s_1 = F.layer_norm(s, s.size(), None, None, 1e-7) * multiplier
            t_1 = F.layer_norm(t, t.size(), None, None, 1e-7) * multiplier
            loss += torch.mean(torch.pow(s_1 - t_1, 2))
        return loss
