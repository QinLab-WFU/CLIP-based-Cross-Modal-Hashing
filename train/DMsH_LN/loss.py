import torch
import torch.nn as nn
from torch.nn import functional as F

from .get_args import get_args
import math
import xlrd


class HyP(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.args = get_args()
        torch.manual_seed(self.args.hypseed)
        # Initialization
        self.proxies = torch.nn.Parameter(torch.randn(self.args.numclass, self.args.output_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        sheet = xlrd.open_workbook('train/DSPH/codetable.xlsx').sheet_by_index(0)
        self.threshold = sheet.row(self.args.output_dim)[math.ceil(math.log(self.args.numclass, 2))].value

    def forward(self, x=None, y=None, label=None):
        P_one_hot = label

        cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos = 1 - cos
        neg = F.relu(cos - self.threshold)

        cos_t = F.normalize(y, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos_t = 1 - cos_t
        neg_t = F.relu(cos_t - self.threshold)

        P_num = len(P_one_hot.nonzero())
        N_num = len((P_one_hot == 0).nonzero())

        pos_term = torch.where(P_one_hot  ==  1, pos.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot  ==  0, neg.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / N_num

        pos_term_t = torch.where(P_one_hot  ==  1, pos_t.to(torch.float32), torch.zeros_like(cos_t).to(torch.float32)).sum() / P_num
        neg_term_t = torch.where(P_one_hot  ==  0, neg_t.to(torch.float32), torch.zeros_like(cos_t).to(torch.float32)).sum() / N_num

        if self.args.alpha > 0:
            index = label.sum(dim = 1) > 1
            label_ = label[index].float()

            x_ = x[index]
            t_ = y[index]

            cos_sim = label_.mm(label_.T)

            if len((cos_sim == 0).nonzero()) == 0:
                reg_term = 0
                reg_term_t = 0
                reg_term_xt = 0
            else:
                x_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)
                t_sim = F.normalize(t_, p = 2, dim = 1).mm(F.normalize(t_, p = 2, dim = 1).T)
                xt_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(t_, p = 2, dim = 1).T)

                neg = self.args.alpha * F.relu(x_sim - self.threshold)
                neg_t = self.args.alpha * F.relu(t_sim - self.threshold)
                neg_xt = self.args.alpha * F.relu(xt_sim - self.threshold)

                reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_t = torch.where(cos_sim == 0, neg_t, torch.zeros_like(t_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_xt = torch.where(cos_sim == 0, neg_xt, torch.zeros_like(xt_sim)).sum() / len((cos_sim == 0).nonzero())
        else:
            reg_term = 0
            reg_term_t = 0
            reg_term_xt = 0

        return pos_term + neg_term + pos_term_t + neg_term_t + reg_term + reg_term_t + reg_term_xt
