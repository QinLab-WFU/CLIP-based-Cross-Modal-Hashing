import torch
import numpy as np


class BPLoss(torch.nn.Module):
    def __init__(self, bit):
        super(DAMHLoss, self).__init__()
        self.y_p = 0.5  #0.5 BasePoint
        self.right = bit / 6  # bit / 2
        self.left = self.right / 2
        self.lowerBound = 0  #0
        self.upperBound = bit / 4  #bit / 4
        self.percent = 9/10   #9 / 10

    def forward(self, u, v, y):
        s = y @ y.t() > 0
        inner = u @ v.t()

        posL = 0
        navL = 0
        count = 0
        for row in range(u.shape[0]):
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                #getting dissimilar-pair and similar-pair
                similar = inner[row][s[row] == 1]
                dissimilar = inner[row][s[row] == 0]
                #sorting
                similar_temp, idx = torch.sort(similar, descending=True)
                dissimilar_temp, idx2 = torch.sort(dissimilar)

                #DAMH_similar
                meanS = torch.mean(similar).clamp(min=self.lowerBound, max=self.upperBound).item()
                # percent can reduce interference of dissimilarMaxInner
                dissimilarMaxInner = dissimilar_temp[int(len(dissimilar_temp) * self.percent):].mean().item()#.clamp(min=self.lowerBound,max=self.upperBound).item()
                #getting xp
                BP = meanS - (self.upperBound - meanS) / self.upperBound * np.abs((meanS - dissimilarMaxInner))
                # BP = dissimilarMaxInner

                # DAMH_dissimilar
                meanDS = torch.mean(dissimilar).clamp(min=self.lowerBound, max=self.upperBound).item()
                similarMinInner = similar_temp[int(len(similar_temp) * self.percent):].mean().item()#.clamp(min=self.lowerBound, max=self.upperBound).item()
                BP_ds = meanDS - meanDS / self.upperBound * np.abs((meanDS - similarMinInner))
                # BP_ds = similarMinInner

                #getting hard or easy samples
                similar_easy = similar[similar > BP]
                similar_hard = similar[similar < BP]
                # calc
                a, c, d, g = self.calcParameter(BP, self.y_p, self.left, self.right)
                f_similar_easy = c * similar_easy + d
                f_similar_hard = a * c * similar_hard + g
                similar_easy_loss = self.DPSHLoss(True, f_similar_easy)
                similar_hard_loss = self.DPSHLoss(True, f_similar_hard)

                # DAMH_dissimilar
                dissimilar_easy = dissimilar[dissimilar < BP_ds]
                dissimilar_hard = dissimilar[dissimilar > BP_ds]
                a, c, d, g = self.calcParameter(BP_ds, self.y_p, self.left, self.right)
                f_dissimilar_easy = c * dissimilar_easy + d
                f_dissimilar_hard = a * c * dissimilar_hard + g
                dissimilar_easy_loss = self.DPSHLoss(False, f_dissimilar_easy)
                dissimilar_hard_loss = self.DPSHLoss(False, f_dissimilar_hard)

                #mean Loss
                similar_loss = torch.cat((similar_easy_loss, similar_hard_loss), dim=0)
                dissimilar_loss = torch.cat((dissimilar_easy_loss, dissimilar_hard_loss), dim=0)
                posL += similar_loss.mean()
                navL += dissimilar_loss.mean()

        if count != 0:
            posL = posL / count
            navL = navL / count
        else:
            posL = 0
            navL = 0

        return posL + navL

    def DPSHLoss(self, s, fx):
        # s * fx + torch.log((1 + torch.exp(-fx)))
        if s == 1:
            losses = fx + torch.log((1 + torch.exp(-fx)))
        else:
            losses = torch.log((1 + torch.exp(-fx)))
        return losses

    #See the paper for details
    def calcParameter(self, BP, y_p, left, right):
        # y1 = 1/(1+e^(cx+d))
        # 1)c=1/right *log((y_p)/99*(1-y_p))
        c = 1 / right * np.log(y_p / (99 * (1 - y_p)))
        # 2)d=log((r-y_p)/y_p)- c*BP
        d = np.log((1 - y_p) / y_p) - c * BP
        # y2 = r/(1+e^(ax+g))
        # 1) a = -1/(left*c) *log( (99*y_p)/(1-y_p) )
        a = -1 / (left * c) * np.log((99 * y_p) / (1 - y_p))
        # 2)g = log((1-y_p)/y_p)-a*c*BP
        g = np.log((1 - y_p) / y_p) - a * c * BP

        return a, c, d, g