import torch, torch.nn as nn, torch.nn.functional as F


class CPF(torch.nn.Module):
    def __init__(self, embed_dim, n_classes, device):
        super(CPF, self).__init__()
        self.device = device

        self.in_features = embed_dim
        self.out_features = n_classes

        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features).to(device))
        nn.init.xavier_uniform_(self.weight)

        self.ls_eps = 0

        self.tau = 0.9
        self.psi = 0.7
        self.sp = 1.3
        self.sn = 1.3
        self.mu = 1.0
        self.b = 2

    def forward(self, image, text, labels):
        one_hot = labels.to(self.device)

        cosine = F.linear(F.normalize(image), F.normalize(self.weight))
        t_cosine = F.linear(F.normalize(text), F.normalize(self.weight))

        tp = ((cosine.clamp(min=0.0) * one_hot) * 2).sum() + self.b
        t_tp = ((t_cosine.clamp(min=0.0) * one_hot) * 2).sum() + self.b

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        lossp = ((1.0 - cosine) * torch.exp((1.0 - cosine) * self.sp).detach() * one_hot).sum()
        t_lossp = ((1.0 - t_cosine) * torch.exp((1.0 - t_cosine) * self.sp).detach() * one_hot).sum()

        mask = cosine > self.tau
        cosine = cosine[mask]
        lossn = ((cosine - self.psi)####qingchufudui
                 * torch.exp((cosine - self.mu) * self.sn).detach()
                 * (1 - one_hot[mask])).sum()

        t_mask = t_cosine > self.tau
        t_cosine = t_cosine[t_mask]
        t_lossn = ((t_cosine - self.psi)
                   * torch.exp((t_cosine - self.mu) * self.sn).detach()
                   * (1 - one_hot[t_mask])).sum()

        loss = (1.0 - (tp) / (tp + lossp + lossn))
        t_loss = (1.0 - (t_tp) / (t_tp + t_lossp + t_lossn))

        return loss + t_loss