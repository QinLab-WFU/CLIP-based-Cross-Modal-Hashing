import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_sim(x, y):
    return x.mm(y.t())


def order_sim(x, y):
    YmX = (y.unsqueeze(1).expand(y.size(0), x.size(0), y.size(1)) - \
           x.unsqueeze(0).expand(y.size(0), x.size(0), y.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def l2norm(x):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)


def rbf(x, y, gamma):
    pdist = torch.norm(x[:, None] - y, dim=2, p=2)
    return torch.exp(-gamma * pdist)


class DPSIHLoss(nn.Module):

    def __init__(self, opt, reduction='mean'):
        super(DPSIHLoss, self).__init__()

        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.num_embeds = opt.num_embeds if hasattr(opt, 'num_embeds') else 1
        self.alpha1 = opt.alpha1 if hasattr(opt, 'alpha1') else 0.
        self.alpha2 = opt.alpha2 if hasattr(opt, 'alpha2') else 0.
        self.sim_fn = order_sim if hasattr(opt, 'order') and opt.order else cosine_sim
        self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
        self.reduction = reduction
        self.MSC_loss = Multi_Semantic_Correlation_Loss(self.margin, "all", False)

        if self.num_embeds > 1:
            self.max_pool = torch.nn.MaxPool2d(self.num_embeds)

    def embedding_diversity_loss(self, x):
        x = l2norm(x)
        gram_x = x.bmm(x.transpose(1, 2))
        I = torch.autograd.Variable((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1)).to(x.device)
        gram_x.masked_fill_(I, 0.0)
        loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (self.num_embeds ** 2)
        return loss.mean() if self.reduction == 'mean' else loss.sum()

    def distribution_consistency_loss(self, x, y, gamma=None):
        if gamma is None:
            gamma = 1. / x.size(-1)
        loss = rbf(x, x, gamma) - 2 * rbf(x, y, gamma) + rbf(y, y, gamma)
        return loss.mean() if self.reduction == 'mean' else loss.sum()

    def forward(self, img, txt, img_r, txt_r, label):
        loss, losses = 0, dict()

        msc_loss = self.MSC_loss(img, label)[0] + self.MSC_loss(txt, label)[0] + self.MSC_loss(img, label, txt)[0]
        loss += msc_loss * 100
        losses['msc_loss'] = msc_loss

        if self.num_embeds > 1 and self.alpha1 > 0:
            dc_loss = self.distribution_consistency_loss(img.view(-1, img.size(-1)), txt.view(-1, txt.size(-1)), gamma=0.5)
            loss += self.alpha1 * dc_loss
            losses['dc_loss'] = dc_loss

        if self.num_embeds > 1 and self.alpha2 > 0:
            ed_loss = self.embedding_diversity_loss(img_r) + self.embedding_diversity_loss(txt_r)
            loss += self.alpha2 * ed_loss
            losses['ed_loss'] = ed_loss

        return loss, losses


class Multi_Semantic_Correlation_Loss(nn.Module):
    def __init__(self, margin, hardness, normalize_embeddings):
        super().__init__()
        self.margin = margin
        self.hardness = hardness
        self.normalize_embeddings = normalize_embeddings

    def forward(self, batch_inputs, batch_labels, inputs=None, labels=None):
        if self.normalize_embeddings:
            batch_inputs = F.normalize(batch_inputs, p=2, dim=1)

        if inputs is None:
            inputs = batch_inputs
        else:
            if self.normalize_embeddings:
                inputs = F.normalize(inputs, p=2, dim=1)

        if labels is None:
            labels = batch_labels

        if batch_inputs.dim() == 2:
            sim_mat = -torch.matmul(batch_inputs, inputs.t())
        else:
            sim_mat = batch_inputs.view(-1, batch_inputs.size(-1)) @ inputs.view(-1, inputs.size(-1)).T
            sim_mat = nn.MaxPool2d(batch_inputs.size(1))(sim_mat.unsqueeze(0)).squeeze()
            sim_mat = -sim_mat

        sames = batch_labels @ labels.T > 0
        diffs = ~sames
        if sames.size(0) == sames.size(1):
            sames.fill_diagonal_(False)

        anchor_idx, positive_idx, negative_idx = torch.where(sames.unsqueeze(2) * diffs.unsqueeze(1))
        ap_dist = sim_mat[anchor_idx, positive_idx]
        an_dist = sim_mat[anchor_idx, negative_idx]
        triplet_margin = an_dist - ap_dist

        idx = int(batch_inputs.shape[0] != inputs.shape[0])
        hardness = self.hardness[idx]

        threshold_condition = triplet_margin <= self.margin  # mining "all"
        if hardness == "semi-hard":
            threshold_condition &= triplet_margin > 0

        anchor_idx, positive_idx, negative_idx = (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

        if len(anchor_idx) == 0:
            return 0, 0

        ap_dists = sim_mat[anchor_idx, positive_idx]
        an_dists = sim_mat[anchor_idx, negative_idx]

        violation = ap_dists - an_dists + self.margin
        loss = F.relu(violation).mean()

        return loss, len(anchor_idx)
