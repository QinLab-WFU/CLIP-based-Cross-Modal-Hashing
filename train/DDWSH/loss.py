import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MarginLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.margin = opt.margin
        self.beta = nn.Parameter(torch.ones(opt.nclass) * opt.beta)

        self.miner = DistanceWeightedMiner(opt)

    def forward(self, batch, labels, y=None):
        batch = F.normalize(batch)
        y = batch if y is None else F.normalize(y)

        cdist = torch.cdist(batch, y).clamp(min=1e-8)

        anc_idxes, pos_idxes, neg_idxes = self.miner(cdist.detach(), labels)

        if len(anc_idxes) == 0:
            # print("no triplets!")
            return 0, 0

        d_ap = cdist[anc_idxes, pos_idxes]
        d_an = cdist[anc_idxes, neg_idxes]

        if self.beta_constant:
            beta = self.beta
        else:
            anchor_labels = labels[anc_idxes]
            if labels.ndim == 2:
                # TODO: should avg beta for multi-labels?
                beta = torch.einsum("nc,c->n", anchor_labels, self.beta) / anchor_labels.sum(dim=1)
                # beta = torch.einsum("nc,c->n", anchor_labels, self.beta)
            else:
                beta = self.beta[anchor_labels.to(int)]

        pos_loss = F.relu(d_ap - beta + self.margin)
        neg_loss = F.relu(beta - d_an + self.margin)

        pair_count = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0))

        loss = torch.sum(pos_loss + neg_loss) if pair_count == 0.0 else torch.sum(pos_loss + neg_loss) / pair_count

        return loss


def inverse_sphere_distances(batch, anchor_to_all_dists, labels, anchor_label):
    dists = anchor_to_all_dists
    bs, dim = batch.shape

    A = 1.0 - 0.25 * (dists.pow(2))
    A = torch.clamp(A, min=1e-8)
    log_q_d_inv = (2.0 - float(dim)) * torch.log(dists) - (float(dim - 3) / 2) * torch.log(A)

    if len(labels.shape) > 1:
        same_idxes = (labels * anchor_label).sum(axis=1) > 0
    else:
        same_idxes = labels == anchor_label
    if same_idxes.sum() == bs:
        return None

    log_q_d_inv[np.where(same_idxes)[0]] = 0
    q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))
    q_d_inv[np.where(same_idxes)[0]] = 0
    q_d_inv = q_d_inv / q_d_inv.sum()

    return q_d_inv.detach().cpu().numpy()


def pdist(A, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res.clamp(min=eps).sqrt()


class DistanceWeightedMiner:

    def __init__(self, tau=0.3, gamma=5.0):
        self.lower_cutoff = 0.5
        self.upper_cutoff = 1.4
        self.contrastive_p = 0.0
        self.tau = tau
        self.gamma = gamma

    def __call__(self, batch, labels, return_separately=True):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances = pdist(batch.detach()).clamp(min=self.lower_cutoff)

        positives, negatives = [], []
        anchors = []

        for i in range(bs):
            if len(labels.shape) > 1:
                pos = (labels * labels[i]).sum(axis=1) > 0
            else:
                pos = labels == labels[i]

            use_contr = np.random.choice(2, p=[1 - self.contrastive_p, self.contrastive_p])
            if np.sum(pos) > 1:
                if use_contr:
                    anchors.append(i)
                    positives.append(i)
                    pos[i] = 0
                    negatives.append(np.random.choice(np.where(pos)[0]))
                else:
                    q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i])

                    if q_d_inv is None:
                        continue
                    anchors.append(i)
                    pos[i] = 0
                    positives.append(np.random.choice(np.where(pos)[0]))
                    negatives.append(np.random.choice(bs, p=q_d_inv))

        if return_separately:
            return anchors, positives, negatives
        else:
            sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
            return sampled_triplets

