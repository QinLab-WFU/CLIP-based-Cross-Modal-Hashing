import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean

from .gnn import GNNDecoder
from .graph_generator import GraphGenerator
from utils.utils import gen_triplets

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

class GeneralPulling(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, embeddings, ref_embeddings, triplets, edge_reprs, J_avg):
        batch_size, n_bits = embeddings.shape

        dist_mat = torch.cdist(embeddings, ref_embeddings)
        dist_mat = torch.clamp(dist_mat, min=1e-6)

        anc_idxes, pos_idxes, neg_idxes = triplets

        D_ap = dist_mat[anc_idxes, pos_idxes]
        D_an = dist_mat[anc_idxes, neg_idxes]

        # n_edges x n_bits -> B x B x n_bits
        edge_reprs = edge_reprs.reshape(batch_size, batch_size, n_bits)
        # λij: n_triplets x n_bits
        lambda_ij = edge_reprs[anc_idxes, neg_idxes, :]

        # no adaptive hardness
        x = 1e6 if J_avg == 0 else J_avg.item()
        lambda_eta = lambda_ij * np.exp(-self.alpha / x)  # λij * η, n=exp(-α/J_avg)
        # lambda_eta = lambda_ij


        # r = (D_ap.unsqueeze(1) + (D_an - D_ap).unsqueeze(1) * lambda_eta) / D_an.unsqueeze(1)  # r = []/d- of Eq. 6

        # -------------------------------------------------------------------------------------------
        r = (1 - lambda_eta) * (D_ap / D_an).unsqueeze(1) + lambda_eta
        # epsilon = 1e-8
        # r = (1 - lambda_eta) * (D_ap / (D_an + epsilon)).unsqueeze(1) + lambda_eta
        # -------------------------------------------------------------------------------------------

        z_i, z_j = embeddings[anc_idxes], ref_embeddings[neg_idxes]
        # z_tile = z_i + (z_j - z_i) * r
        z_tile = (1 - r) * z_i + r * z_j

        # if d+ >= d-
        neg_mask = torch.ge(D_ap, D_an).unsqueeze(1)

        # complete Eq. 6: shape is n_triplets x n_bits
        z_hat = torch.mul(z_j, neg_mask) + torch.mul(z_tile, ~neg_mask)


        return z_hat


class PaperLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.gg = GraphGenerator()
        self.gnn = GNNDecoder(embed_dim=args.output_dim, out_dim=args.output_dim, reduce=1,n_layers=args.n_layers,
                                n_heads=args.n_heads, n_classes=args.nclass, )
        if not args.noCE:
            self.softmax_classifier = nn.Linear(args.output_dim, args.nclass)
            self.ce = nn.CrossEntropyLoss()
        self.pulling = GeneralPulling(args.alpha)


    def triplet_loss(self, embeddings, ref_embeddings, triplets, margin=0.25, neg_embs=None, hardness="all"):

        anc_idxes, pos_idxes, neg_idxes = triplets
        sim_mat = embeddings @ ref_embeddings.T

        S_ap = sim_mat[anc_idxes, pos_idxes]

        if neg_embs is None:
            # normal TripletLoss
            S_an = sim_mat[anc_idxes, neg_idxes]
        else:
            # use synthetic hard negatives
            anc_embs = embeddings[anc_idxes]
            S_an = torch.cosine_similarity(anc_embs, neg_embs)

        losses = F.relu(S_an - S_ap + margin)

        mask = losses > 0
        if hardness == "semi":
            mask &= S_ap >= S_an
        elif hardness == "hard":
            mask &= S_ap < S_an
        loss = 0 if mask.sum() == 0 else losses[mask].mean()

        return loss

    def fwd_stage2(self, img_embs, txt_embs, labels, J_avg, save=0):
        ii_tt_triplets = gen_triplets(labels)
        it_ti_triplets = gen_triplets(labels, labels)  # no fill_dialog

        # save for fwd_stage1
        self.ii_tt_triplets = ii_tt_triplets
        self.it_ti_triplets = it_ti_triplets

        # N-pairLoss or PALoss: Eq. 15~16
        J_r = self.triplet_loss(img_embs, img_embs, ii_tt_triplets)  # j_m in code
        J_r += self.triplet_loss(img_embs, txt_embs, it_ti_triplets)  # j_m in code
        J_r += self.triplet_loss(txt_embs, img_embs, it_ti_triplets)  # j_m in code
        J_r += self.triplet_loss(txt_embs, txt_embs, ii_tt_triplets)  # j_m in code
        J_r /= 4

        if torch.isnan(img_embs).any():
            print("img_embs")
            print(img_embs)
            print("node_attrs nan-")
            exit()
        i_i_edge_attrs, i_i_edge_idxes, i_i_node_attrs = self.gg.get_graph(img_embs)
        # i_i_edge_reprs = i_i_edge_attrs
        i_i_node, i_i_node_preds, i_i_edge_reprs = self.gnn(i_i_node_attrs, i_i_edge_idxes, i_i_edge_attrs, labels)

        # use txts to update imgs
        i_t_edge_attrs, i_t_edge_idxes, i_t_node_attrs = self.gg.get_graph(img_embs, txt_embs)
        # i_t_edge_reprs = i_t_edge_attrs
        i_t_node, i_t_node_preds, i_t_edge_reprs = self.gnn(i_t_node_attrs, i_t_edge_idxes, i_t_edge_attrs, labels)

        # use imgs to update txts
        t_i_edge_attrs, t_i_edge_idxes, t_i_node_attrs = self.gg.get_graph(txt_embs, img_embs)
        # t_i_edge_reprs = t_i_edge_attrs
        t_i_node, t_i_node_preds, t_i_edge_reprs = self.gnn(t_i_node_attrs, t_i_edge_idxes, t_i_edge_attrs, labels)

        t_t_edge_attrs, t_t_edge_idxes, t_t_node_attrs = self.gg.get_graph(txt_embs)
        # t_t_edge_reprs = t_t_edge_attrs
        t_t_node, t_t_node_preds, t_t_edge_reprs = self.gnn(t_t_node_attrs, t_t_edge_idxes, t_t_edge_attrs, labels)


        # Eq. 13
        if not self.args.noCE:
            J_gca = self.ce(i_i_node_preds, labels)  # J_gce in code
            J_gca += self.ce(i_t_node_preds, labels)  # J_gce in code
            J_gca += self.ce(t_i_node_preds, labels)  # J_gce in code
            J_gca += self.ce(t_t_node_preds, labels)  # J_gce in code
            J_gca /= 4
        else:
            J_gca = 0

        # ------------------------------------------------------------------------------------------------

        i_i_syn_embs = self.pulling(img_embs, img_embs, ii_tt_triplets, i_i_edge_reprs.detach(), J_avg)
        i_t_syn_embs = self.pulling(img_embs, txt_embs, it_ti_triplets, i_t_edge_reprs.detach(), J_avg)
        t_i_syn_embs = self.pulling(txt_embs, img_embs, it_ti_triplets, t_i_edge_reprs.detach(), J_avg)
        t_t_syn_embs = self.pulling(txt_embs, txt_embs, ii_tt_triplets, t_t_edge_reprs.detach(), J_avg)

        if save:
            collect_and_save_triplet_vis(self.args.save_dir, img_embs, img_embs, ii_tt_triplets, i_i_syn_embs, save)

        J_syn = self.triplet_loss(img_embs, img_embs, ii_tt_triplets, neg_embs=i_i_syn_embs, hardness="hard")
        J_syn += self.triplet_loss(img_embs, txt_embs, it_ti_triplets, neg_embs=i_t_syn_embs, hardness="hard")
        J_syn += self.triplet_loss(txt_embs, img_embs, it_ti_triplets, neg_embs=t_i_syn_embs, hardness="hard")
        J_syn += self.triplet_loss(txt_embs, txt_embs, ii_tt_triplets, neg_embs=t_t_syn_embs, hardness="hard")
        J_syn /= 4
        # J_syn = 0

        # J_m = J_r + J_gca + J_syn
        return J_r, J_gca, J_syn

    def fwd_classifier(self, embeddings, labels):

        logits = self.softmax_classifier(embeddings.detach())
        loss = self.ce(logits, labels)
        return loss

    def fwd_classifier_(self, embed, embeddings, labels):

        logits_i = self.softmax_classifier_i(embed.detach())
        loss = self.ce(logits_i, labels)
        logits_t = self.softmax_classifier_t(embeddings.detach())
        loss += self.ce(logits_t, labels)
        return loss

    def fwd_stage1(self, img_embs, txt_embs, labels, J_avg):
        # only update GNN

        ii_tt_triplets = self.ii_tt_triplets
        it_ti_triplets = self.it_ti_triplets


        # construct graph again
        i_i_edge_attrs, i_i_edge_idxes, i_i_node_attrs = self.gg.get_graph(img_embs.detach(), img_embs.detach())
        if torch.isnan(i_i_node_attrs).any():
            print("i_i_node_attrs1")
            print("node_attrs nan-")
            exit()
        i_i_node, i_i_node_preds, i_i_edge_reprs = self.gnn(i_i_node_attrs, i_i_edge_idxes, i_i_edge_attrs, labels)

        i_t_edge_attrs, i_t_edge_idxes, i_t_node_attrs = self.gg.get_graph(img_embs.detach(), txt_embs.detach())
        if torch.isnan(i_t_node_attrs).any():
            print("i_t_node_attrs")
            print("node_attrs nan-")
            exit()
        _, i_t_node_preds, i_t_edge_reprs = self.gnn(i_t_node_attrs, i_t_edge_idxes, i_t_edge_attrs, labels)

        t_i_edge_attrs, t_i_edge_idxes, t_i_node_attrs = self.gg.get_graph(txt_embs.detach(), img_embs.detach())
        if torch.isnan(t_i_node_attrs).any():
            print("t_t_node_attrs")
            print("node_attrs nan-")
            exit()
        _, t_i_node_preds, t_i_edge_reprs = self.gnn(t_i_node_attrs, t_i_edge_idxes, t_i_edge_attrs, labels)

        t_t_edge_attrs, t_t_edge_idxes, t_t_node_attrs = self.gg.get_graph(txt_embs.detach(), txt_embs.detach())
        if torch.isnan(t_t_node_attrs).any():
            print("t_t_node_attrs")
            print("node_attrs nan-")
            exit()
        t_t_node, t_t_node_preds, t_t_edge_reprs = self.gnn(t_t_node_attrs, t_t_edge_idxes, t_t_edge_attrs, labels)


        i_i_syn_embs = self.pulling(img_embs.detach(), img_embs.detach(), ii_tt_triplets, i_i_edge_reprs, J_avg)
        i_t_syn_embs = self.pulling(img_embs.detach(), txt_embs.detach(), it_ti_triplets, i_t_edge_reprs, J_avg)
        t_i_syn_embs = self.pulling(txt_embs.detach(), img_embs.detach(), it_ti_triplets, t_i_edge_reprs, J_avg)
        t_t_syn_embs = self.pulling(txt_embs.detach(), txt_embs.detach(), ii_tt_triplets, t_t_edge_reprs, J_avg)

        if self.args.div:
            batch_size, n_bits = img_embs.shape

            r, c = i_i_edge_idxes[:, 0], i_i_edge_idxes[:, 1]
            means = torch.tile(scatter_mean(i_i_edge_reprs, r, dim=0), (1, batch_size)).reshape(-1, n_bits)
            J_div = 1 - ((i_i_edge_reprs - means) ** 2).sum(1).mean().sqrt()

            r, c = i_t_edge_idxes[:, 0], i_t_edge_idxes[:, 1]
            means = torch.tile(scatter_mean(i_t_edge_reprs, r, dim=0), (1, batch_size)).reshape(-1, n_bits)
            J_div += 1 - ((i_t_edge_reprs - means) ** 2).sum(1).mean().sqrt()

            r, c = t_i_edge_idxes[:, 0], t_i_edge_idxes[:, 1]
            means = torch.tile(scatter_mean(t_i_edge_reprs, r, dim=0), (1, batch_size)).reshape(-1, n_bits)
            J_div += 1 - ((t_i_edge_reprs - means) ** 2).sum(1).mean().sqrt()

            r, c = t_t_edge_idxes[:, 0], t_t_edge_idxes[:, 1]
            means = torch.tile(scatter_mean(t_t_edge_reprs, r, dim=0), (1, batch_size)).reshape(-1, n_bits)
            J_div += 1 - ((t_t_edge_reprs - means) ** 2).sum(1).mean().sqrt()

            J_div /= 4
            if torch.isnan(J_div).any():
                print("J_div")
                print(J_div)
        else:
            J_div = 0

        # Eq. 8
        if not self.args.noCE:
            J_ce = 0
            if i_i_syn_embs.shape[0] != 0:
                logits = self.softmax_classifier(i_i_syn_embs)
                J_ce += self.ce(logits, labels[ii_tt_triplets[-1]])

            if i_t_syn_embs.shape[0] != 0:
                logits = self.softmax_classifier(i_t_syn_embs)
                J_ce += self.ce(logits, labels[it_ti_triplets[-1]])

            if t_i_syn_embs.shape[0] != 0:
                logits = self.softmax_classifier(t_i_syn_embs)
                J_ce += self.ce(logits, labels[it_ti_triplets[-1]])

            if t_t_syn_embs.shape[0] != 0:
                logits = self.softmax_classifier(t_t_syn_embs)
                J_ce += self.ce(logits, labels[ii_tt_triplets[-1]])

            J_ce /= 4
        else:
            J_ce = 0

        # Eq. 9
        J_sim = 0
        if not self.args.noSim:
            if i_i_syn_embs.shape[0] != 0:
                J_sim = (1 - torch.cosine_similarity(img_embs[ii_tt_triplets[0]].detach(), i_i_syn_embs)).mean()
            if i_t_syn_embs.shape[0] != 0:
                J_sim += (1 - torch.cosine_similarity(img_embs[it_ti_triplets[0]].detach(), i_t_syn_embs)).mean()
            if t_i_syn_embs.shape[0] != 0:
                J_sim += (1 - torch.cosine_similarity(txt_embs[it_ti_triplets[0]].detach(), t_i_syn_embs)).mean()
            if t_t_syn_embs.shape[0] != 0:
                J_sim += (1 - torch.cosine_similarity(txt_embs[ii_tt_triplets[0]].detach(), t_t_syn_embs)).mean()
            J_sim /= 4

        return J_ce, J_sim, J_div

def collect_and_save_triplet_vis(save_dir, anc_embs, ref_embs, triplets, syn_embs, state=0):
    anc_idx, pos_idx, neg_idx = triplets
    a = anc_embs[anc_idx].detach().cpu()
    p = ref_embs[pos_idx].detach().cpu()
    n_orig = ref_embs[neg_idx].detach().cpu()

    # 注意：syn_embs 已经是 (n_triplets, bits)，直接取就对齐
    n_gen = syn_embs.detach().cpu()

    cos = torch.nn.functional.cosine_similarity
    cos_orig = cos(a, n_orig).numpy()
    cos_gen = cos(a, n_gen).numpy()

    pos_dist = torch.norm(a - p, dim=1).numpy()
    neg_orig_dist = torch.norm(a - n_orig, dim=1).numpy()
    neg_gen_dist = torch.norm(a - n_gen, dim=1).numpy()

    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f"vis-{state}.npz"),
             cos_orig=cos_orig,
             cos_gen=cos_gen,
             pos_dist=pos_dist,
             neg_orig_dist=neg_orig_dist,
             neg_gen_dist=neg_gen_dist)

if __name__ == "__main__":
    from _utils import gen_test_data

    B, C, K = 12, 10, 8
    e, t, l = gen_test_data(B, C, K)

    # pull_func = GeneralPulling()
    # z = pull_func(e, l, torch.randn(B**2, K), 1)
    # print(z.shape)

    criterion = PaperLoss(Namespace(n_bits=K, n_classes=C, n_heads=4, n_layers=2))
    criterion.fwd_stage1(e, l, 0)
