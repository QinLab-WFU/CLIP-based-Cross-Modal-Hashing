import os

import numpy as np
import torch
import time

from timm.utils import AverageMeter
from torch import optim
from tqdm import tqdm

from torch.nn import functional as F

from model.modelbase import BaseBackbone
from train.base import TrainBase
from model.clip.optimization import BertAdam

from .get_args import get_args
from .loss import PaperLoss


class DGHDGHTrainer(TrainBase):

    def __init__(self, args, rank):
        args = get_args(args)
        args.rank = rank
        super(DGHDGHTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        if self.args.is_optuna:
            self.optuna_trial = self.args.optuna_trial
            self.best_map = 0.0
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = BaseBackbone(outputDim=self.args.output_dim, backbone=self.args.backbone,
                                  preload=self.args.preload, writer=self.writer,
                                  logger=self.logger, is_train=True).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"), strict=False)
        self.model.float()

        self.criterion = PaperLoss(self.args).to(self.rank)

        to_optim = [
            {'params': self.model.backbone.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
        ]

        self.optimizer = BertAdam(to_optim, lr=self.args.lr, warmup=self.args.warmup_proportion,
                                  schedule='warmup_cosine',b1=0.9, b2=0.98, e=1e-6,
                                  t_total=len(self.train_loader) * self.args.epochs,
                                  weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.optimizer_g = optim.Adam(self.criterion.gnn.parameters(), lr=self.args.clip_lr, weight_decay=self.args.weight_decay)
        if not self.args.noCE:
            self.optimizer_c =optim.Adam(self.criterion.softmax_classifier.parameters(), lr=self.args.clip_lr)

        self.total_time = 0.0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        stat_meters = {}
        losses = ["J_r", "J_m", "J_gen"] # J_cz
        if not self.args.noCE:
            losses.append("J_cz")
        for x in losses:
            stat_meters[x] = AverageMeter()
        for image, text, label, index in tqdm(self.train_loader):
            start_time = time.time()
            if self.args.backbone == 'clip':
                image = image.to(self.rank, non_blocking=True).float()
                text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()

            if hasattr(self.args, "noise_rate") and self.args.noise_rate > 0:
                label = self.add_noise_to_labels(label)

            embed_i, embed_t = self.model(image, text)
            embed_i, embed_t = F.normalize(embed_i), F.normalize(embed_t)

            # stage-2: Eq. 17
            state = epoch if epoch % 10 == 0 else 0

            J_r, J_gca, J_syn = self.criterion.fwd_stage2(embed_i, embed_t, label, stat_meters["J_r"].avg, state)
            stat_meters["J_r"].update(J_r)

            # gamma_n
            x = 1e6 if stat_meters["J_gen"].avg == 0 else stat_meters["J_gen"].avg.item()
            self.args.lambda3 = 1 - np.exp(-self.args.beta / x)
            self.args.lambda3 = min(max(self.args.lambda3, 0.1), 0.9)
            if self.args.lambda3 not in [0.1, 0.9]:
                print(self.args.lambda3)
                print("take")
                self.args.lambda3 = 0.1
            J_m = self.args.lambda1 * J_r + self.args.lambda2 * J_gca + self.args.lambda3 * J_syn
            stat_meters["J_m"].update(J_m)
            self.optimizer.zero_grad()
            self.optimizer_g.zero_grad()
            if J_m != 0:
                J_m.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.criterion.gnn.parameters(), 1.0)
            self.optimizer_g.step()

            if not self.args.noCE:
                J_cz = self.criterion.fwd_classifier(embed_i, label)
                J_cz += self.criterion.fwd_classifier(embed_t, label)
                J_cz /= 2
                stat_meters["J_cz"].update(J_cz)
                # print("J_cz:", J_cz * self.args.lambda4)
                self.optimizer_c.zero_grad()
                (J_cz * self.args.lambda4).backward()
                self.optimizer_c.step()

            J_ce, J_sim, J_div = self.criterion.fwd_stage1(embed_i, embed_t, label, stat_meters["J_r"].avg)
            J_gen = self.args.lambda5 * J_ce + self.args.lambda6 * J_sim + self.args.lambda7 * J_div
            stat_meters["J_gen"].update(J_gen)
            self.optimizer_g.zero_grad()
            if J_gen != 0:
                J_gen.backward()
            torch.nn.utils.clip_grad_norm_(self.criterion.gnn.parameters(), 1.0)
            self.optimizer_g.step()

            self.total_time += time.time() - start_time
        sm_str = ""
        for x in stat_meters.keys():
            sm_str += f"[{x}:{stat_meters[x].avg:.4f}]"
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {sm_str}, time: {self.total_time}")

    def add_noise_to_labels(self, labels):
        labels = labels.cpu().numpy()
        num_samples, num_labels = labels.shape
        num_noise = int(num_samples * self.args.noise_rate)

        noise_indices = np.random.choice(num_samples, num_noise, replace=False)

        for i in noise_indices:
            ones_indices = np.where(labels[i, :] == 1)[0]
            zeros_indices = np.where(labels[i, :] == 0)[0]

            if len(ones_indices) > 0:
                j = np.random.choice(ones_indices)
                labels[i, j] = 0

            if len(zeros_indices) > 0:
                j = np.random.choice(zeros_indices)
                labels[i, j] = 1

        return torch.tensor(labels, dtype=torch.float32).to(self.rank)  # 转换回 Tensor 并放回 GPU