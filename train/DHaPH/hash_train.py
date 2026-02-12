# DHaPH
# paper [Deep Hierarchy-aware Proxy Hashing with Self-paced Learning for Cross-modal Retrieval, TKDE 2024]
# (https://ieeexplore.ieee.org/document/10530441)

from model.DHaPH import MDHaPH
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
import time

from .hp_model import HPmodel
from .HPloss import HPLoss
from .MSLoss import MSLoss


class DHaPHTrainer(TrainBase):

    def __init__(self, args, rank):
        args = get_args(args)
        args.rank = rank
        super(DHaPHTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = MDHaPH(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.model.float()
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_hash.parameters(), 'lr': self.args.lr}
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.hpmodel = HPmodel(self.args.output_dim, self.args.output_dim).to(self.rank)
        self.optimizer_hpmodel = torch.optim.AdamW(params=self.hpmodel.parameters(), lr=1e-5)
        self.hp = HPLoss(nb_proxies=self.args.HM, sz_embed=self.args.output_dim, mrg=self.args.margin).to(self.rank)
        self.optimizer_hploss = torch.optim.AdamW(params=self.hp.parameters(), lr=1e-5)

        self.msloss = MSLoss(temperature=self.args.tau, totalepoch=self.args.epochs, self_paced=True)

        self.total_time = 0.0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0
        for image, text, label, index in self.train_loader:
            start_time = time.time()
            image.float()

            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)
            label = label.float()

            hash_img, hash_text = self.model(image, text)

            loss1 = self.msloss(hash_img, hash_img, label, epoch + 1)
            loss2 = self.msloss(hash_text, hash_text, label, epoch + 1)
            loss3 = self.msloss(hash_img, hash_text, label, epoch + 1)

            hp_img = self.hpmodel(hash_img.detach())
            hp_text = self.hpmodel(hash_text.detach())
            loss4 = self.hp(hp_img, hp_text, label, self.args.topk)

            loss = loss1 + loss2 + loss3 + self.args.alpha * loss4

            all_loss += loss

            self.optimizer.zero_grad()
            self.optimizer_hpmodel.zero_grad()
            self.optimizer_hploss.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_hpmodel.step()
            self.optimizer_hploss.step()
            self.total_time += time.time() - start_time

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, time: {self.total_time}")