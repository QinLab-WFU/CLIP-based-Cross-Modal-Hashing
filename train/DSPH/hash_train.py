# DSPH
# paper [Deep Semantic-aware Proxy Hashing for Multi-label Cross-modal Retrieval, TCSVT 2023]
# (https://ieeexplore.ieee.org/document/10149001)

from model.DSPH import MDSPH
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
from .loss import HyP
import time


class DSPHTrainer(TrainBase):

    def __init__(self,
                rank=1):
        args = get_args()
        super(DSPHTrainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        self.model = MDSPH(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
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

        self.hyp = HyP().to(self.rank)
        self.optimizer_loss = torch.optim.SGD(params=self.hyp.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
        self.total_time = 0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0
        for image, text, label, index in self.train_loader:
            start_time = time.time()
            self.global_step += 1
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)

            hash_img, hash_text = self.model(image, text)

            loss = self.hyp(hash_img, hash_text, label)

            all_loss += loss

            self.optimizer.zero_grad()
            self.optimizer_loss.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_loss.step()
            self.total_time += time.time() - start_time

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")


