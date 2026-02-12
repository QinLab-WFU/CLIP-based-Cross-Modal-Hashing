# DSPH
# paper [Deep Semantic-aware Proxy Hashing for Multi-label Cross-modal Retrieval, TCSVT 2023]
# (https://ieeexplore.ieee.org/document/10149001)
from tqdm import tqdm

import os
import torch

from model.modelbase import Baseclip
from train.base import TrainBase
from model.base.optimization import BertAdam
from .loss import MarginLoss
from .get_args import get_args
import time


class DDWSHTrainer(TrainBase):

    def __init__(self, args, rank=0):
        args = get_args(args)
        args.rank = rank
        super(DDWSHTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        self.model = Baseclip(outputDim=self.args.output_dim,
                                  clipPath=self.args.clip_path, writer=self.writer, logger=self.logger,
                                  is_train=self.args.is_train).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()

        self.criterion = MarginLoss(self.args).to(self.rank)

        to_opt = [
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
            {'params': self.criterion.parameters(), 'lr': self.args.lr}
        ]

        self.optimizer = BertAdam(to_opt, lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
                                  b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                                  weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.total_time = 0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        for image, text, label, index in tqdm(self.train_loader):
            start_time = time.time()
            self.global_step += 1
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()

            hash_img, hash_text = self.model(image, text)

            loss = self.criterion(hash_img, label) + self.criterion(hash_text, label) + self.criterion(hash_img, label, hash_text)

            all_loss += loss.data
            self.optimizer.zero_grad()
            if loss != 0:
                loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")
