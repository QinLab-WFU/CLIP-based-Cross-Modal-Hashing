# DNpH
# paper [Deep Neighborhood-Preserving Hashing With Quadratic Spherical Mutual Information for Cross-Modal Retrieval, TMM 2024]
# (https://ieeexplore.ieee.org/document/10379137)

from model.DNpH_TMM import MDNpH
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
from .loss import qmi_loss
import time


class DNpHTMMTrainer(TrainBase):

    def __init__(self, args, rank):
        args = get_args(args)
        args.rank = rank
        super(DNpHTMMTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = MDNpH(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
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

        self.total_time = 0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0
        times = 0
        for image, text, label, index in self.train_loader:
            start_time = time.time()
            self.global_step += 1
            times += 1
            image.float()

            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)

            hash_img, hash_text = self.model(image, text)

            loss1 = qmi_loss(images=hash_img, texts=hash_text, targets=label)

            loss = loss1

            all_loss += loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, time: {self.total_time}")