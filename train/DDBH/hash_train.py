# DNpH
# paper [Deep Discriminative Boundary Hashing for Cross-modal Retrieval, TCSVT 2025]
# (https://ieeexplore.ieee.org/document/10379137)

from model.DDBH import MDDBH
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
from .loss import BPLoss
import time


class DDBHTrainer(TrainBase):

    def __init__(self, args, rank):
        args = get_args(args)
        args.rank = rank
        super(DDBHTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = MDDBH(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.bp = BPLoss(bit=self.args.output_dim).to(self.rank)
        
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
            label = label.float()
            index = index.numpy()
            s = (label @ label.t()) > 0
            s = s.float()

            hash_img, hash_text = self.model(image, text)

            intra_lossi = self.bp(hash_img, hash_img, label)
            intra_losst = self.bp(hash_text, hash_text, label)
            inter_loss = self.bp(hash_img, hash_text, label)

            iq_loss = torch.matmul(s, (hash_img - hash_img.sign()).pow(2))
            tq_loss = torch.matmul(s, (hash_text - hash_text.sign()).pow(2))
            iq_loss = iq_loss.mean()
            tq_loss = tq_loss.mean()

            loss = (intra_lossi + intra_losst + inter_loss) + 0.1 * (iq_loss + tq_loss)

            all_loss += loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, time: {self.total_time}")