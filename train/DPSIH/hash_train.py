import os
import torch
import time
from torch import nn
from tqdm import tqdm
from model.DPSIH import MDPSIH
from train.base import TrainBase
from model.base.optimization import BertAdam
from .Loss import DPSIHLoss
from .get_args import get_args


class DPSIHTrainer(TrainBase):

    def __init__(self, args, rank=0):
        args = get_args(args)
        args.rank = rank
        super(DPSIHTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.abs = True if hasattr(self.args, 'order') and self.args.order else False

        self.model = MDPSIH(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                           writer=self.writer, logger=self.logger,
                            num_embeds=self.args.num_embeds,dropout=self.args.dropout,
                           is_train=True).to(self.rank).float()

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        to_optim = [
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.DSIE_i.parameters(), 'lr': self.args.lr},
            {'params': self.model.DSIE_t.parameters(), 'lr': self.args.lr},
        ]

        self.optimizer = BertAdam(to_optim, lr=self.args.lr, warmup=self.args.warmup_proportion,
                                  schedule='warmup_cosine',b1=0.9, b2=0.98, e=1e-6,
                                  t_total=len(self.train_loader) * self.args.epochs,
                                  weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.criterion = DPSIHLoss(self.args, self.rank)

        self.total_time = 0.0

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        for image, text, label, index in tqdm(self.train_loader):
            start_time = time.time()
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)
            label = label.float()
            embed_i, embed_t, attn_i, attn_t, resi_i, resi_t = self.model(image, text)
            loss, loss_dict = self.criterion(embed_i, embed_t, resi_i, resi_t, label)
            all_loss += loss
            self.optimizer.zero_grad()
            if loss != 0:
                loss.backward()
            if self.args.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, time: {self.total_time}")


