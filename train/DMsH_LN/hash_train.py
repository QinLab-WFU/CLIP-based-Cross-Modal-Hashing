

from model.DMsH_LN import MDMsH_LN
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
from .loss import HyP
from  .MSLOSS import  MultiSimilarityLoss
from .labelnet import LabelNet
import time


class DMsH_LNTrainer(TrainBase):

    def __init__(self,
                rank=1):
        args = get_args()
        super(DMsH_LNTrainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        self.model = MDMsH_LN(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()
        self.MSL = MultiSimilarityLoss().to(device='cuda:1')
        self.L_net = LabelNet(code_len=self.args.output_dim, label_dim=self.args.numclass).to(self.rank)
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.MSL.parameters(), 'lr': self.args.lr},
                    {'params': self.L_net.parameters(), 'lr': self.args.lr},
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)


        self.total_time = 0

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0
        self.L_net.set_alpha(epoch)
        for image, text, label, index in self.train_loader:
            start_time = time.time()
            self.global_step += 1
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)
            _, _, label_output = self.L_net(label, device=self.rank)
            hash_img, hash_text = self.model(image, text)
            img_loss1 = self.MSL(hash_img, label_output)
            text_loss1 = self.MSL(hash_text, label_output)
            i_t_loss1 = self.MSL(hash_img, label_output, feat2=hash_text)
            loss = img_loss1 + text_loss1 + i_t_loss1

            all_loss += img_loss1 + text_loss1 + i_t_loss1

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            self.total_time += time.time() - start_time

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")


