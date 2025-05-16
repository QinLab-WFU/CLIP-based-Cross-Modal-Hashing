# DNPH
# paper [Deep Neighborhood-aware Proxy Hashing with Uniform Distribution Constraint for Cross-modal Retrieval, TOMM 2024]
# (https://dl.acm.org/doi/10.1145/3643639)

from model.DNPH_TOMM import MDNPH
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
from .loss import DNPH_out
from .b_reg import rand_unit_rect, gene_noise
import time


class DNPHTOMMTrainer(TrainBase):

    def __init__(self,
                rank=1):
        args = get_args()
        super(DNPHTOMMTrainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = MDNPH(outputDim=self.args.output_dim, num_classes=self.args.numclass, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.model.float()
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.image_pre.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_pre.parameters(), 'lr': self.args.lr}
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine', 
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.DNPH = DNPH_out().to(self.rank)
        self.total_time = 0
        self.optimizer_loss = torch.optim.SGD(params=self.DNPH.parameters(), lr=1e-4)
        # print(self.model)

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
            label = label.float()

            hash_img, pre_img, hash_text, pre_text = self.model(image, text)

            batch_size_, code_length = hash_img.shape
            s_vector = rand_unit_rect(batch_size_, code_length)

            i_noises = gene_noise(hash_img.cpu().detach().numpy(), s_vector)
            t_noises = gene_noise(hash_text.cpu().detach().numpy(), s_vector)

            i_noises = torch.from_numpy(i_noises).float().to(self.rank)
            t_noises = torch.from_numpy(t_noises).float().to(self.rank)
            i_noise_loss = hash_img.mul(i_noises).sum(dim=-1).mean()
            t_noise_loss = hash_text.mul(t_noises).sum(dim=-1).mean()
            noise_loss = i_noise_loss + t_noise_loss
            #
            # loss1 = self.DNPH(hash_img, hash_text, pre_img, pre_text, label, label)
            loss1 = 1

            loss = loss1 - 0.1 * noise_loss
            all_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")

