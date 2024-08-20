# MITH
# paper [Multi-Granularity Interactive Transformer Hashing for Cross-modal Retrieval, ACM MM 2023]
# (https://dl.acm.org/doi/10.1145/3581783.3612411)

from model.MITH import MITH
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
from utils.calc_utils import calc_neighbor
import time
import torch.nn.functional as F
from einops import rearrange


class MITHTrainer(TrainBase):

    def __init__(self,
                rank=1):
        args = get_args()
        super(MITHTrainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = MITH(args=self.args).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.model.float()
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.hash.parameters(), 'lr': self.args.lr},
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine', 
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.k_bits = self.args.output_dim
        self.img_buffer_tokens = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)
        self.img_buffer_cls = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)

        self.txt_buffer_tokens = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)
        self.txt_buffer_cls = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)

        self.total_time = 0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        epoch_avg_loss_dict = {'all_loss': 0}
        for image, text, key_padding_mask, label, index in self.train_loader:
            start_time = time.time()
            self.global_step += 1
            image.float()

            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)

            output_dict = self.model(image, text, key_padding_mask)

            img_cls_hash = output_dict['img_cls_hash']
            txt_cls_hash = output_dict['txt_cls_hash']
            self.img_buffer_cls[index] = img_cls_hash.detach()
            self.txt_buffer_cls[index] = txt_cls_hash.detach()

            img_tokens_hash = output_dict['img_tokens_hash']
            txt_tokens_hash = output_dict['txt_tokens_hash']
            self.img_buffer_tokens[index] = img_tokens_hash.detach()
            self.txt_buffer_tokens[index] = txt_tokens_hash.detach()

            hyper_lambda = self.args.hyper_lambda
            B = torch.sign(
                (img_cls_hash.detach() * hyper_lambda + img_tokens_hash.detach() * (1 - hyper_lambda)) + \
                (txt_cls_hash.detach() * hyper_lambda + txt_tokens_hash.detach() * (1 - hyper_lambda)))

            ALL_LOSS_DICT = self.compute_loss(output_dict, label, B)

            loss = 0
            for key in ALL_LOSS_DICT:
                loss += ALL_LOSS_DICT[key]
                if key in epoch_avg_loss_dict:
                    epoch_avg_loss_dict[key] += ALL_LOSS_DICT[key]
                else:
                    epoch_avg_loss_dict[key] = ALL_LOSS_DICT[key]
            epoch_avg_loss_dict['all_loss'] += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] all loss avg: {epoch_avg_loss_dict['all_loss'].data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")

    def info_nce_loss(self, out_1, out_2, temperature=0.07):
        # out_*: ND
        bz = out_1.size(0)
        targets = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores1, targets)

        return 0.5 * (loss0 + loss1)

    def info_nce_loss_bmm(self, out_1, out_2, temperature=0.07):
        # out1: L,N,D
        # out2: L,N,D
        out_1 = out_1.permute(1, 0, 2)  # NLD
        out_2 = out_2.permute(1, 0, 2)  # NLD
        bz = out_1.size(0)

        sim = torch.bmm(out_1, out_2.permute(0, 2, 1))
        sim /= temperature

        word_num = sim.shape[1]

        sim_1 = rearrange(sim, "b n1 n2 -> (b n1) n2")
        sim_2 = rearrange(sim, "b n1 n2 -> (b n2) n1")

        targets = torch.arange(word_num).type_as(out_1).long().repeat(bz)

        loss_1 = F.cross_entropy(sim_1, targets)
        loss_2 = F.cross_entropy(sim_2, targets)

        return 0.5 * (loss_1 + loss_2)

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        # a: ND
        # b: MD
        # label_sim: NM
        s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
        return b_loss

    def quantization_loss_2(self, hash_feature, B):
        return F.mse_loss(hash_feature, B, reduction='sum') / (hash_feature.shape[0]) / self.k_bits

    def compute_loss(self, output_dict, label, B):
        ALL_LOSS = {}

        label_sim = calc_neighbor(self.train_labels.float().to(self.rank, non_blocking=True), label.float())

        img_tokens_hash = output_dict['img_tokens_hash']
        txt_tokens_hash = output_dict['txt_tokens_hash']

        img_cls_hash = output_dict['img_cls_hash']
        txt_cls_hash = output_dict['txt_cls_hash']

        res_img_cls = output_dict['res_img_cls']
        res_txt_eos = output_dict['res_txt_cls']

        trans_tokens_i = output_dict['trans_tokens_i']
        trans_tokens_t = output_dict['trans_tokens_t']

        # Token Intra
        hyper_tokens_intra = self.args.hyper_tokens_intra
        ALL_LOSS['tokens_intra_likelihood'] = hyper_tokens_intra * \
                                              (self.bayesian_loss(self.img_buffer_tokens, img_tokens_hash, label_sim) + \
                                               self.bayesian_loss(self.txt_buffer_tokens, txt_tokens_hash, label_sim))

        # CLS Inter
        hyper_cls_inter = self.args.hyper_cls_inter
        ALL_LOSS['cls_inter_likelihood'] = hyper_cls_inter * \
                                           (self.bayesian_loss(self.img_buffer_cls, txt_cls_hash, label_sim) + \
                                            self.bayesian_loss(self.txt_buffer_cls, img_cls_hash, label_sim))

        # hash feature
        H_i = img_cls_hash * 0.5 + img_tokens_hash * 0.5
        H_t = txt_cls_hash * 0.5 + txt_tokens_hash * 0.5
        # quantization loss
        hyper_quan = self.args.hyper_quan
        ALL_LOSS['quantization'] = hyper_quan * (self.quantization_loss_2(H_i, B) + self.quantization_loss_2(H_t, B))

        # Contrastive Alignment loss
        hyper_info_nce = self.args.hyper_info_nce
        hyper_alpha = self.args.hyper_alpha
        ALL_LOSS['infoNCE'] = hyper_info_nce * \
                              (self.info_nce_loss(res_img_cls, res_txt_eos) +
                               hyper_alpha * self.info_nce_loss_bmm(trans_tokens_i, trans_tokens_t))

        # 1*gradient back to student.
        item_1 = (F.mse_loss(img_cls_hash.detach(), img_tokens_hash, reduction='sum') +
                  F.mse_loss(txt_cls_hash.detach(), txt_tokens_hash, reduction='sum'))
        # 0.1*gradient back to teacher.
        item_2 = 0.1 * (F.mse_loss(img_cls_hash, img_tokens_hash.detach(), reduction='sum') +
                        F.mse_loss(txt_cls_hash, txt_tokens_hash.detach(), reduction='sum'))
        # distillation loss
        hyper_distill = self.args.hyper_distill
        ALL_LOSS['distillation'] = hyper_distill * (item_1 + item_2) / (img_cls_hash.shape[0])
        return ALL_LOSS

