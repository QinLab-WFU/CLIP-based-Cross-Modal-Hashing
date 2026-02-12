# DCHMT
# paper [Differentiable Cross-modal Hashing via Multimodal Transformers, ACM MM 2022]
# (https://dl.acm.org/doi/10.1145/3503161.3548187)

from model.DCHMT import MDCMHT
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
from utils import calc_neighbor, cosine_similarity, euclidean_similarity


class DCHMTTrainer(TrainBase):

    def __init__(self, args, rank):
        args = get_args(args)
        args.rank = rank
        super(DCHMTTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        self.model = MDCMHT(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
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

        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0
        times = 0
        for image, text, label, index in self.train_loader:
            self.global_step += 1
            times += 1
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            hash_img, hash_text = self.model(image, text)
            if self.args.hash_layer == "select":
                hash_img = torch.cat(hash_img, dim=-1) if isinstance(hash_img, list) else hash_img.view(hash_img.shape[0], -1)
                hash_text = torch.cat(hash_text, dim=-1)if isinstance(hash_text, list) else hash_text.view(hash_text.shape[0], -1)
            loss = self.compute_loss(hash_img, hash_text, label, epoch, times)
            all_loss += loss 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        
        s = torch.matmul(a, b.t())
        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))

        return b_loss
    
    def distribution_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        """
        """
        kl_divergence = torch.mean(a * torch.log(a / (b + 0.001)))
        print("mean", torch.mean(a - b))
        print("kl", kl_divergence)
        return kl_divergence

    def similarity_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor, threshold=0.05):
        
        # $\vartheta$
        vartheta = self.args.vartheta
        if self.args.sim_threshold != 0:
            threshold = self.args.sim_threshold
        similarity = (1 - cosine_similarity(a, b)) if self.args.similarity_function == "cosine" else euclidean_similarity(a, b)
        
        positive_similarity = similarity * label_sim
        # 只要cosine为负值的全都算为计算正确了，因为优化到2确实很难。
        negative_similarity = similarity * (1 - label_sim)
        
        if self.args.similarity_function == "cosine":
            positive_similarity = positive_similarity.clip(threshold) - threshold
            negative_similarity = negative_similarity.clip(max=1.)
            negative_similarity = torch.tensor([1.]).expand_as(negative_similarity).to(self.rank) * (1 - label_sim) - negative_similarity
        elif self.args.similarity_function == "euclidean":
            # 有euclidean距离可知，当有一半长度的hash码不同时，其negative_similarity距离应该是长度（concat操作将outputdim翻倍），所以这里clip掉认为认定的值
            # 人为认定的最大值是一半长度的hash码不同。
            max_value = float(self.args.output_dim * 2 * vartheta) ** 0.5
            negative_similarity = negative_similarity.clip(max=max_value)
            negative_similarity = torch.tensor([max_value]).expand_as(negative_similarity).to(self.rank) * (1 - label_sim) - negative_similarity

        if self.args.loss_type == "l1":
            positive_loss = positive_similarity.mean()
            negative_loss = negative_similarity.mean()
        elif self.args.loss_type == "l2":
            positive_loss = torch.pow(positive_similarity, 2).mean()
            negative_loss = torch.pow(negative_similarity, 2).mean()
        else:
            raise ValueError("argument of loss_type is not support.")
        
        return similarity, positive_loss, negative_loss
        
    def our_loss(self, image, text, label, epoch, times):
        loss = 0

        label_sim = calc_neighbor(label, label)
        if image.is_cuda:
            label_sim = label_sim.to(image.device)
        intra_similarity, intra_positive_loss, intra_negative_loss = self.similarity_loss(image, text, label_sim)
        inter_similarity_i, inter_positive_loss_i, inter_negative_loss_i = self.similarity_loss(image, image, label_sim)
        inter_similarity_t, inter_positive_loss_t, inter_negative_loss_t = self.similarity_loss(text, text, label_sim)

        intra_similarity_loss = (intra_positive_loss + intra_negative_loss) if self.args.similarity_function == "euclidean" else (intra_positive_loss + intra_negative_loss)
        inter_similarity_loss = inter_positive_loss_t + inter_positive_loss_i + (inter_negative_loss_i + inter_negative_loss_t) if self.args.similarity_function == "euclidean" else inter_positive_loss_t + inter_positive_loss_i + inter_negative_loss_i + inter_negative_loss_t
        similarity_loss = inter_similarity_loss + intra_similarity_loss
        
        if self.args.hash_layer != "select":
            quantization_loss = (self.hash_loss(image) + self.hash_loss(text)) / 2
            loss = similarity_loss + quantization_loss
            if self.global_step % self.args.display_step == 0:
                self.logger.info(f">>>>>> Display >>>>>> [{epoch}/{self.args.epochs}], [{times}/{len(self.train_loader)}]: all loss: {loss.data}, "\
                    f"SIMILARITY LOSS, Intra, positive: {intra_positive_loss.data}, negitave: {intra_negative_loss.data}, sum: {intra_similarity_loss.data}, " \
                    f"Inter, image positive: {inter_positive_loss_i.data}, image negitave: {inter_negative_loss_i.data}, "\
                    f"text positive: {inter_positive_loss_t.data}, text negitave: {inter_negative_loss_t.data}, sum: {inter_similarity_loss.data}, "\
                    f"QUATIZATION LOSS, {quantization_loss.data}, "\
                    f"lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")
        else:
            loss = similarity_loss # + self.args.qua_gamma * (image_quantization_loss + text_quantization_loss)
            if self.global_step % self.args.display_step == 0:
                self.logger.info(f">>>>>> Display >>>>>> [{epoch}/{self.args.epochs}], [{times}/{len(self.train_loader)}]: all loss: {loss.data}, "\
                    f"SIMILARITY LOSS, Intra, positive: {intra_positive_loss.data}, negitave: {intra_negative_loss.data}, sum: {intra_similarity_loss.data}, " \
                    f"Inter, image positive: {inter_positive_loss_i.data}, image negitave: {inter_negative_loss_i.data}, "\
                    f"text positive: {inter_positive_loss_t.data}, text negitave: {inter_negative_loss_t.data}, sum: {inter_similarity_loss.data}, "\
                    # f"QUATIZATION LOSS, image: {image_quantization_loss.data}, text: {text_quantization_loss.data}, "\
                    f"lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

        return loss
    
    def compute_loss(self, image, text, label, epoch, times):
        
        loss = self.our_loss(image, text, label, epoch, times)

        return loss


