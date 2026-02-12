import os

import numpy as np
import torch
import time

from .stochman.stochman.laplace import DiagLaplace
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from .stochman.stochman import ContrastiveHessianCalculator, MSEHessianCalculator
from .stochman.stochman.utils import convert_to_stochman
from torch.nn import functional as F

from model.DPBE import MDPBE
from train.base import TrainBase
from model.base.optimization import BertAdam

from .get_args import get_args
from .triplet_miner import TripletMinner


class DPBETrainer(TrainBase):

    def __init__(self, args, rank):
        args = get_args(args)
        args.rank = rank
        super(DPBETrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = MDPBE(use_lam=self.args.use_lam, outputDim=self.args.output_dim,
                           clipPath=self.args.clip_path, writer=self.writer,
                           logger=self.logger, is_train=True).to(self.rank)
        
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"), strict=False)
        self.model.float()

        if self.args.use_lam:
            self.model.image_hash = convert_to_stochman(self.model.image_hash)
            self.model.text_hash = convert_to_stochman(self.model.text_hash)

        to_optim = [
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
        ]

        self.optimizer = BertAdam(to_optim, lr=self.args.lr, warmup=self.args.warmup_proportion,
                                  schedule='warmup_cosine',b1=0.9, b2=0.98, e=1e-6,
                                  t_total=len(self.train_loader) * self.args.epochs,
                                  weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.criterion = torch.nn.MSELoss().to(self.rank)

        if self.args.use_lam:
            if self.args.loss != "acm":
                self.hessian_calculator = ContrastiveHessianCalculator(
                    wrt="weight",
                    shape="diagonal",
                    speed="half",
                    method="fix",
                )
            else:
                self.hessian_calculator = MSEHessianCalculator(  #
                    wrt="weight",
                    shape="diagonal",
                    speed="half",
                    method="",
                )

            self.laplace = DiagLaplace()
            hessian_i = self.laplace.init_hessian(self.args.train_num, self.model.image_hash, self.rank)
            hessian_t = self.laplace.init_hessian(self.args.train_num, self.model.text_hash, self.rank)
            self.scale_hs = self.args.train_num ** 2
            self.model.register_buffer("hessian_i", hessian_i)
            self.model.register_buffer("hessian_t", hessian_t)
            # TODO: notice the special hessian miner
            self.hessian_miner = TripletMinner().to(self.rank)

        self.total_time = 0.0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        for image, text, label, index in tqdm(self.train_loader):
            start_time = time.time()

            image = image.to(self.rank, non_blocking=True).float()
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()

            if hasattr(self.args, "noise-rate") and self.args.warm_up <= epoch and self.args.noise_rate > 0:
                label = self.add_noise_to_labels(label)

            embed_i, embed_t = self.model(image, text)

            if not self.args.use_lam:

                embed_i, embed_t = self.model.hashing(embed_i, embed_t)

                _, aff_norm, aff_label = self.affinity_tag_multi(label.cpu().numpy(), label.cpu().numpy())
                aff_label = torch.Tensor(aff_label).to(self.rank)
                H_i, H_t = F.normalize(embed_i), F.normalize(embed_t)
                
                loss = self.criterion(H_i.mm(H_i.t()), aff_label) + self.criterion(H_t.mm(H_t.t()), aff_label)
                loss += self.criterion(H_i.mm(H_t.t()), aff_label)


            else:
                # get mean and std of posterior
                self.sample()

                loss = 0
                hessian_i = 0
                hessian_t = 0
                for sample_i, sample_t in zip(self.nn_weight_samples_i, self.nn_weight_samples_t):

                    # replace the network parameters with the sampled parameters
                    vector_to_parameters(sample_i, self.model.image_hash.parameters())
                    vector_to_parameters(sample_t, self.model.text_hash.parameters())

                    z_i, z_t = self.model.hashing(embed_i, embed_t)

                    # ensure that we are on unit sphere
                    z_i = z_i / z_i.norm(dim=-1, keepdim=True)
                    z_t = z_t / z_t.norm(dim=-1, keepdim=True)

                    _, aff_norm, aff_label = self.affinity_tag_multi(label.cpu().numpy(), label.cpu().numpy())
                    aff_label = torch.Tensor(aff_label).to(self.rank)
                    H_i, H_t = F.normalize(z_i), F.normalize(z_t)

                    loss = self.criterion(H_i.mm(H_i.t()), aff_label) + self.criterion(H_t.mm(H_t.t()), aff_label)
                    loss += self.criterion(H_i.mm(H_t.t()), aff_label)
                    
                    with torch.inference_mode():

                        # hessian_indices_tuple = self.hessian_miner(z, y)
                        hessian_indices_tuple = self.hessian_miner(label)

                        # randomly choose 5000 pairs if more than 5000 pairs available.
                        # TODO: decide what to do. What pairs should we use to compute the hessian over?
                        # does it matter? What experiments should we run to get a better idea?
                        n_triplets = len(hessian_indices_tuple[0])
                        if n_triplets > self.args.max_pairs:
                            idx = torch.randperm(hessian_indices_tuple[0].size(0))[: self.args.max_pairs]
                            hessian_indices_tuple = (
                                hessian_indices_tuple[0][idx],
                                hessian_indices_tuple[1][idx],
                                hessian_indices_tuple[2][idx],
                            )

                        h_s_i = self.hessian_calculator.compute_hessian(
                            embed_i.detach(), self.model.image_hash, hessian_indices_tuple
                        )
                        h_s_i = self.laplace.scale(h_s_i, min(n_triplets, self.args.max_pairs), self.scale_hs)
                        hessian_i += h_s_i

                        h_s_t = self.hessian_calculator.compute_hessian(
                            embed_t.detach(), self.model.text_hash, hessian_indices_tuple
                        )
                        h_s_t = self.laplace.scale(h_s_t, min(n_triplets, self.args.max_pairs), self.scale_hs)
                        hessian_t += h_s_t

                # reset the network parameters with the mean parameter (MAP estimate parameters)
                vector_to_parameters(self.mu_q_i, self.model.image_hash.parameters())
                vector_to_parameters(self.mu_q_t, self.model.text_hash.parameters())
                loss /= self.args.train_n_samples
                hessian_i = hessian_i / self.args.train_n_samples
                hessian_t = hessian_t / self.args.train_n_samples

                self.model.hessian_i = self.args.hessian_memory_factor * self.model.hessian_i + torch.relu(hessian_i)
                self.model.hessian_t = self.args.hessian_memory_factor * self.model.hessian_t + torch.relu(hessian_t)

            all_loss += loss.data
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, time: {self.total_time}")

    def valid_hook(self):
        if self.args.use_lam:
            self.sample()

    def get_code(self, data_loader, length: int):
        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()

            if not self.args.use_lam:
                z_i, z_t = self.model(image, text)
                z_i, z_t = self.model.hashing(z_i, z_t)
                z_i = torch.sign(z_i)
                z_t = torch.sign(z_t)
                encoder_time = time.time() - start_encoder_time
                img_buffer[index, :] = z_i.data
                text_buffer[index, :] = z_t.data

            else:
                image_embed, text_embed = self.model.encoding(image, text)
                mu_q_i = parameters_to_vector(self.model.image_hash.parameters()).unsqueeze(1)
                mu_q_t = parameters_to_vector(self.model.text_hash.parameters()).unsqueeze(1)

                zs_i = []
                zs_t = []
                for sample_i, sample_t in zip(self.nn_weight_samples_i, self.nn_weight_samples_t):
                    vector_to_parameters(sample_i, self.model.image_hash.parameters())
                    vector_to_parameters(sample_t, self.model.text_hash.parameters())
                    z_i, z_t = self.model.hashing(image_embed, text_embed)
                    z_i = z_i / z_i.norm(dim=-1, keepdim=True)
                    z_t = z_t / z_t.norm(dim=-1, keepdim=True)
                    zs_i.append(z_i)
                    zs_t.append(z_t)

                zs_i = torch.stack(zs_i)
                zs_t = torch.stack(zs_t)
                z_mu_i = torch.sign(zs_i.mean(dim=0))
                z_mu_t = torch.sign(zs_t.mean(dim=0))
                # z_sigma_i = zs_i.std(dim=0)
                # z_sigma_t = zs_t.std(dim=0)

                encoder_time = time.time() - start_encoder_time
                img_buffer[index, :] = z_mu_i.data
                text_buffer[index, :] = z_mu_t.data

                vector_to_parameters(mu_q_i, self.model.image_hash.parameters())
                vector_to_parameters(mu_q_t, self.model.text_hash.parameters())

        return img_buffer, text_buffer, encoder_time

    def sample(self):

        # get mean and std of posterior
        self.mu_q_i = parameters_to_vector(self.model.image_hash.parameters()).unsqueeze(1)
        self.sigma_q_i = self.laplace.posterior_scale(torch.relu(self.model.hessian_i))
        self.mu_q_t = parameters_to_vector(self.model.text_hash.parameters()).unsqueeze(1)
        self.sigma_q_t = self.laplace.posterior_scale(torch.relu(self.model.hessian_t))

        samples_num = self.args.train_n_samples if self.model.training else self.args.valid_n_samples

        # draw samples
        self.nn_weight_samples_i = self.laplace.sample(self.mu_q_i, self.sigma_q_i, samples_num)
        self.nn_weight_samples_t = self.laplace.sample(self.mu_q_t, self.sigma_q_t, samples_num)

    def get_pos_idx(self, target):

        classes = np.unique(target)
        idx = {}
        for c in classes:
            idx[f"{c}"] = {"pos": np.where(target == c)[0]}

        pos_idx = []
        for i in range(len(target)):
            key = f"{target[i].data}"

            pidx = idx[key]["pos"]
            pidx = pidx[pidx != i]  # remove self

            pos_idx.append(pidx)

        return pos_idx

    def zero2eps(self, x):
        x[x == 0] = 1
        return x

    def normalize(self, affinity):
        col_sum = self.zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
        row_sum = self.zero2eps(np.sum(affinity, axis=0))
        out_affnty = affinity / col_sum  # row data sum = 1
        in_affnty = np.transpose(affinity / row_sum)  # col data sum = 1 then transpose
        return in_affnty, out_affnty

    def affinity_tag_multi(self, tag1: np.ndarray, tag2: np.ndarray):
        '''
        Use label or plabel to create the graph.
        :param tag1:
        :param tag2:
        :return:
        '''
        aff = np.matmul(tag1, tag2.T)
        affinity_matrix = np.float32(aff)
        # affinity_matrix[affinity_matrix > 1] = 1
        affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
        affinity_matrix = 2 * affinity_matrix - 1
        in_aff, out_aff = self.normalize(affinity_matrix)

        return in_aff, out_aff, affinity_matrix

    def add_noise_to_labels(self, labels):
        labels = labels.cpu().numpy()  # 转换为 numpy
        num_samples, num_labels = labels.shape
        num_noise = int(num_samples * self.args.noise_rate)

        noise_indices = np.random.choice(num_samples, num_noise, replace=False)

        for i in noise_indices:
            ones_indices = np.where(labels[i, :] == 1)[0]
            zeros_indices = np.where(labels[i, :] == 0)[0]

            if len(ones_indices) > 0:
                j = np.random.choice(ones_indices)
                labels[i, j] = 0

            if len(zeros_indices) > 0:
                j = np.random.choice(zeros_indices)
                labels[i, j] = 1

        return torch.tensor(labels, dtype=torch.float32).to(self.rank)  # 转换回 Tensor 并放回 GPU
