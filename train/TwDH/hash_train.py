# TwDH
# paper [Two-Step Discrete Hashing for Cross-Modal Retrieval, TOMM 2024]
# (https://ieeexplore.ieee.org/document/10487033)

from model.TwDH import MTwDH
import os
import torch

from train.base import TrainBase
from model.base.optimization import BertAdam
from .get_args import get_args
from tqdm import tqdm
from utils.calc_utils import calc_map_k_matrix as calc_map_k


class TwDHTrainer(TrainBase):

    def __init__(self,
                rank=1):
        args = get_args()
        super(TwDHTrainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        self.model = MTwDH(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train,
                               long_center=self.args.long_center, short_center=self.args.short_center,
                               trans=self.args.trans_matrix).to(self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.model.float()
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.img_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.txt_hash.parameters(), 'lr': self.args.lr}
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine', 
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.distributed =False

        self.max_short = {}
        self.best_epoch_short = {}
        for item in self.model.get_short_dims():
            self.max_short.update({item: {"i2t": 0, "t2i": 0}})
            self.best_epoch_short.update({item: {"i2t": 0, "t2i": 0}})

        self.criterion = torch.nn.BCELoss()
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
            index = index.numpy()
            img_long_hash, img_short_hash, txt_long_hash, txt_short_hash, long_center, short_center = self.model(image, text)
            loss = self.compute_loss(img_long_hash, txt_long_hash, img_short_hash, txt_short_hash, label, index, long_center, short_center)
            all_loss += loss 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def hash_convert(self, hash_label):
        if len(hash_label.shape) == 2:
            result = torch.zeros([hash_label.shape[0], hash_label.shape[1], 2])
            hash_label = (hash_label > 0).long()
            i = torch.arange(hash_label.shape[0]).view(hash_label.shape[0], -1).expand_as(hash_label)
            j = torch.arange(hash_label.shape[1]).expand_as(hash_label)
            result[i, j, hash_label] = 1
            result = result.view(hash_label.shape[0], -1)
        elif len(hash_label.shape) == 1:
            result = torch.zeros([hash_label.shape[0], 2])
            hash_label = (hash_label > 0).long()
            result[torch.arange(hash_label.shape[0]), hash_label] = 1
            result = result.view(hash_label.shape[0], -1)
        result = result.to(hash_label.device)
        return result

    def hash_center_multilables(self, labels, Hash_center):
        # if labels.device != Hash_center.device:
        #     Hash_center = Hash_center.to(labels.device)
        is_start = True
        random_center = torch.randint_like(Hash_center[0], 2)
        for label in labels:
            one_labels = (label == 1).nonzero()
            one_labels = one_labels.squeeze(1)
            Center_mean = torch.mean(Hash_center[one_labels], dim=0)
            Center_mean[Center_mean < 0] = -1
            Center_mean[Center_mean > 0] = 1
            random_center[random_center == 0] = -1
            Center_mean[Center_mean == 0] = random_center[Center_mean == 0]
            Center_mean = Center_mean.view(1, -1)

            if is_start:
                hash_center = Center_mean
                is_start = False
            else:
                hash_center = torch.cat((hash_center, Center_mean), 0)
        # hash_center = hash_center.to(labels.device)
        # print(hash_center.device)
        return hash_center

    def soft_argmax_hash_loss(self, code):
        if len(code.shape) < 3:
            code = code.view(code.shape[0], -1, 2)

        hash_loss = 1 - torch.pow(2 * code - 1, 2).mean()
        return hash_loss

    def compute_loss(self, long_img_hash, long_txt_hash, short_img_hash, short_txt_hash, labels, indexs, long_center, short_center):

        long_hash_label = self.hash_convert(self.hash_center_multilables(labels, long_center)).float().to(
            long_img_hash.device, non_blocking=True)
        long_image_loss = self.criterion(long_img_hash, long_hash_label)
        long_text_loss = self.criterion(long_txt_hash, long_hash_label)

        long_nce_loss = (long_image_loss + long_text_loss) / 2

        long_code_image_loss = self.soft_argmax_hash_loss(long_img_hash)
        long_code_text_loss = self.soft_argmax_hash_loss(long_txt_hash)
        long_quan_loss = (long_code_image_loss + long_code_text_loss) / 2

        short_nce_loss = {}
        short_quan_loss = {}
        for k, v in short_center.items():
            short_hash_label = self.hash_convert(self.hash_center_multilables(labels, v)).float().to(long_img_hash.device,
                                                                                           non_blocking=True)

            short_image_loss = self.criterion(short_img_hash[k], short_hash_label)
            # print(short_img_hash[k])
            short_text_loss = self.criterion(short_txt_hash[k], short_hash_label)
            # print(short_txt_hash[k])
            short_nce_loss.update({k: (short_image_loss + short_text_loss) / 2})

            short_code_image_loss = self.soft_argmax_hash_loss(short_img_hash[k])
            short_code_text_loss = self.soft_argmax_hash_loss(short_txt_hash[k])
            short_quan_loss.update({k: (short_code_image_loss + short_code_text_loss) / 2})

        loss = long_nce_loss + self.args.quan_alpha * long_quan_loss
        for k, v in short_nce_loss.items():
            loss += self.args.low_rate * v
        for k, v in short_quan_loss.items():
            loss += self.args.low_rate * v

        short_dict = {}
        for k, v in short_nce_loss.items():
            short_dict.update({k: {"NCE": v, "Quan": short_quan_loss[k]}})

        return loss

    def make_hash_code(cls, code):

        if isinstance(code, list):
            code = torch.stack(code).permute(1, 0, 2)
        elif len(code.shape) < 3:
            code = code.view(code.shape[0], -1, 2)
        else:
            code = code
        hash_code = torch.argmax(code, dim=-1)
        hash_code[torch.where(hash_code == 0)] = -1
        hash_code = hash_code.float()

        return hash_code

    def get_code(self, data_loader, length: int):
        short_dims = self.model.get_short_dims()
        long_img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        long_txt_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        short_img_buffers = {}
        short_txt_buffers = {}
        for dim in short_dims:
            short_img_buffers.update({str(dim): torch.empty(length, dim, dtype=torch.float).to(self.rank)})
            short_txt_buffers.update({str(dim): torch.empty(length, dim, dtype=torch.float).to(self.rank)})

        for image, text, label, index in tqdm(data_loader):
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()

            long_image_hash, short_image_hash = self.model.encode_image(image)
            long_text_hash, short_text_hash = self.model.encode_text(text)

            long_img_buffer[index, :] = self.make_hash_code(long_image_hash.data)
            long_txt_buffer[index, :] = self.make_hash_code(long_text_hash.data)
            for k, v in short_image_hash.items():
                short_img_buffers[k][index, :] = self.make_hash_code(v.data)
            for k, v in short_text_hash.items():
                short_txt_buffers[k][index, :] = self.make_hash_code(v.data)

        return long_img_buffer, long_txt_buffer, short_img_buffers, short_txt_buffers

    def valid(self, epoch, k=None):
        assert self.query_loader is not None and self.retrieval_loader is not None
        self.logger.info("Valid.")

        q_long_img_buffer, q_long_txt_buffer, q_short_img_buffers, q_short_txt_buffers = self.get_code(self.query_loader, self.args.query_num)
        r_long_img_buffer, r_long_txt_buffer, r_short_img_buffers, r_short_txt_buffers = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        self.valid_each(epoch=epoch, query_img=q_long_img_buffer, query_txt=q_long_txt_buffer, retrieval_img=r_long_img_buffer, retrieval_txt=r_long_txt_buffer, k=k)

        for key, v in q_short_img_buffers.items():
            self.valid_each(epoch=epoch, query_img=q_short_img_buffers[key], query_txt=q_short_txt_buffers[key], retrieval_img=r_short_img_buffers[key],
                            retrieval_txt=r_short_txt_buffers[key], k=k, short=key)

    def valid_each(self, epoch, query_img=None, query_txt=None, retrieval_img=None, retrieval_txt=None, k=None, short=None):

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, k)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, k)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, k)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, k)

        if short is None:
            if self.max_mapi2t < mAPi2t:
                self.best_epoch_i = epoch
                if not self.distributed or (self.distributed and self.rank == 0):
                    self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t-long.mat")
            self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
            if self.max_mapt2i < mAPt2i:
                self.best_epoch_t = epoch
                if not self.distributed or (self.distributed and self.rank == 0):
                    self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i-long.mat")
            self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
            self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], Long, {query_img.shape[-1]} Bit, MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, "
                    f"MAX MAP(i->t): {self.max_mapi2t}, epoch: {self.best_epoch_i}, MAX MAP(t->i): {self.max_mapt2i}, epoch: {self.best_epoch_t}")
        else:
            short = int(short)
            if self.max_short[short]["i2t"] < mAPi2t:
                self.best_epoch_short[short]["i2t"] = epoch
                if not self.distributed or (self.distributed and self.rank == 0):
                    self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name=str(short) + "i2t-short.mat")
            self.max_short[short]["i2t"] = max(self.max_short[short]["i2t"], mAPi2t)
            if self.max_short[short]["t2i"] < mAPt2i:
                self.best_epoch_short[short]["t2i"] = epoch
                if not self.distributed or (self.distributed and self.rank == 0):
                    self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name=str(short) + "t2i-short.mat")
            self.max_short[short]["t2i"] = max(self.max_short[short]["t2i"], mAPt2i)
            self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], Short, {query_img.shape[-1]} Bit, MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, "
                    f"MAX MAP(i->t): {self.max_short[short]['i2t']}, epoch: {self.best_epoch_short[short]['i2t']}, MAX MAP(t->i): {self.max_short[short]['t2i']}, epoch: {self.best_epoch_short[short]['t2i']}")
