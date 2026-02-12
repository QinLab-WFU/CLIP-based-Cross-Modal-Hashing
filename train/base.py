import os
from tqdm import tqdm
import torch

from utils import get_logger, get_summary_writer

from train.MITH.data import generate_dataset
from dataset.dataloader import dataloader
from torch.utils.data import DataLoader
import time
from utils.calc_utils import calc_map_k_matrix as calc_map_k
import scipy.io as scio


class TrainBase(object):

    def __init__(self,
                args,
                rank=1):

        self.args = args
        os.makedirs(args.save_dir, exist_ok=True)
        self._init_writer()
        self.logger.info(self.args)
        self.rank = rank   # GPU ID

        self._init_dataset()
        self._init_model()

        self.global_step = 0
        self.max_mapi2t = 0
        self.max_mapt2i = 0
        self.best_epoch_i = 0
        self.best_epoch_t = 0

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        if self.args.dataset == 'flickr':
            data_dir = 'YOUR-FLIE-DIR'
            self.args.nclass = 24
        elif self.args.dataset == 'coco':
            data_dir = 'YOUR-FLIE-DIR'
            self.args.nclass = 80
        elif self.args.dataset == 'nuswide':
            data_dir = 'YOUR-FLIE-DIR'
            self.args.nclass = 21
        elif self.args.dataset == 'iapr':
            data_dir = 'YOUR-FLIE-DIR'
            self.args.nclass = 291
        else:
            raise ValueError("Unknown dataset")

        self.index_file = os.path.join(data_dir, "index.mat")
        if 'nuswide' in self.args.dataset:
            self.caption_file = os.path.join(data_dir, "caption.txt")
        else:
            self.caption_file = os.path.join(data_dir, "caption.mat")
        self.label_file = os.path.join(data_dir, "label.mat")

        if self.args.method == 'MITH':
            train_data, query_data, retrieval_data = generate_dataset(captionFile=self.args.caption_file,
                                                                      indexFile=self.args.index_file,
                                                                      labelFile=self.args.label_file,
                                                                      maxWords=self.args.max_words,
                                                                      imageResolution=self.args.resolution,
                                                                      query_num=self.args.query_num,
                                                                      train_num=self.args.train_num,
                                                                      seed=self.args.seed)
        else:
            train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file,
                                                            indexFile=self.args.index_file,
                                                            labelFile=self.args.label_file,
                                                            maxWords=self.args.max_words,
                                                            imageResolution=self.args.resolution,
                                                            query_num=self.args.query_num,
                                                            train_num=self.args.train_num,
                                                            seed=self.args.seed)

        self.train_labels = train_data.get_all_label().to(self.rank)
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
    
    def _init_model(self):
        self.model = None
        self.model_ddp = None
    
    def _init_writer(self):
        self.logger = get_logger(os.path.join(self.args.save_dir, "train.log" if self.args.is_train else "test.log"))
        self.writer = get_summary_writer(os.path.join(self.args.save_dir, "tensorboard"))

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            self.test()

    def change_state(self, mode):

        if mode == "train":
            self.model.train()
        elif mode == "valid":
            self.model.eval()
    
    def get_code(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            image_hash = torch.sign(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = torch.sign(text_hash)
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer, encoder_time

    def make_hash_code_DCHMT(self, code: list) -> torch.Tensor:

        code = torch.stack(code)
        code = code.permute(1, 0, 2)
        hash_code = torch.argmax(code, dim=-1)
        hash_code[torch.where(hash_code == 0)] = -1
        hash_code = hash_code.float()

        return hash_code

    def get_code_DCHMT(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)

        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            image_hash = self.make_hash_code_DCHMT(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = self.make_hash_code_DCHMT(text_hash)
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer, encoder_time

    def get_code_MITH(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, key_padding_mask, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            output_dict = self.model(image, text, key_padding_mask)

            img_hash_cls = output_dict['img_cls_hash'].detach()
            txt_hash_cls = output_dict['txt_cls_hash'].detach()

            img_tokens_hash = output_dict['img_tokens_hash'].detach()
            txt_tokens_hash = output_dict['txt_tokens_hash'].detach()

            img_buffer[index, :] = torch.sign(img_tokens_hash + img_hash_cls)
            text_buffer[index, :] = torch.sign(txt_tokens_hash + txt_hash_cls)
            encoder_time = time.time() - start_encoder_time

        return img_buffer, text_buffer, encoder_time

    def get_code_DNPH(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash, _ = self.model.encode_image(image)
            image_hash = torch.sign(image_hash)
            text_hash, _ = self.model.encode_text(text)
            text_hash = torch.sign(text_hash)
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer, encoder_time
    
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model-" + str(epoch) + ".pth"))
        self.logger.info("save mode to {}".format(os.path.join(self.args.save_dir, "model-" + str(epoch) + ".pth")))

    def train_epoch(self, epoch):
        raise NotImplementedError("Function of 'train' doesn't implement.")

    def train(self):
        self.logger.info("Start train.")

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
            self.save_model(epoch)
        self.logger.info(
            f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")
        print("1")
    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")

        if self.args.method == 'DCHMT':
            query_img, query_txt, q_encoder_time = self.get_code_DCHMT(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_DCHMT(self.retrieval_loader, self.args.retrieval_num)
        elif self.args.method == 'MITH':
            query_img, query_txt, q_encoder_time = self.get_code_MITH(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_MITH(self.retrieval_loader, self.args.retrieval_num)
        elif self.args.method == 'DNPH-TOMM':
            query_img, query_txt, q_encoder_time = self.get_code_DNPH(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_DNPH(self.retrieval_loader, self.args.retrieval_num)
        else:
            query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)

        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")

        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                            MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}, query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")

    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        if self.args.method == 'DCHMT':
            query_img, query_txt, q_encoder_time = self.get_code_DCHMT(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_DCHMT(self.retrieval_loader,
                                                                               self.args.retrieval_num)
        elif self.args.method == 'MITH':
            query_img, query_txt, q_encoder_time = self.get_code_MITH(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_MITH(self.retrieval_loader,
                                                                              self.args.retrieval_num)
        elif self.args.method == 'DNPH':
            query_img, query_txt, q_encoder_time = self.get_code_DNPH(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_DNPH(self.retrieval_loader,
                                                                              self.args.retrieval_num)
        else:
            query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)

        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(">>>>>> save all data!")

    def compute_loss(self):
        raise NotImplementedError("Function of 'compute_loss' doesn't implement.")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):

        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")
