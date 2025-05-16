import os
import torch
import logging
import torch.nn as nn

from model.modelbase import Baseclip, weights_init_kaiming


def softmax_hash(embed, return_vector=True):
    # assert len(embed.shape) == 2, "the size of input feature must equal to 2"

    if len(embed.shape) == 2:
        embed = embed.view(embed.shape[0], -1, 2)
    else:
        assert embed.shape[-1] == 2, f"softmax hash must input a shape of B,K,2m. It is {embed.shape}"

    embed = embed.view(embed.shape[0], -1, 2)
    hash = torch.softmax(embed, dim=-1).view(embed.shape[0], -1) if return_vector else torch.softmax(embed, dim=-1)
    return hash


def tanh_hash(embed):
    return torch.tanh(embed)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Hash(nn.Module):

    def __init__(self, hash_func=None, merge_func=None):
        """
        merge_func is used for DiHE method.
        """
        super(Hash, self).__init__()
        assert hash_func is not None, "'hash_func': hash function must be provided!"
        if hash_func == 'softmax':
            self.hash_func = softmax_hash
        else:
            self.hash_func = tanh_hash
        self.merge_func = merge_func

    def forward(self, embeds):
        hash = self.hash_func(embeds) if self.merge_func is None else self.hash_func(self.merge_func(embeds))
        return hash


class ModalityHash(Hash):

    def __init__(self, inputDim=2048, outputDim=64, num_heads=8, batch_first=True, layernorm=True, hash_func=None):

        super(ModalityHash, self).__init__(hash_func=hash_func)
        self.bit = outputDim
        self.atten = nn.MultiheadAttention(inputDim, num_heads=num_heads, batch_first=batch_first)
        self.norm = LayerNorm(inputDim) if layernorm else nn.BatchNorm1d(inputDim)
        self.fc2 = nn.Linear(inputDim, outputDim * 2)
        self.fc2.apply(weights_init_kaiming)

    def freezen(self):
        for param in self.atten.parameters():
            param.requires_grid = False
        for param in self.fc2.parameters():
            param.requires_grid = False

    def quantization(self, code):
        return self.hash_func(code)

    def forward(self, data):

        data = data.view(data.shape[0], 1, data.shape[1])
        embed = self.atten(data, data, data, need_weights=False)[0]
        embed = embed.squeeze()
        embed = self.norm(embed)
        embed = self.fc2(embed)
        embed = torch.relu(embed)

        softmax_hash = super().forward(embed)

        return softmax_hash


class MTwDH(Baseclip):

    def __init__(self,
                 outputDim=64,
                 clipPath="./ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 is_train=True,

                 long_center: str = "./TwDH/center/coco/long",
                 short_center: str = "./TwDH/center/coco/short",
                 trans: str = "./TwDH/center/coco/trans",
                 num_heads=8,
                 batch_first=True,
                 hash_func: str = "softmax",
                 quan_alpha: float = 0.5,
                 low_rate: float = 0
                 ):
        super(MTwDH, self).__init__(outputDim=outputDim, clipPath=clipPath, writer=writer,
                                    saveDir=saveDir, logger=logger, is_train=is_train)

        long_dim = outputDim
        # self.output_dim = long_dim

        long_center = os.path.join(long_center, str(long_dim) + ".pkl")
        trans = os.path.join(trans, str(long_dim))

        self.img_hash = ModalityHash(inputDim=self.embedDim, outputDim=long_dim, layernorm=False, num_heads=num_heads,
                                     batch_first=batch_first, hash_func=hash_func)
        self.txt_hash = ModalityHash(inputDim=self.embedDim, outputDim=long_dim, layernorm=True, num_heads=num_heads,
                                     batch_first=batch_first, hash_func=hash_func)

        self.long_center = torch.load(long_center).float()
        if os.path.isfile(short_center):
            key = os.path.basename(short_center).strip().split(".")[0]
            self.short_center = {key: torch.load(short_center).float()}
        else:
            self.short_center = {}
            for item in os.listdir(short_center):
                key = item.strip().split(".")[0]
                self.short_center.update({key: torch.load(os.path.join(short_center, item)).float()})

        if os.path.isfile(trans):
            key = os.path.basename(trans).strip().split(".")[0]
            self.trans = {key: torch.load(trans).float()}
        else:
            self.trans = {}
            for item in os.listdir(trans):
                key = item.strip().split(".")[0]
                self.trans.update({key: torch.load(os.path.join(trans, item)).float()})

        self.quan_alpha = quan_alpha
        self.low_rate = low_rate
        self.criterion = nn.BCELoss()
        self.short_dims = [int(k) for k in self.short_center]

    def get_short_dims(self):
        return self.short_dims

    def encode_image(self, image):

        image_embed = self.clip.encode_image(image)
        long_hash = self.img_hash(image_embed)
        short_hash = {}
        for k, v in self.trans.items():
            v = v.to(long_hash.device)
            short_hash.update({k: self.img_hash.quantization(long_hash.matmul(v))})
        return long_hash, short_hash

    def encode_text(self, text):

        text_embed = self.clip.encode_text(text)
        long_hash = self.txt_hash(text_embed)
        short_hash = {}
        for k, v in self.trans.items():
            v = v.to(long_hash.device)
            short_hash.update({k: self.txt_hash.quantization(long_hash.matmul(v))})

        return long_hash, short_hash

    def forward(self, image, text):
        img_long_hash, img_short_hash = self.encode_image(image)
        txt_long_hash, txt_short_hash = self.encode_text(text)
        return img_long_hash, img_short_hash, txt_long_hash, txt_short_hash, self.long_center, self.short_center