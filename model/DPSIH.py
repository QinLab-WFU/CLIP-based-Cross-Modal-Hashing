import numpy as np
import torch
import logging
import torch.nn as nn
from model.modelbase import Baseclip, weights_init_kaiming


def l2norm(x):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class DSIE(nn.Module):
    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(DSIE, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.num_embeds > 1:
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.layer_norm(out + residual)
        return out, attn, residual


class MDPSIH(Baseclip):

    def __init__(self,
                 outputDim=64,
                 clipPath="/home/user/Projects/ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 num_embeds=2,
                 dropout=0.0,
                 is_train=True,
                 use_part=True):

        self.use_part = use_part
        super(MDPSIH, self).__init__(outputDim=outputDim, clipPath=clipPath, writer=writer,
                                    saveDir=saveDir, logger=logger, is_train=is_train)

        dim = self.dim = [768, 512]
        self.wei = 7
        self.DSIE_i = DSIE(num_embeds, dim[0], outputDim, dim[0] // 2, dropout)
        self.DSIE_t = DSIE(num_embeds, dim[1], outputDim, dim[1] // 2, dropout)

    def encode_image(self, image):
        image_fea, image_re = self.clip.encode_image(image)
        embed_i = self.image_hash(image_fea)
        embed_i, _, _ = self.DSIE_i(embed_i, image_re)
        embed_i = l2norm(embed_i)
        return embed_i

    def encode_text(self, text):
        text_fea, text_re = self.clip.encode_text(text)
        embed_t = self.text_hash(text_fea)
        embed_t, _, _ = self.DSIE_t(embed_t, text_re)
        embed_t = l2norm(embed_t)

        return embed_t


    def forward(self, image, text):
        image_fea, image_re = self.clip.encode_image(image)
        text_fea, text_re = self.clip.encode_text(text)

        embed_i = self.image_hash(image_fea)
        embed_t = self.text_hash(text_fea)

        embed_i, attn_i, resi_i = self.DSIE_i(embed_i, image_re)
        embed_t, attn_t, resi_t = self.DSIE_t(embed_t, text_re)
        embed_i, embed_t = l2norm(embed_i), l2norm(embed_t)

        return embed_i, embed_t, attn_i, attn_t, resi_i, resi_t


if __name__ == "__main__":
    model = MDPSIH(outputDim=16).float()
    image = torch.randn(2, 3, 224, 224)
    text = torch.tensor([[49406, 1237, 2029, 2153, 537, 320, 2029, 2308, 631, 10518,
                       593, 320, 1579, 36679, 525, 320, 17195, 997, 631, 15546,
                       536, 6446, 2403, 1180, 537, 320, 1400, 9648, 22769, 539,
                       320, 49407],
                      [49406, 1237, 1579, 4337, 631, 2862, 536, 518, 1253, 539,
                       320, 10625, 536, 320, 3720, 1237, 5046, 4337, 593, 1746,
                       4804, 320, 2010, 4909, 593, 518, 2102, 530, 518, 5994,
                       49407, 0]], dtype=torch.int64)
    embed_i = model.encode_image(image)
    embed_t = model.encode_text(text)

    print(embed_i.shape, embed_t.shape)
