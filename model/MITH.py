from collections import OrderedDict
from typing import Tuple, Union
import math
import torch
import torch.nn.functional as F
from torch import nn

from model.base.model import LayerNorm, QuickGELU, convert_weights, VisionTransformer, CLIP


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x, attn_mask=None, key_padding_mask=None):
        return self.attn(x, x, x, need_weights=True, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, attn_weight = self.attention(self.ln_1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weight


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        for block in self.resblocks:
            x, attn_weight = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return x, attn_weight


class ViT(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__(input_resolution=input_resolution, patch_size=patch_size, width=width, layers=layers,
                         heads=heads, output_dim=output_dim)

        self.transformer = Transformer(width, layers, heads)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_weight = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        if self.proj is not None:
            x = torch.bmm(x, self.proj.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = x.permute(1, 0, 2)
        seq_tokens = x[1:]
        cls_token = x[0]
        attn_weight = attn_weight[:, 0, 1:]

        return seq_tokens, attn_weight, cls_token  # LND', NS, ND'


class CLIP1(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__(embed_dim=embed_dim, image_resolution=image_resolution, vision_layers=vision_layers,
                         vision_width=vision_width, vision_patch_size=vision_patch_size, context_length=context_length,
                         vocab_size=vocab_size, transformer_width=transformer_width, transformer_heads=transformer_heads,
                         transformer_layers=transformer_layers)

        self.visual = ViT(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=self.vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )

    def encode_text(self, text, key_padding_mask):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x.permute(1, 0, 2)

        attn_mask = self.build_attention_mask(x.shape[0])
        key_padding_mask = key_padding_mask

        x, attn_weight = self.transformer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        EOS = text.argmax(dim=-1)  # N

        attn_weight = attn_weight[torch.arange(x.shape[1]), EOS]
        attn_weight[torch.arange(x.shape[1]), EOS] = 0

        new_key_padding_mask = key_padding_mask + (text == 49407)

        x = self.ln_final(x).type(self.dtype)
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.text_projection.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        x = x.permute(1, 0, 2)

        seq_tokens = x
        EOS_token = x[EOS, torch.arange(x.shape[1])]

        return seq_tokens, attn_weight, new_key_padding_mask, EOS_token

    def forward(self, image, text, key_padding_mask):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, key_padding_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def build_model(state_dict: dict):

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP1(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)

    return model


def load_download_clip(clip_path: str) -> tuple:
    try:
        model = torch.jit.load(clip_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(clip_path, map_location="cpu")

    return build_model(state_dict)


class ResidualMLPs(nn.Module):
    """
    Residual MLPs
    ***D - ***D
    """

    def __init__(self, org_dim, dropout=0., num_layers=2, activation='relu'):
        super().__init__()
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(org_dim, 4 * org_dim),
            self.activation_layer,
            nn.Dropout(p=dropout),
            nn.Linear(4 * org_dim, org_dim),
        ) for i in range(num_layers))

        self.lns = nn.ModuleList(nn.LayerNorm(org_dim) for i in range(num_layers))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x + self.mlps[i](self.lns[i](x))
        return x


class PositionalEncoding(nn.Module):
    """
    Sin-cos position embedding
    LND - LND
    """

    def __init__(self, d_model, dropout=0., max_len=128):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # [max-length, 1, d_model]
        pe = pe / (d_model ** 0.5)  #
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: LND
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BitwiseHashing(nn.Module):
    """
    Bitwise hashing layer
    KND - NK
    """

    def __init__(self, org_dim, k_bits=32, ):
        super().__init__()
        self.k = k_bits
        self.fc_list = nn.ModuleList(nn.Linear(org_dim, 1) for _ in range(k_bits))

    def forward(self, x):
        # x: KND
        x = [self.fc_list[i](x[i, :, :]) for i in range(self.fc_list.__len__())]
        x = torch.stack(x)  # K,N,1
        x = x.permute(1, 0, 2)  # N,K,1
        x = torch.squeeze(x)  # N,K
        return torch.tanh(x)


class GlobalConceptLearning(nn.Module):
    """
    Concept Learning
    ***D - ***K
    """

    def __init__(self, k_concept, org_dim, dropout=0., activation='relu', res_mlp_layers=0):
        super().__init__()

        if res_mlp_layers != 0:
            self.mlp = ResidualMLPs(org_dim=org_dim, dropout=dropout, num_layers=res_mlp_layers, activation=activation)
        else:
            self.mlp = nn.Identity()

        self.common_concept_embedding = nn.Linear(org_dim, k_concept, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        return x, torch.tanh(self.common_concept_embedding(x))


class LocalizedTokenAggregation(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def my_top_k(self, top_k, x):
        val = torch.topk(x, k=top_k, dim=-1).values
        val_min = torch.min(val, dim=-1).values
        val_min = val_min.unsqueeze(-1).repeat(1, 1, x.shape[2])

        ge_ = torch.ge(x, val_min)  # torch.ge: Compute input >= other

        neg_inf = torch.zeros_like(x)
        neg_inf = neg_inf.fill_(float("-inf"))
        result = torch.where(ge_, x, neg_inf)
        return result

    def gen_top_k_label(self, top_k, x):
        top_k_val_with_neg_inf = self.my_top_k(top_k, x)

        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        pseudo_label = torch.where(top_k_val_with_neg_inf > 0, ones, zeros)

        return pseudo_label, top_k_val_with_neg_inf

    def forward(self, x, token_concept_embedding, key_padding_mask=None):
        # x: LND
        # token_concept_embedding: LNK(K concept)
        # return: KND
        sim = token_concept_embedding.detach()  # no grad need.

        if key_padding_mask is not None:
            # set sim to '-inf' by key padding mask
            key_pad = torch.where(key_padding_mask, float('-inf'), 0.)  # NL
            key_pad = key_pad.unsqueeze(dim=1).repeat(1, sim.shape[2], 1)  # NKL
            key_pad = key_pad.permute(2, 0, 1)  # LNK

            sim += key_pad

        # make neg_value to neg_inf
        neg_inf = torch.zeros_like(sim)
        neg_inf = neg_inf.fill_(float("-inf"))
        sim = torch.where(sim > 0, sim, neg_inf)

        # select top_k for each token
        pseudo_label, sim = self.gen_top_k_label(self.top_k, sim)

        # softmax
        sim = torch.softmax(sim, dim=0)  # sim: LNK
        sim = torch.where(torch.isnan(sim), torch.full_like(sim, 0), sim)  # sim: LNK

        # x: LND
        # sim: LNK
        merge_val = torch.bmm(
            sim.permute(1, 2, 0),  # NKL
            x.permute(1, 0, 2)  # NLD
        )  # NKD
        merge_val = merge_val.permute(1, 0, 2)  # NKD - KND
        return merge_val, pseudo_label  # KND, LNK


class LocalConceptTransforming(nn.Module):
    def __init__(self, clip_embed_dim, k_bits, transformer_layers, dropout, top_k):
        super().__init__()
        self.lta = LocalizedTokenAggregation(top_k=top_k)
        self.position = PositionalEncoding(clip_embed_dim, dropout=dropout, max_len=k_bits)
        self.transformer = Transformer(
            width=clip_embed_dim,
            layers=transformer_layers,
            heads=clip_embed_dim // 64,
        )
        self.hashing = BitwiseHashing(org_dim=clip_embed_dim, k_bits=k_bits)

    def forward(self, x, token_concept_embedding, key_padding_mask=None):
        # x: LND
        # token_concept_embedding: LNK (K concept)
        x, pseudo_label = self.lta(x, token_concept_embedding, key_padding_mask)
        x, _ = self.transformer(self.position(x))
        return self.hashing(x), pseudo_label, x


class HashingModel(nn.Module):
    """
    Hashing model
    """
    def __init__(self, clip_embed_dim=512, args=None):
        super().__init__()

        self.k_bits = k_bits = args.output_dim
        self.dropout = dropout = args.dropout
        self.transformer_layers = transformer_layers = args.transformer_layers
        self.activation = activation = args.activation
        self.top_k_label = top_k_label = args.top_k_label
        self.res_mlp_layers = res_mlp_layers = args.res_mlp_layers

        # share weight.
        self.gcl_i = self.gcl_t = GlobalConceptLearning(k_concept=k_bits, org_dim=clip_embed_dim, dropout=dropout,
                                                        activation=activation, res_mlp_layers=res_mlp_layers)

        self.lct_i = LocalConceptTransforming(clip_embed_dim=clip_embed_dim, k_bits=k_bits,
                                              transformer_layers=transformer_layers, dropout=0,
                                              top_k=top_k_label)
        self.lct_t = LocalConceptTransforming(clip_embed_dim=clip_embed_dim, k_bits=k_bits,
                                              transformer_layers=transformer_layers, dropout=0,
                                              top_k=top_k_label)

        self.img_concept_proj = nn.Linear(clip_embed_dim, clip_embed_dim)
        self.txt_concept_proj = nn.Linear(clip_embed_dim, clip_embed_dim)

    def forward(self, img_tokens, txt_tokens, img_cls, txt_eos, key_padding_mask):
        output_dict = {}

        gcl_i = self.gcl_i
        gcl_t = self.gcl_t
        lct_i = self.lct_i
        lct_t = self.lct_t

        res_img_cls, img_cls_hash = gcl_i(img_cls)
        res_txt_cls, txt_cls_hash = gcl_t(txt_eos)

        output_dict['img_cls_hash'] = img_cls_hash
        output_dict['txt_cls_hash'] = txt_cls_hash

        output_dict['res_img_cls'] = F.normalize(res_img_cls, dim=-1)
        output_dict['res_txt_cls'] = F.normalize(res_txt_cls, dim=-1)

        tokens_hash_i, _, trans_tokens_i = lct_i(img_tokens, gcl_i(img_tokens)[1].detach())
        tokens_hash_t, _, trans_tokens_t = lct_t(txt_tokens, gcl_t(txt_tokens)[1].detach(), key_padding_mask)

        output_dict['img_tokens_hash'] = tokens_hash_i
        output_dict['txt_tokens_hash'] = tokens_hash_t

        output_dict['trans_tokens_i'] = F.normalize(self.img_concept_proj(trans_tokens_i), dim=-1)
        output_dict['trans_tokens_t'] = F.normalize(self.txt_concept_proj(trans_tokens_t), dim=-1)

        return output_dict


class MITH(nn.Module):
    def __init__(self, args=None):
        super(MITH, self).__init__()
        self.args = args
        self.clip = load_download_clip(self.args.clip_path)
        self.hash = HashingModel(clip_embed_dim=512, args=args)

    def forward(self, image, text, key_padding_mask):
        img_tokens, _, img_cls = self.clip.encode_image(image)
        txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
        output_dict = self.hash(img_tokens, txt_tokens, img_cls, txt_eos, new_key_padding_mask)
        return output_dict
