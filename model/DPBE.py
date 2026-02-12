import logging

from torch import nn

from model.modelbase import Baseclip, Pre_Layer
from stochman.nnj import L2Norm
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class MDPBE(Baseclip):

    def __init__(self,
                 use_lam=True,
                 outputDim=64,
                 clipPath="./ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 is_train=True):
        super(MDPBE, self).__init__(outputDim=outputDim, clipPath=clipPath, writer=writer,
                                    saveDir=saveDir, logger=logger, is_train=is_train)

        if use_lam:
            self.image_hash = nn.Sequential(nn.Linear(self.embedDim, outputDim), L2Norm())
            self.text_hash = nn.Sequential(nn.Linear(self.embedDim, outputDim), L2Norm())

    def encoding(self, image, text):
        image_embed = self.clip.encode_image(image)
        text_embed = self.clip.encode_text(text)
        return image_embed, text_embed

    def hashing(self, image_embed, text_embed):
        image_hash = self.image_hash(image_embed)
        text_hash = self.text_hash(text_embed)
        return image_hash, text_hash

    def forward(self, image, text):

        image_embed = self.clip.encode_image(image)
        text_embed = self.clip.encode_text(text)
        image_embed = self.image_hash(image_embed)
        text_embed = self.text_hash(text_embed)
        return image_embed, text_embed
