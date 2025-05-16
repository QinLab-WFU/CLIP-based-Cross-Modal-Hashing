import logging
import torch.nn as nn

from model.modelbase import Baseclip


class Pre_Layer(nn.Module):
    def __init__(self, inputdim=2048, nb_class=64):
        super(Pre_Layer, self).__init__()
        self.fc = nn.Linear(inputdim, nb_class)

    def forward(self, data):
        pre = self.fc(data)
        return pre


class MDNPH(Baseclip):

    def __init__(self,
                 outputDim=64,
                 num_classes=80,
                 clipPath="./ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 is_train=True):
        super(MDNPH, self).__init__(outputDim=outputDim, clipPath=clipPath, writer=writer,
                                    saveDir=saveDir, logger=logger, is_train=is_train)

        self.image_pre = Pre_Layer(inputdim=self.embedDim, nb_class=num_classes)
        self.text_pre = Pre_Layer(inputdim=self.embedDim, nb_class=num_classes)

    def encode_image(self, image):
        image_fea = self.clip.encode_image(image)
        image_embed = self.image_hash(image_fea)

        image_pre = self.image_pre(image_fea)

        return image_embed, image_pre

    def encode_text(self, text):
        text_fea = self.clip.encode_text(text)

        text_embed = self.text_hash(text_fea)
        text_pre = self.text_pre(text_fea)

        return text_embed, text_pre

    def forward(self, image, text):
        image_embed, image_pre = self.encode_image(image)
        text_embed, text_pre = self.encode_text(text)
        return image_embed, image_pre, text_embed, text_pre