import torch
import logging
import torch.nn as nn

from model.modelbase import Baseclip, weights_init_kaiming


class HashLayer(nn.Module):
    LINEAR_EMBED = 128
    SIGMOID_ALPH = 10

    def __init__(self, inputDim=2048, outputDim=64):
        super(HashLayer, self).__init__()
        self.fc = nn.Linear(inputDim, self.LINEAR_EMBED)
        self.fc.apply(weights_init_kaiming)
        self.hash_list = nn.ModuleList([nn.Linear(self.LINEAR_EMBED, 2) for _ in range(outputDim)])
        for item in self.hash_list:
            item.apply(weights_init_kaiming)

    def forward(self, data):
        embed = self.fc(data)
        embed = torch.relu(embed)

        softmax_list = [torch.softmax(item(embed), dim=-1) for item in self.hash_list]

        return softmax_list


class MDCMHT(Baseclip):

    def __init__(self,
                 outputDim=64,
                 clipPath="./ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 is_train=True):
        super(MDCMHT, self).__init__(outputDim=outputDim, clipPath=clipPath, writer=writer,
                                    saveDir=saveDir, logger=logger, is_train=is_train)

        self.image_hash = HashLayer(inputDim=self.embedDim, outputDim=outputDim)
        self.text_hash = HashLayer(inputDim=self.embedDim, outputDim=outputDim)

    def forward(self, image, text):
        return self.encode_image(image), self.encode_text(text)