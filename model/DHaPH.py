import logging
from model.modelbase import Baseclip


class MDHaPH(Baseclip):

    def __init__(self,
                 outputDim=64,
                 clipPath="./ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 is_train=True):
        super(MDHaPH, self).__init__(outputDim=outputDim, clipPath=clipPath, writer=writer,
                                    saveDir=saveDir, logger=logger, is_train=is_train)

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        return image_embed, text_embed