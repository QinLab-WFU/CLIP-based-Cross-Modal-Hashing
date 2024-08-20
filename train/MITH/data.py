import scipy.io as scio
from dataset.dataloader import split_data, BaseDataset
import torch
import random
from model.base.simple_tokenizer import SimpleTokenizer as Tokenizer
import numpy as np


class BasDataset(BaseDataset):
    def __init__(self, captions: dict,
            indexs: dict,
            labels: dict,
            is_train=True,
            tokenizer=Tokenizer(),
            maxWords=32,
            imageResolution=224):

        # BaseDataset._init_(captions, indexs, labels, is_train, tokenizer, maxWords, imageResolution)
        super(BasDataset, self).__init__(captions, indexs, labels, is_train, tokenizer, maxWords, imageResolution)

    def _load_text(self, index: int):
        captions = self.captions[index]
        use_cap = captions[random.randint(0, len(captions) - 1)]
        words = self.tokenizer.tokenize(use_cap)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)
        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask

    def __getitem__(self, index):
        image = self._load_image(index)
        caption, key_padding_mask = self._load_text(index)
        label = self._load_label(index)

        return image, caption, key_padding_mask, label, index


def generate_dataset(captionFile: str,
                     indexFile: str,
                     labelFile: str,
                     maxWords=32,
                     imageResolution=224,
                     query_num=2000,
                     train_num=10000,
                     seed=None,
                     ):
    if captionFile.endswith("mat"):
        captions = scio.loadmat(captionFile)["caption"]
        captions = captions[0] if captions.shape[0] == 1 else captions
    elif captionFile.endswith("txt"):
        with open(captionFile, "r") as f:
            captions = f.readlines()
        captions = np.asarray([[item.strip()] for item in captions])
    else:
        raise ValueError("the format of 'captionFile' doesn't support, only support [txt, mat] format.")
    indexs = scio.loadmat(indexFile)["index"]
    labels = scio.loadmat(labelFile)["category"]

    split_indexs, split_captions, split_labels = split_data(captions, indexs, labels, query_num=query_num, train_num=train_num, seed=seed)

    query_data = BasDataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0],
                             maxWords=maxWords, imageResolution=imageResolution, is_train=False)
    train_data = BasDataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1],
                             maxWords=maxWords, imageResolution=imageResolution)
    retrieval_data = BasDataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2],
                                 maxWords=maxWords, imageResolution=imageResolution, is_train=False)

    return train_data, query_data, retrieval_data