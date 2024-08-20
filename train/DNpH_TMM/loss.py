import torch
import numpy as np


def qmi_loss(images, texts, targets, sigma=3, M=0, eps=1e-8, use_cosine=True, use_square_clamp=True):

    """
    :param images: the images representation
    :param texts: the texts representation
    :param targets: the sample labels
    :param sigma: scaling factor for the Gaussian kernel (if used)
    :param eps: a small number used to ensure the stability of the cosine similarity
    :param M: number of information needs (assuming that each one is equiprobable)
    :param use_cosine: Set to true to use QSMI, otherwise QMI is used
    :param use_square_clamp: Set to true to used the square clamping method
    """

    if use_cosine:
        images = images / (torch.sqrt(torch.sum(images ** 2, dim=1, keepdim=True)) + eps)
        Y = torch.mm(images, images.t())
        Y = 0.5 * (Y + 1)

        texts = texts / (torch.sqrt(torch.sum(texts ** 2, dim=1, keepdim=True)) + eps)
        T = torch.mm(texts, texts.t())
        T = 0.5 * (T + 1)

        YT = torch.mm(images, texts.t())
        YT = 0.5 * (YT + 1)
    else:
        Y = squared_pairwise_distances(images)
        Y = torch.exp(-Y / (2 * sigma ** 2))

        T = squared_pairwise_distances(texts)
        T = torch.exp(-T / (2 * sigma ** 2))

        YT = squared_pairwise_distances(images, texts)
        YT = torch.exp(-YT / (2 * sigma ** 2))
    # Get the indicator matrix \Delta
    targets = targets.float()
    D = targets.mm(targets.transpose(0, 1)) > 0
    D = D.type(torch.cuda.FloatTensor)

    if M == 0:
        M = D.size(1) ** 2 / torch.sum(D)

    if use_square_clamp:
        Qy_in = (D * Y - 1) ** 2
        Qy_btw = (1.0 / M) * Y ** 2

        Qt_in = (D * T - 1) ** 2
        Qt_btw = (1.0 / M) * T ** 2

        Qyt_in = (D * YT - 1) ** 2
        Qyt_btw = (1.0 / M) * YT ** 2
        # Minimize clamped loss
        loss = torch.sum(Qy_in + Qy_btw + Qt_in + Qt_btw + Qyt_in + Qyt_btw)
    else:
        Qy_in = D * Y
        Qy_btw = (1.0 / M) * Y

        Qt_in = D * T
        Qt_btw = (1.0 / M) * T

        Qyt_in = D * YT
        Qyt_btw = (1.0 / M) * YT
        # Maximize QMI/QSMI
        loss1 = -torch.sum(Qy_in - Qy_btw)
        loss2 = -torch.sum(Qt_in - Qt_btw)
        loss3 = -torch.sum(Qyt_in - Qyt_btw)
        loss = loss1 + loss2 + loss3

    return loss


def squared_pairwise_distances(a, b=None):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = torch.sum(a ** 2, dim=1)
    bb = torch.sum(b ** 2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf)

    return dists