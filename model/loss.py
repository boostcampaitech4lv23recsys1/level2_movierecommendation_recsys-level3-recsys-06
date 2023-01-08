import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def ce_loss(output, target):
    return F.cross_entropy(output, target, ignore_index = 0)