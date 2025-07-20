import torch.nn as nn


def calc_loss_M2SDF(pred, target, metrics):

    criterion_MSE = nn.MSELoss(reduction='mean')
    loss = criterion_MSE(pred, target)
    metrics['loss'] += loss

    return loss

