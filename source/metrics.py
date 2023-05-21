import torch
import numpy as np


def rate(pred, target, mask, threshold):
    error = torch.abs(target - pred)
    error = (error < threshold) & mask
    error = torch.flatten(error, 1).float().sum(-1)
    count = torch.flatten(mask, 1).sum(-1)
    rate = error / count * 100
    return torch.mean(rate[count > 0])


def epe(pred, target, mask):
    error = torch.abs(target - pred) * mask
    error = torch.flatten(error, 1).sum(-1)
    count = torch.flatten(mask, 1).sum(-1)
    epe = error / count
    return epe[count > 0]


def bad(pred_depth, true_depth, thresholds):
    abs_diff = torch.abs(pred_depth - true_depth)
    sorted_diff = torch.sort(abs_diff.view(-1))[0]
    num_pixels = abs_diff.numel()
    bad_pixels = int(num_pixels * thresholds)
    bad = torch.mean(sorted_diff[-bad_pixels:] ** 2)
    return bad


def rmse(predictions, targets):
    rmse = ((predictions - targets) ** 2).mean().sqrt()
    return rmse
