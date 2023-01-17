#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8

import torch
from torch import nn
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.update import to_one_hot


def test(net_g, datatest, args):
    net_g.to(args.device)
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (images, labels, texts) in enumerate(data_loader):
        labels = torch.tensor(labels)

        images, labels, texts = images.to(args.device), labels.to(args.device), texts.to(args.device)
        labels_oh = to_one_hot(labels, args.num_classes).to(args.device)

        # images, labels = images.to(args.device), labels.to(args.device)
        net_g.zero_grad()
        log_probs = net_g(images, texts)

        log_probs = log_probs.to(args.device)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, labels_oh, reduction='sum').item()
        # get the index of the max log-probability


        y_pred = log_probs.data.max(1, keepdim=True)[1]

        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


