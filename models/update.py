#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset

from torch.optim import Adam

from utils.setting import args_parser


args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


def to_one_hot(labels, d_classes):
    """
    :param labels: integer tensor of shape (d_batch, *)
    :param d_classes : number of classes
    :return: float tensor of shape (d_batch, *, d_classes), one hot representation of the labels
    """
    return torch.zeros(*labels.size(), d_classes, device=labels.device).scatter_(-1, labels.unsqueeze(-1), 1)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, text = self.dataset[self.idxs[item]]
        return image, label, text


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, texts) in enumerate(self.ldr_train):
                labels = torch.tensor(labels)

                images, labels, texts = images.to(args.device), labels.to(args.device), texts.to(args.device)
                labels_oh = to_one_hot(labels,args.num_classes).to(args.device)
                net.zero_grad()
                log_probs = net(images, texts)

                loss = self.loss_func(log_probs, labels_oh)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)





