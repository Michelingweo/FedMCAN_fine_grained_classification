#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import pandas as pd

from utils.sampling import iid_sample
from utils.setting import args_parser
from utils.dataset import get_oxford_flowers_102
from models.update import LocalUpdate
from models.aggregation import FedAvg
from models.test import test
from models.model import MCAN

# remove the comment if OMP error
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)
    print(f'rand seed:{args.seed}')
    print(f'communication round:{args.epochs}')
    print(f'local epochs:{args.local_ep}')
    print(f'dataset:{args.dataset}')
    print(f'model:{args.model}')
    print(f'training batch size:{args.local_bs}')
    print(f'comment:{args.comment}')


    # load dataset and split users

    if args.dataset == 'flower':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(256,256)),
                                   ])
        # 7370 train images
        train_set = datasets.Flowers102(root='./data',split='train', download=True, transform=trans, target_transform=transforms.Lambda(lambda x: (x / 255.) * 2. - 1.))
        # 819 test images
        test_set = datasets.Flowers102(root='./data',split='test', download=True, transform=trans, target_transform=transforms.Lambda(lambda x: (x / 255.) * 2. - 1.))

        # 7370 train images
        train_set, train_loader = get_oxford_flowers_102(image_set=train_set,split='train_val', d_batch=args.local_bs)
        # 819 test images
        test_set, test_loader = get_oxford_flowers_102(image_set=test_set, split='test', d_batch=args.bs)

        if args.iid:
            dict_users = iid_sample(train_set, args.num_users)
        else:
            dict_users = noniid_sample(train_set, args.num_users)
    else:
        exit('Error: unrecognized dataset')


    img_size = train_set[0][0].shape

    print(f'image size:{img_size}')
    print(int(img_size[-1]))
    print('Dataset is loaded.')



    # build model
    if args.model == 'mcan':

        net_glob = MCAN(embed_dim=256, num_heads=4, num_classes=args.num_classes, num_layers=2, text_dim=256, image_dim=img_size[-1])

    else:
        exit('Error: unrecognized model')

    print('Model is loaded.')

    print(net_glob)
    net_glob.train()


    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    log_loss = []


    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:

            local = LocalUpdate(args=args, dataset=train_set, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('Round {:3d}, Average loss {:.2f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # save loss log
        if iter % 30 == 0:
            net_glob.eval()
            acc_train, loss_ = test(net_glob, train_set, args)
            acc_test, loss_t = test(net_glob, test_set, args)
            print(f'Round{iter}, Train Acc:{acc_train:.2f}')
            print(f'Round{iter}, Test Acc:{acc_test:.2f}')

            net_glob.train()

        else:
            acc_train = None
            acc_test = None
        log_loss.append([iter, loss_avg, acc_train, acc_test])



    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_C{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.comment))

    # testing
    net_glob.eval()


    print('======evaluation begin======')
    acc_train, loss_train = test(net_glob, train_set, args)
    acc_test, loss_test = test(net_glob, test_set, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    log_loss.append([args.epochs, loss_test, acc_train, acc_test])
    df = pd.DataFrame(log_loss, columns=['round', 'loss', 'acc_train', 'acc_test'])
    df.to_excel('./save/fed_{}_{}_{}_C{}_{}.xlsx'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                args.comment), index=False)

    print('=======END========')