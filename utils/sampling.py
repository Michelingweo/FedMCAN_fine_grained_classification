import numpy as np
from torchvision import datasets, transforms
import math


def iid_sample(dataset,  num_users):
    """
    Sample I.I.D. client data from given dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # the number of data samples of each client
    num_items = int(len(dataset)/num_users)
    # all_idxs contains the idx of all the data samples
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # randomly select num_items data from all_indxs without replacement
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])#update all_idxs
        # dict_users = {user_idx : set(data_idx) }
    return dict_users





