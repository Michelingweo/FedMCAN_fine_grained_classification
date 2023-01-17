
import os
import math
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
# import spacy
# spacy.load('en_core_web_sm')
from spacy.lang.en import English
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import Dataset, DataLoader
from utils.setting import args_parser


args = args_parser()
DATASETS_PATH = './data/'

META_PATH = os.path.join(DATASETS_PATH, 'metadata.pth')

class OxfordFlowers102(Dataset):
    """If should_pad is True, need to also provide a pad_to_length. Padding also adds <START> and <END> tokens to captions."""
    def __init__(self, image, split='all', d_image_size=64, transform=None, should_pad=True, pad_to_length=256, no_start_end=False, **kwargs):
        super().__init__()

        assert split in ('all', 'train_val', 'test')
        assert d_image_size in (64, 128, 256)
        if should_pad:
            assert pad_to_length >= 3 # <START> foo <END> need at least length 3.

        self.split = split
        self.d_image_size = d_image_size
        self.transform = transform
        self.should_pad = should_pad
        self.pad_to_length = pad_to_length
        self.no_start_end = no_start_end

        metadata = torch.load(META_PATH)

        # labels
        self.img_id_to_class_id = metadata['img_id_to_class_id']

        self.class_id_to_class_name = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
        self.class_name_to_class_id = {c: i for i, c in enumerate(self.class_id_to_class_name)}

        # captions
        self.img_id_to_encoded_caps = metadata['img_id_to_encoded_caps']
        self.word_id_to_word = metadata['word_id_to_word']
        self.word_to_word_id = metadata['word_to_word_id']
        self.pad_token     = 0
        self.start_token   = 1747
        self.end_token     = 1748
        self.unknown_token = 1749

        self.d_vocab = metadata['num_words']
        self.num_captions_per_image = metadata['num_captions_per_image']

        nlp = English()
        self.tokenizer = nlp.tokenizer # Create a Tokenizer with the default settings for English including punctuation rules and exceptions

        # images
        self.img_ids = metadata['img_ids']


        self.imgs = image

    def encode_caption(self, cap):
        words = [token.text for token in self.tokenizer(cap)]
        return [self.word_to_word_id.get(word, self.unknown_token) for word in words]

    def decode_caption(self, cap):
        if isinstance(cap, torch.Tensor):
            cap = cap.tolist()
        return ' '.join([self.word_id_to_word[word_id] for word_id in cap])

    def pad_caption(self, cap):
        max_len = self.pad_to_length - 2 # 2 since we need a start token and an end token.
        cap = cap[:max_len] # truncate to maximum length.
        cap = [self.start_token] + cap + [self.end_token]
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap = list(cap)
        padding = list(padding)

        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def pad_without_start_end(self, cap):
        cap = cap[:self.pad_to_length] # truncate to maximum length.
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx][0]
        if self.transform:
            img = self.transform(img)

        img_id = self.img_ids[idx]
        class_id = int(self.img_id_to_class_id[img_id]) - 1
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        cap_idx = torch.randint(low=0, high=self.num_captions_per_image, size=(1,)).item()
        encoded_cap = encoded_caps[cap_idx]
        if self.should_pad:
            if self.no_start_end:
                encoded_cap, cap_len = self.pad_without_start_end(encoded_cap)
            else:
                encoded_cap, cap_len = self.pad_caption(encoded_cap)
            return img, class_id, encoded_cap
        return img, class_id, encoded_cap



def get_oxford_flowers_102(image_set, split='train_val',args=args, d_batch=4, should_pad=True, shuffle=True, **kwargs):

    train_set = OxfordFlowers102(image=image_set, split=split, transform=None, should_pad=should_pad, **kwargs)

    if split=='train_val':
        d_batch = args.local_bs
    elif split == 'test':
        d_batch = args.bs

    if not should_pad:
        def collate_fn(samples):
            imgs, class_ids, caps = zip(*samples)
            imgs = torch.stack(imgs)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
            return imgs, class_ids, caps
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=0, pin_memory=True)
    return train_set, train_loader