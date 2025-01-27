import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data

import torchvision
import torch
from torchvision.transforms import transforms
from semilearn.datasets.augmentation import RandAugment, CutoutAbs
from semilearn.datasets.utils import get_onehot


def get_small_imagenet(args, alg, name, num_labels, num_classes, data_dir='./data',
                       include_lb_to_ulb=True, return_strong_labeled_set=False, seed=0):
    img_size = args.img_size

    assert img_size == 32 or img_size == 64, 'img size should only be 32 or 64!!!'
    root = os.path.join(data_dir, f'{name}-{img_size}x{img_size}'.lower())
    base_dataset = SmallImageNet(root, img_size, True)
    # compute dataset mean and std
    dataset_mean = (0.48109809, 0.45747185, 0.40785507)  # np.mean(base_dataset.data, axis=(0, 1, 2)) / 255
    print(dataset_mean)

    dataset_std = (0.26040889, 0.2532126, 0.26820634)  # np.std(base_dataset.data, axis=(0, 1, 2)) / 255
    print(dataset_std)

    # construct data augmentation
    # Augmentations.
    transform_weak = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 4),  # includes CutOut as well
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    # select labeled data and construct labeled dataset
    num_classes = len(set(base_dataset.targets))
    num_data_per_cls = [0 for _ in range(num_classes)]
    for l in base_dataset.targets:
        num_data_per_cls[l] += 1

    num_data = int(sum(num_data_per_cls))
    percentage = num_labels / num_data
    num_labeled_data_per_cls = [int(np.around(n * percentage)) for n in num_data_per_cls]
    num_unlabeled_data_per_cls = [n-int(np.around(n * percentage)) for n in num_data_per_cls]

    print('total number of labeled data is ', sum(num_labeled_data_per_cls))
    print("lb count: {}".format(num_labeled_data_per_cls))
    print("ulb count: {}".format(num_unlabeled_data_per_cls))
    print("total count: {}".format(num_data_per_cls))

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, num_labeled_data_per_cls, num_unlabeled_data_per_cls, num_classes, seed, include_lb_to_ulb=include_lb_to_ulb)

    train_labeled_dataset = SmallImageNet(root, img_size, True, ulb=False, alg=alg, transform=transform_weak, lb_index=train_labeled_idxs, percentage=percentage)
    train_unlabeled_dataset = SmallImageNet(root, img_size, True, ulb=True, alg=alg, transform=transform_weak, lb_index=train_unlabeled_idxs, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb)
    test_dataset = SmallImageNet(root, img_size, False, ulb=False, alg=alg, transform=transform_val)

    if return_strong_labeled_set:
        train_strong_labeled_dataset = SmallImageNet(root, img_size, True, ulb=False, alg=alg, transform=transform_strong, lb_index=train_labeled_idxs, percentage=percentage)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_strong_labeled_dataset
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, num_classes, seed=0, include_lb_to_ulb=True):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        if seed != 0:
            np.random.shuffle(idxs)

        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])

        if include_lb_to_ulb:
            train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        else:
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]: n_labeled_per_class[i] + n_unlabeled_per_class[i]])

    return train_labeled_idxs, train_unlabeled_idxs


class SmallImageNet(data.Dataset):
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10']
    test_list = ['val_data']

    def __init__(self, file_path, imgsize, train, ulb=False, alg=None, transform=None, strong_transform=None, onehot=False, percentage=-1, include_lb_to_ulb=True, lb_index=None):
        # assert imgsize == 32 or imgsize == 64, 'imgsize should only be 32 or 64'
        self.imgsize = imgsize
        self.train = train

        self.onehot = onehot
        self.data = []
        self.targets = []
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # extra flags
        self.alg = alg
        self.is_ulb = ulb
        self.percentage = percentage
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index

        self.transform = transform
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"

        # now load the picked numpy arrays
        for filename in downloaded_list:
            file = os.path.join(file_path, filename)
            with open(file, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])  # Labels are indexed from 1
        self.targets = [i - 1 for i in self.targets]
        self.data = np.vstack(self.data).reshape((len(self.targets), 3, self.imgsize, self.imgsize))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if lb_index is not None:
            self.data = self.data[lb_index]
            self.targets = np.array(self.targets)[lb_index]

    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        return img, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target = self.__sample__(idx)

        if self.transform is None and self.strong_transform is None:
            return {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            if self.transform is not None:
                img_w = self.transform(img)
            else:
                img_w = None

            if not self.is_ulb:
                if self.strong_transform is not None:
                    img_s = self.strong_transform(img)
                else:
                    img_s = None

                img_default = img_w if (img_w is not None) else img_s
                out = {'idx_lb': idx, 'x_lb': img_default, 'y_lb': target}
                if (img_w is not None) and (img_s is not None):
                    out['x_lb_s'] = img_s

                return out
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': idx, 'x_ulb_w':img_w}
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)}
                else:
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)}

    def __len__(self):
        return len(self.data)