# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import numpy as np 
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import get_onehot


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 lb_in_ulb_mask=None,
                 *args, 
                 **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
        self.lb_in_ulb_mask = lb_in_ulb_mask

        self.transform = transform
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"
    
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

        if self.lb_in_ulb_mask is None:
            return img, target
        else:
            return img, target, self.lb_in_ulb_mask[idx]

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        if self.lb_in_ulb_mask is None:
            img, target = self.__sample__(idx)
            lb_in_ulb_mask = None
        else:
            img, target, lb_in_ulb_mask = self.__sample__(idx)

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
                    result_dict = {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    result_dict = {'idx_ulb': idx, 'x_ulb_w':img_w}
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    result_dict = {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    result_dict = {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch' or self.alg == 'overclustered' or self.alg == 'fixcutmixmatch':
                    result_dict = {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)}
                else:
                    result_dict = {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)}

                if self.lb_in_ulb_mask is not None:
                    result_dict['lb_in_ulb_mask'] = lb_in_ulb_mask

                return result_dict



    def __len__(self):
        return len(self.data)