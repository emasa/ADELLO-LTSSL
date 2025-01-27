# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import numpy as np

import torch
from .utils import FlexDAConsistencyLoss, ADELLOHook, FlexDASupervisedLoss

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS, get_data_loader
from semilearn.algorithms.utils import SSL_Argument, str2bool


@IMB_ALGORITHMS.register('adello')
class ADELLO(ImbAlgorithmBase):
    """
        ADELLO algorithm (supports FlexDA and can be combined with complementary consistency regularization).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - adello_alpha_min (float):
                alpha_min in ADELLO
            - adello_k (float):
                k in adello
            - adello_ema_p (float):
                momentum
            - adello_mode (str):
                 Possible modes: ['adello', 'adello_gt']
            - adello_alpha_from_epoch (bool)
                Whether you use the current epoch or current iteration to compute alpha. Default: True (epoch).
            - More options below (UPDATE DOCSTRINGS)
    """

    def __init__(self, args, **kwargs):
        assert args.adello_mode in ['adello', 'adello_gt']
        self.imb_init(args.adello_alpha_min, args.adello_k, args.adello_ema_p, args.adello_mode, args.adello_alpha_from_epoch)
        super().__init__(args, **kwargs)
        assert args.algorithm not in ['mixmatch', 'meanteacher', 'pimodel'], \
            "ADELLO not supports {} as the base algorithm.".format(args.algorithm)

        self.p_data, num_samples_lb = self.compute_labeled_prior()
        self.gt_u_prior, num_samples_ulb = self.compute_unlabeled_prior()

        if ('_gt' in args.adello_mode) and (self.gt_u_prior is not None):
            self.print_fn('Using ground-truth unlabeled prior.')
            p_target = self.gt_u_prior
        elif args.adello_mode == 'adello':
            self.print_fn('Target prior will be dynamically estimated.')
            p_target = None  # ignored
        else:  # deprecated (similar to UDAL)
            self.print_fn('Assuming target prior equal to labeled prior.')
            p_target = self.p_data

        num_samples_lb = num_samples_lb if args.include_lb_to_ulb else 0.0
        lb_weight = (num_samples_lb / num_samples_ulb)
        self.print_fn(f'Portion of labeled data in unlabeled set: {lb_weight}')

        self.p_hat = torch.tensor(np.ones((self.num_classes,)) / self.num_classes).to(self.gpu)

        self.ce_loss = FlexDASupervisedLoss(alpha_min=self.alpha_min, k=self.k, use_epochs=self.alpha_from_epoch,
                                            p_data=self.p_data, p_target=p_target, target_mode=self.mode,
                                            num_samples_lb=num_samples_lb, num_samples_ulb=num_samples_ulb)
        self.consistency_loss = FlexDAConsistencyLoss(alpha_min=self.alpha_min, k=self.k,
                                                      use_epochs=self.alpha_from_epoch,
                                                      p_data=self.p_data, p_target=p_target, target_mode=self.mode,
                                                      num_samples_lb=num_samples_lb, num_samples_ulb=num_samples_ulb)

    def compute_labeled_prior(self):
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.print_fn(f'GT labeled prior: {lb_class_dist}')
        # normalize distribution
        p_data = torch.tensor(lb_class_dist / lb_class_dist.sum()).to(self.gpu)
        return p_data, len(self.dataset_dict['train_lb'])

    def compute_unlabeled_prior(self):
        ulb_class_dist = None

        if hasattr(self.dataset_dict['train_ulb'], 'targets'):
            ulb_class_dist = [0 for _ in range(self.num_classes)]
            for c in self.dataset_dict['train_ulb'].targets:
                ulb_class_dist[c] += 1
            ulb_class_dist = np.array(ulb_class_dist)
            self.print_fn(f'GT unlabeled prior: {ulb_class_dist}')
            # normalize distribution
            ulb_class_dist = torch.tensor(ulb_class_dist / ulb_class_dist.sum()).to(self.gpu)

        return ulb_class_dist, len(self.dataset_dict['train_ulb'])

    def imb_init(self, alpha_min=0.1, k=2.0, ema_p=0.999, mode='adello', alpha_from_epoch=True):
        self.alpha_min = alpha_min
        self.k = k
        self.ema_p = ema_p
        self.mode = mode
        self.alpha_from_epoch = bool(alpha_from_epoch)

    def set_hooks(self):
        self.register_hook(ADELLOHook(), "ADELLOHook", priority="NORMAL")
        super().set_hooks()

    def get_p_target(self):
        return self.consistency_loss.get_target_dist()

    def get_p_data(self):
        return self.consistency_loss.p_data

    def get_p_hat(self):
        return self.p_hat

    def compute_prob(self, u_logits, lb_in_ulb_mask=None, **kwargs):
        probs = super().compute_prob(u_logits)

        if (lb_in_ulb_mask is not None):
            mask_unl_data = lb_in_ulb_mask < 1
            if mask_unl_data.sum() > 0:  # there is true unlabeled data
                delta_p = probs[mask_unl_data].mean(dim=0)
            else:
                delta_p = None
        else:
            delta_p = probs.mean(dim=0)

        delta_p = delta_p if (delta_p is not None) else self.p_hat  # it doesn't update when delta_p is None
        self.p_hat = self.ema_p * self.p_hat + (1 - self.ema_p) * delta_p

        return probs

    def set_dataset(self):
        dataset_dict = super().set_dataset()

        if self.args.eval_pl_accuracy:
            dataset_dict['eval_ulb_privileged'] = copy.deepcopy(dataset_dict['train_ulb'])
            dataset_dict['eval_ulb_privileged'].is_ulb = False

        return dataset_dict

    def set_data_loader(self):
        loader_dict = super().set_data_loader()

        if self.args.eval_pl_accuracy:
            # add unlabeled evaluation data loader
            loader_dict['eval_ulb_privileged'] = get_data_loader(self.args,
                                                                 self.dataset_dict['eval_ulb_privileged'],
                                                                 self.args.eval_batch_size,
                                                                 data_sampler=None,
                                                                 shuffle=False,
                                                                 num_workers=self.args.num_workers,
                                                                 drop_last=False)

        return loader_dict

    def get_alpha_factor(self):
        return self.consistency_loss.get_alpha_factor()

    @staticmethod
    def get_argument():
        return [
            # ADELLO flags
            SSL_Argument('--adello_mode', str, 'adello'),
            SSL_Argument('--adello_alpha_from_epoch', str2bool, False),
            SSL_Argument('--adello_alpha_min', float, 0.1),
            SSL_Argument('--adello_k', float, 2.0),
            SSL_Argument('--adello_ema_p', float, 0.999),

            # for debugging purposes
            SSL_Argument('--eval_pl_accuracy', str2bool, False),
        ]
