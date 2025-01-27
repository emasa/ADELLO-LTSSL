# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import concat_all_gather


class DistAlignEMAHook(Hook):
    """
    Distribution Alignment Hook for conducting distribution alignment
    """
    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None,
                 smooth_p_target=False, smooth_p_model=False):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum
        self.eps = 1e-6

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)    
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None
        # distribution smoothing
        self.smooth_p_target = smooth_p_target
        self.smooth_p_model = smooth_p_model
        self.smooth_factor = None

    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)

        # optionally smooth distributions
        p_target, p_model = self.apply_distribution_smoothing(self.p_target, self.p_model)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (p_target + self.eps) / (p_model + self.eps)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned
    

    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        if algorithm.distributed and algorithm.world_size > 1:
            if probs_x_lb is not None and self.update_p_target:
                probs_x_lb = concat_all_gather(probs_x_lb)
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)

        if hasattr(algorithm, 'smooth_factor'):
            self.smooth_factor = algorithm.smooth_factor

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes, )) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes, ))/ self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
        
        return update_p_target, p_target

    def apply_distribution_smoothing(self, p_target, p_model):
        if self.smooth_p_target and self.smooth_factor is not None:
            p_target = p_target ** self.smooth_factor
            p_target = p_target / p_target.sum(dim=-1)

        if self.smooth_p_model and self.smooth_factor is not None:
            p_model = p_model ** self.smooth_factor
            p_model = p_model / p_model.sum(dim=-1)

        return p_target, p_model

class DistAlignQueueHook(Hook):
    """
    Distribution Alignment Hook for conducting distribution alignment
    """
    def __init__(self, num_classes, queue_length=128, p_target_type='uniform', p_target=None,
                 smooth_p_target=False, smooth_p_model=False):
        super().__init__()
        self.num_classes = num_classes
        self.queue_length = queue_length
        self.eps = 1e-6

        # p_target
        self.p_target_ptr, self.p_target = self.set_p_target(p_target_type, p_target)    
        print('distribution alignment p_target:', self.p_target.mean(dim=0))
        # p_model
        self.p_model = torch.zeros(self.queue_length, self.num_classes, dtype=torch.float)
        self.p_model_ptr = torch.zeros(1, dtype=torch.long)
        # distribution smoothing
        self.smooth_p_target = smooth_p_target
        self.smooth_p_model = smooth_p_model
        self.smooth_factor = None

    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):
        """
        Args:
            algorithm: base algorithm
            probs_x_ulb: unlabeled batch probs
        """

        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)
        # estimate distributions from queues
        p_target, p_model = self.p_target.mean(dim=0), self.p_model.mean(dim=0)
        # optionally smooth distributions
        p_target, p_model = self.apply_distribution_smoothing(p_target, p_model)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (p_target + self.eps) / (p_model + self.eps)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned
    
    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):
        # TODO: think better way?
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)
            if self.p_target_ptr is not None:
                self.p_target_ptr = self.p_target_ptr.to(probs_x_ulb.device)
        
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)
            self.p_model_ptr = self.p_model_ptr.to(probs_x_ulb.device)


        if algorithm.distributed and algorithm.world_size > 1:
            if probs_x_lb is not None and self.p_target_ptr is not None:
                probs_x_lb = concat_all_gather(probs_x_lb)
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        probs_x_ulb = probs_x_ulb.detach()
        p_model_ptr = int(self.p_model_ptr)
        self.p_model[p_model_ptr] = probs_x_ulb.mean(dim=0)
        self.p_model_ptr[0] = (p_model_ptr + 1) % self.queue_length

        if self.p_target_ptr is not None:
            assert probs_x_lb is not None
            p_target_ptr = int(self.p_target_ptr)
            self.p_target[p_target_ptr] = probs_x_lb.mean(dim=0)
            self.p_target_ptr[0] = (p_target_ptr + 1) % self.queue_length

        if hasattr(algorithm, 'smooth_factor'):
            self.smooth_factor = algorithm.smooth_factor

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        p_target_ptr = None
        if p_target_type == 'uniform':
            p_target = torch.ones(self.queue_length, self.num_classes, dtype=torch.float) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.zeros((self.queue_length, self.num_classes), dtype=torch.float)
            p_target_ptr =  torch.zeros(1, dtype=torch.long)
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
            p_target = p_target.unsqueeze(0).repeat((self.queue_length, 1))
        
        return p_target_ptr, p_target

    def apply_distribution_smoothing(self, p_target, p_model):
        if self.smooth_p_target and self.smooth_factor is not None:
            p_target = p_target ** self.smooth_factor
            p_target = p_target / p_target.sum(dim=-1)

        if self.smooth_p_model and self.smooth_factor is not None:
            p_model = p_model ** self.smooth_factor
            p_model = p_model / p_model.sum(dim=-1)

        return p_target, p_model
