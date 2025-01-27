# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn.functional as F

from semilearn.core.hooks import Hook
from semilearn.core.criterions import ConsistencyLoss, CELoss


def compute_divergences(p1, p2):
    fwd_kl_div = F.kl_div(p1.log(), p2, reduction='sum')
    bwd_kl_div = F.kl_div(p2.log(), p1, reduction='sum')
    js_div = (fwd_kl_div + bwd_kl_div) / 2.0
    return fwd_kl_div, bwd_kl_div, js_div

class ADELLOHook(Hook):

    def before_train_step(self, algorithm):
        algorithm.consistency_loss.set_params(p_hat=algorithm.get_p_hat(),
                                              cte_iter=algorithm.it,
                                              max_iter=algorithm.num_train_iter,
                                              num_iter_per_epoch=algorithm.num_iter_per_epoch)
        algorithm.ce_loss.set_params(p_hat=algorithm.get_p_hat(),
                                     cte_iter=algorithm.it,
                                     max_iter=algorithm.num_train_iter,
                                     num_iter_per_epoch=algorithm.num_iter_per_epoch)
        algorithm.smooth_factor = algorithm.consistency_loss.get_alpha_factor()

    def after_train_step(self, algorithm):
        algorithm.log_dict['train/cte_alpha'] = algorithm.consistency_loss.get_alpha_factor()

        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            self._track_unl_prior_estimation(algorithm)

        if algorithm.args.eval_pl_accuracy and (self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm)):
            algorithm.print_fn("evaluating unlabeled data...")
            eval_dict = algorithm.evaluate('eval_ulb_privileged', return_logits=False, track_mean_acc=False)
            algorithm.log_dict.update(eval_dict)

    def _track_unl_prior_estimation(self, algorithm):
        p_hat = algorithm.get_p_hat()
        p_data = algorithm.p_data
        p_unif = torch.ones_like(p_hat) / algorithm.num_classes
        gt_prior = algorithm.gt_u_prior
        p_target = algorithm.get_p_target()

        if p_hat is not None:
            fwd_kl_div_p_data, _, js_div_p_data = compute_divergences(p_hat, p_data)
            algorithm.log_dict['train/kl_div_ulb_vs_lb_prior'] = fwd_kl_div_p_data
            fwd_kl_div_p_data_tgt, _, _ = compute_divergences(p_target, p_data)
            algorithm.log_dict['train/kl_div_tgt_vs_lb_prior'] = fwd_kl_div_p_data_tgt

            fwd_kl_div_p_unif, _, _ = compute_divergences(p_hat, p_unif)
            algorithm.log_dict['train/kl_div_ulb_vs_unif'] = fwd_kl_div_p_unif
            fwd_kl_div_p_unif_tgt, _, _ = compute_divergences(p_target, p_unif)
            algorithm.log_dict['train/kl_div_tgt_vs_unif'] = fwd_kl_div_p_unif_tgt

            fwd_kl_div_p_data_vs_unif, _, js_div_p_data_vs_unif = compute_divergences(p_data, p_unif)
            algorithm.log_dict['train/kl_div_p_data_vs_unif'] = fwd_kl_div_p_data_vs_unif
            algorithm.log_dict['train/abs_diff_kl_div'] = torch.abs(fwd_kl_div_p_data_vs_unif - fwd_kl_div_p_data)
            algorithm.log_dict['train/abs_diff_js_div'] = torch.abs(js_div_p_data_vs_unif - js_div_p_data)

            if gt_prior is not None:
                fwd_kl_div, _, _ = compute_divergences(p_hat, gt_prior)
                algorithm.log_dict['train/kl_div_ulb_prior'] = fwd_kl_div
                fwd_kl_div_tgt, _, _ = compute_divergences(p_target, gt_prior)
                algorithm.log_dict['train/kl_div_tgt_vs_gt'] = fwd_kl_div_tgt

            with np.printoptions(threshold=np.inf):
                algorithm.print_fn('ADELLO unlabeled prior:\n' + np.array_str(p_hat.cpu().numpy()))

def compute_alpha_factor(current_epoch, max_epoch, a_min=0.0, k=1.0, a_max=1.0):
    return a_max - (a_max - a_min) * (current_epoch / max_epoch) ** k


def compute_adello_adjustment_dist(current_dist, p_target, current_epoch, max_epoch, a_min=0.1, k=2.0, a_max=1.0, eps=1e-9):
    a_factor = compute_alpha_factor(current_epoch, max_epoch, a_min=a_min, k=k, a_max=a_max)

    # normalization ensures the argument sums to 1
    target_dist = p_target ** a_factor
    target_dist = target_dist / target_dist.sum(dim=-1)
    return (current_dist + eps) / (target_dist + eps)


def adjusted_mixed_prior(mixed_prior, lb_prior, lb_weight):
    # it's not working -> it likely produces "negative probabilities"
    unl_prior = mixed_prior - lb_weight * lb_prior
    return unl_prior / unl_prior.sum(dim=-1)

class FlexDASupervisedLoss(CELoss):
    def __init__(self, alpha_min=0.1, k=2.0, p_data=None, p_hat=None, p_target=None, target_mode='adello', use_epochs=True,
                 num_samples_lb=None, num_samples_ulb=None):
        super().__init__()
        self.progressive_alpha_min = alpha_min
        self.progressive_k = k

        self.p_data = p_data
        self.p_hat = p_hat
        self.p_target = p_target
        self.target_mode = target_mode

        self.cte_iter = None
        self.max_iter = None
        self.num_iter_per_epoch = None
        self.use_epochs = use_epochs
        self.num_samples_lb = num_samples_lb
        self.num_samples_ulb = num_samples_ulb

    def set_params(self, p_hat=None, cte_iter=None, max_iter=None, num_iter_per_epoch=None):
        self.p_hat = p_hat
        self.cte_iter = cte_iter
        self.max_iter = max_iter
        self.num_iter_per_epoch = num_iter_per_epoch

    def get_progress_values(self):
        if self.use_epochs:
            cte_epoch = int(self.cte_iter // self.num_iter_per_epoch)
            max_epoch = int(np.ceil(self.max_iter // self.num_iter_per_epoch))
            return cte_epoch, max_epoch
        else:
            return self.cte_iter, self.max_iter

    def get_alpha_factor(self):
        cte_val, max_val = self.get_progress_values()
        return compute_alpha_factor(cte_val, max_val, a_min=self.progressive_alpha_min, k=self.progressive_k)

    def forward(self, logits, targets, reduction='mean', T_src=None):
        assert self.target_mode in ['adello', 'adello_gt']

        p_target = self.get_target_dist()

        cte_val, max_val = self.get_progress_values()

        distr_ratio = compute_adello_adjustment_dist(self.p_data, p_target, cte_val, max_val,
                                                     a_min=self.progressive_alpha_min, k=self.progressive_k)

        adjusted_logits = logits + torch.log(distr_ratio)

        if T_src is not None:
            adjusted_logits = adjusted_logits / T_src

        return super().forward(adjusted_logits, targets, reduction=reduction)

    def get_target_dist(self):
        assert self.target_mode in ['adello', 'adello_gt']

        if self.target_mode in ['adello']:
            p_target = self.p_hat
        else:
            p_target = self.p_target

        return p_target


class FlexDAConsistencyLoss(ConsistencyLoss):
    def __init__(self, alpha_min=0.1, k=2.0, p_data=None, p_hat=None, p_target=None, target_mode='adello',
                 use_epochs=True, num_samples_lb=None, num_samples_ulb=None):
        super().__init__()
        self.progressive_alpha_min = alpha_min
        self.progressive_k = k

        self.p_data = p_data
        self.p_hat = p_hat
        self.p_target = p_target
        self.target_mode = target_mode

        self.cte_iter = None
        self.max_iter = None
        self.num_iter_per_epoch = None
        self.use_epochs = use_epochs
        self.num_samples_lb = num_samples_lb
        self.num_samples_ulb = num_samples_ulb

    def set_params(self, p_hat=None, cte_iter=None, max_iter=None, num_iter_per_epoch=None):
        self.p_hat = p_hat
        self.cte_iter = cte_iter
        self.max_iter = max_iter
        self.num_iter_per_epoch = num_iter_per_epoch

    def get_progress_values(self):
        if self.use_epochs:
            cte_epoch = int(self.cte_iter // self.num_iter_per_epoch)
            max_epoch = int(self.max_iter // self.num_iter_per_epoch)
            return cte_epoch, max_epoch
        else:
            return self.cte_iter, self.max_iter

    def get_alpha_factor(self):
        cte_val, max_val = self.get_progress_values()
        return compute_alpha_factor(cte_val, max_val, a_min=self.progressive_alpha_min, k=self.progressive_k)

    def forward(self, logits, targets, name='ce', mask=None, T_src=None, T_tgt=None):
        assert self.target_mode in ['adello', 'adello_gt']

        p_target = self.get_target_dist()

        cte_val, max_val = self.get_progress_values()

        distr_ratio = compute_adello_adjustment_dist(self.p_hat, p_target, cte_val, max_val,
                                                     a_min=self.progressive_alpha_min, k=self.progressive_k)

        adjusted_logits = logits + torch.log(distr_ratio)
        adjusted_targets = targets

        if (T_src is not None) or (T_tgt is not None):
            assert adjusted_targets.dim() > 1

            T_src = T_src if (T_src is not None) else 1.0
            T_tgt = T_tgt if (T_tgt is not None) else 1.0

            adjusted_logits = adjusted_logits / T_src
            adjusted_targets = adjusted_targets ** (1.0 / T_tgt)
            adjusted_targets = adjusted_targets / adjusted_targets.sum(dim=-1, keepdim=True)

        return super().forward(adjusted_logits, adjusted_targets, name, mask)

    def get_target_dist(self):
        assert self.target_mode in ['adello', 'adello_gt']

        if self.target_mode in ['adello_gt']:
            p_target = self.p_target
        elif self.target_mode in ['adello']:
            p_target = self.p_hat
        else:
            p_target = None

        return p_target
