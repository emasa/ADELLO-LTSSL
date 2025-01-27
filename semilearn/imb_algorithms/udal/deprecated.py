import copy

import numpy as np
import torch
from torch.nn import functional as F

from semilearn import get_data_loader
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core import ImbAlgorithmBase
from semilearn.core.criterions import ConsistencyLoss, CELoss
from semilearn.core.hooks import Hook
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.imb_algorithms.udal.utils import compute_divergences, compute_alpha_factor, compute_udal_adjustment_dist


@IMB_ALGORITHMS.register('generalized-udal')
class GeneralizedUDAL(ImbAlgorithmBase):
    """
        Generalized UDAL algorithm (with support for ADELLO's distribution alignment, FlexDA).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - udal_alpha_min (float):
                alpha_min in UDAL
            - udal_k (float):
                k in UDAL
            - udal_ema_p (float):
                momentum
            - udal_mode (str):
                 Possible modes: ['udal', 'udal_dynamic', 'udal_gt', 'our_udal', 'our_udal_gt', 'udal_uniform_dynamic', 'decoupled_udal']
            - udal_alpha_from_epoch (bool)
                Whether you use the current epoch or current iteration to compute alpha. Default: True (epoch).
            - More options below (UPDATE DOCSTRINGS)
    """

    def __init__(self, args, **kwargs):
        assert args.udal_mode in ['udal', 'udal_dynamic', 'udal_gt', 'our_udal', 'our_udal_gt', 'our_udal_adjusted_mix',
                                  'udal_uniform_dynamic', 'decoupled_udal', 'decoupled_udal_uniform_dynamic']
        self.imb_init(args.udal_alpha_min, args.udal_k, args.udal_ema_p, args.udal_mode, None, args.udal_alpha_from_epoch)
        super().__init__(args, **kwargs)
        assert args.algorithm not in ['mixmatch', 'meanteacher',
                                      'pimodel'], "UDAL not supports {} as the base algorithm.".format(args.algorithm)

        self.p_data, num_samples_lb = self.compute_labeled_prior()
        self.gt_u_prior, num_samples_ulb = self.compute_unlabeled_prior()

        if ('_gt' in args.udal_mode) and (self.gt_u_prior is not None):
            self.print_fn('Using ground-truth unlabeled prior.')
            p_target = self.gt_u_prior

        elif args.udal_mode == 'udal' and args.udal_precomputed_u_prior_file is not None:
            self.print_fn('Loading precomputed unlabeled prior from: {}'.format(args.udal_precomputed_u_prior_file))
            darp_u_prior = np.load(args.udal_precomputed_u_prior_file)

            self.print_fn(f'Precomputed prior: {darp_u_prior}')

            p_target = torch.tensor(darp_u_prior / darp_u_prior.sum()).to(self.gpu)
        elif args.udal_mode in ('udal_uniform_dynamic', 'decoupled_udal_uniform_dynamic'):
            self.print_fn('Assuming target prior is uniform.')
            p_target = torch.ones_like(self.p_data) / self.num_classes
        else:
            self.print_fn('Assuming target prior equal to labeled prior.')
            p_target = self.p_data

        num_samples_lb = num_samples_lb if args.include_lb_to_ulb else 0.0
        lb_weight = (num_samples_lb / num_samples_ulb)
        self.print_fn(f'Portion of labeled data in unlabeled set: {lb_weight}')

        self.p_hat_sup = torch.tensor(np.ones((self.num_classes,)) / self.num_classes).to(self.gpu)
        self.p_hat_unsup = torch.tensor(np.ones((self.num_classes,)) / self.num_classes).to(self.gpu)

        self.ce_loss = UDALSupervisedLoss(alpha_min=self.alpha_min, k=self.k, use_epochs=self.alpha_from_epoch,
                                          p_data=self.p_data, p_target=p_target, target_mode=self.mode,
                                          num_samples_lb=num_samples_lb, num_samples_ulb=num_samples_ulb)
        self.consistency_loss = UDALConsistencyLoss(alpha_min=self.alpha_min, k=self.k,
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

    def imb_init(self, alpha_min=0.1, k=2.0, ema_p=0.999, mode='our_udal', precomputed_u_prior=None,
                 alpha_from_epoch=True, ema_hard_pl=False):
        self.alpha_min = alpha_min
        self.k = k
        self.ema_p = ema_p
        self.mode = mode
        self.precomputed_u_prior = precomputed_u_prior
        self.alpha_from_epoch = bool(alpha_from_epoch)
        self.ema_hard_pl = ema_hard_pl

    def set_hooks(self):
        self.register_hook(GeneralizedUDALHook(), "UDALHook", priority="NORMAL")
        super().set_hooks()

    def get_p_hat_sup(self):
        return self.p_hat_sup

    def get_p_target(self):
        return self.consistency_loss.get_target_dist()

    def get_p_data(self):
        return self.consistency_loss.p_data

    def get_p_hat(self):
        if self.mode in ('decoupled_udal', 'decoupled_udal_uniform_dynamic'):
            return self.p_hat_unsup
        else:
            return self.p_hat_sup

    def compute_prob(self, u_logits, mode='sup', lb_in_ulb_mask=None):
        # callback called on logits before creating pseudo-labels (by default with semi-supervised classifier (mode=sup))
        # for 'decoupled_udal', this callback needs to be called manually (using weakly-supervised classifier (mode=unsup))
        assert mode in ('sup', 'unsup')

        if 'decoupled' not in self.mode:  # mask is useful only for (decoupled_udal or decoupled_udal_uniform_dynamic)
            lb_in_ulb_mask = None

        if self.args.udal_prior_correction and (mode == 'sup'):
            adj_u_logits = u_logits - self.get_correction_bias()
        else:
            adj_u_logits = u_logits

        if (mode == 'sup') and (self.args.udal_prior_temp is not None):  # change temperature of predictions
            adj_u_logits = adj_u_logits / self.args.udal_prior_temp

        probs = super().compute_prob(u_logits)
        adj_probs = super().compute_prob(adj_u_logits)

        if self.ema_hard_pl:  # use one-hot sharpened predictions
            _, max_cls = torch.max(adj_probs, dim=-1)
            adj_probs = F.one_hot(max_cls.view(-1, 1), self.num_classes).float()

        if (lb_in_ulb_mask is not None) and (mode == 'sup'):
            mask_unl_data = lb_in_ulb_mask < 1
            if mask_unl_data.sum() > 0:  # there is true unlabeled data
                delta_p = adj_probs[mask_unl_data].mean(dim=0)
            else:
                delta_p = None
        else:
            delta_p = adj_probs.mean(dim=0)

        if mode == 'sup':  # update p_hat from supervised classifier (can also be semi-supervised)
            delta_p = delta_p if (delta_p is not None) else self.p_hat_sup  # don't update when delta_p is None
            self.p_hat_sup = self.ema_p * self.p_hat_sup + (1 - self.ema_p) * delta_p
        elif mode == 'unsup':  # update p_hat from weakly supervised classifier
            delta_p = delta_p if (delta_p is not None) else self.p_hat_unsup  # don't update when delta_p is None
            self.p_hat_unsup = self.ema_p * self.p_hat_unsup + (1 - self.ema_p) * delta_p

        return probs

    def get_correction_bias(self):
        mode = self.args.udal_prior_correction_mode

        if (mode is None) or (mode == 'p_hat'):
            p_correct = self.get_p_hat()
        else:
            p_correct = self.get_p_target()

        return self.get_alpha_factor() * torch.log(p_correct)

    def compute_prob_unsup(self, u_logits):
        return self.compute_prob(u_logits, mode='unsup')

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
            # FlexDA and UDAL flags
            SSL_Argument('--udal_mode', str, 'udal'), # set udal_mode=our_udal for ADELLO (FlexDA)
            SSL_Argument('--udal_alpha_from_epoch', str2bool, True), # set udal_alpha_from_epoch=False for ADELLO (FlexDA)
            SSL_Argument('--udal_alpha_min', float, 0.1),
            SSL_Argument('--udal_k', float, 2.0),
            SSL_Argument('--udal_ema_p', float, 0.999),

            # dev options (deprecated)
            SSL_Argument('--udal_precomputed_u_prior_file', str, None),
            SSL_Argument('--udal_ema_hard_pl', str2bool, False),
            SSL_Argument('--udal_prior_correction', str2bool, False),
            SSL_Argument('--udal_prior_correction_mode', str, None),
            SSL_Argument('--udal_prior_temp', float, None),
            # for debugging purposes
            SSL_Argument('--eval_pl_accuracy', str2bool, False),
        ]


class GeneralizedUDALHook(Hook):

    def before_train_step(self, algorithm):
        algorithm.consistency_loss.set_params(p_hat=algorithm.get_p_hat(), p_hat_sup=algorithm.get_p_hat_sup(),
                                              cte_iter=algorithm.it,
                                              max_iter=algorithm.num_train_iter,
                                              num_iter_per_epoch=algorithm.num_iter_per_epoch)
        algorithm.ce_loss.set_params(p_hat=algorithm.get_p_hat(), p_hat_sup=algorithm.get_p_hat_sup(),
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
                algorithm.print_fn('UDAL unlabeled prior:\n' + np.array_str(p_hat.cpu().numpy()))


class UDALSupervisedLoss(CELoss):
    def __init__(self, alpha_min=0.1, k=2.0, p_data=None, p_hat=None, p_hat_sup=None, p_target=None, target_mode='our_udal', use_epochs=True,
                 num_samples_lb=None, num_samples_ulb=None):
        super().__init__()
        self.progressive_alpha_min = alpha_min
        self.progressive_k = k

        self.p_data = p_data
        self.p_hat = p_hat
        self.p_hat_sup = p_hat_sup
        self.p_target = p_target
        self.target_mode = target_mode

        self.cte_iter = None
        self.max_iter = None
        self.num_iter_per_epoch = None
        self.use_epochs = use_epochs
        self.num_samples_lb = num_samples_lb
        self.num_samples_ulb = num_samples_ulb

    def set_params(self, p_hat=None, cte_iter=None, max_iter=None, num_iter_per_epoch=None, p_hat_sup=None):
        self.p_hat = p_hat
        self.p_hat_sup = p_hat_sup

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
        assert self.target_mode in ['udal', 'udal_dynamic', 'udal_gt', 'our_udal', 'our_udal_gt', 'our_udal_adjusted_mix',
                                    'udal_uniform_dynamic', 'decoupled_udal', 'decoupled_udal_uniform_dynamic']

        p_target = self.get_target_dist()

        cte_val, max_val = self.get_progress_values()

        distr_ratio = compute_udal_adjustment_dist(self.p_data, p_target, cte_val, max_val,
                                                   a_min=self.progressive_alpha_min, k=self.progressive_k)

        adjusted_logits = logits + torch.log(distr_ratio)

        if T_src is not None:
            adjusted_logits = adjusted_logits / T_src

        return super().forward(adjusted_logits, targets, reduction=reduction)

    def get_target_dist(self):
        assert self.target_mode in ['udal', 'udal_dynamic', 'udal_gt', 'our_udal', 'our_udal_gt', 'our_udal_adjusted_mix',
                                    'udal_uniform_dynamic', 'decoupled_udal', 'decoupled_udal_uniform_dynamic']

        if self.target_mode in ['udal', 'udal_dynamic', 'udal_gt']:
            p_target = self.p_data
        elif self.target_mode in ['our_udal']:
            p_target = self.p_hat
        elif self.target_mode in ['decoupled_udal']:
            p_target = self.p_hat_sup
        elif self.target_mode in ['our_udal_adjusted_mix']:
            p_target = adjusted_mixed_prior(self.num_samples_ulb * self.p_hat, self.num_samples_lb * self.p_data,
                                            self.num_samples_lb / self.num_samples_ulb)
        else:
            p_target = self.p_target

        return p_target

class UDALConsistencyLoss(ConsistencyLoss):
    def __init__(self, alpha_min=0.1, k=2.0, p_data=None, p_hat=None, p_hat_sup=None, p_target=None, target_mode='our_udal', use_epochs=True,
                 num_samples_lb=None, num_samples_ulb=None):
        super().__init__()
        self.progressive_alpha_min = alpha_min
        self.progressive_k = k

        self.p_data = p_data
        self.p_hat = p_hat
        self.p_hat_sup = p_hat_sup
        self.p_target = p_target
        self.target_mode = target_mode

        self.cte_iter = None
        self.max_iter = None
        self.num_iter_per_epoch = None
        self.use_epochs = use_epochs
        self.num_samples_lb = num_samples_lb
        self.num_samples_ulb = num_samples_ulb

    def set_params(self, p_hat=None, cte_iter=None, max_iter=None, num_iter_per_epoch=None, p_hat_sup=None):
        self.p_hat = p_hat
        self.p_hat_sup = p_hat_sup

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

    def forward(self, logits, targets, name='ce', mask=None, T_src=None, T_tgt=None, adjust_target=False):
        assert self.target_mode in ['udal', 'udal_dynamic', 'udal_gt', 'our_udal', 'our_udal_gt', 'our_udal_adjusted_mix',
                                    'udal_uniform_dynamic', 'decoupled_udal', 'decoupled_udal_uniform_dynamic']

        p_target = self.get_target_dist()

        cte_val, max_val = self.get_progress_values()

        distr_ratio = compute_udal_adjustment_dist(self.p_hat, p_target, cte_val, max_val,
                                                   a_min=self.progressive_alpha_min, k=self.progressive_k)

        adjusted_logits = logits + torch.log(distr_ratio)
        adjusted_targets = targets

        if (T_src is not None) or (T_tgt is not None) or adjust_target:
            assert adjusted_targets.dim() > 1

            T_src = T_src if (T_src is not None) else 1.0
            T_tgt = T_tgt if (T_tgt is not None) else 1.0

            adjusted_logits = adjusted_logits / T_src

            if adjust_target:
                adjusted_targets = adjusted_targets * distr_ratio
            adjusted_targets = adjusted_targets ** (1.0 / T_tgt)
            adjusted_targets = adjusted_targets / adjusted_targets.sum(dim=-1, keepdim=True)

        return super().forward(adjusted_logits, adjusted_targets, name, mask)

    def get_target_dist(self):
        assert self.target_mode in ['udal', 'udal_dynamic', 'udal_gt', 'our_udal', 'our_udal_gt', 'our_udal_adjusted_mix',
                                    'udal_uniform_dynamic', 'decoupled_udal', 'decoupled_udal_uniform_dynamic']

        if self.target_mode in ['udal', 'udal_gt', 'our_udal_gt']:
            p_target = self.p_target
        elif self.target_mode in ['our_udal', 'udal_dynamic', 'udal_uniform_dynamic']:
            p_target = self.p_hat
        elif self.target_mode in ['decoupled_udal', 'decoupled_udal_uniform_dynamic']:
            p_target = self.p_hat_sup
        elif self.target_mode in ['our_udal_adjusted_mix']:
            p_target = adjusted_mixed_prior(self.num_samples_ulb * self.p_hat, self.num_samples_lb * self.p_data,
                                            self.num_samples_lb / self.num_samples_ulb)
        else:
            p_target = None

        return p_target


def adjusted_mixed_prior(mixed_prior, lb_prior, lb_weight):
    # it's not working -> it likely produces "negative probabilities"
    unl_prior = mixed_prior - lb_weight * lb_prior
    return unl_prior / unl_prior.sum(dim=-1)
