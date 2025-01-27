# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.imb_algorithms.udal.utils import compute_divergences


@ALGORITHMS.register('fixmatch_distill')
class FixMatchDistill(AlgorithmBase):
    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685) + distillation support for ADELLO.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - More options below (UPDATE DOCSTRINGS)
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self._distill_T_fixed = None

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")

        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, lb_in_ulb_mask=None):
        num_lb = y_lb.shape[0]
        distill_temperature = 1.0

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            p_cutoff = self.get_p_cutoff()

            # callback (usually computes softmax. DebiasPL, UDAL and ADELLO keep track of running average)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach(), lb_in_ulb_mask=lb_in_ulb_mask)

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w)

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False,
                                  p_cutoff=p_cutoff)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook(
                "gen_ulb_targets", "PseudoLabelingHook",
                logits=probs_x_ulb_w if self.use_hard_label else torch.softmax(torch.log(probs_x_ulb_w) / self.T, dim=-1),
                use_hard_label=self.use_hard_label, softmax=False,
            )

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

            # is distillation enabled?
            do_distill = self.args.distill_complement or self.args.distill_all
            # is it time for distillation?
            begin_distill = (self.args.distill_start_iter is None) or (self.it >= self.args.distill_start_iter)
            do_distill = do_distill and begin_distill

            if do_distill:
                distill_mask = self.get_distill_mask(mask, probs_x_ulb_w)
                distill_loss, distill_temperature, distill_mask = self.compute_distill_loss(logits_x_ulb_s, probs_x_ulb_w, distill_mask)
                total_loss = total_loss + self.args.lambda_distill * distill_loss

        log_dict = dict(sup_loss=sup_loss.item(),
                        unsup_loss=unsup_loss.item(),
                        total_loss=total_loss.item(),
                        effective_p_cutoff=p_cutoff,
                        util_ratio=mask.float().mean().item())

        if do_distill:
            distill_temperature = torch.as_tensor(distill_temperature).float().mean()
            log_dict.update(dict(distill_loss=distill_loss.item(),
                                 distill_temperature=distill_temperature.item(),
                                 distill_util_ratio=distill_mask.float().mean().item()))

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(**log_dict)
        return out_dict, log_dict

    def get_p_cutoff(self):
        return self.p_cutoff

    def get_distill_mask(self, mask, probs_x_ulb_w):
        if self.args.distill_complement:
            distill_mask = 1 - mask
        else:
            distill_mask = torch.ones_like(mask)

        return distill_mask

    def compute_distill_loss(self, logits_x_ulb_s, probs_x_target, mask):
        T = 1.0 if (self.args.distill_T is None) else self.args.distill_T

        if self.args.distill_T is None:
            distill_loss = self.consistency_loss(logits_x_ulb_s,
                                                 probs_x_target,
                                                 'ce',
                                                 mask=mask)

        else:
            if self.args.imbalance_distill_with_warmup:
                # estimate temperature at the beginning of the distillation stage (p_target can differ from p_model)
                after_warmup = ((self.args.distill_start_iter is not None) and (self.it >= self.args.distill_start_iter))
                if after_warmup:
                    if self._distill_T_fixed is None:
                        p_hat = self.get_p_hat()
                        p_unif = torch.ones_like(p_hat) / self.num_classes
                        kl_div_imb, _, _ = compute_divergences(p_hat, p_unif)
                        self._distill_T_fixed = torch.exp(kl_div_imb)

                    T = self._distill_T_fixed

            # distill at the same temperature
            distill_loss = self.consistency_loss(logits_x_ulb_s,
                                                 probs_x_target,
                                                 'ce',
                                                 mask=mask,
                                                 T_src=T, T_tgt=T)

        return distill_loss, T, mask

    @staticmethod
    def get_argument():
        return [
            # FixMatch flags
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--T', float, 0.5),

            # Complementary Consistency Regularization (CCR) flags
            SSL_Argument('--distill_complement', str2bool, True),  # set distill_complement=True for ADELLO
            SSL_Argument('--lambda_distill', float, 1.0),  # need to distill_complement=True to work
            SSL_Argument('--imbalance_distill_with_warmup', str2bool, True),  # set imbalance_distill_with_warmup=True for ADELLO
            SSL_Argument('--distill_start_iter', int, None),  # set accordingly (Default: None -> distill from iter=0)

            # dev flags (deprecated)
            SSL_Argument('--distill_T', float, None),  # (Default: None -> distill with T=1)
            SSL_Argument('--distill_all', str2bool, False),
        ]
