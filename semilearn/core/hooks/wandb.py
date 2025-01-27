# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import wandb
from .hook import Hook


class WANDBHook(Hook):
    """
    Wandb Hook
    """

    def __init__(self, log_all_keys=True):
        super().__init__()
        self.log_key_list = ['train/sup_loss', 'train/unsup_loss', 'train/total_loss', 'train/util_ratio', 
                             'train/run_time', 'train/prefetch_time', 'lr',
                             'eval/top-1-acc', 'eval/precision', 'eval/recall', 'eval/F1']

        self.log_all_keys = log_all_keys

    def before_run(self, algorithm):
        # job_id = '_'.join(algorithm.args.save_name.split('_')[:-1])
        name = algorithm.save_name
        project = algorithm.save_dir.split('/')[-1]

        # tages
        benchmark = f'benchmark: {project}'
        dataset = f'dataset: {algorithm.args.dataset}'
        data_setting = f'setting: {algorithm.args.dataset}_lb{algorithm.args.num_labels}_{algorithm.args.lb_imb_ratio}_ulb{algorithm.args.ulb_num_labels}_{algorithm.args.ulb_imb_ratio}'
        alg = f'alg: {algorithm.args.algorithm}'
        imb_alg = f'imb_alg: {algorithm.args.imb_algorithm}'
        tags = [benchmark, dataset, data_setting, alg, imb_alg] 
        if algorithm.args.resume:
            resume = 'auto'
        else:
            resume = 'never'
        # resume = 'never'
        
        save_dir = os.path.join(algorithm.args.save_dir, 'wandb', algorithm.args.save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.run = wandb.init(name=name, 
                              tags=tags, 
                              config=algorithm.args.__dict__, 
                              project=project, 
                              resume=resume,
                              dir=save_dir)

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            log_dict = {}
            for key, item in algorithm.log_dict.items():
                if key in self.log_key_list or self.log_all_keys:
                    log_dict[key] = item
            self.run.log(log_dict, step=self._get_it(algorithm))
    
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            self.run.log({'eval/best-acc': algorithm.best_eval_acc}, step=self._get_it(algorithm))
    
    def after_run(self, algorithm):
        self.run.finish()

    def _get_it(self, algorithm):
        if hasattr(algorithm, 'global_it'):
            # to support CreST multi-generation approach
            return algorithm.global_it
        else:
            return algorithm.it