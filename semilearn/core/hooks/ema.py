# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .hook import Hook
from semilearn.core.utils import EMA, EMAN


class EMAHook(Hook):
    """
    EMA model Hook for updating ema version of the model
    """

    def before_run(self, algorithm):
        algorithm.ema = EMA(algorithm.model, algorithm.ema_m)
        algorithm.ema.register()
        if algorithm.resume == True:
            algorithm.ema.load(algorithm.ema_model)

    def after_train_step(self, algorithm):
        if algorithm.ema is not None:
            algorithm.ema.update()
            algorithm.ema_model.load_state_dict(algorithm.model.state_dict())
            algorithm.ema_model.load_state_dict(algorithm.ema.shadow, strict=False)


class EMANHook(Hook):
    """
    Improved EMA model Hook for updating ema version of the model (includes warmup and update of BN buffers)
    """

    def before_run(self, algorithm):
        algorithm.ema = EMAN(algorithm.model, algorithm.ema_m)
        algorithm.ema.register()
        if algorithm.resume == True:
            algorithm.ema.load(algorithm.ema_model)

    def after_train_step(self, algorithm):
        if algorithm.ema is not None:
            algorithm.ema.update(step=algorithm.it)
            algorithm.ema_model.load_state_dict(algorithm.model.state_dict())
            algorithm.ema_model.load_state_dict(algorithm.ema.shadow, strict=False)