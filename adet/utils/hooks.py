import datetime
import itertools
import logging
import math
import operator
import os
import tempfile
import time
import warnings
from collections import Counter
import torch
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.timer import Timer
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
import os.path as osp
import detectron2.utils.comm as comm
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.solver import LRMultiplier
from detectron2.utils.events import EventStorage, EventWriter
from detectron2.utils.file_io import PathManager
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.engine.train_loop import HookBase
from bisect import bisect_right

__all__ = ['StepRecord', 'MeanTeacher',]

"""
Customized hooks for the SSL.
"""


class PseudoLabelHook(HookBase):
    #implement of Pseudolabel generate hook to generate PL to be saved as an attr of model
    #support for Teacher-student framework without EMA,while teacher's params can be updated by epoch/iter or freezed
    #the format of PL is different from the original groundtruth for easier utilization , default to be the orignal predict of Model
    #in some cases only the top-k PL would be used ,which can be implemented by add semi_5s function in Model,while called from here to generate PL
    def __init__(
        self,
        strategy:'freeze',
        iter=None,
        epoch=None,
        save_dir=None,
        save=False,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
        clone_teacher=False
    ):
        self.strategy = strategy
        if isinstance(strategy, str):
            strategy = [strategy]
        allowed_strategies = ['freeze', 'iter', 'epoch', 'EMA']
        if not set(strategy).issubset(set(allowed_strategies)):
            raise KeyError(f'metrics {strategy} is not supported')
        logger = logging.getLogger(__name__)
        logger.info(f"use {self.strategy} strategy for PseudoLabel generating")

        self.pred_allow = True
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        elif os.listdir(save_dir):
            self.pred_allow = False   #if PL already exist， then forbid pred at the first round
            self.pseudo_init(save_dir)
        self.save_pl = save

        self.epoch_base=False
        self.iter_base=False
        self.EMA= self.strategy == 'EMA'

        self.iter = iter
        if self.iter is not None:
            self.iter_base = True

        self.epoch = epoch
        if self.epoch is not None:
            self.epoch_base =True

        if self.EMA :
            assert momentum >= 0 and momentum <= 1
            self.momentum = momentum
            assert isinstance(interval, int) and interval > 0
            self.warm_up = warm_up
            self.interval = interval
            assert isinstance(decay_intervals, list) or decay_intervals is None
            self.decay_intervals = decay_intervals
            self.decay_factor = decay_factor
        self.clone_teacher = clone_teacher

    def before_train(self):
        model = self.trainer.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher")
        assert hasattr(model, "student")
        # only do it at initial stage
        if self.clone_teacher:
            if self.trainer.iter == 0:
                self.model_clone(model)

        if  self.pred_allow:
            self.generate_PL(model)


    def before_step(self):
        """whether to Update  parameter every interval."""
        if self.strategy == 'freeze':
            pass
        elif self.iter_base :
            curr_step = self.trainer.iter
            if curr_step % self.iter != 0:
                return
            self.model_clone(self.trainer.model)

        elif self.epoch_base:
            curr_epoch = self.trainer.epoch
            if curr_epoch % self.epoch != 0:
                return
            self.model_clone(self.trainer.model)
        elif self.EMA:
            self.EMA_before_step()
    def EMA_before_step(self):
        """Update ema parameter every self.interval iterations."""
        curr_step = self.trainer.iter
        if curr_step % self.interval != 0:
            return
        model = self.trainer.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        if momentum < self.momentum:
            logger = logging.getLogger(__name__)
            logger.info(
                f"warming up momentum to {self.momentum}, current value is {momentum} at {curr_step} step."
            )
        self.momentum_update(model, momentum)

    def after_step(self):
        if self.strategy == 'freeze':
            pass
        elif self.iter_base:
            pass
        elif self.epoch_base:
            pass
        elif self.EMA:
            self.EMA_after_step()
    def EMA_after_step(self):
        curr_step = self.trainer.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) / self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
    def model_clone(self,model):
        logger = logging.getLogger(__name__)
        logger.info("Clone all parameters of student to teacher...")

        self.momentum_update(model, 0)


    def generate_PL(self,model): #refer to eval hook
        self.model.Pseudo = 'to be implemented'
        if self.save_pl:
            pass
        #save pl to save_dir
    def pseudo_init(self,save_dir):
        pass
        #load pl from save_dir
        self.trainer.model.pseudo='to be implemented'



class StepRecord(HookBase):
    def __init__(
        self,
        normalize=False,
        name="curr_step"
    ):
        self.normalize = normalize
        self.name = name

    def after_step(self):
        """
        Called after each iteration.
        """
        curr_step = self.trainer.iter
        model = self.trainer.model
        if is_module_wrapper(model):
            model = model.module

        assert hasattr(model, self.name)
        setattr(model, self.name, curr_step/10000 if self.normalize else curr_step)


class MeanTeacher(HookBase):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
        clone_teacher=False
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor
        self.clone_teacher = clone_teacher

    def before_train(self):
        model = self.trainer.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher")
        assert hasattr(model, "student")
        # only do it at initial stage
        if self.clone_teacher:
            if self.trainer.iter == 0:
                logger = logging.getLogger(__name__)
                logger.info("Clone all parameters of student to teacher...")

                self.momentum_update(model, 0)

    def before_step(self):
        """Update ema parameter every self.interval iterations."""
        curr_step = self.trainer.iter
        if curr_step % self.interval != 0:
            return
        model = self.trainer.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        if momentum < self.momentum:
            logger = logging.getLogger(__name__)
            logger.info(
                f"warming up momentum to {self.momentum}, current value is {momentum} at {curr_step} step."
            )
        self.momentum_update(model, momentum)

    def after_step(self):
        curr_step = self.trainer.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) / self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)


def is_module_wrapper(module: torch.nn.Module) -> bool:
    """Check if a module is a module wrapper.

    The following 3 modules in MMCV (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version). You may add you own
    module wrapper by registering it to mmcv.parallel.MODULE_WRAPPERS or
    its children registries.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """
    return isinstance(module, (DataParallel, DistributedDataParallel))

    # def is_module_in_wrapper(module, module_wrapper):
    #     module_wrappers = tuple(module_wrapper.module_dict.values())
    #     if isinstance(module, module_wrappers):
    #         return True
    #     for child in module_wrapper.children.values():
    #         if is_module_in_wrapper(module, child):
    #             return True
    #     return False
    #
    # return is_module_in_wrapper(module, (DataParallel, DistributedDataParallel))