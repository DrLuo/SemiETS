from typing import List
import time
import logging
import numpy as np
import torch
from torch import nn
import pickle
import copy
import torch.nn.functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances
from adet.layers.pos_encoding import PositionalEncoding2D
# from adet.modeling.model.losses import SetCriterion, SetAdaptiveO2MCriterionFull
from adet.modeling.model.matcher import build_matcher, build_matcher_o2m, CtrlPointCost, build_matcher_o2m_full
from adet.modeling.utils.dist_utils import concat_all_gather_with_various_shape

try:
    from ctcdecode import CTCBeamDecoder
except ImportError:
    CTCBeamDecoder = None


@META_ARCH_REGISTRY.register()
class MultiStreamSpotter(nn.Module):
    def __init__(self, cfg):
        super(MultiStreamSpotter, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.submodules = ['student', 'teacher']

        # create student model
        self.student = self.build_model(cfg)

        # create teacher model
        self.teacher = self.build_model(cfg)

        # inference model
        self.inference_on = 'teacher'

        # warm up using only labeled data
        self.label_warm_up = 0


    def build_model(self, cfg):
        model = build_model(cfg)
        return model

    # select the model to forward
    def model(self, **kwargs):
        if "submodule" in kwargs:
            assert (
                    kwargs["submodule"] in self.submodules
            ), "Detector does not contain submodule {}".format(kwargs["submodule"])
            model = getattr(self, kwargs["submodule"])
        else:
            model = getattr(self, self.inference_on)
        return model

    # freeze the sub-model during training
    def freeze(self, model_ref: str):
        assert model_ref in self.submodules
        model = getattr(self, model_ref)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False


    def split_ssl_batch(self, batched_inputs):
        """
        Split batched inputs into labeled and unlabeled samples.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
        Returns:
            labeled_batched_inputs (list[dict]): same as in :meth:`forward`
            unlabeled_batched_inputs (list[dict]): same as in :meth:`forward`
        """
        results = {'sup': [], 'unsup_teacher': [], 'unsup_student': []}
        for d in batched_inputs:
            semi_flag = d['semi']
            if semi_flag == 'sup':
                results['sup'].append(d)
            elif semi_flag == 'unsup':
                results['unsup_teacher'].append(d['weak'])
                results['unsup_student'].append(d['strong'])

        return results

    def inference(self, batched_inputs):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
        Returns:
            same as in :meth:`forward`.
        """
        if self.inference_on == 'student':
            return self.student.forward(batched_inputs)
        elif self.inference_on == 'teacher':
            return self.teacher.forward(batched_inputs)
        else:
            raise NotImplementedError


def reverse_sigmoid(y):
    return -torch.log(1 / y - 1)
