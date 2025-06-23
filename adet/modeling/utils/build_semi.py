# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.logger import _log_api_usage
# from detectron2.utils.registry import Registry
#
# META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
# META_ARCH_REGISTRY.__doc__ = """
# Registry for meta-architectures, i.e. the whole model.
#
# The registered object will be called with `obj(cfg)`
# and expected to return a `nn.Module` object.
# """


def build_semi_wrapper(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.SSL.SEMI_WRAPPER
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.semi_supervised_method." + meta_arch)
    return model