# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .text_spotter import TransformerPureDetector, TransformerPureDetectorV2
from .SemiETS import SemiETSTextSpotter


_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
