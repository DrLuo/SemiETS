import random
from typing import Tuple
import sys
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from fvcore.transforms import transform as T
from detectron2.data.transforms import RandomCrop, StandardAugInput, RandomFlip, RandomRotation, ResizeTransform
from detectron2.structures import BoxMode
import torch
from detectron2.data.transforms import Augmentation, PadTransform, BlendTransform, PILColorTransform
from fvcore.transforms.transform import Transform, NoOpTransform
from .geo_utils import GeometricTransformationBase as GTrans
from detectron2.data.transforms.transform import RotationTransform


def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 25:
                raise ValueError(
                    "Cannot finished cropping adjustment within 25 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                return T.CropTransform(0, 0, image_size[1], image_size[0])

    return T.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True, record=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance  # relative range
        self.input_args = ("image", "boxes")
        self.record = record  # record the random crop paramter to do the following gt bbox alignment

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )

    def enable_record(self, mode: bool = True):
        self.record = mode


class Pad(Augmentation):
    def __init__(self, divisible_size = 32):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        ori_h, ori_w = img.shape[:2]  # h, w
        if ori_h % 32 == 0:
            pad_h = 0
        else:
            pad_h = 32 - ori_h % 32
        if ori_w % 32 == 0:
            pad_w = 0
        else:
            pad_w = 32 - ori_w % 32
        # pad_h, pad_w = 32 - ori_h % 32, 32 - ori_w % 32
        return PadTransform(
            0, 0, pad_w, pad_h, pad_value=0
        )



class RandomRotationWithRecord(Augmentation):
    """
    override the original method to get the matrix of this operator
    """

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None , record=True):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self.record = record  # record the random crop paramter to do the following gt bbox alignment
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            if self.record:
                self.matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
            return NoOpTransform()


        tfm = RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)
        if self.record:
            # mat = np.concatenate(
            #     [tfm.rm_image, np.array([0, 0, 1]).reshape((1, 3))], axis=0
            # )
            mat = np.concatenate(
                [tfm.rm_coords, np.array([0, 0, 1]).reshape((1, 3))], axis=0
            )
            self.matrix = mat


        return tfm

    def enable_record(self, mode: bool = True):
        self.record = mode


class ResizeShortestEdgeWithRecord(Augmentation):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    @torch.jit.unused
    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, record=True
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())
        self.record = record

    @torch.jit.unused
    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            if self.record:
                self.matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
            return NoOpTransform()

        newh, neww = ResizeShortestEdgeWithRecord.get_output_shape(h, w, size, self.max_size)
        scale  = size / min(h, w)
        if self.record:
            self.matrix =GTrans._get_scale_matrix(sx=scale , sy=scale ,inverse=False)
        return ResizeTransform(h, w, newh, neww, self.interp)

    def enable_record(self, mode: bool = True):
        self.record = mode

    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


# TODO: RandShear and random RandTranslate
class RandomSheerWithMatrix(Augmentation):
    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None , record=True):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self.record = record  # record the random crop paramter to do the following gt bbox alignment
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            if self.record:
                self.matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
            return NoOpTransform()


        tfm = RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)
        if self.record:
            # mat = np.concatenate(
            #     [tfm.rm_image, np.array([0, 0, 1]).reshape((1, 3))], axis=0
            # )
            mat = np.concatenate(
                [tfm.rm_coords, np.array([0, 0, 1]).reshape((1, 3))], axis=0
            )
            self.matrix = mat


        return tfm

    def enable_record(self, mode: bool = True):
        self.record = mode


''' 
-------------------------------------
 Implemented from MMdetection 
-------------------------------------
'''


class PILEnhancer(object):
    def __init__(self, op, factor):
        self.op = op
        self.factor = factor

    def __call__(self, pil_img):
        return self.op(pil_img).enhance(self.factor)


class RandomSharpness(Augmentation):
    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        op = PILEnhancer(ImageEnhance.Sharpness, w)
        return PILColorTransform(op)


class RandomEqualize(Augmentation):
    def __init__(self, p):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        if np.random.uniform(0, 1) < self.p:
            op = ImageOps.equalize
            return PILColorTransform(op)
        else:
            return NoOpTransform()


class OneOf(Augmentation):
    def __init__(self, augs: list):
        self.augs = augs

    def get_transform(self, image):
        transforms = [NoOpTransform()]
        for aug in self.augs:
            transforms.append(aug.get_transform(image))
        return np.random.choice(transforms)


