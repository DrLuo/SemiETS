import copy
import logging
import os.path as osp

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import MapDataset, AspectRatioGroupedDataset
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance, ResizeShortestEdgeWithRecord, RandomRotationWithRecord
from .detection_utils import (annotations_to_instances, build_augmentation,
                              build_augmentation_weak, build_augmentation_strong,
                              transform_instance_annotations)
import warnings

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis", "DatasetMapperForSSL"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m

def filter_empty_instances(instances):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
        return_mask (bool): whether to return boolean mask of filtered instances

    Returns:
        Instances: the filtered instances.
        tensor[bool], optional: boolean mask of filtered instances
    """
    pass
    r = []
    r.append(instances.gt_boxes.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    return instances[m]


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train and cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
        if cfg.INPUT.ROTATE and is_train:
            if cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
                self.augmentation.insert(0, T.RandomRotation(angle=[-45, 45]))
            else:
                self.augmentation.insert(0, T.RandomRotation(angle=[-90, 90]))

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        ######################################################################
        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        ######################################################################

        # aug_input = T.StandardAugInput(image)
        aug_input = T.StandardAugInput(image, boxes=boxes)

        transforms = aug_input.apply_augmentations(self.augmentation)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # dataset_dict["instances"] = instances
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict



class DatasetMapperForSSL(DatasetMapper):
    """
    This caller enables the DatasetMapperWithBasis to build dataset pipeline for semi-supervised learning
    It produces two augmented images from a single image (unlabeled)

    The callable implement the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`

    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the semi-supervised augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        self.strong_augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train and cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
        if cfg.INPUT.ROTATE and is_train:
            if cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
                self.augmentation.insert(0, T.RandomRotation(angle=[-45, 45]))
            else:
                self.augmentation.insert(0, T.RandomRotation(angle=[-90, 90]))

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # semi_anno = dataset_dict['semi']
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        ######################################################################
        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        ######################################################################

        # aug_input = T.StandardAugInput(image)
        aug_input = T.StandardAugInput(image, boxes=boxes)

        transforms = aug_input.apply_augmentations(self.augmentation)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # dataset_dict["instances"] = instances
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


class MultiBranchDatasetMapper(DatasetMapper):
    """
    This caller enables the DatasetMapperWithBasis to build dataset pipeline for semi-supervised learning
    It produces two augmented images from a single image (unlabeled)

    The callable implement the following:

    1. Read the image from "file_name"
    2. Applies weak augmentation to the image and annotations
    2. Applies strong augmentation cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`

    As unlabeled data does not have annotations, we abandon the instance-ware crop
    Strong augmentation:
        - Resize
        -


    Weak augmentation:
        - Resize

    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the semi-supervised augmentations. The previous augmentations will be overridden."
        )

        # if data from different domain, replace parameters
        unlabel_cfg = cfg.clone()
        try:
            if cfg.INPUT.REPLACE:
                unlabel_cfg.defrost()
                unlabel_cfg.INPUT.MIN_SIZE_TRAIN = cfg.INPUT_UNLABEL.MIN_SIZE_TRAIN if cfg.INPUT_UNLABEL.MIN_SIZE_TRAIN is not None else cfg.INPUT.MIN_SIZE_TRAIN
                unlabel_cfg.INPUT.MAX_SIZE_TRAIN = cfg.INPUT_UNLABEL.MAX_SIZE_TRAIN if cfg.INPUT_UNLABEL.MAX_SIZE_TRAIN is not None else cfg.INPUT.MAX_SIZE_TRAIN
                unlabel_cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT_UNLABEL.MIN_SIZE_TEST if cfg.INPUT_UNLABEL.MIN_SIZE_TEST is not None else cfg.INPUT.MIN_SIZE_TEST
                unlabel_cfg.INPUT.MAX_SIZE_TEST = cfg.INPUT_UNLABEL.MAX_SIZE_TEST if cfg.INPUT_UNLABEL.MAX_SIZE_TEST is not None else cfg.INPUT.MAX_SIZE_TEST
                unlabel_cfg.INPUT.ROTATE = cfg.INPUT_UNLABEL.ROTATE if cfg.INPUT_UNLABEL.ROTATE is not None else cfg.INPUT.ROTATE
                unlabel_cfg.freeze()

        except:
            warnings.WarningMessage(
                "Warning: [cfg.INPUT.REPLACE] Not found, therefore augmentation parameters remains the same",
                category=UserWarning,
                filename=__name__,
                lineno=0
            )
        self.weak_augmentation = build_augmentation_weak(unlabel_cfg, is_train)

        self.strong_augmentation = build_augmentation_strong(unlabel_cfg, is_train)


        if cfg.INPUT.ROTATE and is_train:
            if cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
                self.strong_augmentation.insert(0, RandomRotationWithRecord(angle=[-45, 45]))
            else:
                self.strong_augmentation.insert(0, RandomRotationWithRecord(angle=[-90, 90]))

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # semi_anno = dataset_dict['semi']
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )   # numpy
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        ######################################################################
        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        ######################################################################
        strong_dict = copy.copy(dataset_dict)
        weak_dict = copy.copy(dataset_dict)

        def apply_augmentations(image, boxes, dataset_dict, augmentation):
            aug_input = T.StandardAugInput(image, boxes=boxes)
            transforms = aug_input.apply_augmentations(augmentation)
            image = aug_input.image

            image_shape = image.shape[:2]
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )

            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                dataset_dict.pop("pano_seg_file_name", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    if not self.use_instance_mask:
                        anno.pop("segmentation", None)
                    if not self.use_keypoint:
                        anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    transform_instance_annotations(
                        obj,
                        transforms,
                        image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    )
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format
                )

                # dataset_dict["instances"] = instances
                dataset_dict["instances"] = utils.filter_empty_instances(instances)

            return dataset_dict

        strong_dict = apply_augmentations(image, boxes, strong_dict, self.strong_augmentation)
        weak_dict = apply_augmentations(image, boxes, weak_dict, self.weak_augmentation)

        # get the transform matrix of geometric operator
        for obj in self.strong_augmentation:
            trans_matrix = obj.matrix if hasattr(obj, 'matrix') else None
            if trans_matrix is not None:
                if "transform_matrix" not in strong_dict:
                    strong_dict["transform_matrix"] = trans_matrix
                else:
                    base_transformation = strong_dict["transform_matrix"]
                    strong_dict["transform_matrix"] = np.dot(trans_matrix, base_transformation)

        for obj in self.weak_augmentation:
            trans_matrix = obj.matrix if hasattr(obj, 'matrix') else None
            if trans_matrix is not None:
                if "transform_matrix" not in weak_dict:
                    weak_dict["transform_matrix"] = trans_matrix
                else:
                    base_transformation = weak_dict["transform_matrix"]
                    weak_dict["transform_matrix"] = np.dot(trans_matrix, base_transformation)



        # return [strong_dict, weak_dict]

        return dict(strong=strong_dict, weak=weak_dict, semi=dataset_dict['semi'])


class DatasetMapperWithMatrix(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train and cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping  used in training: " + str(self.augmentation[0])
            )
        if cfg.INPUT.ROTATE and is_train:
            if cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
                self.augmentation.insert(0, RandomRotationWithRecord(angle=[-45, 45]))
            else:
                self.augmentation.insert(0, RandomRotationWithRecord(angle=[-90, 90]))
            logging.getLogger(__name__).info(
                "Cropping  used in training: " + str(self.augmentation[0])
            )
        # if cfg.INPUT.ROTATE : #debug to visualize /rotate when testing
        #     if cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
        #         self.augmentation.insert(0, Modified_RandomRotation(angle=[-45, 45]))
        #     else:
        #         self.augmentation.insert(0, Modified_RandomRotation(angle=[-90, 90]))

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        ######################################################################
        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        ######################################################################

        # aug_input = T.StandardAugInput(image)
        aug_input = T.StandardAugInput(image, boxes=boxes)

        transforms = aug_input.apply_augmentations(self.augmentation)
        #get the transform matrix of geometric operator
        for obj in self.augmentation:
            trans_matrix = obj.matrix  if hasattr(obj,'matrix') else None
            if trans_matrix is not None:
                if "transform_matrix" not in dataset_dict:
                    dataset_dict["transform_matrix"] = trans_matrix
                else:
                    base_transformation = dataset_dict["transform_matrix"]
                    dataset_dict["transform_matrix"] = np.dot(trans_matrix, base_transformation)
        # dataset_dict["transform_matrix"]=[torch.from_numpy(dataset_dict["transform_matrix"]).float().to(feat[0][0].device) for meta in img_metas ]
        #feat refers to coordinates of pserdo pnts

        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            # dataset_dict.pop("transform_matrix", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # dataset_dict["instances"] = instances
            dataset_dict["instances"] = utils.filter_empty_instances(instances)



        return dataset_dict

class MultiBranchPseudoDatasetMapper(DatasetMapper):
    """
    This caller enables the DatasetMapperWithBasis to build dataset pipeline for semi-supervised learning
    It produces two augmented images from a single image (unlabeled)

    The callable implement the following:

    1. Read the image from "file_name"
    2. Applies weak augmentation to the image and annotations
    2. Applies strong augmentation cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`

    As unlabeled data does not have annotations, we abandon the instance-ware crop
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the semi-supervised augmentations. The previous augmentations will be overridden."
        )
        # self.weak_augmentation = build_augmentation_weak(cfg, is_train)
        self.weak_augmentation =build_augmentation_weak(cfg, is_train) if cfg.SSL.PL.STAC else  []
        self.pure = (not cfg.SSL.PL.STAC)

        self.strong_augmentation = build_augmentation_strong(cfg, is_train)

        if cfg.INPUT.ROTATE and is_train:
            if cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
                self.strong_augmentation.insert(0, RandomRotationWithRecord(angle=[-45, 45]))
            else:
                self.strong_augmentation.insert(0, RandomRotationWithRecord(angle=[-90, 90]))

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # semi_anno = dataset_dict['semi']
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        ######################################################################
        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        ######################################################################
        strong_dict = copy.copy(dataset_dict)
        weak_dict = copy.copy(dataset_dict)

        def apply_augmentations(image, boxes, dataset_dict, augmentation):
            aug_input = T.StandardAugInput(image, boxes=boxes)
            transforms = aug_input.apply_augmentations(augmentation)
            image = aug_input.image

            image_shape = image.shape[:2]
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )

            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                dataset_dict.pop("pano_seg_file_name", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    if not self.use_instance_mask:
                        anno.pop("segmentation", None)
                    if not self.use_keypoint:
                        anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    transform_instance_annotations(
                        obj,
                        transforms,
                        image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    )
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format
                )

                # dataset_dict["instances"] = instances
                dataset_dict["instances"] = utils.filter_empty_instances(instances)

            return dataset_dict

        strong_dict = apply_augmentations(image, boxes, strong_dict, self.strong_augmentation)
        weak_dict = apply_augmentations(image, boxes, weak_dict, self.weak_augmentation)

        # get the transform matrix of geometric operator
        for obj in self.strong_augmentation:
            trans_matrix = obj.matrix if hasattr(obj, 'matrix') else None
            if trans_matrix is not None:
                if "transform_matrix" not in strong_dict:
                    strong_dict["transform_matrix"] = trans_matrix
                else:
                    base_transformation = strong_dict["transform_matrix"]
                    strong_dict["transform_matrix"] = np.dot(trans_matrix, base_transformation)

        # weak_dict["transform_matrix"]=None #if weak_aug is None ,matrix <- None
        for obj in self.weak_augmentation:
            trans_matrix = obj.matrix if hasattr(obj, 'matrix') else None
            if trans_matrix is not None:
                if "transform_matrix" not in weak_dict:
                    weak_dict["transform_matrix"] = trans_matrix
                else:
                    base_transformation = weak_dict["transform_matrix"]
                    weak_dict["transform_matrix"] = np.dot(trans_matrix, base_transformation)
        if self.pure:
            weak_dict["transform_matrix"]=None #if weak_aug is None ,matrix <- None


        return dict(strong=strong_dict, weak=weak_dict, semi=dataset_dict['semi'])