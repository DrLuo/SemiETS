import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
# from mmdet.core.mask.structures import BitmapMasks
from torch.nn import functional as F
import shapely.geometry.polygon as plg


def get_union(pD, pG):
    areaA = pD.area();
    areaB = pG.area();
    return areaA + areaB - get_intersection(pD, pG);
def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG);
    except:
        return 0
def get_poly_iouMat(bd_tgts,bd_preds):
    assert isinstance(bd_tgts,np.ndarray)
    assert isinstance(bd_preds, np.ndarray)
    b,n_gt = bd_tgts.shape[0],bd_tgts.shape[1]
    n_pred = bd_preds.shape[1]
    bsiou=[]
    for i in range(b):
        bd_tgt =bd_tgts[i]
        bd_pred = bd_preds[i]
        tgtPols = []
        detPols = []
        for bd in bd_tgt:
            poly = pnt_to_Polygon(bd)
            tgtPol=polygon_from_points(poly)
            tgtPols.append(tgtPol)
        for bd in bd_pred:
            poly = pnt_to_Polygon(bd)
            detPol = polygon_from_points(poly)
            detPols.append(detPol)
        outputShape = [len(tgtPols), len(detPols)]
        iouMat = np.empty(outputShape)
        for tgtNum in range(len(tgtPols)):
            for detNum in range(len(detPols)):
                pG = tgtPols[tgtNum]
                pD = detPols[detNum]
                iouMat[tgtNum, detNum] = get_intersection_over_union(pD, pG)
        bsiou.append(iouMat)


    return bsiou

def pnt_to_Polygon(bd_pnt):
    bd_pnt = np.hsplit(bd_pnt, 2)
    poly = np.vstack([bd_pnt[0], bd_pnt[1][::-1]])
    return poly.tolist()


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    num_points = len(points)
    # resBoxes=np.empty([1,num_points],dtype='int32')
    resBoxes = np.empty([1, num_points], dtype='float32')
    for inp in range(0, num_points, 2):
        resBoxes[0, int(inp / 2)] = float(points[int(inp)])
        resBoxes[0, int(inp / 2 + num_points / 2)] = float(points[int(inp + 1)])
    pointMat = resBoxes[0].reshape([2, int(num_points / 2)]).T
    return plg.Polygon(pointMat)

def resize_image(inputs, resize_ratio=0.5):
    down_inputs = F.interpolate(inputs,
                                scale_factor=resize_ratio,
                                mode='nearest')

    return down_inputs


def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
        [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
    ).reshape(
        -1, 2
    )  # n*4,2


def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor."""
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))


def normal_transform_pixel(
        height: int,
        width: int,
        eps: float = 1e-14,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(
        dst_pix_trans_src_pix: torch.Tensor,
        dsize_src: Tuple[int, int],
        dsize_dst: Tuple[int, int],
) -> torch.Tensor:
    check_is_tensor(dst_pix_trans_src_pix)

    if not (
            len(dst_pix_trans_src_pix.shape) == 3
            or dst_pix_trans_src_pix.shape[-2:] == (3, 3)
    ):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(
                dst_pix_trans_src_pix.shape
            )
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(
        dst_pix_trans_src_pix
    )
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix.float()).to(
        src_norm_trans_src_pix.dtype
    )
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(
        dst_pix_trans_src_pix
    )

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
            dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def warp_affine(
        src: torch.Tensor,
        M: torch.Tensor,
        dsize: Tuple[int, int],
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        raise TypeError(
            "Input src type is not a torch.Tensor. Got {}".format(type(src))
        )

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        message: str = (
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpAffine."
        )
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine

    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), dsize)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm.float())

    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        [B, C, dsize[0], dsize[1]],
        align_corners=align_corners,
    )

    return F.grid_sample(
        src.float(),
        grid,
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode,
    ).to(src.dtype)


class Transform2D:
    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_coords(coords, M, out_shape):
        if isinstance(coords, Sequence):
            assert len(coords) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(coords, M, out_shape)
            ]
        else:
            if coords.shape[0] == 0:
                return coords
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]

            return (
                warp_affine(img.float(), M[None, ...], out_shape, mode="nearest")
                    .squeeze()
                    .to(img.dtype)
            )


def filter_invalid(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None and thr is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask


def filter_invalid_with_index(bbox, label=None, score=None, index=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if index is not None:
            index = index[valid]
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if index is not None:
            index = index[valid]
    return bbox, label, index


def filter_ignore(bbox, label=None, score=None, mask=None, up_thr=0.0, ig_thr=0.1, min_size=0):
    """
    filter the ignore pseudo bbox from the detection bboxes.
    """
    if score is not None:
        valid = torch.logical_and(score > ig_thr, score < up_thr)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask


def filter_invalid_class_wise(bbox, label=None, score=None, mask=None, class_wise_thr=None, min_size=0):
    """
    use the class-wise threshold to filter the  pseudo bbox
    """
    if score is not None:
        valid = score > class_wise_thr[label]
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask


def filter_ignore_class_wise(bbox, label=None, score=None, mask=None, up_thr=0.0, ig_thr=0.1, min_size=0):
    """
    filter the ignore pseudo bbox from the detection bboxes with the classwise threshold.
    the up_thr now is a classwise threshold with the shape of [num_class,]
    """
    if score is not None:
        valid = torch.logical_and(score > ig_thr, score < up_thr[label])
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask


def filter_invalid_soft_label(bbox, label=None, soft_label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if soft_label is not None:
            soft_label = soft_label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if soft_label is not None:
            soft_label = soft_label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, soft_label, mask