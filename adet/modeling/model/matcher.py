"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from adet.utils.curve_utils import BezierSampler
import numpy as np

def gen_relative_distance_mat(outputs , targets):
    out_ctrls = outputs["pred_ctrl_points"].flatten(0, 1)
    tgt_ctrls = torch.cat([v["ctrl_points"] for v in targets])
    out_bds = outputs["pred_bd_points"].flatten(0,1)
    out_bds = out_bds.view(out_bds.shape[0],-1,2)
    tgt_bds = torch.cat([v["bd_points"] for v in targets])
    tgt_bds = tgt_bds.view(tgt_bds.shape[0],-1,2)
    out_mean_ctrls = out_ctrls.mean(dim=1)
    tgt_mean_ctrls = tgt_ctrls.mean(dim=1)
    center_dis = torch.cdist(out_mean_ctrls,tgt_mean_ctrls,p=2) # 200 , gt_nums
    # norm_factor = [torch.cdist(out_bds[i],tgt_bds[j],p=2).max() for i, j in zip(range(len(out_bds)),range(len(tgt_bds)))]
    norm_factor_matrix = torch.ones_like(center_dis)
    indices = [0,24,25,-1]
    for i in range(len(out_bds)):
        for j in range(len(tgt_bds)):
            out_bds_pts = out_bds[i][indices,:]
            tgt_bds_pts = tgt_bds[j][indices,:]
            norm_factor_matrix[i,j] = torch.cdist(out_bds_pts,tgt_bds_pts,p=2).max()
            # norm_factor_matrix[i,j] = torch.cdist(out_bds[i],tgt_bds[j],p=2).max()
    relative_ctrl_dis_mat = center_dis / norm_factor_matrix
    return relative_ctrl_dis_mat


class CtrlPointCost(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

            out_texts = F.log_softmax(outputs['pred_text_logits'], dim=-1)  # (bs, n_q, n_pts, voc+1)
            n_pts, voc = out_texts.shape[2], out_texts.shape[-1] - 1
            target_texts = torch.cat([v["texts"] for v in targets])
            target_lengths = (target_texts != voc).long().sum(dim=-1)
            target_texts = torch.split(target_texts, sizes, dim=0)
            target_lengths = torch.split(target_lengths, sizes)
            texts_cost_list = []
            for out_texts_batch, targe_texts_batch, target_len_batch in zip(out_texts, target_texts, target_lengths):
                out_texts_batch_temp = out_texts_batch.repeat(targe_texts_batch.shape[0], 1, 1).permute(1, 0, 2)
                input_len = torch.full((out_texts_batch_temp.size(1),), out_texts_batch_temp.size(0), dtype=torch.long)
                if targe_texts_batch.shape[0] > 0:
                    targe_texts_batch_temp = torch.cat([
                        t[:target_len_batch[t_idx]].repeat(num_queries) for t_idx, t in enumerate(targe_texts_batch)
                    ])

                    target_len_batch_temp = target_len_batch.reshape((-1, 1)).repeat(1, num_queries).reshape(-1)
                    text_cost = F.ctc_loss(
                        out_texts_batch_temp,
                        targe_texts_batch_temp,
                        input_len,
                        target_len_batch_temp,
                        blank=voc,
                        zero_infinity=True,
                        reduction='none'
                    )
                    if float('inf') in text_cost:
                        print("inf in CTC Loss")
                    text_cost.div_(target_len_batch_temp)
                    if float('inf') in text_cost:
                        print("0 in CTC target_len_batch_temp")
                        print(target_len_batch_temp)
                    text_cost_cpu = text_cost.reshape((-1, num_queries)).transpose(1, 0).cpu()
                else:
                    text_cost_cpu = torch.zeros((num_queries, 0), dtype=torch.float32)
                texts_cost_list.append(text_cost_cpu)

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)

            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)

            C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))


            indices = []
            costs = []
            for i, c in enumerate(C.split(sizes, -1)):
                try:
                    matched_row_inds, matched_col_inds = linear_sum_assignment(
                        c[i] + self.text_weight * texts_cost_list[i]
                    ) if sizes[i] > 0 else (np.array([]).astype(np.int64), np.array([])
                                            )

                    matched_row_inds = torch.as_tensor(matched_row_inds, dtype=torch.int64)
                    matched_col_inds = torch.as_tensor(matched_col_inds, dtype=torch.int64)

                    indices.append((matched_row_inds, matched_col_inds))
                    costs.append(c[i][matched_row_inds, matched_col_inds])

                except:
                    raise RuntimeError

            return indices, costs

class DetectionCost(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets, idx, indices):
        with torch.no_grad():
            #################################################################
            src_ctrl_points = outputs['pred_ctrl_points'][idx]
            target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            src_logits = outputs["pred_logits"][idx].sigmoid()
            #######################################################################

            out_prob = src_logits

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = src_ctrl_points.flatten(-2)
            tgt_pts = target_ctrl_points.flatten(-2)
            out_1 = torch.full_like(out_prob,0.95)

            out_2 = torch.full_like(out_prob,0.05)

            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            neg_avg_factor = (1 - self.alpha) * (out_1 ** self.gamma) * \
                (-(1 - out_1.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_avg_factor = self.alpha * ((1 - out_2) ** self.gamma) * \
                (-(out_2.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0])
            # turn positive
            norm_cost_class = (cost_class + neg_avg_factor[...,0]) / (neg_avg_factor[...,0] + pos_avg_factor[...,0])

            norm_cost_class = norm_cost_class.mean(-1,keepdims=True)
            # cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)

            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)
            norm_cost_kpts = cost_kpts / tgt_pts.shape[1]

            C = (self.class_weight * norm_cost_class + self.coord_weight * norm_cost_kpts) / (self.class_weight+self.coord_weight)
            C = torch.clamp(C,0,1)

            return C


class CtrlPointHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets, match_idx=None):
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

            out_texts = F.log_softmax(outputs['pred_text_logits'], dim=-1)  # (bs, n_q, n_pts, voc+1)
            n_pts, voc = out_texts.shape[2], out_texts.shape[-1] - 1
            target_texts = torch.cat([v["texts"] for v in targets])
            target_lengths = (target_texts != voc).long().sum(dim=-1)
            target_texts = torch.split(target_texts, sizes, dim=0)
            target_lengths = torch.split(target_lengths, sizes)
            texts_cost_list = []
            for out_texts_batch, targe_texts_batch, target_len_batch in zip(out_texts, target_texts, target_lengths):
                out_texts_batch_temp = out_texts_batch.repeat(targe_texts_batch.shape[0], 1, 1).permute(1, 0, 2)
                input_len = torch.full((out_texts_batch_temp.size(1),), out_texts_batch_temp.size(0), dtype=torch.long)
                if targe_texts_batch.shape[0] > 0:
                    targe_texts_batch_temp = torch.cat([
                        t[:target_len_batch[t_idx]].repeat(num_queries) for t_idx, t in enumerate(targe_texts_batch)
                    ])
                # except:
                #     targe_texts_batch_temp = torch.zeros((0), dtype=torch.int32, device=out_texts_batch_temp.device)

                    target_len_batch_temp = target_len_batch.reshape((-1, 1)).repeat(1, num_queries).reshape(-1)
                    text_cost = F.ctc_loss(
                        out_texts_batch_temp,
                        targe_texts_batch_temp,
                        input_len,
                        target_len_batch_temp,
                        blank=voc,
                        zero_infinity=True,
                        reduction='none'
                    )
                    if float('inf') in text_cost:
                        print("inf in CTC Loss")
                    if True in torch.isnan(text_cost):
                        print("nan in CTC Loss")
                    text_cost.div_(target_len_batch_temp)
                    if float('inf') in text_cost:
                        print("0 in CTC target_len_batch_temp")
                        print(target_len_batch_temp)
                    text_cost_cpu = text_cost.reshape((-1, num_queries)).transpose(1, 0).cpu()
                else:
                    text_cost_cpu = torch.zeros((num_queries, 0), dtype=torch.float32)
                texts_cost_list.append(text_cost_cpu)

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)

            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)

            C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))

            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                try:
                    indices.append(linear_sum_assignment(
                            c[i] + self.text_weight * texts_cost_list[i]
                        ) if sizes[i]>0 else (np.array([]).astype(np.int64), np.array([])
                    ))
                    # print(c[i].shape)
                except:

                    np.save("cm.npy", c[i])
                    np.save("tm.npy", texts_cost_list[i])
                    raise RuntimeError



            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class CtrlPointDetectionCostMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets, match_idx=None):
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)

            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)

            C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))

            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                try:
                    indices.append(linear_sum_assignment(
                            c[i]
                        ) if sizes[i]>0 else (np.array([]).astype(np.int64), np.array([])
                    ))
                    # print(c[i].shape)
                except:

                    np.save("cm.npy", c[i])
                    raise RuntimeError


            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



class CtrlPointRecCostMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets,match_idx=None):
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]


            out_texts = F.log_softmax(outputs['pred_text_logits'], dim=-1)  # (bs, n_q, n_pts, voc+1)
            n_pts, voc = out_texts.shape[2], out_texts.shape[-1] - 1
            target_texts = torch.cat([v["texts"] for v in targets])
            target_lengths = (target_texts != voc).long().sum(dim=-1)
            target_texts = torch.split(target_texts, sizes, dim=0)
            target_lengths = torch.split(target_lengths, sizes)
            texts_cost_list = []
            for out_texts_batch, targe_texts_batch, target_len_batch in zip(out_texts, target_texts, target_lengths):
                out_texts_batch_temp = out_texts_batch.repeat(targe_texts_batch.shape[0], 1, 1).permute(1, 0, 2)
                input_len = torch.full((out_texts_batch_temp.size(1),), out_texts_batch_temp.size(0), dtype=torch.long)
                if targe_texts_batch.shape[0] > 0:
                    targe_texts_batch_temp = torch.cat([
                        t[:target_len_batch[t_idx]].repeat(num_queries) for t_idx, t in enumerate(targe_texts_batch)
                    ])

                    target_len_batch_temp = target_len_batch.reshape((-1, 1)).repeat(1, num_queries).reshape(-1)
                    text_cost = F.ctc_loss(
                        out_texts_batch_temp,
                        targe_texts_batch_temp,
                        input_len,
                        target_len_batch_temp,
                        blank=voc,
                        zero_infinity=True,
                        reduction='none'
                    )
                    if float('inf') in text_cost:
                        print("inf in CTC Loss")
                    if True in torch.isnan(text_cost):
                        print("nan in CTC Loss")
                    text_cost.div_(target_len_batch_temp)
                    if float('inf') in text_cost:
                        print("0 in CTC target_len_batch_temp")
                        print(target_len_batch_temp)
                    text_cost_cpu = text_cost.reshape((-1, num_queries)).transpose(1, 0).cpu()
                else:
                    text_cost_cpu = torch.zeros((num_queries, 0), dtype=torch.float32)
                texts_cost_list.append(text_cost_cpu)

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)

            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)

            C = self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))

            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                try:
                    indices.append(linear_sum_assignment(
                        c[i] + self.text_weight * texts_cost_list[i]
                    ) if sizes[i] > 0 else (np.array([]).astype(np.int64), np.array([])
                                            ))
                    # print(c[i].shape)
                except:

                    np.save("cm.npy", c[i])
                    np.save("tm.npy", texts_cost_list[i])
                    raise RuntimeError

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class BezierHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            num_sample_points: int = 100,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.num_sample_points = num_sample_points
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_beziers = outputs["pred_beziers"].flatten(0, 1).view(-1, 4, 2)  # (batch_size * num_queries, 4, 2)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_beziers = torch.cat([v["beziers"] for v in targets])  # (g, 4, 2)

            # Compute the classification cost.
            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost betweeen sampled points on Bezier curve
            cost_coord = torch.cdist(
                (self.bezier_sampler.get_sample_points(out_beziers)).flatten(start_dim=-2),
                (self.bezier_sampler.get_sample_points(tgt_beziers)).flatten(start_dim=-2),
                p=1
            )

            sizes = [len(v["beziers"]) for v in targets]

            C = self.class_weight * cost_class + self.coord_weight * cost_coord
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))

            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class BezierPointsHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            num_sample_points: int = 100,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.num_sample_points = num_sample_points
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)

    def forward(self, outputs, targets, match_idx=None):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_beziers = outputs["pred_beziers"].flatten(0, 1).view(-1, 4, 2)  # (batch_size * num_queries, 4, 2)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            has_bezier = 'beziers' in targets[0]
            if has_bezier:
                tgt_beziers = torch.cat([v["beziers"] for v in targets])  # (g, 4, 2)
                sizes = [len(v["beziers"]) for v in targets]
            else:
                tgt_beziers = torch.cat([v["ctrl_points"] for v in targets], dim=0)  # (g, 25, 2)
                sizes = [len(v["ctrl_points"]) for v in targets]

            # Compute the classification cost.
            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost betweeen sampled points on Bezier curve
            cost_coord = torch.cdist(
                (self.bezier_sampler.get_sample_points(out_beziers)).flatten(start_dim=-2),
                (self.bezier_sampler.get_sample_points(tgt_beziers)).flatten(
                    start_dim=-2) if has_bezier else tgt_beziers.flatten(start_dim=-2),
                p=1
            )

            C = self.class_weight * cost_class + self.coord_weight * cost_coord
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))

            if has_bezier:
                sizes = [len(v["beziers"]) for v in targets]
            else:
                sizes = [len(v["ctrl_points"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i]) if sizes[i]>0 else (np.array([]).astype(np.int64), np.array([]).astype(np.int64))
                for i, c in enumerate(C.split(sizes, -1))
            ]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



class One2ManyBezierPointsHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            num_sample_points: int = 100,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            match_num: int = 13
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.num_sample_points = num_sample_points
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.match_num = match_num
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)

    def forward(self, outputs, targets,match_idx=None):
        INF = 100000000
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_beziers = outputs["pred_beziers"].flatten(0, 1).view(-1, 4, 2)  # (batch_size * num_queries, 4, 2)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            has_bezier = 'beziers' in targets[0]
            if has_bezier:
                tgt_beziers = torch.cat([v["beziers"] for v in targets])  # (g, 4, 2)
            else:
                tgt_beziers = torch.cat([v["ctrl_points"] for v in targets], dim=0)  # (g, 25, 2)

            # Compute the classification cost.
            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost betweeen sampled points on Bezier curve
            cost_coord = torch.cdist(
                (self.bezier_sampler.get_sample_points(out_beziers)).flatten(start_dim=-2),
                (self.bezier_sampler.get_sample_points(tgt_beziers)).flatten(
                    start_dim=-2) if has_bezier else tgt_beziers.flatten(start_dim=-2),
                p=1
            )

            if has_bezier:
                sizes = [len(v["beziers"]) for v in targets]
            else:
                sizes = [len(v["ctrl_points"]) for v in targets]

            C = self.class_weight * cost_class + self.coord_weight * cost_coord
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))

            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                topk_match = torch.topk(c[i], self.match_num, dim=0, largest=False)
                indices.append((topk_match[1].transpose(0, 1).flatten(),
                                torch.arange(c[i].shape[1], dtype=torch.int64).view(-1, 1).repeat(1,
                                                                                                  self.match_num).flatten()) if
                               sizes[i] > 0 else (
                    torch.tensor([]).to(dtype=torch.int64), torch.tensor([]).to(dtype=torch.int64)))

            # deal with a single candidate assigned to multiple gt_bboxes
            o2m_indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                if len(indices[i][0]) == 0:
                    o2m_indices.append(indices[i])
                else:
                    temp_cost = c[i].detach()
                    temp_cost = temp_cost.detach()
                    cost_inf = torch.full_like(temp_cost, INF)
                    cost_inf[indices[i]] = temp_cost[indices[i]]
                    source_indices = torch.unique(indices[i][0])
                    min_values, min_indices = cost_inf.min(dim=1)

                    target_indices = min_indices[source_indices]

                    o2m_indices.append((source_indices, target_indices))

            return o2m_indices


class One2ManyBezierPointsHungarianMatcher_withdynamic(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            num_sample_points: int = 100,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            match_num: int = 13,
            topk: int = 13,
            use_dynamic_k=False,
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.num_sample_points = num_sample_points
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.match_num = match_num
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)
        self.topk = topk
        self.use_dynamic_k = use_dynamic_k

    def forward(self, outputs, targets):
        INF = 100000000
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_beziers = outputs["pred_beziers"].flatten(0, 1).view(-1, 4, 2)  # (batch_size * num_queries, 4, 2)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            has_bezier = 'beziers' in targets[0]
            if has_bezier:
                tgt_beziers = torch.cat([v["beziers"] for v in targets])  # (g, 4, 2)
            else:
                tgt_beziers = torch.cat([v["ctrl_points"] for v in targets], dim=0)  # (g, 25, 2)

            # Compute the classification cost.
            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost betweeen sampled points on Bezier curve
            cost_coord = torch.cdist(
                (self.bezier_sampler.get_sample_points(out_beziers)).flatten(start_dim=-2),
                (self.bezier_sampler.get_sample_points(tgt_beziers)).flatten(
                    start_dim=-2) if has_bezier else tgt_beziers.flatten(start_dim=-2),
                p=1
            )

            if has_bezier:
                sizes = [len(v["beziers"]) for v in targets]
            else:
                sizes = [len(v["ctrl_points"]) for v in targets]

            C = self.class_weight * cost_class + self.coord_weight * cost_coord
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))

            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                topk_match = torch.topk(c[i], self.match_num, dim=0, largest=False)
                indices.append((topk_match[1].transpose(0, 1).flatten(),
                                torch.arange(c[i].shape[1], dtype=torch.int64).view(-1, 1).repeat(1,
                                                                                                  self.match_num).flatten()) if
                               sizes[i] > 0 else (
                    torch.tensor([]).to(dtype=torch.int64), torch.tensor([]).to(dtype=torch.int64)))

            # deal with a single candidate assigned to multiple gt_bboxes
            o2m_indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                if len(indices[i][0]) == 0:
                    o2m_indices.append(indices[i])
                else:
                    temp_cost = c[i].detach()
                    temp_cost = temp_cost.detach()
                    cost_inf = torch.full_like(temp_cost, INF)
                    cost_inf[indices[i]] = temp_cost[indices[i]]
                    source_indices = torch.unique(indices[i][0])
                    min_values, min_indices = cost_inf.min(dim=1)

                    target_indices = min_indices[source_indices]

                    o2m_indices.append((source_indices, target_indices))

            return o2m_indices


class One2ManyCtrlPointHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            match_num: int = 13
    ):
        """Creates the One-to-many matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        # super().__init__()
        # self.class_weight = class_weight
        # self.coord_weight = coord_weight
        # self.num_sample_points = num_sample_points
        # self.alpha = focal_alpha
        # self.gamma = focal_gamma
        # assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        # self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.match_num = match_num
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets,match_idx=None):
        INF = 100000000
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

            out_texts = F.log_softmax(outputs['pred_text_logits'], dim=-1)  # (bs, n_q, n_pts, voc+1)
            n_pts, voc = out_texts.shape[2], out_texts.shape[-1] - 1
            target_texts = torch.cat([v["texts"] for v in targets])
            target_lengths = (target_texts != voc).long().sum(dim=-1)
            target_texts = torch.split(target_texts, sizes, dim=0)
            target_lengths = torch.split(target_lengths, sizes)
            texts_cost_list = []
            for out_texts_batch, targe_texts_batch, target_len_batch in zip(out_texts, target_texts, target_lengths):
                out_texts_batch_temp = out_texts_batch.repeat(targe_texts_batch.shape[0], 1, 1).permute(1, 0, 2)
                input_len = torch.full((out_texts_batch_temp.size(1),), out_texts_batch_temp.size(0), dtype=torch.long)
                if targe_texts_batch.shape[0] > 0:
                    targe_texts_batch_temp = torch.cat([
                        t[:target_len_batch[t_idx]].repeat(num_queries) for t_idx, t in enumerate(targe_texts_batch)
                    ])
                    # except:
                    #     targe_texts_batch_temp = torch.zeros((0), dtype=torch.int32, device=out_texts_batch_temp.device)

                    target_len_batch_temp = target_len_batch.reshape((-1, 1)).repeat(1, num_queries).reshape(-1)
                    text_cost = F.ctc_loss(
                        out_texts_batch_temp,
                        targe_texts_batch_temp,
                        input_len,
                        target_len_batch_temp,
                        blank=voc,
                        zero_infinity=True,
                        reduction='none'
                    )
                    if float('inf') in text_cost:
                        print("inf in CTC Loss")
                    text_cost.div_(target_len_batch_temp)
                    if float('inf') in text_cost:
                        print("0 in CTC target_len_batch_temp")
                        print(target_len_batch_temp)
                    text_cost_cpu = text_cost.reshape((-1, num_queries)).transpose(1, 0).cpu()
                else:
                    text_cost_cpu = torch.zeros((num_queries, 0), dtype=torch.float32)
                texts_cost_list.append(text_cost_cpu)

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)

            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)

            C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).detach().cpu()

            indices = []
            # cost_candidates = []
            for i, c in enumerate(C.split(sizes, -1)):
                try:
                    topk_match = torch.topk(c[i] + self.text_weight * texts_cost_list[i],
                                               self.match_num,
                                               dim=0,
                                               largest=False)
                    indices.append((topk_match[1].transpose(0,1).flatten(),
                                    torch.arange(c[i].shape[1], dtype=torch.int64).view(-1,1).repeat(1,self.match_num).flatten()) if sizes[i] > 0 else (
                    torch.tensor([]).to(dtype=torch.int64), torch.tensor([]).to(dtype=torch.int64)))
                    # cost_candidates.append(topk_match[0].transpose(0,1).flatten() if sizes[i] > 0 else torch.tensor([]).to(dtype=torch.float32))
                    # print(c[i].shape)
                except:

                    np.save("cm.npy", c[i])
                    np.save("tm.npy", texts_cost_list[i])
                    raise RuntimeError

            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            # deal with a single candidate assigned to multiple gt_bboxes
            o2m_indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                if len(indices[i][0]) == 0:
                    o2m_indices.append(indices[i])
                else:
                    temp_cost = (c[i] + self.text_weight * texts_cost_list[i]).detach()
                    temp_cost = temp_cost.detach()
                    cost_inf = torch.full_like(temp_cost, INF)
                    cost_inf[indices[i]] = temp_cost[indices[i]]
                    source_indices = torch.unique(indices[i][0])
                    min_values, min_indices = cost_inf.min(dim=1)

                    target_indices = min_indices[source_indices]

                    o2m_indices.append((source_indices, target_indices))

            return o2m_indices

class One2ManyCtrlPointHungarianMatcher_withdynamic(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            match_num: int = 5,
            topk: int =13,
            use_dynamic_k = False,
    ):
        """Creates the One-to-many matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        # super().__init__()
        # self.class_weight = class_weight
        # self.coord_weight = coord_weight
        # self.num_sample_points = num_sample_points
        # self.alpha = focal_alpha
        # self.gamma = focal_gamma
        # assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        # self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.match_num = match_num
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.topk = topk
        self.use_dynamic_k = use_dynamic_k



    def forward(self, outputs, targets, specific_match_num = None):
        INF = 100000000
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

            out_texts = F.log_softmax(outputs['pred_text_logits'], dim=-1)  # (bs, n_q, n_pts, voc+1)
            n_pts, voc = out_texts.shape[2], out_texts.shape[-1] - 1
            target_texts = torch.cat([v["texts"] for v in targets])
            target_lengths = (target_texts != voc).long().sum(dim=-1)
            target_texts = torch.split(target_texts, sizes, dim=0)
            target_lengths = torch.split(target_lengths, sizes)
            texts_cost_list = []
            for out_texts_batch, targe_texts_batch, target_len_batch in zip(out_texts, target_texts, target_lengths):
                out_texts_batch_temp = out_texts_batch.repeat(targe_texts_batch.shape[0], 1, 1).permute(1, 0, 2)
                input_len = torch.full((out_texts_batch_temp.size(1),), out_texts_batch_temp.size(0), dtype=torch.long)
                if targe_texts_batch.shape[0] > 0:
                    targe_texts_batch_temp = torch.cat([
                        t[:target_len_batch[t_idx]].repeat(num_queries) for t_idx, t in enumerate(targe_texts_batch)
                    ])
                    # except:
                    #     targe_texts_batch_temp = torch.zeros((0), dtype=torch.int32, device=out_texts_batch_temp.device)

                    target_len_batch_temp = target_len_batch.reshape((-1, 1)).repeat(1, num_queries).reshape(-1)
                    text_cost = F.ctc_loss(
                        out_texts_batch_temp,
                        targe_texts_batch_temp,
                        input_len,
                        target_len_batch_temp,
                        blank=voc,
                        zero_infinity=True,
                        reduction='none'
                    )
                    if float('inf') in text_cost:
                        print("inf in CTC Loss")
                    text_cost.div_(target_len_batch_temp)
                    if float('inf') in text_cost:
                        print("0 in CTC target_len_batch_temp")
                        print(target_len_batch_temp)
                    text_cost_cpu = text_cost.reshape((-1, num_queries)).transpose(1, 0).cpu()
                else:
                    text_cost_cpu = torch.zeros((num_queries, 0), dtype=torch.float32)
                texts_cost_list.append(text_cost_cpu)

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)

            dtype = out_prob.dtype
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob.to(torch.float32) + 1e-8).log()).to(dtype)
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)


            C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu() if sum(sizes) > 0 else torch.zeros((bs, num_queries, 0))
            # D1 = cost_kpts.view(C.shape) / tgt_pts.shape[1]
            D = C
            if self.use_dynamic_k:
                try:
                    D = gen_relative_distance_mat(outputs, targets)
                    D = D.view(bs, num_queries, -1).cpu()
                # bd_pts = torch.cat([v["bd_points"] for v in targets]).view(bs,-1,25,4).detach().cpu().numpy()
                # pred_bd_pts = outputs["pred_bd_points"].flatten(0, 1).view(bs,-1,25,4).detach().cpu().numpy()
                # iou_mat = get_poly_iouMat(bd_pts,pred_bd_pts)
                #lack img info to de_normalize the coords
                except:
                    print('None_valid_targets_occur!!!!!!!!!')

            indices = []
            # cost_candidates = []
            for i, (c, d) in enumerate(zip(C.split(sizes, -1), D.split(sizes, -1))):
                if self.use_dynamic_k:
                #calculate dynamic_k upon 1-L1 dis, and the top-k sum of semi_5s metric is used for dynamic_k for each gt
                    cost = c[i] + self.text_weight * texts_cost_list[i]
                    d , temp_idx , temp_cat = d[i] , [] , []
                    candidate_topk = min(self.topk, d.size(0))
                    topk_mets, _ = torch.topk( 1-d , candidate_topk , dim=0 , largest=True)
                    dynamic_ks = torch.clamp(topk_mets.sum(0).int(), min=1)
                    # if dynamic_ks is not None:
                    #     print(f"current dynamic k-means is [{int(dynamic_ks.sum(0)/len(dynamic_ks))}]")
                    for gt_idx in range(len(dynamic_ks)):
                        _, pos_idx = torch.topk(cost[:, gt_idx],
                                                k=dynamic_ks[gt_idx].item(),
                                                largest=False)
                        temp_idx.extend(pos_idx)
                        temp_cat.extend([gt_idx for l in range(len(pos_idx))])
                    temp_idx = torch.Tensor(temp_idx)
                    temp_cat = torch.Tensor(temp_cat)
                    indices.append((temp_idx,temp_cat)
                                   if sizes[i] > 0 else (
                    torch.tensor([]).to(dtype=torch.int64), torch.tensor([]).to(dtype=torch.int64)))
                else:
                    try:
                        topk_match = torch.topk(c[i] + self.text_weight * texts_cost_list[i],
                                                   specific_match_num if specific_match_num is not None else self.match_num,
                                                   dim=0,
                                                   largest=False)
                        indices.append((topk_match[1].transpose(0,1).flatten(),
                                        torch.arange(c[i].shape[1], dtype=torch.int64).view(-1,1).repeat(1,specific_match_num if specific_match_num is not None else self.match_num,).flatten()) if sizes[i] > 0 else (
                        torch.tensor([]).to(dtype=torch.int64), torch.tensor([]).to(dtype=torch.int64)))
                        # cost_candidates.append(topk_match[0].transpose(0,1).flatten() if sizes[i] > 0 else torch.tensor([]).to(dtype=torch.float32))
                        # print(c[i].shape)
                    except:

                        np.save("cm.npy", c[i])
                        np.save("tm.npy", texts_cost_list[i])
                        raise RuntimeError

            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            # deal with a single candidate assigned to multiple gt_bboxes
            o2m_indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                if len(indices[i][0]) == 0:
                    o2m_indices.append(indices[i])
                else:
                    temp_cost = c[i] + self.text_weight * texts_cost_list[i].detach()
                    temp_cost=temp_cost.detach()
                    cost_inf = torch.full_like(temp_cost, INF)
                    cost_inf[indices[i]] = temp_cost[indices[i]]
                    source_indices = torch.unique(indices[i][0])
                    min_values, min_indices = cost_inf.min(dim=1)

                    target_indices = min_indices[source_indices]

                    o2m_indices.append((source_indices, target_indices))

            return o2m_indices




def build_matcher(cfg):
    cfg = cfg.MODEL.TRANSFORMER.LOSS
    # return BezierHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
    #                               coord_weight=cfg.BEZIER_COORD_WEIGHT,
    #                               num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
    #                               focal_alpha=cfg.FOCAL_ALPHA,
    #                               focal_gamma=cfg.FOCAL_GAMMA), \
    return BezierPointsHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
                                        coord_weight=cfg.BEZIER_COORD_WEIGHT,
                                        num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
                                        focal_alpha=cfg.FOCAL_ALPHA,
                                        focal_gamma=cfg.FOCAL_GAMMA), \
           CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                     coord_weight=cfg.POINT_COORD_WEIGHT,
                                     text_weight=cfg.POINT_TEXT_WEIGHT,
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA)


def build_matcher_o2m_full(cfg):
    cfg = cfg.MODEL.TRANSFORMER.LOSS
    # return BezierHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
    #                               coord_weight=cfg.BEZIER_COORD_WEIGHT,
    #                               num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
    #                               focal_alpha=cfg.FOCAL_ALPHA,
    #                               focal_gamma=cfg.FOCAL_GAMMA), \
    return BezierPointsHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
                                  coord_weight=cfg.BEZIER_COORD_WEIGHT,
                                  num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
                                  focal_alpha=cfg.FOCAL_ALPHA,
                                  focal_gamma=cfg.FOCAL_GAMMA), \
           CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                     coord_weight=cfg.POINT_COORD_WEIGHT,
                                     text_weight=cfg.POINT_TEXT_WEIGHT,
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA), \
           One2ManyBezierPointsHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
                                        coord_weight=cfg.BEZIER_COORD_WEIGHT,
                                        num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
                                        match_num=cfg.O2M_ENC_MATCH_NUM,
                                        focal_alpha=cfg.FOCAL_ALPHA,
                                        focal_gamma=cfg.FOCAL_GAMMA), \
           One2ManyCtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                    coord_weight=cfg.POINT_COORD_WEIGHT,
                    text_weight=cfg.POINT_TEXT_WEIGHT,
                    match_num=cfg.O2M_MATCH_NUM,
                    focal_alpha=cfg.FOCAL_ALPHA,
                    focal_gamma=cfg.FOCAL_GAMMA)


def build_all_matcher_semi(cfg):
    cfg = cfg.MODEL.TRANSFORMER.LOSS
    # return BezierHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
    #                               coord_weight=cfg.BEZIER_COORD_WEIGHT,
    #                               num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
    #                               focal_alpha=cfg.FOCAL_ALPHA,
    #                               focal_gamma=cfg.FOCAL_GAMMA), \
    return BezierPointsHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
                                  coord_weight=cfg.BEZIER_COORD_WEIGHT,
                                  num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
                                  focal_alpha=cfg.FOCAL_ALPHA,
                                  focal_gamma=cfg.FOCAL_GAMMA), \
           CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                     coord_weight=cfg.POINT_COORD_WEIGHT,
                                     text_weight=cfg.POINT_TEXT_WEIGHT,
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA), \
           One2ManyBezierPointsHungarianMatcher_withdynamic(class_weight=cfg.BEZIER_CLASS_WEIGHT,
                                        coord_weight=cfg.BEZIER_COORD_WEIGHT,
                                        num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
                                        match_num= cfg.O2M_ENC_MATCH_NUM,
                                        focal_alpha=cfg.FOCAL_ALPHA,
                                        focal_gamma=cfg.FOCAL_GAMMA,
                                        use_dynamic_k=cfg.USE_DYNAMIC_K), \
           One2ManyCtrlPointHungarianMatcher_withdynamic(class_weight=cfg.POINT_CLASS_WEIGHT,
                    coord_weight=cfg.POINT_COORD_WEIGHT,
                    text_weight=cfg.POINT_TEXT_WEIGHT,
                    match_num=cfg.O2M_MATCH_NUM,
                    focal_alpha=cfg.FOCAL_ALPHA,
                    focal_gamma=cfg.FOCAL_GAMMA,
                    use_dynamic_k=cfg.USE_DYNAMIC_K), \
           CtrlPointDetectionCostMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                     coord_weight=cfg.POINT_COORD_WEIGHT,
                                     text_weight=cfg.POINT_TEXT_WEIGHT,
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA), \
           CtrlPointRecCostMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                     coord_weight=cfg.POINT_COORD_WEIGHT,
                                     text_weight=cfg.POINT_TEXT_WEIGHT,
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA)



