import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from adet.utils.misc import accuracy, is_dist_avail_and_initialized
from detectron2.utils.comm import get_world_size
from adet.utils.curve_utils import BezierSampler
import numpy as np
from adet.utils.polygon_utils import make_valid_poly, get_intersection_over_union, pnt_to_Polygon,get_intersection_over_union_from_pnts
from shapely.geometry import  Polygon
from rapidfuzz import string_metric
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

@torch.no_grad()
def gen_relative_distance_mat(outputs , targets, idx, indices,valid_idx=None):
    epsilon = 1e-15 # avoid max boundary distance equals to zero which makes RCD nan
    out_ctrls = outputs["pred_ctrl_points"][idx]
    tgt_ctrls = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    out_bds = outputs["pred_bd_points"][idx]
    tgt_bds = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    try:
        out_bds = out_bds.view(out_bds.shape[0], -1, 2)
        tgt_bds = tgt_bds.view(tgt_bds.shape[0],-1, 2)
    except:
        if not torch.all(
                torch.Tensor([out_bds.shape[0], tgt_bds.shape[0], out_ctrls.shape[0], tgt_ctrls.shape[0]]) > 0):
            return None  # Empty instances
        else:
            print('unknown error')
            print(f'out_ctrls{out_ctrls} shape{out_ctrls.shape}')
            print(f'tgt_ctrls{tgt_ctrls} shape{tgt_ctrls.shape}')
            print(f'out_bds{out_bds} shape{out_bds.shape}')
            print(f'tgt_bds{tgt_bds} shape{tgt_bds.shape}')
    if valid_idx is not None:
        out_ctrls = out_ctrls[valid_idx]
        tgt_ctrls = tgt_ctrls[valid_idx]
        out_bds = out_bds[valid_idx]
        tgt_bds = tgt_bds[valid_idx]
    out_mean_ctrls = out_ctrls.mean(dim=1)
    tgt_mean_ctrls = tgt_ctrls.mean(dim=1)
    #already matched case
    center_dis = torch.cdist(out_mean_ctrls,tgt_mean_ctrls,p=2).diag()
    norm_factor = torch.cdist(out_bds, tgt_bds, p=2).flatten(1).max(1)[0]
    # relative_center_distance = center_dis / (norm_factor + epsilon)
    relative_center_distance = torch.clip(center_dis / (norm_factor + epsilon),min=0,max=1-epsilon)
    return relative_center_distance



@torch.no_grad()
def gen_polygon_IOU_mat(outputs , targets, idx, indices,valid_idx=None):
    #IOU between output and targets after matching
    out_bds = outputs["pred_bd_points"][idx]
    tgt_bds = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    if not torch.all(
            torch.Tensor([out_bds.shape[0], tgt_bds.shape[0]]) > 0):
        return None  # Empty instances

    if valid_idx is not None:
        out_bds = out_bds[valid_idx]
        tgt_bds = tgt_bds[valid_idx]

    out_bds = out_bds.detach().cpu().numpy() #N,25,4
    tgt_bds = tgt_bds.detach().cpu().numpy() #N.25,4

    detPols = []
    gtPols = []
    #normalized bd points -> anti clock wise pnts
    for t in tgt_bds:
        poly_pnt = pnt_to_Polygon(t) # 50
        temp_poly = make_valid_poly(poly_pnt) #check and fix to get valid polygon using buffer


        gtPols.append(temp_poly)

    for o in out_bds:
        poly_pnt = pnt_to_Polygon(o)
        temp_poly = make_valid_poly(poly_pnt)

        detPols.append(temp_poly)

    total_num = len(tgt_bds)
    iouMat = [get_intersection_over_union(gtPols[i],detPols[i]) for i in range(total_num)]
    # check whether the convex can avoid manually reranking point orders and get simillar results
    # iouMat2 = [get_intersection_over_union_from_pnts(tgt_bds_2[i].tolist(),out_bds_2[i].tolist()) for i in range(total_num)]
    return  iouMat


def adaptive_weight_logger(weights, name):

    if weights is not None:
        ins = len(weights)
        min_value = min(weights)
        max_value = max(weights)
        mean_value = torch.mean(weights)
        std_value = torch.std(weights)
        logger = logging.getLogger(__name__)
        logger.info(f'{name}: num_inst: {ins}, min: {min_value:.2f}, max: {max_value:.2f}, mean: {mean_value:.2f}, std: {std_value:.2f}')
    else:
        return
    pass
    return

def sigmoid_focal_loss(inputs, targets, num_inst, alpha: float = 0.25, gamma: float = 2, weights=None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if weights is not None:
        if weights.ndim == 2:
            weights = weights[..., None, None]
        elif weights.ndim == 3:
            weights = weights[..., None]
        elif weights.ndim == 4:
            weights = weights
        else:
            raise NotImplementedError(f"Unsupported dim {weights.ndim}")
        loss = loss * weights

    if loss.ndim == 4:
        return loss.mean((1, 2)).sum() / num_inst
    elif loss.ndim == 3:
        return loss.mean(1).sum() / num_inst
    else:
        raise NotImplementedError(f"Unsupported dim {loss.ndim}")


class SetCriterion(nn.Module):
    """
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            num_classes,
            enc_matcher,
            dec_matcher,
            o2m_matcher,
            weight_dict,
            enc_losses,
            num_sample_points,
            dec_losses,
            voc_size,
            num_ctrl_points,
            focal_alpha=0.25,
            focal_gamma=2.0
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as suffix the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.enc_matcher = enc_matcher
        self.dec_matcher = dec_matcher
        self.o2m_matcher = o2m_matcher
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)
        self.dec_losses = dec_losses
        self.voc_size = voc_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points

    def loss_labels(self, outputs, targets, indices, num_inst, log=False):
        """Classification loss (NLL)
        targets dicts must contain the suffix "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:-1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(shape,
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        # src_logits, target_classes_onehot: (bs, nq, n_ctrl_pts, 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_inst,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_beziers(self, outputs, targets, indices, num_inst):
        # may FIX: (1) seg valid points
        assert 'pred_beziers' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_beziers = outputs['pred_beziers'][idx]
        src_beziers = self.bezier_sampler.get_sample_points(src_beziers.view(-1, 4, 2))

        has_beziers = 'beziers' in targets[0]
        if has_beziers:
            target_beziers = torch.cat(
                [t['beziers'][i] for t, (_, i) in zip(targets, indices)],
                dim=0
            )
            target_beziers = self.bezier_sampler.get_sample_points(target_beziers)
        else:
            target_beziers = torch.cat(
                [t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)],
                dim=0
            )
        if target_beziers.numel() == 0:
            target_beziers = src_beziers.clone().detach()
        loss_bezier = F.l1_loss(src_beziers, target_beziers, reduction='none')
        losses = {}
        losses['loss_bezier'] = loss_bezier.sum() / num_inst
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_inst):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_texts(self, outputs, targets, indices, num_inst):
        # CTC loss for classification of points
        assert 'pred_text_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        src_texts = src_texts.permute(1, 0, 2)
        src = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)

        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
        input_lengths = torch.full((src.size(1),), src.size(0), dtype=torch.long)
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)
        if target_lengths.sum() == 0:
            return {'loss_texts': torch.tensor(0.).to(src_texts.device)}
        else:
            target_texts = torch.cat([t[:l] for t, l in zip(target_texts, target_lengths)])

            return {
                'loss_texts': F.ctc_loss(
                    src,
                    target_texts,
                    input_lengths,
                    target_lengths,
                    blank=self.voc_size,
                    zero_infinity=True
                )
            }

    def loss_ctrl_points(self, outputs, targets, indices, num_inst):
        """Compute the L1 regression loss
        """
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_ctrl_points = F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='sum')
        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    def loss_bd_points(self, outputs, targets, indices, num_inst):
        assert 'pred_bd_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bd_points = outputs['pred_bd_points'][idx]
        target_bd_points = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bd_points = F.l1_loss(src_bd_points, target_bd_points, reduction='sum')
        losses = {'loss_bd_points': loss_bd_points / num_inst}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'beziers': self.loss_beziers,
            'texts': self.loss_texts,
            'bd_points': self.loss_bd_points
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_inst, **kwargs)

    def get_loss_o2m(self, loss, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'beziers': self.loss_beziers,
            'texts': self.loss_texts,
            'bd_points': self.loss_bd_points
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        empty_tensor = torch.tensor([], dtype=torch.int64)
        losses = {}
        for i, indice in enumerate(indices):
            indice_o, indice_t = indice
            for j in range(len(indice_o)):
                i_temp = [(indice_o[j], indice_t)]
                for n in range(len(targets)):
                    if n != i:
                        i_temp.insert(n, (empty_tensor, empty_tensor))

                loss_temp = loss_map[loss](outputs, targets, i_temp, num_inst, **kwargs)
                for k in loss_temp.keys():
                    if k in losses.keys():
                        losses[k] += loss_temp[k]
                    else:
                        losses[k] = (loss_temp[k])
        if losses == {}:
            losses = loss_map[loss](outputs, targets, indices, num_inst, **kwargs)
        return losses

    def forward(self, outputs, targets, o2m=False):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_inst = sum(len(t['ctrl_points']) for t in targets)
        num_inst = torch.as_tensor(
            [num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()

        if o2m:
            assert self.o2m_matcher is not None, 'if using o2m loss, o2m_matcher should not be None'
            indices = self.o2m_matcher(outputs_without_aux, targets)
        else:
            indices = self.dec_matcher(outputs_without_aux, targets)
        losses = {}
        for loss in self.dec_losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_inst, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i_aux, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.o2m_matcher(aux_outputs, targets) if o2m else self.dec_matcher(aux_outputs, targets)

                for loss in self.dec_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_inst, **kwargs)
                    l_dict = {k + f'_{i_aux}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']

            indices = self.enc_matcher(enc_outputs, targets)
            for loss in self.enc_losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, targets, indices, num_inst, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses



class SetAdaptiveO2MCriterionSemi(nn.Module):
    """
    Loss criterion for SemiETS
        1) add o2m_matcher
        2) add dual thresholds for loss_labels(PSA PART II , PART I IS IN SemiETS.py)
        3) add adaptive weights for loss_bd_points,loss_ctrl_points(CRC)
        4) add adaptive weights for loss_text(SCI)

    """

    def __init__(
            self,
            num_classes,
            enc_matcher,
            dec_matcher,
            o2m_matcher_enc,
            o2m_matcher_dec,
            det_based_point_matcher,
            rec_based_point_matcher,
            weight_dict,
            enc_losses,
            num_sample_points,
            o2m_dec_losses,
            o2o_dec_losses,
            voc_size,
            num_ctrl_points,
            focal_alpha=0.25,
            focal_gamma=2.0,
            leven_dis_alpha=2.0,
            cost_alpha = 2.0,
            rec_threshold = None,
            det_threshold = None,
            use_combined_thr = False,
            o2m_text_o2o = False,
            use_o2m_enc = False,
            det_adaptive_type = 'edit_distance',
            rec_adaptive_type = 'polygon_diou',
            use_task_specific_matcher = False,
            precise_teacher = False,

    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as suffix the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.enc_matcher = enc_matcher
        self.dec_matcher = dec_matcher
        self.o2m_matcher_enc = o2m_matcher_enc
        self.o2m_matcher_dec = o2m_matcher_dec
        self.det_based_point_matcher = det_based_point_matcher
        self.rec_based_point_matcher = rec_based_point_matcher
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)
        self.voc_size = voc_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points
        self.rec_threshold = rec_threshold
        self.det_threshold = det_threshold
        self.use_combined_thr = use_combined_thr
        self.o2m_text_o2o = o2m_text_o2o
        self.use_o2m_enc = use_o2m_enc
        if self.voc_size == 96:
            self.CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1',
                             '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C',
                             'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                             'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                             'z', '{', '|', '}', '~']
        elif self.voc_size == 37:
            self.CTLABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            import pickle
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
            # voc_size includes the unknown class, which is not in self.CTABLES
        assert (int(self.voc_size - 1) == len(
            self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1),
                                                                                              len(self.CTLABELS))
        self.leven_dis_alpha = leven_dis_alpha
        self.cost_alpha = cost_alpha
        assert det_adaptive_type in ['edit_distance' , 'score' , 'polygon_diou', 'merge'],f"Are you sure you want to use {det_adaptive_type} for det_adaptive_loss?"
        self.det_adaptive_type = det_adaptive_type
        self.rec_adaptive_type = rec_adaptive_type
        self.use_task_specific_matcher = use_task_specific_matcher
        self.o2m_dec_losses = o2m_dec_losses
        self.o2o_dec_losses = o2o_dec_losses
        self.precise_teacher = precise_teacher

    def loss_labels(self, outputs, targets, indices, extra_info,target_info=None,log=False):
        """Classification loss (NLL)
        targets dicts must contain the suffix "labels" containing a tensor of dim [nb_target_boxes]
        """
        num_inst = extra_info['num_inst']
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:-1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(shape,
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        # src_logits, target_classes_onehot: (bs, nq, n_ctrl_pts, 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_inst,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses



    def loss_beziers(self, outputs, targets, indices, extra_info,target_info=None):
        # may FIX: (1) seg valid points
        num_inst = extra_info['num_inst']
        assert 'pred_beziers' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_beziers = outputs['pred_beziers'][idx]
        src_beziers = self.bezier_sampler.get_sample_points(src_beziers.view(-1, 4, 2))

        has_beziers = 'beziers' in targets[0]
        if has_beziers:
            target_beziers = torch.cat(
                [t['beziers'][i] for t, (_, i) in zip(targets, indices)],
                dim=0
            )
            target_beziers = self.bezier_sampler.get_sample_points(target_beziers)
        else:
            target_beziers = torch.cat(
                [t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)],
                dim=0
            )
        if target_beziers.numel() == 0:
            target_beziers = src_beziers.clone().detach()
        loss_bezier = F.l1_loss(src_beziers, target_beziers, reduction='none')
        losses = {}
        losses['loss_bezier'] = loss_bezier.sum() / num_inst
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, extra_info):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_texts(self, outputs, targets, indices, extra_info,target_info=None):
        # CTC loss for classification of points
        assert 'pred_text_logits' in outputs
        # assert self.rec_threshold is not None, "to use this loss,you should set the threshold"
        if 'text_indices' in extra_info.keys():
            indices = extra_info['text_indices']
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)

        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length


        src_texts = src_texts.permute(1, 0, 2)
        src_texts = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)
        input_lengths = torch.full((src_texts.size(1),), src_texts.size(0), dtype=torch.long)
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)
        if target_lengths.sum() == 0:
            return {'loss_texts': torch.tensor(0.).to(src_texts.device)}
        else:
            # beta = 1.0
            target_texts = torch.cat([t[:l] for t, l in zip(target_texts, target_lengths)])
            loss_text = F.ctc_loss(
                src_texts,
                target_texts,
                input_lengths,
                target_lengths,
                blank=self.voc_size,
                zero_infinity=True,
                reduction="none"
            )
            # loss_text = loss_text * cls_scores
            # loss_text = loss_text * (ctc_scores ** beta)  # reliable targets weight
            loss_text = loss_text / target_lengths
            weights = torch.ones_like(loss_text)
            self.adaptive_weight_logger(weights=weights, name='loss_text')
            return {
                'loss_texts': torch.mean(loss_text)
            }


    def loss_texts_psa(self, outputs, targets, indices, extra_info, target_info=None):
        # CTC loss for classification of points
        # add rcs based weighting strategy
        assert 'pred_text_logits' in outputs
        assert self.rec_threshold is not None, "to use this loss,you should set the threshold"
        assert hasattr(self, 'decoder_matcher')
        if 'text_indices' in extra_info.keys():
            indices = extra_info['text_indices']
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
        ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])  # n,
        if self.use_combined_thr:
            det_scores = torch.cat([t['scores'][i] for t, (_, i) in zip(targets, indices)])
            valid_idx = (ctc_scores > self.rec_threshold) & (det_scores > self.rec_threshold)
        elif self.precise_teacher:
            if self.student_ctc_already_calculate:
                student_ctc_scores = self.student_ctc_scores
            else:
                # quick version
                student_ctc_scores = self._get_match_student_ctc(outputs, idx, ctc_scores)
                self.student_ctc_already_calculate = True
                self.student_ctc_scores = student_ctc_scores
            valid_idx = (ctc_scores > self.rec_threshold) & (ctc_scores > student_ctc_scores)
        else:
            valid_idx = (ctc_scores > self.rec_threshold)

        src_texts = src_texts[valid_idx]
        target_texts = target_texts[valid_idx]

        src_texts = src_texts.permute(1, 0, 2)
        src_texts = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)

        input_lengths = torch.full((src_texts.size(1),), src_texts.size(0), dtype=torch.long)
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)

        if target_lengths.sum() == 0:
            return {'loss_texts': torch.tensor(0.).to(src_texts.device)}
        else:
            # beta = 1.0
            target_texts = torch.cat([t[:l] for t, l in zip(target_texts, target_lengths)])
            loss_text = F.ctc_loss(
                src_texts,
                target_texts,
                input_lengths,
                target_lengths,
                blank=self.voc_size,
                zero_infinity=True,
                reduction="none"
            )
            # loss_text = loss_text * cls_scores
            # loss_text = loss_text * (ctc_scores ** beta)  # reliable targets weight
            loss_text = loss_text / target_lengths
            weights = torch.ones_like(loss_text)
            self.adaptive_weight_logger(weights=weights, name='loss_text')
            return {
                'loss_texts': torch.mean(loss_text)
            }

    def _ctc_decode_recognition(self, rec):
        """ decode a sequence from CTC output to pseudo label """
        text_length = len(rec)
        last_char = 100
        s = []
        for c in rec:
            # c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    s.append(c)
                    last_char = c
            else:
                last_char = 100

        if len(s) < text_length:

            s.extend([self.voc_size] * (text_length-len(s)))

        s = torch.tensor(s).to(rec)

        return s

    @torch.no_grad()
    def _get_match_student_ctc(self, outputs,idx, teacher_ctc=None):
        # quick version code to speed up ctc student fetcher especially in o2m
        # only fetch the case where ctc_t > rec_thr for student
        # others will be replaced by zero for convenience since they make no difference to code bellow

        text_logits = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        ctc_scores = torch.full((text_logits.shape[0],), 100.0, device=self.device)
        #get subset for speeding up
        if teacher_ctc is not None:
            need_compute_idx = teacher_ctc > self.rec_threshold
            text_logits = text_logits[need_compute_idx]
            ctc_scores_sub = torch.full((text_logits.shape[0],), 100.0, device=self.device)

        # get student ctc for subset of student logits
        src_texts = F.log_softmax(text_logits, dim=-1)  # shape: (length, n, voc_size+1)
        text_pred = src_texts.topk(1)[1].squeeze(-1)
        recs = text_pred
        if len(recs) > 0:
            recs = torch.stack([self._ctc_decode_recognition(rc) for rc in recs])

        text_logits = torch.softmax(text_logits, dim=-1)
        #pass the ctc_decoder from o2m_text_spotter to criterition
        beam_results, beam_scores, _, out_lens = self.ctc_decoder.decode(text_logits)
        beam_results,out_lens = beam_results.to(self.device), out_lens.to(self.device),
        beam_scores = beam_scores.to(self.device)
        #target length
        valid_length = (recs != self.voc_size).sum(-1)
        valid_length = valid_length.unsqueeze(1).expand(-1, beam_results.size(1), )

        recs_expanded = recs.unsqueeze(1).expand(-1, beam_results.size(1), -1)
        recs_expanded = recs_expanded.to(dtype=beam_results.dtype)

        # VERSION 2 ctc score fetch
        difference = torch.abs(recs_expanded - beam_results)
        mask = torch.arange(beam_results.size(2)).expand(beam_results.shape[0], beam_results.shape[1], -1).to(
            self.device)
        mask = mask < valid_length.unsqueeze(-1)
        valid_difference = difference * mask  # N,100,25/50
        score_idx = (out_lens == valid_length) & (valid_difference.sum(dim=-1) == 0)  # N,100
        success_match = score_idx.any(dim=-1)
        #sub to full
        if teacher_ctc is not None:
            ctc_scores_sub[success_match] = beam_scores[score_idx]
            ctc_scores[need_compute_idx] = ctc_scores_sub
        else:
            ctc_scores[success_match] = beam_scores[score_idx]

        result_ctc_score = 1 / torch.exp(ctc_scores)

        return result_ctc_score


    def loss_texts_adaptive_sci(self, outputs, targets, indices, extra_info, target_info=None):
        # CTC loss for classification of points
        # add rcs based weighting strategy
        assert 'pred_text_logits' in outputs
        assert  self.rec_threshold is not None,"to use this loss,you should set the threshold"
        assert hasattr(self, 'decoder_matcher')
        if 'text_indices' in extra_info.keys():
            indices = extra_info['text_indices']
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
        ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])  # n,
        if self.use_combined_thr:
            det_scores = torch.cat([t['scores'][i] for t, (_, i) in zip(targets, indices)])
            valid_idx = (ctc_scores > self.rec_threshold) & (det_scores > self.rec_threshold)
        elif self.precise_teacher:
            if self.student_ctc_already_calculate:
                student_ctc_scores = self.student_ctc_scores
            else:
                student_ctc_scores = self._get_match_student_ctc(outputs,idx,ctc_scores)
                self.student_ctc_already_calculate = True
                self.student_ctc_scores = student_ctc_scores
            valid_idx = (ctc_scores > student_ctc_scores) & (ctc_scores > self.rec_threshold)
        else:
            valid_idx = (ctc_scores > self.rec_threshold)


        src_texts = src_texts[valid_idx]
        target_texts = target_texts[valid_idx]
        # student_ctc_scores = student_ctc_scores[valid_idx]
        # ctc_scores = ctc_scores[valid_idx]

        src_texts = src_texts.permute(1, 0, 2)
        src_texts = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)

        input_lengths = torch.full((src_texts.size(1),), src_texts.size(0), dtype=torch.long)
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)

        if target_lengths.sum() == 0:
            return {'loss_texts': torch.tensor(0.).to(src_texts.device)}
        else:

            target_texts = torch.cat([t[:l] for t, l in zip(target_texts, target_lengths)])
            # adaptive weight based on detection consistency
            if self.rec_adaptive_type == 'polygon_diou':
                iou_matrix = torch.Tensor(gen_polygon_IOU_mat(outputs, targets, idx, indices,valid_idx))
                relative_center_offset_weight = gen_relative_distance_mat(outputs, targets, idx, indices,valid_idx)
                # reliable_weight = torch.clip(iou_matrix.to(relative_center_offset_weight.device)\
                #                   - relative_center_offset_weight,0) #reliability learning
                reliable_weight = 1 + iou_matrix.to(relative_center_offset_weight.device) \
                                             - relative_center_offset_weight
            elif self.rec_adaptive_type == 'polygon_iou':
                iou_matrix = torch.Tensor(gen_polygon_IOU_mat(outputs, targets, idx, indices, valid_idx))
                reliable_weight = 1 + iou_matrix.to(src_texts.device)


            loss_text = F.ctc_loss(
                src_texts,
                target_texts,
                input_lengths,
                target_lengths,
                blank=self.voc_size,
                zero_infinity=True,
                reduction="none"
            )
            weights = reliable_weight


            self.adaptive_weight_logger(weights=weights,name=f'{self.rec_adaptive_type}_weight')
            loss_text = loss_text / target_lengths * weights  # reliability
            loss_text = torch.sum(loss_text / torch.sum(weights))

            return {
                'loss_texts': loss_text
            }

    def gen_targets_label(self,indices,valid_idx):
        batch_num = len(indices)
        batch_category = torch.Tensor([torch.unique(indices[i][1]).shape[0] for i in range(batch_num)])
        batch_category = torch.cat((torch.Tensor([0]),torch.cumsum(batch_category,0)))
        label = torch.cat([batch_category[i]+j for i,(_, j) in  enumerate(indices)])
        label = label.to(valid_idx.device)
        label = label[valid_idx]
        return label


    def edit_distance_weight(self, outputs, targets, indices, extra_info):
        """Compute the L1 regression loss
        """
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)


        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        #adaptive weight
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        src_texts = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)
        text_pred = src_texts.topk(1)[1].squeeze(-1)
        text_pred_list = [self._ctc_decode_recognition_pred_logits(tx) for tx in text_pred]
        text_target = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
        text_target_list = [self._ctc_decode_recognition_pred(t) for t in text_target]
        assert  len(text_pred_list) == len(text_target_list) , 'length error'

        edit_distance = torch.tensor([string_metric.levenshtein(text_pred_list[k].upper(), text_target_list[k].upper()) for k in range(len(text_target_list))]).to(target_ctrl_points)
        norm_factor = torch.tensor([max(len(text_pred_list[k]),len(text_target_list[k]))  for k in range(len(text_target_list))]).to(target_ctrl_points)
        normalized_distance = edit_distance / norm_factor
        # normalized_distance = (distance - distance.min()) / (distance.max() - distance.min())
        weights = (1.0 + self.leven_dis_alpha * normalized_distance)

        return weights

    def geo_distance_weight(self, outputs, targets, indices, extra_info):
        # CTC loss for classification of points
        assert 'pred_text_logits' in outputs
        assert  self.rec_threshold is not None,"to use this loss,you should set the threshold"
        assert hasattr(self, 'decoder_matcher')
        if 'text_indices' in extra_info.keys():
            indices = extra_info['text_indices']
        idx = self._get_src_permutation_idx(indices)
        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
        ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])  # n,
        if self.use_combined_thr:
            det_scores = torch.cat([t['scores'][i] for t, (_, i) in zip(targets, indices)])
            valid_idx = (ctc_scores > self.rec_threshold) & (det_scores > self.rec_threshold)
        else:
            valid_idx = (ctc_scores > self.rec_threshold)

        target_texts = target_texts[valid_idx]
        # ctc_scores = ctc_scores[valid_idx]
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)

        if target_lengths.sum() == 0:
            return  None
        else:
            gamma = 1.0
            # adaptive weight based on detection consistency
            iou_matrix = torch.Tensor(gen_polygon_IOU_mat(outputs, targets, idx, indices, valid_idx))
            relative_center_offset_weight = gen_relative_distance_mat(outputs, targets, idx, indices, valid_idx)
            reliable_weight = iou_matrix.to(relative_center_offset_weight.device) \
                              - relative_center_offset_weight
            relative_geo_distance = 1 - reliable_weight
            # relative_center_offset_weight = gen_relative_distance_mat(outputs, targets, idx, indices,valid_idx)
            # normalized_weight = ( 1 - relative_center_offset_weight)**gamma  #reliability learning
            normalized_weight = relative_geo_distance**gamma  #hard example learning
            weights = self.cost_alpha * normalized_weight + 1
            return weights


    def hm_adaptive_weight_cal(self, outputs, targets, indices, extra_info):
        edit_weight = self.edit_distance_weight(outputs,targets, indices, extra_info)
        geo_weight = self.geo_distance_weight(outputs , targets , indices, extra_info)
        return edit_weight, geo_weight

    def adaptive_weight_logger(self, weights, name , interval=50):
        log_sta = (self.curr_step % interval == 0 )
        if weights is not None and log_sta and weights.shape[0]>0:
            ins = len(weights)
            min_value = min(weights) if len(weights) > 0 else 0
            max_value = max(weights) if len(weights) > 0 else 0
            mean_value = torch.mean(weights) if len(weights) > 0 else 0
            std_value = torch.std(weights) if len(weights) > 0 else 0
            logger = logging.getLogger(__name__)
            logger.info(
                f'{name}: step:{self.curr_step} num_inst: {ins}, min: {min_value:.3f}, max: {max_value:.3f}, mean: {mean_value:.3f}, std: {std_value:.3f}')
        else:
            return



    def loss_ctrl_points(self, outputs, targets, indices, extra_info,target_info=None):
        """Compute the L1 regression loss
        """
        num_inst = extra_info['num_inst']
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_ctrl_points = F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='sum')
        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        weights = torch.ones_like(src_ctrl_points[:,0,0])
        self.adaptive_weight_logger(weights=weights, name='loss_ctrl_point')
        return losses

    def loss_ctrl_points_adaptive_crc(self, outputs, targets, indices, extra_info, target_info=None):
        """Compute the L1 regression loss with adaptive weigting use normalized edit_distance
        If targets are in different set which means
        """
        num_inst = extra_info['num_inst']
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])
        loss_ctrl_points = F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='none')

        #adaptive weight
        if self.det_adaptive_type in ['edit_distance' , 'merge']:
            if self.edit_distance_already_calculate:
                normalized_distance_all = self.norm_edit_distance
            else:
                src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
                src_texts = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)
                text_pred = src_texts.topk(1)[1].squeeze(-1)
                text_pred_list = [self._ctc_decode_recognition_pred_logits(tx) for tx in text_pred]
                text_target = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
                text_target_list = [self._ctc_decode_recognition_pred(t) for t in text_target]
                assert  len(text_pred_list) == len(text_target_list) , 'length error'

                edit_distance = torch.tensor([string_metric.levenshtein(text_pred_list[k].upper(), text_target_list[k].upper()) for k in range(len(text_target_list))]).to(target_ctrl_points)
                norm_factor = torch.tensor([max(len(text_pred_list[k]),len(text_target_list[k]))  for k in range(len(text_target_list))]).to(target_ctrl_points)
                normalized_distance_all = edit_distance / norm_factor
                self.norm_edit_distance = normalized_distance_all
                self.edit_distance_already_calculate = True
            # ctc filter only for edit distance
            # threshold turn to be a switch for whether to use hard-mining weight for target instance
            ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])
            if self.precise_teacher:
                if self.student_ctc_already_calculate:
                    student_ctc_scores = self.student_ctc_scores
                else:
                    student_ctc_scores = self._get_match_student_ctc(outputs, idx, ctc_scores)
                    self.student_ctc_scores = student_ctc_scores
                    self.student_ctc_already_calculate = True
                valid_idx = (ctc_scores > student_ctc_scores) & (ctc_scores > self.rec_threshold)
            else:
                valid_idx = (ctc_scores > self.rec_threshold)

            normalized_distance = torch.zeros_like(normalized_distance_all)
            normalized_distance[valid_idx] = normalized_distance_all[valid_idx]

        elif self.det_adaptive_type == 'score':
            prob_score = outputs['pred_logits'][idx].mean(-2).sigmoid().squeeze(1)
            target_score = torch.cat([t['scores'][i] for t, (_, i) in zip(targets, indices)])
            normalized_distance = torch.abs(target_score-prob_score)
        elif self.det_adaptive_type == 'polygon_diou':
            normalized_distance = gen_relative_distance_mat(outputs, targets, idx, indices)
            if normalized_distance is None:
                return {'loss_ctrl_points': torch.tensor(0.).to(src_ctrl_points.device)}
            else:
                iou_matrix = torch.Tensor(gen_polygon_IOU_mat(outputs, targets, idx, indices))
                #for diou we need to normalize it to [0,1]
                normalized_distance = 1 - iou_matrix.to(normalized_distance.device) \
                                      + normalized_distance
                normalized_distance = normalized_distance / 2

        hard_mining_weight = (1.0 + self.leven_dis_alpha * normalized_distance)
        if self.det_adaptive_type == 'merge':
            #cal 1-iou+rcd as supplemented

            relative_center_distance = gen_relative_distance_mat(outputs, targets, idx, indices)
            if relative_center_distance is None:
                return {'loss_ctrl_points': torch.tensor(0.).to(src_ctrl_points.device)}
            else:
                iou_matrix = torch.Tensor(gen_polygon_IOU_mat(outputs, targets, idx, indices))

                # for diou we need to normalize it to [0,1]
                relative_center_distance = 1 - iou_matrix.to(relative_center_distance.device) \
                                           + relative_center_distance
                relative_center_distance = relative_center_distance / 2

            hard_mining_weight_rcd = (1.0 + self.leven_dis_alpha * relative_center_distance)
            weights = (hard_mining_weight + hard_mining_weight_rcd) / 2
            # weights = hard_mining_weight + hard_mining_weight_rcd
        else:
            weights = hard_mining_weight

        self.adaptive_weight_logger(weights=weights, name=self.det_adaptive_type)
        weights = weights[:, None, None].repeat(1,loss_ctrl_points.shape[1],loss_ctrl_points.shape[2])
        loss_ctrl_points = torch.sum(weights * loss_ctrl_points)
        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    def loss_bd_points(self, outputs, targets, indices, extra_info,target_info=None):
        num_inst = extra_info['num_inst']
        assert 'pred_bd_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bd_points = outputs['pred_bd_points'][idx]
        target_bd_points = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bd_points = F.l1_loss(src_bd_points, target_bd_points, reduction='sum')
        losses = {'loss_bd_points': loss_bd_points / num_inst}
        return losses

    def loss_bd_points_adaptive_crc(self, outputs, targets, indices, extra_info, target_info=None):
        num_inst = extra_info['num_inst']
        assert 'pred_bd_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bd_points = outputs['pred_bd_points'][idx]
        target_bd_points = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])

        loss_bd_points = F.l1_loss(src_bd_points, target_bd_points, reduction='none')

        if self.det_adaptive_type in ['edit_distance' , 'merge']:
            if self.edit_distance_already_calculate:
                normalized_distance_all = self.norm_edit_distance
            else:
                src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
                src_texts = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)
                text_pred = src_texts.topk(1)[1].squeeze(-1)
                text_pred_list = [self._ctc_decode_recognition_pred_logits(tx) for tx in text_pred]
                text_target = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
                text_target_list = [self._ctc_decode_recognition_pred(t) for t in text_target]
                assert len(text_pred_list) == len(text_target_list), 'length error'

                edit_distance = torch.tensor(
                    [string_metric.levenshtein(text_pred_list[k].upper(), text_target_list[k].upper()) for k in
                     range(len(text_target_list))]).to(target_bd_points)
                norm_factor = torch.tensor(
                    [max(len(text_pred_list[k]), len(text_target_list[k])) for k in range(len(text_target_list))]).to(
                    target_bd_points)
                normalized_distance_all = edit_distance / norm_factor
                self.norm_edit_distance = normalized_distance_all
                self.edit_distance_already_calculate = True
            #ctc filter
            ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])  # n,
            if self.precise_teacher:
                if self.student_ctc_already_calculate:
                    student_ctc_scores = self.student_ctc_scores
                else:
                    student_ctc_scores = self._get_match_student_ctc(outputs, idx, ctc_scores)
                    self.student_ctc_already_calculate = True
                    self.student_ctc_scores = student_ctc_scores

                valid_idx = (ctc_scores > student_ctc_scores) & (ctc_scores > self.rec_threshold)
            else:
                valid_idx = (ctc_scores > self.rec_threshold)
            # threshold turn to be a switch for whether to use hard-mining weight for target instance
            normalized_distance = torch.zeros_like(normalized_distance_all)
            normalized_distance[valid_idx] = normalized_distance_all[valid_idx]

        elif self.det_adaptive_type == 'score':
            prob_score = outputs['pred_logits'][idx].mean(-2).sigmoid().squeeze(1)
            target_score = torch.cat([t['scores'][i] for t, (_, i) in zip(targets, indices)])
            normalized_distance = torch.abs(target_score - prob_score)
        elif self.det_adaptive_type == 'polygon_diou':
            normalized_distance = gen_relative_distance_mat(outputs, targets, idx, indices)
            if normalized_distance is None:
                return {'loss_bd_points': torch.tensor(0.).to(src_bd_points.device)}
            else:
                iou_matrix = torch.Tensor(gen_polygon_IOU_mat(outputs, targets, idx, indices))

                # for diou we need to normalize it to [0,1]
                normalized_distance = 1 - iou_matrix.to(normalized_distance.device) \
                                      + normalized_distance
                normalized_distance = normalized_distance / 2

        hard_mining_weight = (1.0 + self.leven_dis_alpha * normalized_distance)
        if self.det_adaptive_type == 'merge':
            relative_center_distance = gen_relative_distance_mat(outputs, targets, idx, indices)
            if relative_center_distance is None:
                return {'loss_bd_points': torch.tensor(0.).to(src_bd_points.device)}
            else:
                iou_matrix = torch.Tensor(gen_polygon_IOU_mat(outputs, targets, idx, indices))

                # for diou we need to normalize it to [0,1]
                relative_center_distance = 1 - iou_matrix.to(relative_center_distance.device) \
                                           + relative_center_distance
                relative_center_distance = relative_center_distance / 2

            hard_mining_weight_rcd = (1.0 + self.leven_dis_alpha * relative_center_distance)
            weights = (hard_mining_weight + hard_mining_weight_rcd) / 2
            # weights = hard_mining_weight + hard_mining_weight_rcd
        else:
            weights = hard_mining_weight

        self.adaptive_weight_logger(weights=weights, name=self.det_adaptive_type)
        weights = weights[:, None, None].repeat(1, loss_bd_points.shape[1], loss_bd_points.shape[2])
        loss_bd_points = torch.sum(weights * loss_bd_points)
        losses = {'loss_bd_points': loss_bd_points / num_inst}
        return losses


    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _ctc_decode_recognition_pred(self, rec):
        # last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                # if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        # last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        # last_char = c
            else:
                last_char = '###'
        return s

    def _ctc_decode_recognition_pred_logits(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = '###'
        return s

    def get_num_inst(self, indices , outputs):
        num_inst = sum(len(i[1]) for i in indices)
        num_inst = torch.as_tensor(
            [num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()
        return num_inst

    def get_loss_multi_task(self, loss, outputs, targets,indices, extra_info,target_info, **kwargs):
        loss_map = {
            'labels': self.loss_labels,#det prob loss
            'beziers': self.loss_beziers,#enc loss
            'ctrl_points': self.loss_ctrl_points,
            'bd_points': self.loss_bd_points,
            'ctrl_points_adaptive_crc': self.loss_ctrl_points_adaptive_crc,  # CRC, thr filter
            'bd_points_adaptive_crc': self.loss_bd_points_adaptive_crc,  # CRC, thr filter
            'texts': self.loss_texts,#original thr filter for ctc loss
            'texts_psa': self.loss_texts_psa,#orignial thr filter and ts compare both optional
            'texts_adaptive_sci':self.loss_texts_adaptive_sci,# sci weighting o2o, thr filter
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs,targets,indices,extra_info,target_info)


    def filter_targets(self,targets,idx):
        #NMS to be implemented
        #one batch targets
        new_targets=dict()
        for k,v in targets.items():
            if k not in ['gt_instances',]:
                new_targets[k] = v[idx]
            else:
                new_targets[k] = v

        return new_targets


    def prepare_extra_info(self,outputs,target_info,o2m,num_inst_outputs):
        #match the outputs for each set of Pseudos and record the num_inst
        extra_info={}
        if o2m:
            det_only_indices = self.det_based_point_matcher(outputs, target_info['det_targets'])
            rec_only_indices = self.rec_based_point_matcher(outputs, target_info['rec_targets'])
            e2e_indices = self.o2m_matcher_dec(outputs, target_info['e2e_targets'])
            indices =  self.o2m_matcher_dec(outputs, target_info['cls_targets'])

        else:
            det_only_indices = self.det_based_point_matcher(outputs, target_info['det_targets'])
            rec_only_indices = self.rec_based_point_matcher(outputs, target_info['rec_targets'])
            e2e_indices = self.dec_matcher(outputs, target_info['e2e_targets'])
            indices = self.dec_matcher(outputs, target_info['cls_targets'])


        extra_info['det_only_indices'] = det_only_indices
        extra_info['rec_only_indices'] = rec_only_indices
        extra_info['e2e_indices'] = e2e_indices
        extra_info['indices'] = indices

        extra_info['det_num_inst'] = self.get_num_inst(det_only_indices, num_inst_outputs)
        extra_info['rec_num_inst'] = self.get_num_inst(rec_only_indices, num_inst_outputs)
        extra_info['e2e_num_inst'] = self.get_num_inst(e2e_indices, num_inst_outputs)
        extra_info['num_inst'] = self.get_num_inst(indices, num_inst_outputs)

        return extra_info

    def prepare_targets(self,targets):
        target_info = dict()
        det_scores_list = [t['scores']for t in targets]
        ctc_scores_list = [t['ctc_scores'] for t in targets]

        #DIFFERENT filtering strategy
        valid_class_idx = [ (det_sc > self.det_threshold) for det_sc in det_scores_list]
        valid_detection_idx =[ (det_sc > self.det_threshold) & (ctc_sc <= self.rec_threshold) for det_sc,ctc_sc in zip(det_scores_list,ctc_scores_list)]
        valid_recognition_idx = [ (det_sc <= self.det_threshold) & (ctc_sc > self.rec_threshold) for det_sc,ctc_sc in zip(det_scores_list,ctc_scores_list)]
        valid_e2e_idx =  [ (det_sc > self.det_threshold) & (ctc_sc > self.rec_threshold) for det_sc,ctc_sc in zip(det_scores_list,ctc_scores_list)]
        cls_targets = []
        det_targets=[]
        rec_targets=[]
        e2e_targets=[]


        for tgt,cls_idx_per,det_idx_per,rec_idx_per,e2e_idx_per in zip(targets,valid_class_idx,valid_detection_idx,valid_recognition_idx,valid_e2e_idx):
            cls_targets.append(self.filter_targets(tgt,cls_idx_per))
            det_targets.append(self.filter_targets(tgt,det_idx_per))
            rec_targets.append(self.filter_targets(tgt,rec_idx_per))
            e2e_targets.append(self.filter_targets(tgt,e2e_idx_per))
        target_info['cls_targets'] = cls_targets
        target_info['det_targets'] = det_targets
        target_info['rec_targets'] = rec_targets
        target_info['e2e_targets'] = e2e_targets

        return target_info

    def forward(self, outputs, targets, o2m=False):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # adaptive dec loss ctrl
        if o2m:
            dec_losses = self.o2m_dec_losses
        else:
            dec_losses = self.o2o_dec_losses
        # Retrieve the matching between the outputs of the last layer and the targets


        #Target filter for different matcher, and the filter thr in coarse label is discarded
        #Divide the targets into different set
        if self.use_task_specific_matcher :
            target_info = self.prepare_targets(targets)
            extra_info = self.prepare_extra_info(outputs_without_aux, target_info, o2m, outputs)
        else:
            # main targets with its matching indices, with 0.4 thr in coarse label
            target_info = dict(cls_targets = targets)
            if o2m:
                indices = self.o2m_matcher_dec(outputs_without_aux, target_info['cls_targets'])
            else:
                indices = self.dec_matcher(outputs_without_aux, target_info['cls_targets'])
            num_inst = self.get_num_inst(indices , outputs)
            extra_info = dict(indices = indices , num_inst = num_inst,)

        # losses.update(self.get_loss(loss, outputs, targets, indices, extra_info,** kwargs))
        # Compute all the requested losses
        losses = {}
        self.edit_distance_already_calculate = False
        self.student_ctc_already_calculate = False
        for loss in dec_losses:
            kwargs = {}
            losses.update(self.get_loss_multi_task(loss, outputs, target_info['cls_targets'],extra_info['indices'], extra_info,target_info,**kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.use_task_specific_matcher:
                    extra_info = self.prepare_extra_info(aux_outputs, target_info, o2m, outputs)
                else:
                    if o2m:
                        indices = self.o2m_matcher_dec(outputs_without_aux, target_info['cls_targets'])
                    else:
                        indices = self.dec_matcher(outputs_without_aux, target_info['cls_targets'])
                    num_inst = self.get_num_inst(indices, outputs)
                    extra_info = dict(indices=indices, num_inst=num_inst,)
                self.edit_distance_already_calculate = False
                self.student_ctc_already_calculate = False
                for loss in dec_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    # l_dict = self.get_loss(
                    #     loss, aux_outputs, targets, indices, extra_info, **kwargs)
                    l_dict=self.get_loss_multi_task(
                        loss, aux_outputs, target_info['cls_targets'],extra_info['indices'], extra_info, target_info,**kwargs)

                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            if o2m and self.use_o2m_enc:
                indices = self.o2m_matcher_enc(enc_outputs, target_info['cls_targets'])
            else:
                indices = self.enc_matcher(enc_outputs, target_info['cls_targets'])

            num_inst = self.get_num_inst(indices, outputs)
            extra_info = dict(indices=indices, num_inst=num_inst, )

            for loss in self.enc_losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                # l_dict = self.get_loss(
                #     loss, enc_outputs, targets, indices, extra_info, **kwargs)
                l_dict = self.get_loss_multi_task(
                    loss, enc_outputs, target_info['cls_targets'],extra_info['indices'], extra_info, target_info, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)


        return losses


class SetAdaptiveO2MCriterionFull(nn.Module):
    """
    Modified from SetCriterion
        1) add o2m_matcher
        2) add dual thresholds for loss_labels
        3) add adaptive weights for loss_labels

    """

    def __init__(
            self,
            num_classes,
            enc_matcher,
            dec_matcher,
            o2m_matcher_enc,
            o2m_matcher_dec,
            weight_dict,
            enc_losses,
            num_sample_points,
            dec_losses,
            voc_size,
            num_ctrl_points,
            focal_alpha=0.25,
            focal_gamma=2.0
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as suffix the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.enc_matcher = enc_matcher
        self.dec_matcher = dec_matcher
        self.o2m_matcher_enc = o2m_matcher_enc
        self.o2m_matcher_dec = o2m_matcher_dec
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)
        self.dec_losses = dec_losses
        self.voc_size = voc_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points

    def loss_labels(self, outputs, targets, indices, num_inst, log=False):
        """Classification loss (NLL)
        targets dicts must contain the suffix "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:-1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(shape,
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        # src_logits, target_classes_onehot: (bs, nq, n_ctrl_pts, 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_inst,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_adaptive(self, outputs, targets, indices, num_inst, log=False):
        """Classification loss (NLL) for partial predictions
        targets dicts must contain the suffix "labels" containing a tensor of dim [nb_target_boxes]
        also contains the suffix "neg_idx" to assign the background

        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:-1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        if 'neg_idx' in targets:
            neg_idx = targets['neg_idx']
            target_classes[neg_idx] = self.num_classes



        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(shape,
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        # src_logits, target_classes_onehot: (bs, nq, n_ctrl_pts, 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_inst,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_beziers(self, outputs, targets, indices, num_inst):
        # may FIX: (1) seg valid points
        assert 'pred_beziers' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_beziers = outputs['pred_beziers'][idx]
        src_beziers = self.bezier_sampler.get_sample_points(src_beziers.view(-1, 4, 2))

        has_beziers = 'beziers' in targets[0]
        if has_beziers:
            target_beziers = torch.cat(
                [t['beziers'][i] for t, (_, i) in zip(targets, indices)],
                dim=0
            )
            target_beziers = self.bezier_sampler.get_sample_points(target_beziers)
        else:
            target_beziers = torch.cat(
                [t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)],
                dim=0
            )
        if target_beziers.numel() == 0:
            target_beziers = src_beziers.clone().detach()
        loss_bezier = F.l1_loss(src_beziers, target_beziers, reduction='none')
        losses = {}
        losses['loss_bezier'] = loss_bezier.sum() / num_inst
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_inst):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_texts(self, outputs, targets, indices, num_inst):
        # CTC loss for classification of points
        assert 'pred_text_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        src_texts = src_texts.permute(1, 0, 2)
        src = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)

        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
        input_lengths = torch.full((src.size(1),), src.size(0), dtype=torch.long)
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)
        if target_lengths.sum() == 0:
            return {'loss_texts': torch.tensor(0.).to(src_texts.device)}
        else:
            target_texts = torch.cat([t[:l] for t, l in zip(target_texts, target_lengths)])

            return {
                'loss_texts': F.ctc_loss(
                    src,
                    target_texts,
                    input_lengths,
                    target_lengths,
                    blank=self.voc_size,
                    zero_infinity=True
                )
            }

    def loss_texts_adaptive(self, outputs, targets, indices, num_inst):
        # CTC loss for classification of points
        assert 'pred_text_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        # src_texts = src_texts.permute(1, 0, 2)
        src_texts = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)

        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)])  # n, length
        # ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])  # n,
        cls_scores = torch.cat([t["scores"][J] for t, (_, J) in zip(targets, indices)])

        src_texts = src_texts.permute(1, 0, 2)

        input_lengths = torch.full((src_texts.size(1),), src_texts.size(0), dtype=torch.long)
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)

        if target_lengths.sum() == 0:
            return {'loss_texts': torch.tensor(0.).to(src_texts.device)}
        else:
            target_texts = torch.cat([t[:l] for t, l in zip(target_texts, target_lengths)])

            loss_text = F.ctc_loss(
                src_texts,
                target_texts,
                input_lengths,
                target_lengths,
                blank=self.voc_size,
                zero_infinity=True,
                reduction="none"
            )
            loss_text = loss_text * cls_scores

            return {
                'loss_texts': torch.mean(loss_text)
            }

    def loss_ctrl_points(self, outputs, targets, indices, num_inst):
        """Compute the L1 regression loss
        """
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_ctrl_points = F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='sum')
        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    def loss_ctrl_points_adaptive(self, outputs, targets, indices, num_inst):
        """Compute the L1 regression loss
        """
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])
        loss_ctrl_points = F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='none')
        weights = ctc_scores[:,None, None].repeat(1, loss_ctrl_points.shape[1], loss_ctrl_points.shape[2])
        loss_ctrl_points = torch.sum(weights * loss_ctrl_points)
        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    def loss_bd_points(self, outputs, targets, indices, num_inst):
        assert 'pred_bd_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bd_points = outputs['pred_bd_points'][idx]
        target_bd_points = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bd_points = F.l1_loss(src_bd_points, target_bd_points, reduction='sum')
        losses = {'loss_bd_points': loss_bd_points / num_inst}
        return losses

    def loss_bd_points_adaptive(self, outputs, targets, indices, num_inst):
        assert 'pred_bd_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bd_points = outputs['pred_bd_points'][idx]
        target_bd_points = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ctc_scores = torch.cat([t['ctc_scores'][i] for t, (_, i) in zip(targets, indices)])

        loss_bd_points = F.l1_loss(src_bd_points, target_bd_points, reduction='none')
        weights = ctc_scores[:, None, None].repeat(1, loss_bd_points.shape[1], loss_bd_points.shape[2])
        loss_bd_points = torch.sum(weights * loss_bd_points)
        losses = {'loss_bd_points': loss_bd_points / num_inst}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    def get_num_inst(self, indices , outputs):
        num_inst = sum(len(i[1]) for i in indices)
        num_inst = torch.as_tensor(
            [num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()
        return num_inst

    def get_loss(self, loss, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'labels_adaptive': self.loss_labels_adaptive,
            'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'ctrl_points_adaptive': self.loss_ctrl_points_adaptive,
            'beziers': self.loss_beziers,
            'texts': self.loss_texts,
            'texts_adaptive': self.loss_texts_adaptive,
            'bd_points': self.loss_bd_points,
            'bd_points_adaptive': self.loss_bd_points_adaptive,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_inst, **kwargs)

    def forward(self, outputs, targets, o2m=False):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # try:
        if o2m:
            indices = self.o2m_matcher_dec(outputs_without_aux, targets)
        else:
            indices = self.dec_matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_inst = self.get_num_inst(indices, outputs)


        # Compute all the requested losses
        losses = {}

        for loss in self.dec_losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_inst, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if o2m:
                    indices = self.o2m_matcher_dec(aux_outputs, targets)
                else:
                    indices = self.dec_matcher(aux_outputs, targets)
                num_inst = self.get_num_inst(indices, outputs)
                for loss in self.dec_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_inst, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)



        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            # indices = self.enc_matcher(enc_outputs, targets)
            if o2m:
                indices = self.o2m_matcher_enc(enc_outputs, targets)
            else:
                indices = self.enc_matcher(enc_outputs, targets)
            num_inst = self.get_num_inst(indices, outputs)
            for loss in self.enc_losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, targets, indices, num_inst, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses



import math
class NLLoss(nn.Module):
    """ """

    def __init__(self):
        super(NLLoss, self).__init__()

    def forward(
        self,
        input,
        input_std,
        target,
        weight=None,
        iou_weight=None,
        beta=1.0,
        loss_denorm=None,
        method="weight_ctr_sum",
    ):
        """
        Args:
            pred: Nx4 predicted bounding boxes; before sigmoid
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        # TODO: check bbox_inside_weights, bbox_outside_weights, getlossscale
        mean = input
        sigma_sq = torch.square(input_std)

        # smooth l1 ?
        # Gradient explosion and predict log(2*sigma) instead?
        first_term = torch.square(target - mean).detach() / (2 * sigma_sq)
        second_term = 0.5 * torch.log(sigma_sq)
        sum_before_iou = (first_term + second_term).sum(-1).sum(-1) + 2 * torch.log(
            2 * torch.Tensor([math.pi]).cuda()
        )
        loss_mean = (sum_before_iou * iou_weight).sum()
        return loss_mean
