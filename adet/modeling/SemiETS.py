import numpy as np
import torch
from torch import nn
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances
from adet.modeling.model.losses import SetAdaptiveO2MCriterionSemi
from adet.modeling.model.matcher import CtrlPointCost, build_all_matcher_semi, DetectionCost
from adet.utils.polygon_utils import SPOTTING_NMS

import pickle
from .semi_text_spotter import MultiStreamSpotter

from adet.utils.structure_utils import weighted_loss

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

try:
    from ctcdecode import CTCBeamDecoder
except ImportError:
    CTCBeamDecoder = None


@META_ARCH_REGISTRY.register()
class SemiETSTextSpotter(MultiStreamSpotter):

    def __init__(self, cfg):
        super(SemiETSTextSpotter, self).__init__(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        try:
            self.train_cfg = cfg.SSL
            self.cfg = cfg
        except:
            self.train_cfg = None

        if self.train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.UNSUP_WEIGHT
            self.inference_on = self.train_cfg.INFERENCE_ON
            self.label_warm_up = self.train_cfg.WARM_UP
            self.stage_warm_up = self.train_cfg.STAGE_WARM_UP
            self.decoder_only = self.train_cfg.DECODER_ONLY


        self.to(self.device)

        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        if self.voc_size == 96:
            self.CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1',
                             '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C',
                             'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                             'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                             'z', '{', '|', '}', '~', '*', '-']
        elif self.voc_size == 37:
            self.CTLABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '*', '-']
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
        assert (int(self.voc_size + 1) == len(
            self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1),
                                                                                              len(self.CTLABELS))
        self.covariance_type = 'diag'
        self.curr_step = 0
        self.is_loss = True
        #prepare ctc decoder for pseudo label gen
        self.ctc_decoder = CTCBeamDecoder(
            labels=self.CTLABELS,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=100,
            num_processes=4,
            blank_id=self.voc_size,
            log_probs_input=False
        )

        #loss list initialization
        if self.decoder_only:
            enc_losses = []
        else:
            enc_losses = ["labels", "beziers"]

        o2m_dec_losses = self.train_cfg.DECODER_LOSS
        o2o_dec_losses = self.train_cfg.O2O_DECODER_LOSS
        print('o2m_dec_losses: ', o2m_dec_losses)
        print('o2o_dec_losses: ', o2o_dec_losses)

        # rewrite pseudo label loss weight
        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {
            "loss_ce": loss_cfg.POINT_CLASS_WEIGHT,
            "loss_texts": loss_cfg.POINT_TEXT_WEIGHT,
            "loss_ctrl_points": loss_cfg.POINT_COORD_WEIGHT,
            "loss_bd_points": loss_cfg.BOUNDARY_WEIGHT,
        }

        enc_weight_dict = {
            "loss_bezier": loss_cfg.BEZIER_COORD_WEIGHT,
            "loss_ce": loss_cfg.BEZIER_CLASS_WEIGHT
        }

        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # aux_weight_dict_clear=[]
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()}
                )
                # aux_weight_dict_clear.extend([k + f'_{i}' for k in weight_dict_clear])
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()}
            )
            weight_dict.update(aux_weight_dict)



        bezier_matcher, point_matcher, o2m_matcher_enc, o2m_matcher_dec, det_based_point_matcher, rec_based_point_matcher = build_all_matcher_semi(
            cfg)

        self.criterion = SetAdaptiveO2MCriterionSemi(
            self.student.detection_transformer.num_classes,
            bezier_matcher,
            point_matcher,
            o2m_matcher_enc,
            o2m_matcher_dec,
            det_based_point_matcher,
            rec_based_point_matcher,
            weight_dict,
            enc_losses,
            cfg.MODEL.TRANSFORMER.LOSS.BEZIER_SAMPLE_POINTS,
            o2m_dec_losses,
            o2o_dec_losses,
            cfg.MODEL.TRANSFORMER.VOC_SIZE,
            self.student.detection_transformer.num_points,
            focal_alpha=loss_cfg.FOCAL_ALPHA,
            focal_gamma=loss_cfg.FOCAL_GAMMA,
            leven_dis_alpha=loss_cfg.LEVEN_ALPHA,
            cost_alpha=loss_cfg.COST_ALPHA,
            rec_threshold=self.train_cfg.PSEUDO_LABEL_FINAL_SCORE_THR,
            det_threshold= self.train_cfg.PSEUDO_LABEL_INITIAL_SCORE_THR,
            use_o2m_enc = self.train_cfg.USE_O2M_ENC,
            use_combined_thr= self.train_cfg.USE_COMBINED_THR,
            o2m_text_o2o = self.train_cfg.O2M_TEXT_O2O,
            det_adaptive_type= loss_cfg.DET_ADAPTIVE_TYPE,
            rec_adaptive_type= loss_cfg.REC_ADAPTIVE_TYPE,
            use_task_specific_matcher=self.train_cfg.USE_SEPERATE_MATCHER,
            precise_teacher= loss_cfg.PRECISE_TEACHER,
        )
        self.criterion.ctc_decoder = self.ctc_decoder
        self.criterion.voc_size = self.voc_size
        self.criterion.device = self.device

        self.use_task_specific_matcher = self.train_cfg.USE_SEPERATE_MATCHER
        self.eval_count = 0
        self.decoder_matcher = CtrlPointCost()
        self.criterion.decoder_matcher = DetectionCost()
        self.o2m_current_stat = False
        # if hasattr(self.student, 'use_o2m'):
        #     self.student.use_o2m = False
        self.num_points = cfg.MODEL.TRANSFORMER.NUM_POINTS
        self.use_sup_o2m = cfg.MODEL.TRANSFORMER.LOSS.USE_SUP_O2M


    def forward(self, batched_inputs):
        """
        Forward logic of the model, unsupervised training mode.
        """
        if self.training:
            return self.forward_train(batched_inputs)
        else:
            return self.forward_test(batched_inputs)



    def forward_test(self, batched_inputs):
        """
        Forward logic of the model, unsupervised training mode.
        """

        return super().inference(batched_inputs)



    def forward_train(self, batched_inputs):
        """
        Forward logic of the model, containing 2 steps:
            - 1. forward the labeled data
            - 2. forward the unlabeled data
        """

        # split the batched inputs for teacher and student
        input_dict = self.split_ssl_batch(batched_inputs)

        if hasattr(self.student, 'use_o2m'):
            if self.use_sup_o2m:
                if self.curr_step >= self.label_warm_up:
                    use_o2m = (self.curr_step < self.stage_warm_up)
                else:
                    use_o2m = False
                self.student.use_o2m = use_o2m
            else:
                self.student.use_o2m = False
                self.teacher.use_o2m = False

        loss = {}
        # forward the labeled data
        if 'sup' in input_dict:
            label_loss_dict = self.student.forward(input_dict['sup'])
            loss.update(add_prefix(label_loss_dict, 'sup'))

        # forward the unlabeled data
        if self.curr_step >= self.label_warm_up:
            if "unsup_student" in input_dict:
                input_teacher = input_dict['unsup_teacher']
                input_student = input_dict['unsup_student']

                unlabel_loss_dict = self.forward_unlabeled(input_teacher, input_student)

                unlabel_loss_dict = weighted_loss(unlabel_loss_dict, self.unsup_weight)

                loss.update(add_prefix(unlabel_loss_dict, 'unsup'))

        return loss

    def forward_unlabeled(self, input_teacher, input_student):
        ''' pseudo label generation and consistency loss calculation'''

        # forward the teacher model

        with torch.no_grad():
            # 1. get pseudo bbox from the weak augmented images
            teacher_info = self.extract_teacher_info(input_teacher)

        # 2. get the prediction from the strong augmented images
        student_info = self.extract_student_info(input_student)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        # 1. convert the weak augmented pseudo bbox into the strong augmented pseudo bbox
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        # pseudo_copy = deepcopy(teacher_info['pseudo_labels'])
        pseudo_labels = teacher_info['pseudo_labels']
        for pl, si in zip(pseudo_labels, student_info['gt_instances']):
            pl['gt_instances'] = si
        targets = []
        for instances, mat in zip(pseudo_labels, M):
            targets.append(self._geo_trans_for_pseudo(instances, mat))

        # 2. loss with pseudo bbox
        if self.curr_step >= self.label_warm_up:
            use_o2m = (self.curr_step < self.stage_warm_up)
        else:
            use_o2m = False

        self.criterion.curr_step = self.curr_step
        unsup_loss = self.loss_by_output(
            student_info['output'],
            targets,
            use_o2m,
        )

        return unsup_loss


    def gen_coarse_label(
            self,
            ctrl_point_cls,
            ctrl_point_coord,
            ctrl_point_text,
            bd_points,
            image_sizes,
            threshold=None,
            use_task_specific_matcher=False,
            use_spotting_NMS = False,
    ):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []
        # cls shape: (b, nq, n_pts, voc_size)
        ctrl_point_text = torch.softmax(ctrl_point_text, dim=-1)
        scores_ctrl_point = ctrl_point_cls.clone().sigmoid()
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)


        # roughly filter using mean and std
        if bd_points is not None:
            for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, bd, image_size, scores_per_point in zip(
                    scores, labels, ctrl_point_coord, ctrl_point_text, bd_points, image_sizes, scores_ctrl_point
            ):

                if threshold is None:
                    avg_score = torch.mean(scores_per_image)
                    std_score = torch.std(scores_per_image)
                    pseudo_thr = avg_score + std_score
                else:
                    pseudo_thr = threshold

                # filter_reset in task assignment setting
                if use_task_specific_matcher:
                    pseudo_thr = 0.1
                # filter the pseudo points
                selector = scores_per_image >= pseudo_thr

                # selector = scores_per_image >= self.test_score_threshold
                scores_per_image = scores_per_image[selector]
                labels_per_image = labels_per_image[selector]
                ctrl_point_per_image = ctrl_point_per_image[selector]
                ctrl_point_text_per_image = ctrl_point_text_per_image[selector]
                bd = bd[selector]
                scores_per_point = scores_per_point[selector]


                beam_results, beam_scores, _, out_lens = self.ctc_decoder.decode(ctrl_point_text_per_image)
                _, text_pred = ctrl_point_text_per_image.topk(1)
                recs = text_pred.squeeze(-1)
                if len(recs) > 0:
                    recs = torch.stack([self._ctc_decode_recognition(rc) for rc in recs])

                valid_length = (recs != self.voc_size).sum(-1)
                valid_length = valid_length.unsqueeze(1).expand(-1, beam_results.size(1), )

                beam_results, out_lens = beam_results.to(self.device), out_lens.to(self.device),
                beam_scores = beam_scores.to(self.device)

                recs_expanded = recs.unsqueeze(1).expand(-1, beam_results.size(1), -1)
                recs_expanded = recs_expanded.to(dtype=beam_results.dtype)

                # VERSION 2 ctc score fetch
                difference = torch.abs(recs_expanded - beam_results)
                mask = torch.arange(beam_results.size(2)).expand(beam_results.shape[0], beam_results.shape[1], -1).to(
                    self.device)
                mask = mask < valid_length.unsqueeze(-1)
                valid_difference = difference * mask #N,100,25/50
                score_idx = (out_lens == valid_length) & (valid_difference.sum(dim=-1) == 0 )#N,100
                success_match = score_idx.any(dim=-1)
                ctc_scores = torch.full((recs.shape[0],), 100.0, device=self.device)
                ctc_scores[success_match] = beam_scores[score_idx]
                result_ctc_score = 1 / torch.exp(ctc_scores)
                #############################

                if use_spotting_NMS :
                    VALID_INDEX = SPOTTING_NMS(bds=bd,scs=scores_per_image,ctcs=result_ctc_score,recs=recs,
                                               iou_threshold=0.7,voc_size=self.voc_size)
                    selector = VALID_INDEX
                    scores_per_image = scores_per_image[selector]
                    labels_per_image = labels_per_image[selector]
                    ctrl_point_per_image = ctrl_point_per_image[selector]
                    ctrl_point_text_per_image = ctrl_point_text_per_image[selector]
                    bd = bd[selector]
                    scores_per_point = scores_per_point[selector]
                    result_ctc_score = result_ctc_score[selector]
                    recs = recs[selector]
                ###############################################################

                result = Instances(image_size)
                result.scores = scores_per_image
                result.point_scores = scores_per_point
                result.pred_classes = labels_per_image
                result.rec_scores = ctrl_point_text_per_image
                ctrl_point_per_image[..., 0] *= image_size[1]
                ctrl_point_per_image[..., 1] *= image_size[0]
                result.ctrl_points = ctrl_point_per_image.flatten(1)

                result.recs = recs
                result.ctc_scores = result_ctc_score
                bd[..., 0::2] *= image_size[1]
                bd[..., 1::2] *= image_size[0]
                result.bd = bd

                results.append(result)

            return results
        else:
            for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, image_size, scores_per_point in zip(
                    scores, labels, ctrl_point_coord, ctrl_point_text, image_sizes, scores_ctrl_point
            ):
                if threshold is None:
                    avg_score = torch.mean(scores_per_image)
                    std_score = torch.std(scores_per_image)

                    pseudo_thr = avg_score + std_score
                else:
                    pseudo_thr = threshold

                selector = scores_per_image >= pseudo_thr
                scores_per_image = scores_per_image[selector]
                labels_per_image = labels_per_image[selector]
                ctrl_point_per_image = ctrl_point_per_image[selector]
                ctrl_point_text_per_image = ctrl_point_text_per_image[selector]

                result = Instances(image_size)
                result.scores = scores_per_image
                result.point_scores = scores_per_point
                result.pred_classes = labels_per_image
                result.rec_scores = ctrl_point_text_per_image
                ctrl_point_per_image[..., 0] *= image_size[1]
                ctrl_point_per_image[..., 1] *= image_size[0]
                result.ctrl_points = ctrl_point_per_image.flatten(1)
                _, text_pred = ctrl_point_text_per_image.topk(1)
                result.recs = text_pred.squeeze(-1)
                result.bd = [None] * len(scores_per_image)
                results.append(result)
            return results

    def gen_coarse_label_full(
            self,
            ctrl_point_cls,
            ctrl_point_coord,
            ctrl_point_text,
            bd_points,
            image_sizes,
            threshold=None,
            use_task_specific_matcher=False,
            use_spotting_nms = False
    ):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []
        # cls shape: (b, nq, n_pts, voc_size)
        ctrl_point_text = torch.softmax(ctrl_point_text, dim=-1)
        scores_ctrl_point = ctrl_point_cls.clone().sigmoid()
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        # roughly filter using mean and std
        if bd_points is not None:
            for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, bd, image_size, scores_per_point in zip(
                    scores, labels, ctrl_point_coord, ctrl_point_text, bd_points, image_sizes, scores_ctrl_point
            ):

                if threshold is None:
                    avg_score = torch.mean(scores_per_image)
                    std_score = torch.std(scores_per_image)

                    pseudo_thr = avg_score + std_score
                else:
                    pseudo_thr = threshold

                # filter the pseudo points
                selector = scores_per_image >= pseudo_thr

                # selector = scores_per_image >= self.test_score_threshold
                scores_per_image = scores_per_image[selector]
                labels_per_image = labels_per_image[selector]
                ctrl_point_per_image = ctrl_point_per_image[selector]
                ctrl_point_text_per_image = ctrl_point_text_per_image[selector]
                bd = bd[selector]
                scores_per_point = scores_per_point[selector]
                ################################################################

                text_logits = ctrl_point_text_per_image
                character_uncertainty = torch.sum(-text_logits * torch.log(text_logits), dim=-1)

                character_norm_factor = torch.ones_like(character_uncertainty)
                instance_character_uncertainty = torch.mean(character_uncertainty /character_norm_factor, dim=-1)


                beam_results, beam_scores, _, out_lens = self.ctc_decoder.decode(ctrl_point_text_per_image)
                ###################################

                beam_width = 5

                hypo_ctc_sc = beam_scores[:,:beam_width]
                hypo_ctc_sc = 1 / torch.exp(hypo_ctc_sc.to(self.device))
                #
                hypo_prob = torch.softmax(hypo_ctc_sc/0.1,dim=-1)

                instance_beam_search_uncertainty = torch.sum(-hypo_prob*torch.log(hypo_prob),dim=-1)

                beam_factor =torch.ones_like(instance_beam_search_uncertainty)
                instance_beam_search_uncertainty = instance_beam_search_uncertainty /beam_factor

                instance_total_uncertainty = instance_character_uncertainty

                ###################################
                ctc_score = []
                _, text_pred = ctrl_point_text_per_image.topk(1)
                recs = text_pred.squeeze(-1)
                if len(recs) > 0 :
                    recs = torch.stack([self._ctc_decode_recognition(rc) for rc in recs])

                for i in range(len(beam_results)):
                    rec_i = torch.nonzero(recs[i] != self.voc_size).squeeze(-1)
                    rec = recs[i][rec_i].int().to(torch.device('cpu'))
                    for j in range(len(beam_results[i])):
                        if torch.equal(beam_results[i][j][:out_lens[i, j]], rec):
                            ctc_score.append(beam_scores[i, j])
                            break
                        if j == len(beam_results[i]) - 1:
                            ctc_score.append(100.0)
                result_ctc_score = torch.tensor(ctc_score)
                result_ctc_score = 1 / torch.exp(result_ctc_score.to(self.device))


                ###############################################################
                #   SPOTTING NMS IMPLEMENTATION
                if use_spotting_nms:
                    VALID_INDEX = SPOTTING_NMS(bds=bd, scs=scores_per_image, ctcs=result_ctc_score,recs=recs,
                                               iou_threshold=0.7,voc_size=self.voc_size )
                    selector = VALID_INDEX
                    scores_per_image = scores_per_image[selector]
                    labels_per_image = labels_per_image[selector]
                    ctrl_point_per_image = ctrl_point_per_image[selector]
                    ctrl_point_text_per_image = ctrl_point_text_per_image[selector]
                    bd = bd[selector]
                    scores_per_point = scores_per_point[selector]
                    result_ctc_score = result_ctc_score[selector]
                    recs = recs[selector]
                    instance_total_uncertainty= instance_total_uncertainty[selector]

                ###############################################################
                # filter out the empty recognition results
                result = Instances(image_size)
                result.scores = scores_per_image
                result.point_scores = scores_per_point
                result.pred_classes = labels_per_image
                result.rec_scores = ctrl_point_text_per_image
                ctrl_point_per_image[..., 0] *= image_size[1]
                ctrl_point_per_image[..., 1] *= image_size[0]
                result.ctrl_points = ctrl_point_per_image.flatten(1)

                result.recs = recs
                result.ctc_scores = result_ctc_score
                bd[..., 0::2] *= image_size[1]
                bd[..., 1::2] *= image_size[0]
                result.bd = bd
                #uncertainty
                result.uncertainty = instance_total_uncertainty

                results.append(result)
            return results
        else:
            for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, image_size, scores_per_point in zip(
                    scores, labels, ctrl_point_coord, ctrl_point_text, image_sizes, scores_ctrl_point
            ):
                if threshold is None:
                    avg_score = torch.mean(scores_per_image)
                    std_score = torch.std(scores_per_image)

                    pseudo_thr = avg_score + std_score
                else:
                    pseudo_thr = threshold

                selector = scores_per_image >= pseudo_thr
                scores_per_image = scores_per_image[selector]
                labels_per_image = labels_per_image[selector]
                ctrl_point_per_image = ctrl_point_per_image[selector]
                ctrl_point_text_per_image = ctrl_point_text_per_image[selector]

                result = Instances(image_size)
                result.scores = scores_per_image
                result.point_scores = scores_per_point
                result.pred_classes = labels_per_image
                result.rec_scores = ctrl_point_text_per_image
                ctrl_point_per_image[..., 0] *= image_size[1]
                ctrl_point_per_image[..., 1] *= image_size[0]
                result.ctrl_points = ctrl_point_per_image.flatten(1)
                _, text_pred = ctrl_point_text_per_image.topk(1)
                result.recs = text_pred.squeeze(-1)
                result.bd = [None] * len(scores_per_image)
                results.append(result)
            return results


    def create_empty_instances(self, r):
        """
        Create an empty Instances object with the given image size.
        """
        empty_r = Instances(r.image_size)
        num_pt = self.num_points
        shape_mapper = {
            'scores': (0),
            'pred_classes': (0),
            'ctrl_points': (0, num_pt*2),
            'rec_scores': (0, num_pt, self.voc_size+1),
            'recs': (0, num_pt),
            'bd': (0, num_pt, 4),
            'dec_recs': (0, num_pt),
            'ctc_scores':(0),
            'uncertainty':(0),
            'sampling_loc':(0,num_pt*128,2),
            'attn_weights':(0,num_pt*128),
        }
        for k in r.get_fields().keys():
            if k in shape_mapper:
                empty_r.set(k, torch.zeros(shape_mapper[k], dtype=torch.int64).to(self.device))
        return empty_r
    
    def check_invalid_rec(self, r, invalid_rec_idx):
        """ check if the recognition is valid """
        reduced_r = Instances(r.image_size)
        for k in r.get_fields().keys():
            value = r.get(k)
            value = value.split(1, dim=0)
            temp_value = []
            for i, v in enumerate(value):
                if i in invalid_rec_idx:
                    pass
                else:
                    temp_value.append(v)
            value = torch.cat(temp_value, dim=0)
            reduced_r.set(k, value)
        return reduced_r
    
    def _gen_text_pseudo(self, recs ,vocs):
        """ generate pseudo text labels """
        dec_recs = torch.ones_like(recs) * vocs
        invalid_rec_idx = []
        for i in range(len(recs)):
            # dec_recs[i] = self._ctc_decode_recognition(recs[i])
            dec_recs[i] = recs[i] #ctc decode on gen_coarse_label
            if sum(dec_recs[i] == vocs) ==  len(dec_recs[i]):
                invalid_rec_idx.append(i)
        return dec_recs, invalid_rec_idx

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

    def loss_by_output(self, output, targets, use_o2m=False):
        loss_dict = self.criterion(output, targets, o2m=use_o2m)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]

        return loss_dict


    def extract_teacher_info(self, batched_inputs):
        teacher_info = {}
        pred, images = self.teacher.forward_model(batched_inputs)
        teacher_info['img'] = images
        image_sizes = images.image_sizes

        
        results = self.gen_coarse_label(**pred, image_sizes=image_sizes, threshold=self.train_cfg.PSEUDO_LABEL_INITIAL_SCORE_THR,
                                        use_task_specific_matcher=self.train_cfg.USE_SEPERATE_MATCHER,
                                        use_spotting_NMS=self.train_cfg.USE_SPOTTING_NMS,)
        processed_results = []
        for results_per_image, input_per_image in zip(results, batched_inputs):
            r = results_per_image
            r.dec_recs, invalid_rec_idx = self._gen_text_pseudo(r.recs,self.voc_size)
            if len(invalid_rec_idx) > 0:
                if len(invalid_rec_idx) == len(r):
                    r = self.create_empty_instances(r)
                else:
                    r = self.check_invalid_rec(r, invalid_rec_idx)
            # r = self.check_invalid_rec(r)
            if len(r) == 0:
                r = self.create_empty_instances(r)
            processed_results.append({"instances": r})
        
        key_mapper = {
            'pred_classes': 'labels',       # unchanged
            'bd': 'bd_points',               # unchanged
            'ctrl_points': 'ctrl_points',   # reshape to [N, 25, 2]
            'dec_recs': 'texts',                 # unchanged
        }

        def map_pseudo_labels(result, source, target,num_pt):
            if source == 'ctrl_points':
                temp = result['instances'].get(source).reshape(-1, num_pt, 2)
                result['instances'].get_fields()[source] = temp
            else:
                result['instances'].get_fields()[target] = result['instances'].get_fields().pop(source)
            return result


        pseudo_labels = []
        # origin_keys = result.get('instances').get_fields().keys()
        for result in processed_results:
            pseudo_labels_per_image = {}
            origin_keys = list(result.get('instances').get_fields().keys())
            for k in origin_keys:
                if k in key_mapper:
                    result = map_pseudo_labels(result, k, key_mapper[k],num_pt = self.num_points)
                    pseudo_labels_per_image[key_mapper[k]] = result['instances'].get(key_mapper[k])
                else:
                    pseudo_labels_per_image[k] = result['instances'].get(k)
            pseudo_labels.append(pseudo_labels_per_image)

        teacher_info['pseudo_labels'] = pseudo_labels

        # get the transform matrix to convert the pseudo labels (described by points)
        teacher_info["transform_matrix"] = [
            torch.from_numpy(x["transform_matrix"]).float()
            for x in batched_inputs
        ]
        # teacher_info["img_metas"] = img_metas
        return teacher_info

    def extract_student_info(self, batched_inputs):
        """Only get some data info of student model
        """
        student_info = {}
        images = self.student.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        output = self.student.detection_transformer(images)

        student_info["img"] = images
        student_info["gt_instances"] = gt_instances
        student_info["output"] = output
        student_info["transform_matrix"] = [
            torch.from_numpy(x["transform_matrix"]).float()
            for x in batched_inputs
        ]
        # prediction results of the student model

        return student_info


    def _get_match_student_ctc(self,match_student):
        text_logits = match_student['text_logits_raw']
        recs = match_student['recs']
        if len(recs)>0:
            recs = torch.stack([self._ctc_decode_recognition(rc) for rc in recs])
        text_logits = torch.softmax(text_logits,dim=-1)
        beam_results, beam_scores, _, out_lens = self.ctc_decoder.decode(text_logits)
        ctc_score = []
        for i in range(len(beam_results)):
            rec_i = torch.nonzero(recs[i] != self.voc_size).squeeze(-1)
            rec = recs[i][rec_i].int().to(torch.device('cpu'))
            for j in range(len(beam_results[i])):
                if torch.equal(beam_results[i][j][:out_lens[i, j]], rec):
                    ctc_score.append(beam_scores[i, j])
                    break
                if j == len(beam_results[i]) - 1:
                    ctc_score.append(100.0)
        result_ctc_score = torch.tensor(ctc_score)
        result_ctc_score = 1 / torch.exp(result_ctc_score.to(self.device))
        match_student['ctc_scores'] = result_ctc_score

        return match_student

    def _geo_trans_for_pseudo_attn(self, pseudo_labels, transform_matrix, keys=['ctrl_points', 'bd_points','sampling_loc']):
        """transform the pseudo labels to the original image space
        """
        device = pseudo_labels[keys[0]].device
        h, w = pseudo_labels['gt_instances'].image_size
        for k in keys:
            pseudo_labels[k] = self._transform_coords(pseudo_labels[k], transform_matrix.to(device))
            if pseudo_labels[k].shape[-1] == 2:
                pseudo_labels[k] = pseudo_labels[k] / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            elif pseudo_labels[k].shape[-1] == 4:
                pseudo_labels[k] = pseudo_labels[k] / torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)[None, None, :]

        return pseudo_labels

    def _geo_trans_for_pseudo(self, pseudo_labels, transform_matrix, keys=['ctrl_points', 'bd_points',]):
        """transform the pseudo labels to the original image space
        """
        device = pseudo_labels[keys[0]].device
        h, w = pseudo_labels['gt_instances'].image_size
        for k in keys:
            pseudo_labels[k] = self._transform_coords(pseudo_labels[k], transform_matrix.to(device))
            if pseudo_labels[k].shape[-1] == 2:
                pseudo_labels[k] = pseudo_labels[k] / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            elif pseudo_labels[k].shape[-1] == 4:
                pseudo_labels[k] = pseudo_labels[k] / torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)[None, None, :]

        return pseudo_labels

    # @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_coords(self, coords, trans_mat):

        B, N, C = coords.size()

        coords = coords.reshape(-1, C)
        if C > 2:
            assert C % 2 == 0
            coords = coords.reshape(-1, 2)
        coords = torch.cat([coords, coords.new_ones(coords.shape[0], 1)], dim=1)
        coords = torch.matmul(trans_mat, coords.t()).t()
        coords = coords[:, :2] / coords[:, 2:3]
        if C > 2:
            coords = coords.reshape(B*N, C)
        coords = coords.reshape(B, N, C)
        return coords

    # @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs