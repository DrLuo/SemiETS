import numpy as np
import pickle

import torch
from detectron2.utils.visualizer import Visualizer,VisImage
import matplotlib.colors as mplc
import matplotlib.font_manager as mfm
import matplotlib as mpl
import matplotlib.figure as mplfigure
import random
from shapely.geometry import LineString
import math
import operator
from functools import reduce
from torch import cat,device


class TextVisualizer(Visualizer):
    def __init__(self, image, metadata, instance_mode, cfg , with_gt = False):
        Visualizer.__init__(self, image, metadata, instance_mode=instance_mode)
        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.use_customer_dictionary = cfg.MODEL.TRANSFORMER.CUSTOM_DICT
        if with_gt :
            self.output_gt = VisImage(self.img, scale=1.0)
        if self.voc_size == 96:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        elif self.voc_size == 37:
            self.CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
        # voc_size includes the unknown class, which is not in self.CTABLES
        assert(int(self.voc_size - 1) == len(self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1), len(self.CTLABELS))

    def draw_instance_predictions(self, predictions):
        ctrl_pnts = predictions['ctrl_points'].numpy()
        scores = predictions['scores'].tolist()
        recs = predictions["recs"]
        bd_pts = np.asarray(predictions["bd_points"])
        ctc_scores = predictions['ctc_score']
        self.overlay_instances(ctrl_pnts, recs, bd_pts,scores=scores,ctc_scores = ctc_scores)

        return self.output
    def draw_instance_predictions_withGT(self,anno):
        self.output = self.output_gt#clear
        bd_gt =np.array([instance['boundary'].reshape(25,4) for instance in anno])
        ctrl_gt = np.array([instance['polyline'].reshape(-1) for instance in anno])
        recs_gt = np.array([instance['text'] for instance in anno])
        self.overlay_instances(ctrl_gt, recs_gt, bd_gt)
        return self.output
    def draw_ts(self, predictions, img_size = None):
        if img_size is not None:
            img_size = img_size.__reversed__()
            ctrl_pnts = (predictions['ctrl_points'].to(device('cpu')) * img_size).numpy()
            bd_pts = np.asarray(predictions['bd_points'].to(device('cpu')) * cat([img_size, img_size]))
            recs = predictions['texts'].to(device('cpu')).numpy()
            self.overlay_instances_ts(ctrl_pnts, bd_pts, recs)
        else:
            ctrl_pnts = (predictions['ctrl_points'].to(device('cpu'))).numpy()
            bd_pts = np.asarray(predictions['bd_points'].to(device('cpu')))
            recs = predictions['texts'].to(device('cpu')).numpy()
            self.overlay_instances_ts(ctrl_pnts, bd_pts, recs)
        return self.output

    def draw_ref_points(self, ref_points, img_size):
        colors = [(0,0.5,0),(0,0.75,0),(1,0,1),(0.75,0,0.75),(0.5,0,0.5),(1,0,0),(0.75,0,0),(0.5,0,0),
        (0,0,1),(0,0,0.75),(0.75,0.25,0.25),(0.75,0.5,0.5),(0,0.75,0.75),(0,0.5,0.5),(0,0.3,0.75)]
        img_size = img_size.__reversed__()
        ref_points = (ref_points.to(device('cpu')) * img_size).numpy()
        for i, ref_point in enumerate(ref_points):
            color = random.choice(colors)
            line = self._process_ctrl_pnt(ref_point)
            self.draw_line(
                line[:, 0],
                line[:, 1],
                color=color,
                linewidth=2
            )
            for pt in line:
                self.draw_circle(pt, 'r', radius=6)
            self.draw_text(
                        f'{i}',
                        line[-1] + np.array([0, 10]),
                        color=color,
                        horizontal_alignment='left',
                        font_size=self._default_font_size,
                        draw_chinese=False if self.voc_size == 37 or self.voc_size == 96 else True
                    )
        return self.output

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points

    def _ctc_decode_recognition(self, rec):
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

    def overlay_instances(self, ctrl_pnts, recs, bd_pnts, alpha=0.4,scores=None,ctc_scores=None):
        colors = [(0,0.5,0),(0,0.75,0),(1,0,1),(0.75,0,0.75),(0.5,0,0.5),(1,0,0),(0.75,0,0),(0.5,0,0),
        (0,0,1),(0,0,0.75),(0.75,0.25,0.25),(0.75,0.5,0.5),(0,0.75,0.75),(0,0.5,0.5),(0,0.3,0.75)]
        instance_num = ctrl_pnts.shape[0]
        scores = [1 for i in range(instance_num)] if scores is None else scores
        ctc_scores=[1 for i in range(instance_num)] if ctc_scores is None else ctc_scores
        for ctrl_pnt, rec, bd , sc , ctc_sc in zip(ctrl_pnts, recs, bd_pnts , scores , ctc_scores):
            color = random.choice(colors)

            # draw polygons
            if bd is not None:
                bd = np.hsplit(bd, 2)
                bd = np.vstack([bd[0], bd[1][::-1]])
                self.draw_polygon(bd, color, alpha=alpha)

            # draw center lines
            line = self._process_ctrl_pnt(ctrl_pnt)
            line_ = LineString(line)
            center_point = np.array(line_.interpolate(0.5, normalized=True).coords[0], dtype=np.int32)
            self.draw_line(
                line[:, 0],
                line[:, 1],
                color=color,
                linewidth=2
            )
            for pt in line:
                self.draw_circle(pt, 'w', radius=4)
                self.draw_circle(pt, 'r', radius=2)

            # draw text
            text = self._ctc_decode_recognition(rec)
            if self.voc_size == 37:
                text = text.upper() #大写
            # text = "{}".format(text)
            text = f'{text} {ctc_sc:.4f}'
            det_text = f'{sc:.4f}'
            det_text_pos = center_point
            lighter_color = self._change_color_brightness(color, brightness_factor=0)
            if bd is not None:
                text_pos = bd[0] - np.array([0,15])
            else:
                text_pos = center_point
            horiz_align = "left"
            font_size = self._default_font_size
            self.draw_text(
                        text,
                        text_pos,
                        color=lighter_color,
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                        draw_chinese=False if self.voc_size == 37 or self.voc_size == 96 else True
                    )
            self.draw_text(
                det_text,
                det_text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
                draw_chinese=False if self.voc_size == 37 or self.voc_size == 96 else True
            )
            
    def overlay_instances_ts(self, ctrl_pnts, bd_pnts, recs, alpha=0.4):
        colors = [(0,0.5,0),(0,0.75,0),(1,0,1),(0.75,0,0.75),(0.5,0,0.5),(1,0,0),(0.75,0,0),(0.5,0,0),
        (0,0,1),(0,0,0.75),(0.75,0.25,0.25),(0.75,0.5,0.5),(0,0.75,0.75),(0,0.5,0.5),(0,0.3,0.75)]

        for ctrl_pnt, rec, bd in zip(ctrl_pnts, recs, bd_pnts):
            color = random.choice(colors)

            # draw polygons
            if bd is not None:
                bd = np.hsplit(bd, 2)
                bd = np.vstack([bd[0], bd[1][::-1]])
                self.draw_polygon(bd, color, alpha=alpha)

            # draw center lines
            line = self._process_ctrl_pnt(ctrl_pnt)
            line_ = LineString(line)
            center_point = np.array(line_.interpolate(0.5, normalized=True).coords[0], dtype=np.int32)
            self.draw_line(
                line[:, 0],
                line[:, 1],
                color=color,
                linewidth=2
            )
            for pt in line:
                self.draw_circle(pt, 'w', radius=4)
                self.draw_circle(pt, 'r', radius=2)

            # draw text
            text = self._ctc_decode_recognition(rec)
            if self.voc_size == 37:
                text = text.upper()
            # text = "{:.2f}: {}".format(score, text)
            text = "{}".format(text)
            lighter_color = self._change_color_brightness(color, brightness_factor=0)
            if bd is not None:
                text_pos = bd[0] - np.array([0,15])
            else:
                text_pos = center_point
            horiz_align = "left"
            font_size = self._default_font_size
            self.draw_text(
                        text,
                        text_pos,
                        color=lighter_color,
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                        draw_chinese=False if self.voc_size == 37 or self.voc_size == 96 else True
                    )

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
        draw_chinese=False
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        
        x, y = position
        if draw_chinese:
            font_path = "./simsun.ttc"
            prop = mfm.FontProperties(fname=font_path)
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
                fontproperties=prop
            )
        else:
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
            )
        return self.output