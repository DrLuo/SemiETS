import os
import argparse
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .datasets.text import register_text_instances, register_text_instances_ssl
from adet.config import get_cfg, get_cfg_semi
from detectron2.engine import default_argument_parser

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    # 37 voc_size
    "syntext1": ("syntext1/train_images", "syntext1/annotations/train_37voc.json"),
    "syntext2": ("syntext2/train_images", "syntext2/annotations/train_37voc.json"),
    "mlt": ("mlt2017/train_images", "mlt2017/train_37voc.json"),
    "totaltext_train": ("totaltext/train_images", "totaltext/train_37voc.json"),

    # semi-supervised settings
    "totaltext_train_0.5_label": ("totaltext/train_images", "totaltext/train_37voc_0.5_labeled.json"),
    "totaltext_train_0.5_unlabel": ("totaltext/train_images", "totaltext/train_37voc_0.5_unlabeled.json"),
    "totaltext_train_1_label": ("totaltext/train_images", "totaltext/train_37voc_1_labeled.json"),
    "totaltext_train_1_unlabel": ("totaltext/train_images", "totaltext/train_37voc_1_unlabeled.json"),
    "totaltext_train_2_label": ("totaltext/train_images", "totaltext/train_37voc_2_labeled.json"),
    "totaltext_train_2_unlabel": ("totaltext/train_images", "totaltext/train_37voc_2_unlabeled.json"),
    "totaltext_train_5_label": ("totaltext/train_images", "totaltext/train_37voc_5_labeled.json"),
    "totaltext_train_5_unlabel": ("totaltext/train_images", "totaltext/train_37voc_5_unlabeled.json"),
    "totaltext_train_10_label": ("totaltext/train_images", "totaltext/train_37voc_10_labeled.json"),
    "totaltext_train_10_unlabel": ("totaltext/train_images", "totaltext/train_37voc_10_unlabeled.json"),

    "totaltext_train_full_label": ("totaltext/train_images", "totaltext/train_37voc_full_labeled.json"),
    "totaltext_train_full_unlabel": ("totaltext/train_images", "totaltext/train_37voc_full_unlabeled.json"),

    "textocr1_unlabel": ("textocr/train_images", "textocr/train_37voc_1_unlabeled.json"),
    "textocr2_unlabel": ("textocr/train_images", "textocr/train_37voc_2_unlabeled.json"),
    "textocr_val": ("textocr/train_images", "textocr/train_37voc_2.json"),
    # "textocr_test": ("textocr/test_images", "textocr/train_37voc_2.json"),

    "textocr_train_100_label": ("textocr/train_images", "textocr/train_37voc_1_100_labeled.json"),
    "textocr_train_1000_unlabel": ("textocr/train_images", "textocr/train_37voc_1_1000_unlabeled.json"),
    "textocr_train_2000_unlabel": ("textocr/train_images", "textocr/train_37voc_1_2000_unlabeled.json"),
    "textocr_train_3000_unlabel": ("textocr/train_images", "textocr/train_37voc_1_3000_unlabeled.json"),
    "textocr_train_4000_unlabel": ("textocr/train_images", "textocr/train_37voc_1_4000_unlabeled.json"),
    "textocr_train_5000_unlabel": ("textocr/train_images", "textocr/train_37voc_1_5000_unlabeled.json"),
    "textocr_train_10000_unlabel": ("textocr/train_images", "textocr/train_37voc_1_10000_unlabeled.json"),
    "textocr_train_15000_unlabel": ("textocr/train_images", "textocr/train_37voc_1_15000_unlabeled.json"),
    "textocr_train_20000_unlabel": ("textocr/train_images", "textocr/train_37voc_1_20000_unlabeled.json"),
    "textocr_train_5000_sim_unlabel": ("textocr/train_images", "COCO-Text/ocr_train_37voc_1_sim_5k_unlabeled.json"),


    "ic15_train_0.5_label": ("ic15/train_images", "ic15/train_37voc_0.5_labeled.json"),
    "ic15_train_0.5_unlabel": ("ic15/train_images", "ic15/train_37voc_0.5_unlabeled.json"),
    "ic15_train_1_label": ("ic15/train_images", "ic15/train_37voc_1_labeled.json"),
    "ic15_train_1_unlabel": ("ic15/train_images", "ic15/train_37voc_1_unlabeled.json"),
    "ic15_train_2_label": ("ic15/train_images", "ic15/train_37voc_2_labeled.json"),
    "ic15_train_2_unlabel": ("ic15/train_images", "ic15/train_37voc_2_unlabeled.json"),
    "ic15_train_5_label": ("ic15/train_images", "ic15/train_37voc_5_labeled.json"),
    "ic15_train_5_unlabel": ("ic15/train_images", "ic15/train_37voc_5_unlabeled.json"),
    "ic15_train_10_label": ("ic15/train_images", "ic15/train_37voc_10_labeled.json"),
    "ic15_train_10_unlabel": ("ic15/train_images", "ic15/train_37voc_10_unlabeled.json"),

    "ic15_train_full_label": ("ic15/train_images", "ic15/train_37voc_full_labeled.json"),
    "ic15_train_full_unlabel": ("ic15/train_images", "ic15/train_37voc_full_unlabeled.json"),

    "ctw1500_train_0.5_label": ("ctw1500/train_images", "ctw1500/train_96voc_0.5_labeled.json"),
    "ctw1500_train_0.5_unlabel": ("ctw1500/train_images", "ctw1500/train_96voc_0.5_unlabeled.json"),
    "ctw1500_train_1_label": ("ctw1500/train_images", "ctw1500/train_96voc_1_labeled.json"),
    "ctw1500_train_1_unlabel": ("ctw1500/train_images", "ctw1500/train_96voc_1_unlabeled.json"),
    "ctw1500_train_2_label": ("ctw1500/train_images", "ctw1500/train_96voc_2_labeled.json"),
    "ctw1500_train_2_unlabel": ("ctw1500/train_images", "ctw1500/train_96voc_2_unlabeled.json"),
    "ctw1500_train_5_label": ("ctw1500/train_images", "ctw1500/train_96voc_5_labeled.json"),
    "ctw1500_train_5_unlabel": ("ctw1500/train_images", "ctw1500/train_96voc_5_unlabeled.json"),
    "ctw1500_train_10_label": ("ctw1500/train_images", "ctw1500/train_96voc_10_labeled.json"),
    "ctw1500_train_10_unlabel": ("ctw1500/train_images", "ctw1500/train_96voc_10_unlabeled.json"),


    "ic13_train": ("ic13/train_images", "ic13/train_37voc.json"),
    "ic15_train": ("ic15/train_images", "ic15/train_37voc.json"),
    "textocr1": ("textocr/train_images", "textocr/train_37voc_1.json"),
    "textocr2": ("textocr/train_images", "textocr/train_37voc_2.json"),


    # 96 voc_size
    "syntext1_96voc": ("syntext1/train_images", "syntext1/annotations/train_96voc.json"),
    "syntext2_96voc": ("syntext2/train_images", "syntext2/annotations/train_96voc.json"),
    "mlt_96voc": ("mlt2017/train_images", "mlt2017/train_96voc.json"),
    "totaltext_train_96voc": ("totaltext/train_images", "totaltext/train_96voc.json"),
    "ic13_train_96voc": ("ic13/train_images", "ic13/train_96voc.json"),
    "ic15_train_96voc": ("ic15/train_images", "ic15/train_96voc.json"),
    "ctw1500_train_96voc": ("ctw1500/train_images", "ctw1500/train_96voc.json"),

    # Chinese
    "chnsyn_train": ("chnsyntext/syn_130k_images", "chnsyntext/chn_syntext.json"),
    "rects_train": ("ReCTS/ReCTS_train_images", "ReCTS/rects_train.json"),
    "rects_val": ("ReCTS/ReCTS_val_images", "ReCTS/rects_val.json"),
    "lsvt_train": ("LSVT/rename_lsvtimg_train", "LSVT/lsvt_train.json"),
    "art_train": ("ArT/rename_artimg_train", "ArT/art_train.json"),

    # evaluation, just for reading images, annotations may be empty
    "totaltext_test": ("totaltext/test_images", "totaltext/test.json"),
    "ic15_test": ("ic15/test_images", "ic15/test.json"),
    "ctw1500_test": ("ctw1500/test_images", "ctw1500/test.json"),
    # "inversetext_test": ("inversetext/test_images", "inversetext/test.json"),
    # "rects_test": ("ReCTS/ReCTS_test_images", "ReCTS/rects_test.json"),
    # "textocr_test": ("textocr/test_images", "textocr/test.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets", voc_size_cfg=37, num_pts_cfg=25):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            voc_size_cfg,
            num_pts_cfg
        )

def register_all_coco_semi(root="datasets", voc_size_cfg=37, num_pts_cfg=25):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances_ssl(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            voc_size_cfg,
            num_pts_cfg
        )


# get the vocabulary size and number of point queries in each instance
# to eliminate blank text and sample gt according to Bezier control points
parser = default_argument_parser()
# add the following argument to avoid some errors while running demo/demo.py
parser.add_argument("--input", nargs="+", help="A list of space separated input images")
parser.add_argument(
    "--output",
    help="A file or directory to save output visualizations. "
    "If not given, will show output in an OpenCV window.",
)
parser.add_argument(
    "--opts",
    help="Modify config options using the command-line 'KEY VALUE' pairs",
    default=[],
    nargs=argparse.REMAINDER,
    )
parser.add_argument("--refer", action="store_true", help="whether use the anno in builtin dataset to better visualize")
parser.add_argument("--TSA", action="store_true", help="whether use TSA PL strategy")
args = parser.parse_args()
# cfg = get_cfg()
cfg = get_cfg_semi()
cfg.merge_from_file(args.config_file)
# register_all_coco(voc_size_cfg=cfg.MODEL.TRANSFORMER.VOC_SIZE, num_pts_cfg=cfg.MODEL.TRANSFORMER.NUM_POINTS)

register_all_coco_semi(voc_size_cfg=cfg.MODEL.TRANSFORMER.VOC_SIZE, num_pts_cfg=cfg.MODEL.TRANSFORMER.NUM_POINTS)