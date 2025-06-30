# SemiETS

👋 Welcome to the official code of [SemiETS: Integrating Spatial and Content Consistencies for Semi-Supervised End-to-end Text Spotting](https://arxiv.org/abs/2504.09966) (CVPR 2025)

This work explored semi-supervised text spotting (SSTS) to reduce the expensive annotation costs for text spotting. We observe two challenges in SSTS: 1) inconsistent pseudo labels between detection and recognition tasks, and 2) sub-optimal supervisions caused by inconsistency between teacher/student. Addressing them, we proposed SemiETS. It gradually generates reliable hierarchical pseudo labels for each task, thereby reducing noisy labels. Meanwhile, it extracts important information in text locations and transcriptions from bidirectional flows to improve consistency.

<div align="center">
  <img src="figs/framework.jpg" width=85%/>
</div>


## 📖 Usage

### 🛠️ Dependencies and Installation

* **Environment**

```
Python 3.8 + Pytorch 1.9.0 + CUDA 11.1 + Detectron2 (v0.6) + ctcdecode
```

1. **Install SemiETS**

```
# 1. Clone depository
git clone git@github.com:DrLuo/SemiETS.git
cd SemiETS

# 2. Create conda environment
conda create -n semiets python=3.8 -y
conda activate semiets

# 3. Install PyTorch and other dependencies using conda
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
python setup.py build develop
```

2. **Install ctcdecode** from [source](https://github.com/parlance/ctcdecode)

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```


### 🧱 Preparation

1. **Download datasets** from [here](https://github.com/ViTAE-Transformer/DeepSolo/blob/main/DeepSolo/README.md#preparation). Data splits are in ```SemiETS/datasets```.

<details>
<summary>Dataset Orgnization</summary>

*Some image files need to be renamed.* Organize them as follows (lexicon files are not listed here):

```
|- ./datasets
   |- syntext1
   |  |- train_images
   |  └  annotations
   |       |- train_37voc.json
   |       └  train_96voc.json   
   |- syntext2
   |  |- train_images
   |  └  annotations
   |       |- train_37voc.json
   |       └  train_96voc.json
   |- totaltext
   |  |- train_images
   |  |- test_images
   |  |- train_37voc.json
   |  |- train_96voc.json
   |  |- train_37voc_0.5_labeled.json
   |  |- train_37voc_0.5_unlabeled.json
   |  |- train_37voc_1_labeled.json
   |  |- train_37voc_1_unlabeled.json
   |  |- train_37voc_2_labeled.json
   |  |- train_37voc_2_unlabeled.json
   |  |- train_37voc_5_labeled.json
   |  |- train_37voc_5_unlabeled.json
   |  |- train_37voc_10_labeled.json
   |  |- train_37voc_10_unlabeled.json  
   |  └  test.json
   |- ic15
   |  |- train_images
   |  |- test_images
   |  |- train_37voc.json
   |  |- train_96voc.json
   |  |- train_37voc_0.5_labeled.json
   |  |- train_37voc_0.5_unlabeled.json
   |  |- train_37voc_1_labeled.json
   |  |- train_37voc_1_unlabeled.json
   |  |- train_37voc_2_labeled.json
   |  |- train_37voc_2_unlabeled.json
   |  |- train_37voc_5_labeled.json
   |  |- train_37voc_5_unlabeled.json
   |  |- train_37voc_10_labeled.json
   |  |- train_37voc_10_unlabeled.json  
   |  └  test.json
   |- ctw1500
   |  |- train_images
   |  |- test_images
   |  |- train_96voc.json
   |  |- train_96voc_0.5_labeled.json
   |  |- train_96voc_0.5_unlabeled.json
   |  |- train_96voc_1_labeled.json
   |  |- train_96voc_1_unlabeled.json
   |  |- train_96voc_2_labeled.json
   |  |- train_96voc_2_unlabeled.json
   |  |- train_96voc_5_labeled.json
   |  |- train_96voc_5_unlabeled.json
   |  |- train_96voc_10_labeled.json
   |  |- train_96voc_10_unlabeled.json  
   |  └  test.json
   |- evaluation
   |  |- gt_*.zip
```
</details>



2. **Download pretrained weights** to for initialization from [Google Drive](https://drive.google.com/drive/folders/1ix416PtjenJxvDm_2KlS6z1vo6z5gI1K?usp=drive_link)

The checkpoints were pretrained using only Synth150K.
Place them under the folder ```./output/R50/150k_tt/pretrain/```.




### 🚀 Training

```
python tools/train_semi.py --config-file ${CONFIG_FILE} --num-gpus 4  --dist-url 'auto'
```

For example:
```
python tools/train_semi.py --config-file configs/R_50/TotalText/SemiETS/SemiETS_2s.yaml --num-gpus 4  --dist-url 'auto'
```

The configuration files are named following the format: ```SemiETS_{DATA_PROPORTION}s.yaml```



### 📈 Evaluation

```
python tools/train_semi.py --config-file ${CONFIG_FILE} --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```


## 🔗 Citation
If you find [SemiETS](https://arxiv.org/abs/2504.09966) useful for your research and applications, please cite using this BibTeX:

```
@article{luo2025semiets,
  title={SemiETS: Integrating Spatial and Content Consistencies for Semi-Supervised End-to-end Text Spotting},
  author={Luo, Dongliang and Zhu, Hanshen and Zhang, Ziyang and Liang, Dingkang and Xie, Xudong and Liu, Yuliang and Bai, Xiang},
  journal={CVPR},
  year={2025}
}
```

## Acknowledgement
This project is based on [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo) and [Adelaidet](https://github.com/aim-uofa/AdelaiDet). We appreciate their wonderful codebase. For academic use, this project is licensed under the 2-clause BSD License.

