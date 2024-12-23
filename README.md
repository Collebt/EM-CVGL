# Learning Cross-view Visual Geo-localization without Ground Truth


## Project Page:
See this [page](https://collebt.github.io/EM-CVGL).

## Overview
A self-supervised learning framework to train a learnable adapter for a frozen Foundation Model (FM) on the task of drone-to-satellite cross-view geo-localization.


## Installation

```
pip install -r requirements.txt
```

## Dataset & Preparation

- Dataset: Download the University-1652 dataset from [here] (https://github.com/layumi/University1652-Baseline) and put them on `~/data/`
- Extract Initial Features: 
```
python test.py configs/base_anyloc_D2S.yml --save_dir data/University-Release/train/anyloc_gap
```



## Training and Evaluation

Configurate the `data_path` in file `configs/base_autoen_lr00001_randomD_share_recon_ssl_701.yml`

```
python train.py configs/base_autoen_lr00001_randomD_share_recon_ssl_701.yml
```
```
python test.py configs/base_autoen_lr00001_randomD_share_recon_ssl_701.yml --param_path path/to/outputs/model
```



## BibTeX
```
@article{li2024learning,
  title={Learning Cross-view Visual Geo-localization without Ground Truth},
  author={Li, Haoyuan and Xu, Chang and Yang, Wen and Yu, Huai and Xia, Gui-Song},
  journal={arXiv preprint arXiv:2403.12702},
  year={2024}
}
```

