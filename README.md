# TBFormer

**This is the official repo for paper: *TBFormer: Two-Branch Transformer for Image Forgery Localization***

## Overview
Image forgery localization aims to identify forged regions by capturing subtle traces from high-quality discriminative features. 
In this paper, we propose a Transformer-style network with two feature extraction branches for image forgery localization, and it is named as Two-Branch Transformer (TBFormer). 
Firstly, two feature extraction branches are elaborately designed, taking advantage of the discriminative stacked Transformer layers, for both RGB and noise domain features. 
Secondly, an Attention-aware Hierarchical-feature Fusion Module (AHFM) is proposed to effectively fuse hierarchical features from two different domains. 
Although the two feature extraction branches have the same architecture, their features have significant differences since they are extracted from different domains. 
We adopt position attention to embed them into a unified feature domain for hierarchical feature investigation. 
Finally, a Transformer decoder is constructed for feature reconstruction to generate the predicted mask. 
Extensive experiments on publicly available datasets demonstrate the effectiveness of the proposed model.

## Installation 
Our project is developed based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation). 
1. Please install MMSegmentation follow the official tutorial.
2. Move the files provided here to the folder corresponding to MMSegmentation.

## Pre-trained models
Please download our pre-trained modle here and place it under */checkpoint* directory.
- [TBFormer](https://pan.baidu.com/s/1d4gFyF7d7vMuL1yBvF5XZQ)(Extraction code: tbfm)

## Demo
- You can run a simple demo:
```bash
python demo/demo.py
```

## Synthesized dataset
Our synthesized dataset can be downloaded here. 
- [Mydata](https://pan.baidu.com/s/1CI_WL-BLbpdwS1sEp8FLbw)(Extraction code: lbmd; Decompression password: mdtvtlb22)

## Contact
If you enounter any questions, please contact `lv-bin-bin@outlook.com`


