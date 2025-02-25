# [NoisyNN: Exploring the Impact of Information Entropy Change in Learning Systems](https://arxiv.org/pdf/2309.10625)

### Project in Progress
-The learning theory proposed in this work primarily enhances model performance in single-modality classification tasks, including image classification, domain adaptation/generalization, semi-supervised classification, and text classification.

-Applications of NoisyNN in semi-supervised learning [InterLUDE](https://arxiv.org/pdf/2403.10658) and domain adaptation [FFTAT](https://arxiv.org/pdf/2411.07794v1) have been accepted at ICML 2024 and WACV 2025. 

-NoisyNN shows significant potential for other learning tasks, which I will explore further.

<p align="left"> 
<img width="800" src="https://github.com/Shawey94/NoisyNN/blob/main/NoisyNNMethod.png">
</p>

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages (see requirements.txt)
pip install -r requirements.txt
```

### Pretrained ViT
NoisyViT-B_16-384 (pre-trained on ImageNet-21K) achieved a top 1 accuracy of over 95% and a top 5 accuracy of 99.9% on ImageNet-1K:
<p align="left"> 
<img width="500" src="https://github.com/Shawey94/NoisyNN/blob/main/ResImageNet.png">
</p>

### Datasets:

- Download the [ImageNet-1K(ILSVRC2012)](https://www.image-net.org/download.php) dataset.

- The ImageNet-1K folder has a structure like this:

```
ImageNet1K/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_18.JPEG
│   │   ├── n01440764_36.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
│   └── n01484850/
├── val/
│   ├── n01440764/
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
│   └── n01484850/
```
- Use 'unzip_tra.sh' and 'preprocess.py' for data preprocessing:
``` bash
sh unzip_tra.sh
python preprocess.py
```
### Training:

Commands can be found in `runScript.txt`. An example:
```
python Main.py --lr 0.000001 --epochs 50 --batch_size 16 --layer 11 --gpu_id 0 --res 384 --patch_size 16 --scale base --noise_type linear --datasets ImageNet --num_classes 1000 --tra 0 --inf 1 --OptimalQ 1
```

### Citation:
```
@article{Yu2023NoisyNN,
  title={NoisyNN: Exploring the Impact of Information Entropy Change in Learning Systems},
  author={Yu, Xiaowei and Huang, Zhe and Xue, Yao and Zhang, Lu and Wang, Li and Liu, Tianming and Dajiang Zhu},
  journal={arXiv preprint arXiv:2309.10625},
  year={2023}
}

@article{Huang2024InterLUDE,
  title={InterLUDE: Interactions between Labeled and Unlabeled Data to Enhance Semi-Supervised Learning},
  author={Huang, Zhe and Yu, Xiaowei and Zhu, Dajiang and Michael C. Hughes},
  journal={International Conference on Machine Learning},
  year={2024}
}

@article{Yu2025FFTAT,
  title={Feature Fusion Transferability Aware Transformer for Unsupervised Domain Adaptation},
  author={Yu, Xiaowei and Huang, Zhe and Zao Zhang},
  journal={IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2025}
}
```
Our code is largely borrowed from [Timm](https://github.com/huggingface/pytorch-image-models/tree/main/timm)

Github: [InterLUDE: Interactions between Labeled and Unlabeled Data to Enhance Semi-Supervised Learning](https://github.com/tufts-ml/InterLUDE)

Github: [Feature Fusion Transferability Aware Transformer for Unsupervised Domain Adaptation](https://github.com/Shawey94/WACV2025-FFTAT)
