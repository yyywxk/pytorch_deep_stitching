<h2 align="center">pytorch_deep_stitching</h2>
<p align="center">
    <!-- <a href="https://github.com/yyywxk/pytorch_deep_stitching/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
    <a href="https://github.com/yyywxk/pytorch_deep_stitching/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/yyywxk/pytorch_deep_stitching">
    </a>
    <a href="https://github.com/yyywxk/pytorch_deep_stitching/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/yyywxk/pytorch_deep_stitching">
    </a>
    <a href="https://github.com/yyywxk/pytorch_deep_stitching/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/yyywxk/pytorch_deep_stitching?color=pink">
    </a>
    <a href="https://github.com/yyywxk/pytorch_deep_stitching">
        <img alt="issues" src="https://img.shields.io/github/stars/yyywxk/pytorch_deep_stitching">
    </a>
    <a href="mailto: qiulinwei@buaa.edu.cn">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>


## Introduction

This is the PyTorch reimplementation of following papers:

- UDIS: [Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images](http://arxiv.org/abs/2106.12859) [TIP 2021]

<details>
<summary>Fig</summary>
<img src=./assets/UDIS.png border=0 width=500>
</details>

- UDIS2: [Parallax-Tolerant Unsupervised Deep Image Stitching](https://arxiv.org/abs/2302.08207) [ICCV 2023]

<details>
<summary>Fig</summary>
<img src=./assets/UDIS++.png border=0 width=500>
</details>

## Requirements

- Packages
  
  The code was tested with Anaconda and Python 3.10.13. The Anaconda environment is:
  
  - pytorch = 2.1.1
  - torchvision = 0.16.1
  - cudatoolkit = 11.8
  - tensorboard = 2.17.0
  - tensorboardX = 2.6.2.2
  - opencv-python = 4.9.0.80
  - numpy = 1.26.4
  - pillow = 10.3.0

Install dependencies:

- For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.
- For custom dependencies:
  
  ```bash
  conda install tensorboard tensorboardx
  pip install tqdm opencv-python thop scikit-image lpips scipy
  ```
- We implement this work with Ubuntu 18.04, NVIDIA Tesla V100, and CUDA11.8.

## Datasets

- Put data in `../dataset` folder or  configure your dataset path in the `my_path` function of  `dataloaders/__inint__.py`.
- These codes support the **UDIS-D** ([Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images](http://arxiv.org/abs/2106.12859) [TIP 2021]). You can download it at [Google Drive](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/13KZ29e487datgtMgmb9laQ) (Extraction code: 1234).
- For UDIS, it is needed to download the [**WarpedCOCO**](https://pan.baidu.com/s/1MVn1VFs_6-9dNRVnG684og) (Baidu Cloud Extraction Code: 1234) dataset for pre-training.
- The details of the dataset **UDAIS-D** (Unsupervised Deep Stitching of Aerial Images Dataset) can be found in our paper ([Unsupervised Deep Image Stitching for Remote Sensing: Aligning Features Across Large-Scale Geometric Distortions](https://ieeexplore.ieee.org/document/11164964) [JSTARS 2025]). You can download it at [Baidu Cloud](https://pan.baidu.com/s/1U0Bw7DZGM5J8mAK8mwsxFw?pwd=1234) (Extraction code: 1234).
- The details of the dataset **UDAIS-D+** can be found in our paper ([Radiation-Tolerant Unsupervised Deep Image Stitching for Remote Sensing](https://ieeexplore.ieee.org/document/11146793) [TGRS 2025]). You can download it at [Baidu Cloud](https://pan.baidu.com/s/1cXSOosWZbIg9CSu_iRmSyw?pwd=1234) (Extraction code: 1234).

## Unsupervised Alignment or Warp

For input arguments: (see full input arguments via `python train_align.py --help` or `python test_align.py --help`):

### Model Training

Follow steps below to train your model (UDIS or UDIS2):

**Step 0 (Only for UDIS): Unsupervised pre-training on Stitched MS-COCO**

```bash
CUDA_VISIBLE_DEVICES=0 python train_align.py --dataset MS-COCO --model UDIS --epochs 150 -b 16 --loss-type UDIS --freq_record 600 --height 128 --width 128 --run_path ./run_align/ --lam_lp1 16.0 --lam_lp2 4.0 --lam_lp3 1.0
```

**Step 1: Unsupervised training on UDIS-D**

For UDIS, you can finetune your model with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python train_align.py --dataset UDIS-D --model UDIS --epochs 50 -b 16 --loss-type UDIS --freq_record 600 --height 128 --width 128 --run_path ./run_align/ --lam_lp1 16.0 --lam_lp2 4.0 --lam_lp3 1.0 --resume ./run_align/MS-COCO/UDIS/experiment_0/epoch150_model.pth --ft
```

For UDIS2, you can train your model with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python train_align.py --dataset UDIS-D --model UDIS2 --epochs 100 -b 4 --loss-type UDIS2 --freq_record 600 --height 512 --width 512 --run_path ./run_align/ --lam_lp1 3.0 --lam_lp2 1.0 --lam_grid 10
```

Note: You can change the dataset from **UDIS-D**  to **UDAIS-D** or **UDAIS-D+** by changing the `--dataset` argument.

### Model Testing

**Step 2: Evaluating the alignment results**

Using the trained model, you can evaluate the alignment results with the following command in the `test` mode:

```bash
% UDIS
CUDA_VISIBLE_DEVICES=0 python test_align.py --dataset UDIS-D --model UDIS --height 128 --width 128 --mode test --model_path ./run_align/UDIS-D/UDIS/experiment_0/epoch50_model.pth

% UDIS2
CUDA_VISIBLE_DEVICES=0 python test_align.py --dataset UDIS-D --model UDIS2 --height 512 --width 512 --mode test --model_path ./run_align/UDIS-D/UDIS2/experiment_0/epoch100_model.pth
```

**Step 3: Generate the warped images and corresponding masks for UDIS-D**

Using the trained model, you can generate the warped images and corresponding masks for training and testing sets with the following command in the `test_output` mode:

```bash
% UDIS
CUDA_VISIBLE_DEVICES=0 python test_align.py --dataset UDIS-D --model UDIS --height 128 --width 128 --mode test_output --model_path ./run_align/UDIS-D/UDIS/experiment_0/epoch50_model.pth --save_path ./Warp/UDIS-D/UDIS/align/training/ --get_warp_path training

CUDA_VISIBLE_DEVICES=0 python test_align.py --dataset UDIS-D --model UDIS --height 128 --width 128 --mode test_output --model_path ./run_align/UDIS-D/UDIS/experiment_0/epoch50_model.pth --save_path ./Warp/UDIS-D/UDIS/align/testing/ --get_warp_path testing

% UDIS2
CUDA_VISIBLE_DEVICES=0 python test_align.py --dataset UDIS-D --model UDIS2 --height 512 --width 512 --mode test_output --model_path ./run_align/UDAIS-D/UDIS2/experiment_0/epoch100_model.pth --save_path ./Warp/UDIS-D/UDIS2/align/training/ --get_warp_path training

CUDA_VISIBLE_DEVICES=0 python test_align.py --dataset UDIS-D --model UDIS2 --height 512 --width 512 --mode test_output --model_path ./run_align/UDIS-D/UDIS2/experiment_0/epoch100_model.pth --save_path ./Warp/UDIS-D/UDIS2/align/testing/ --get_warp_path testing
```

### Inference

**Step 4: Generate the warped images and corresponding masks for custom datasets**

Using the trained model, you can generate the warped images and corresponding masks for custom datasets with the following command in the `test_other` mode:

```bash
% UDIS
CUDA_VISIBLE_DEVICES=0 python test_align.py --dataset UDIS-D --model UDIS --height 128 --width 128 --mode test_other --model_path ./run_align/UDIS-D/UDIS/experiment_0/epoch50_model.pth --save_path {your_save_path} --input_path {your_custom_dataset_path}

% UDIS2
CUDA_VISIBLE_DEVICES=0 python test_align.py --dataset UDIS-D --model UDIS2 --height 512 --width 512 --mode test_other --model_path ./run_align/UDIS-D/UDIS2/experiment_0/epoch100_model.pth --save_path {your_save_path} --input_path {your_custom_dataset_path}
```

Note: 
1. You can finetune the UDIS2 model on the single image by adding `--ft` to the command. 
2. The `--save_path` is the path to save the warped images and corresponding masks. 
3. The `--input_path` is the path to the custom dataset.
4. Make sure to put your custom data files as the following structure:
   
   ```
   input_path
   ├── input1
   |   ├── 001.png
   │   ├── 002.png
   │   ├── 003.png
   │   ├── 004.png
   │   ├── ...
   |
   ├── input2
   |   ├── 001.png
   │   ├── 002.png
   │   ├── 003.png
   │   ├── 004.png
   |   ├── ...
   ```

## Unsupervised Reconstruction or Composition

For input arguments: (see full input arguments via `python train_compo.py --help` or `python test_compo.py --help`):

### Model Training

**Step 5: Unsupervised training on UDIS-D after unsupervised alignment or warp**

Using the aligned or warped images, you can train the model with the following command:

```bash
% UDIS
CUDA_VISIBLE_DEVICES=0 python train_compo.py --dataset UDIS-D --train_path ./Warp/UDIS-D/UDIS/align/training --model UDIS --epochs 30 -b 4 --loss-type UDIS --freq_record 300 --height 640 --width 640 --run_path ./run_com/ --resize

% UDIS2
CUDA_VISIBLE_DEVICES=0 python train_compo.py --dataset UDIS-D --train_path ./Warp/ UDIS-D/UDIS2/align/training --model UDIS2 --epochs 50 -b 16 --loss-type UDIS2 --freq_record 300 --height 512 --width 512 --run_path ./run_com/ --lam_bt 10000 --lam_st1 1000 --lam_st2 1000 --resize
```

Note: You can change the dataset from **UDIS-D**  to **UDAIS-D** or **UDAIS-D+** by changing the `--dataset` argument.

### Model Testing

**Step 6: Generating the finally stitched results for UDIS-D**

Using the trained model, you can generate the finally stitched results for training and testing sets with the following command in the `test_output` mode:

```bash
% UDIS
CUDA_VISIBLE_DEVICES=0 python test_compo.py --dataset UDIS-D --model UDIS --height 640 --width 640 --mode test_output --test_path ./Warp/UDIS-D/UDIS/align/testing --model_path ./run_com/UDIS-D/UDIS/experiment_0/epoch30_model.pth --save_path ./Final/UDIS-D/UDIS/

% UDIS2
CUDA_VISIBLE_DEVICES=0 python test_compo.py --dataset UDIS-D --model UDIS2 --height 512 --width 512 --mode test_output --test_path ./Warp/UDIS-D/UDIS2/align/testing --model_path ./run_com/UDIS-D/UDIS2/experiment_0/epoch50_model.pth --save_path ./Final/UDIS-D/UDIS2/
```

Note: 
1. The `--test_path` is the path to put the warped images and corresponding masks. 
2. The `--model_path` is the path to the trained model.
3. The `--save_path` is the path to save the finally stitched results.
4. You can change the dataset from **UDIS-D**  to **UDAIS-D** or **UDAIS-D+** by changing the `--dataset` argument.

## Citation

If our work is useful for your research, please consider citing and staring our work:

```tex
@article{qiu2025unsupervised,
  author={Qiu, Linwei and Liu, Chang and Li, Gongzhe and Dong, Xiaomeng and Xie, Fengying and Shi, Zhenwei},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Unsupervised Deep Image Stitching for Remote Sensing: Aligning Features Across Large-Scale Geometric Distortions}, 
  year={2025},
  pages={1-19},
  doi={10.1109/JSTARS.2025.3609808}
}

@article{qiu2025radiation,
  author={Qiu, Linwei and Xie, Fengying and Liu, Chang and Che, Xiaoling and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Radiation-Tolerant Unsupervised Deep Image Stitching for Remote Sensing}, 
  year={2025},
  volume={63},
  pages={1-21},
  doi={10.1109/TGRS.2025.3605270}
}

@misc{qiu2024pytorch_deep_stitching,
  author = {Qiu, Linwei},
  title = {pytorch_deep_stitching},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yyywxk/pytorch_deep_stitching}}
}
```

## Questions

Please contact [qiulinwei@buaa.edu.cn](mailto:qiulinwei@buaa.edu.cn).

## Acknowledgement

[UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching)

[UDIS2](https://github.com/nie-lang/UDIS2)

[UnsupDIS-pytorch](https://github.com/kail8/UnsupDIS-pytorch)


[UDRSIS](https://github.com/yyywxk/UDRSIS)

[RT-UDRSIS](https://github.com/yyywxk/RT-UDRSIS)

