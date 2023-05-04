# Implicit Stacked Autoregressive Model for Video Prediction
### [Project Page](-) | [Video](-) | [Paper](https://arxiv.org/pdf/2303.07849.pdf) | 
 [Minseok Seo](https://sites.google.com/view/minseokcv/%ED%99%88),
 [Hakjin Lee](https://github.com/nijkah),
 [Doyi Kim](-),
 [Junghoon Seo](https://mikigom.github.io),
<br>
 SI Analytics  

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/implicit-stacked-autoregressive-model-for-1/video-prediction-on-human36m)](https://paperswithcode.com/sota/video-prediction-on-human36m?p=implicit-stacked-autoregressive-model-for-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/implicit-stacked-autoregressive-model-for-1/video-prediction-on-moving-mnist)](https://paperswithcode.com/sota/video-prediction-on-moving-mnist?p=implicit-stacked-autoregressive-model-for-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/implicit-stacked-autoregressive-model-for-1/weather-forecasting-on-sevir)](https://paperswithcode.com/sota/weather-forecasting-on-sevir?p=implicit-stacked-autoregressive-model-for-1)


---
<p align="center">
<img width="704" alt="image" src="https://user-images.githubusercontent.com/33244972/225778106-3c5c6a62-8a8c-46dc-93ea-92a82fd8ab95.png">
</p>

## NEWS

### Our typhoon data set and experimental results will be published soon. [2023.06.01] 
(This is not included in our report. prediction +10 hours)

![pred](https://user-images.githubusercontent.com/33244972/231080870-760ca587-7a48-4c6e-9587-d91c3c6d7cfb.gif)

## Requirements
```
scikit-image==0.16.1
```
## Moving MNIST

```
sh data/moving_mnist/download_mmnist.sh
python3 main.py
```

## Pre-trained weight
(We are in the process of reproducing the performance by running the refactored code and updating ckpt.)

| Dataset  | Epoch | MSE | Weight | 
| ------------- | ------------- | ------------- | ------------- |
| Moving MNIST  | 100  | 31.04  | [ckpt](https://drive.google.com/file/d/1mqIHwh-DLhvGfRWfaBoj5obtPCO4mNaL/view?usp=share_link)  |
| Moving MNIST  | 200  | 27.04  | [ckpt](https://drive.google.com/file/d/1ERemZ49GD5nuFs_epBnHfuFumzlEsBHu/view?usp=share_link)  |
| Moving MNIST  | 300  | 24.92  | [ckpt](https://drive.google.com/file/d/15m-T-dzmqnvy7vreQFtl47c5JDtiunJI/view?usp=share_link)  |
| Moving MNIST  | 400  | 23.56  | [ckpt](https://drive.google.com/file/d/1soT_a1Ycq8BZz_hYtXT9_Old6BeHL4ie/view?usp=share_link)  |
| Moving MNIST  | 500  | 22.57  | [ckpt](https://drive.google.com/file/d/1jh_XCu4ofkvS7V525WNtEzTXQECdWG7n/view?usp=share_link)  |
| Moving MNIST  | 600  | 21.67  | [ckpt](https://drive.google.com/file/d/1pflos2XC9AU2i3gSMXxsrnXCFTvHNXBV/view?usp=share_link)  |
| Moving MNIST  | 800  | 20.50  | [ckpt](-)  |
| Moving MNIST  | 1,000  | 19.53  | [ckpt](-)  |
| Moving MNIST  | 1,300  | 18.57  | [ckpt](-)  |
| Moving MNIST  | 1,500  | 18.03  | [ckpt](-)  |
| Moving MNIST  | 2,000  | 17.20 (paper:16.9)  | [ckpt](-)  |

## Web Demo
IAM4VP web deomo available at https://ovision.ai/ [it will plan 2023.08.01]

## Thank to

This code is heavily based on [SimVP](https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction).

We thank the authors of that code.

## Citation

```
@article{seo2023implicit,
  title={Implicit Stacked Autoregressive Model for Video Prediction},
  author={Seo, Minseok and Lee, Hakjin and Kim, Doyi and Seo, Junghoon},
  journal={arXiv preprint arXiv:2303.07849},
  year={2023}
}
```
