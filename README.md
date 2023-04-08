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

### The current code takes longer to train than this paper, but has been updated to a much higher performance version.


![gt](https://user-images.githubusercontent.com/33244972/228428350-33d705c7-61ee-4f08-b5f3-8b42c57a6ffb.gif)


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

TODO

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
