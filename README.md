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

## TODO

Need to add readme file.
Need to add fine-tuning LP(Learnd Prior).

## Moving MNIST

```
sh data/moving_mnist/download_mmnist.sh
python3 main.py
```


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
