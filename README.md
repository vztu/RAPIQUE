[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rapique-rapid-and-accurate-video-quality/video-quality-assessment-on-konvid-1k)](https://paperswithcode.com/sota/video-quality-assessment-on-konvid-1k?p=rapique-rapid-and-accurate-video-quality)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rapique-rapid-and-accurate-video-quality/video-quality-assessment-on-youtube-ugc)](https://paperswithcode.com/sota/video-quality-assessment-on-youtube-ugc?p=rapique-rapid-and-accurate-video-quality)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rapique-rapid-and-accurate-video-quality/video-quality-assessment-on-live-vqc)](https://paperswithcode.com/sota/video-quality-assessment-on-live-vqc?p=rapique-rapid-and-accurate-video-quality)

# RAPIQUE
An official implementation of Rapid and Accurate Video Quality Evaluator (RAPIQUE) proposed in [IEEE OJSP2021] RAPIQUE: Rapid and Accurate Video Quality Prediction of User Generated Content. [Arxiv](https://arxiv.org/abs/2101.10955). [IEEExplore](https://ieeexplore.ieee.org/document/9463703)(Open Access!) and [PCS2021] Efficient User-Generated Video Quality Prediction. [IEEExplore](https://ieeexplore.ieee.org/abstract/document/9477483). Note that the temporal features can be used as standalone features in company with spatial models to boost performance on motion-intensive models. Check out the temporal-only modules in [ICIP21] A Temporal Statistics Model For UGC Video Quality Prediction. [IEEExplore](https://ieeexplore.ieee.org/abstract/document/9506669)

Check out our BVQA resource list and performance benchmark/leaderboard results in https://github.com/vztu/BVQA_Benchmark.

For more evaluation codes, please check out [VIDEVAL](https://github.com/vztu/VIDEVAL)

## Requirements

- MATLAB >= 2019
  - Deep learning toolbox (ResNet-50)
- Python3
- Sklearn
- FFmpeg
- Git LFS

## Performances

### SRCC / PLCC

|    Methods   | KoNViD-1k | LIVE-VQC             | YouTube-UGC         | All-Combined |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| TLVQM        | 0.7101 / 0.7037 | 0.7988 / 0.8025  | 0.6693 / 0.6590 | 0.7271 / 0.7342  |
| VIDEVAL      | 0.7832 / 0.7803 | 0.7522 / 0.7514  | 0.7787 / 0.7733 | 0.7960 / 0.7939  |
| MDVSFA       | 0.7812 / 0.7856 | 0.7382 / 0.7728  |  - / - | - / - |
| RAPIQUE      | 0.8031 / 0.8175 | 0.7548 / 0.7863  | 0.7591 / 0.7684 | 0.8070 / 0.8229 |

Scatter plots and fitted logistic curves on these datasets:


| KonVid-1k             |  LIVE-VQC  | YouTube-UGC | All-Combined |
|:-------------------------:|:-------------------------:|:-------:|:-------:|
|![](https://github.com/vztu/RAPIQUE/blob/main/figures/KONVID_1K_kfCV_corr.png)  |  ![](https://github.com/vztu/RAPIQUE/blob/main/figures/LIVE_VQC_kfCV_corr.png) | ![](https://github.com/vztu/RAPIQUE/blob/main/figures/YOUTUBE_UGC_kfCV_corr.png) | ![](https://github.com/vztu/RAPIQUE/blob/main/figures/ALL_COMBINED_kfCV_corr.png)


### Speed

The unit is average `secs/video`. 

|    Methods   |  540p | 720p | 1080p | 4k@60  |
|:-----------:|:----:|:----:|:------:|:--------:|
| Video-BLIINDS | 341.1 | 839.1 | 1989.9 | 16129.2 |
| VIDEVAL      |   61.9   |  146.5   |  354.5   | 1716.3  |
| TLVQM        | 34.5  | 78.9 | 183.8 | 969.3 |
| RAPIQUE      | 13.5 | 17.3 | 18.3 | 112 |

<p float="left">
  <img src="https://github.com/vztu/RAPIQUE/blob/main/figures/speed_scales.jpg" width="400" />
</p>

### Performance vs. Speed

<p float="left">
  <img src="https://github.com/vztu/RAPIQUE/blob/main/figures/perf_n_speed.jpg" width="400" />
</p>

## Demos

#### Feature Extraction Only

```
demo_compute_RAPIQUE_feats.m
```
You need to specify the parameters

#### Evaluation of BVQA Model

We proposed several evaluation methods for BIQA/BVQA models. Please check out [ICASSP21] Regression or classification? New methods to evaluate no-reference picture and video quality models [IEEExplore](https://ieeexplore.ieee.org/abstract/document/9414232/) for details.

* For regression evaluation:
```
$ python evaluate_bvqa_features_regression.py
```

* For binary classification evaluation:
```
$ python evaluate_bvqa_features_binary_classification.py
```

* For ordinal classification evaluation:
```
$ python evaluate_bvqa_features_ordinal_classification.py
```

## Citation

If you use this code for your research, please cite our papers.

```
@article{tu2021rapique,
  title={RAPIQUE: Rapid and accurate video quality prediction of user generated content},
  author={Tu, Zhengzhong and Yu, Xiangxu and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  journal={IEEE Open Journal of Signal Processing},
  volume={2},
  pages={425--440},
  year={2021},
  publisher={IEEE}
}
@article{tu2021ugc,
  title={UGC-VQA: Benchmarking blind video quality assessment for user generated content},
  author={Tu, Zhengzhong and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
@inproceedings{tu2021efficient,
  title={Efficient User-Generated Video Quality Prediction},
  author={Tu, Zhengzhong and Chen, Chia-Ju and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  booktitle={2021 Picture Coding Symposium (PCS)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
@inproceedings{tu2021temporal,
  title={A Temporal Statistics Model For UGC Video Quality Prediction},
  author={Tu, Zhengzhong and Chen, Chia-Ju and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={1454--1458},
  year={2021},
  organization={IEEE}
}
@inproceedings{tu2021regression,
  title={Regression or classification? New methods to evaluate no-reference picture and video quality models},
  author={Tu, Zhengzhong and Chen, Chia-Ju and Chen, Li-Heng and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2085--2089},
  year={2021},
  organization={IEEE}
}

```

