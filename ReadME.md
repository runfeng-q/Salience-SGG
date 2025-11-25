# Salience-SGG: Enhancing Unbiased Scene Graph Generation with Iterative Salience Estimation (WACV 2026)

## Environment Installation Step by Step
- conda create -n salience_sgg python==3.10
- conda install nvidia/label/cuda-11.8.0::cuda-nvcc
- conda install conda-forge::cuda-version==11.8
- conda install nvidia/label/cuda-11.8.0::cuda-toolkit
- pip install cython==3.0.12
- pip install pycocotools==2.0.8
- pip install tqdm==4.67.1
- pip install scipy==1.15.2
- cd models/ops/
- sh ./make.sh
- cd ../../
- conda install yaml=0.2.5=h7b6447c_0
- conda install pyyaml=6.0.2=py310h5eee18b_0
- pip install opencv-python==4.11.0.86
- pip install opencv-python-headless==4.11.0.86
- pip install imageio==2.35.1
- pip install lightning==2.4.0
- cd lib/fpn
- sh make.sh
- cd ../../
- install seaborn==0.13.2
- pip install tensorboard==2.17.1

## Datasets
To download the datasets please refer to below links, Then change data/root under the corresponding dataset configuration file to your own data path.
- VG: https://github.com/yrcong/RelTR/blob/main/data/README.md
- OIv6: https://github.com/Scarecrow0/SGTR/blob/main/DATASET.md
- GQA-200: https://github.com/dongxingning/SHA-GCL-for-SGG/blob/master/DATASET.md

## Pre-trained models:
We provide pre-trained object detectors and Salience-SGGs on VG, OIv6 and GQA-200 datasets, you can download them [here](https://drive.google.com/drive/folders/1Ge2IjWldzDW3NSv3UFLkmlmzsOkxgHIi?usp=sharing).

## Training
You need to train an object detector for each data first with the following command.

```python3 run.py --config=configs/DABDeformableDETR_{vg/OI/GQA}.yaml --command=fit```

After this, put the path of your pre-trained detector to the model/resume/ckpt in the corresponding dataset configuration file. And run:

```python3 run.py --config=configs/SalienceSGG_{vg/OI/GQA}.yaml --command=fit```

## Evaluation
Put the path of your Salience-SGG model to the model/resume/ckpt in the corresponding dataset configuration file. And run:

```python3 run.py --config=configs/SalienceSGG_{vg/OI/GQA}.yaml --command=eval```

If you run the evaluation with the models provided above, the following results should be obtained.
```
VG:
R@20:  0.218538   mR@20:  0.128150   f_R@20: 0.1615616594558
R@50:  0.287625   mR@50:  0.179607   f_R@50: 0.2211298172001
R@100: 0.334015  mR@100:  0.215732   f_R@100: 0.2621486755907
GQA-200:
R@20:  0.189095   mR@20:  0.130885   f_R@20: 0.1546958584396
R@50:  0.236024   mR@50:  0.161901   f_R@50: 0.1920593743214
R@100: 0.265685  mR@100:  0.183736   f_R@100: 0.2172391765454
OIv6:
microR@50: 0.7807422969187674 
w_rel_mAP: 0.45733594448023845 
w_phr_mAP: 0.45109483948958284
score: 0.519520772971682
```
## References
- https://github.com/naver-ai/egtr
