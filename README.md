# glomerular_classification

## Ryohei Yamaguchi

'Glomerular Classification'
====
## [Glomerular classification using convolutional neural networks based on defined annotation criteria and concordance evaluation among clinicians](https://www.sciencedirect.com/science/article/pii/S2468024920317940?via%3Dihub#tbl6)
### !上のリンクは最終的に論文の正式URLに変更する(この文章はアップロード確定後に削除する)

## Overview
1. Train a CNN to classify a remark (e.g. fibrouscrescent) for a glomerular image.

2. Show the CNN's attention for a glomerular image by using Grad-CAM.

## Description
### Dataset
Our image datasets are **<u>NOT</u>** allowed to be public by the IRBs. So please prepare the images by yourself. 

* All images must be saved as (.npy), having four dimensions.(n of images,r,g,b)<br> 
* All scores(e.g positive or negative) should be saved as (.npy), having two dimensions.<br>
* All filename must be saved as (.npy)

### CNN experiment
You have to download the pretrained model of ResNet50 for chainer. Please download the pretrained model from [https://github.com/KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)
, and place it on /root/.chainer/dataset/pfnet/chainer/models/ResNet-50-model.caffemodel


### Grad-CAM
* Showing CNN's attention f


## Requirement

Python3
--------------- -------
chainer         4.0.0
<br>
cupy-cuda80     4.0.0
<br>
cycler          0.10.0
<br>
fastrlock       0.4
<br>
filelock        3.0.8
<br>
h5py            2.10.0
<br>
kiwisolver      1.1.0
<br>
matplotlib      3.0.3
<br>
nano            0.10.0
<br>
numpy           1.15.2
<br>
opencv-python   4.2.0.34
<br>
pandas          0.24.2
<br>
Pillow          7.2.0
<br>
pip             20.1.1
<br>
protobuf        3.6.1
<br>
pyparsing       2.4.7
<br>
python-dateutil 2.8.1
<br>
pytz            2020.1
<br>
setuptools      20.7.0
<br>
six             1.11.0
<br>
wheel           0.29.0
--------------- -------

## Usage
### train a CNN
python3 yama_cnn.py -remark [remark] -e [epoch]
### Grad-CAM(after training finished)
python3 yama_gradcam.py - remark [remark]

<img src="/Users/home_pc/Desktop/図1.png" width="120">
<img src="/Users/home_pc/Desktop/図2.png" width="120">

## Install
$ docker pull ryohei\_yamaguchi/glomerular\_classification


## Licence
MIT license

## Author
Ryohei Yamaguchi
