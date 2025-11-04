# PyTorch SimCLR + OADS

Forked from https://github.com/sthalles/SimCLR and adapted for OADS

Features:
* Linear Regression model to predict EEG ERPs over the SimCLR representations.
* Analysis and evalution pipelines including classifications and multiple datasets.
* OADS dataset and dataloader.

Other improvements:
* Checkpoint loading.
* Feature activation extration and plotting.
* Hyperparameter adaptation.
* Bugfixes over the original repository.


![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)


## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Config file

Before running SimCLR, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -data ./datasets --dataset-name stl10 --log-every-n-steps 100 --epochs 100 

```

If you want to run it on CPU (for debugging purposes) use the ```--disable-cuda``` option.

For 16-bit precision GPU training, there **NO** need to to install [NVIDIA apex](https://github.com/NVIDIA/apex). Just use the ```--fp16_precision``` flag and this implementation will use [Pytorch built in AMP training](https://pytorch.org/docs/stable/notes/amp_examples.html).


## Analysis

The analysis folder includes scripts for:
* Linear regression to compare model's representations with EEG data.
* Linear SVM regression for object classification.
* Linear SVm regression for determining the shape/texture bias of a model.

