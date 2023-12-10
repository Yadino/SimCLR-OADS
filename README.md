# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

Forked from https://github.com/sthalles/SimCLR and adapted for OADS

Other improvements:
* OADS dataset and dataloader 
* Checkpoint loading
* Feature activation extration and plotting
* Minor bugfix


![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)

### See also [PyTorch Implementation for BYOL - Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://github.com/sthalles/PyTorch-BYOL).

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
