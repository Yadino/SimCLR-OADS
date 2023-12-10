import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

print(model_names)