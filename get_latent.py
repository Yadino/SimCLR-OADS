import torch
import torchvision
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


checkpoint_path = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Practice\checkpoints (runs)\Nov19_23-33-24_node404\checkpoint_1000.pth.tar"
base_model = 'resnet50'
out_dim = 128
batch_size = 4


# INIT

# copied from the code
'''state = {
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
           }'''

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Initialize model
model = ResNetSimCLR(base_model=base_model, out_dim=out_dim)

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["state_dict"])
epoch = checkpoint['epoch']

# Mode
model.eval()

#print(model)

# move to GPU?

# Use the same transform series as the training
transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_pipeline_transform(32), 2)
#transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_OADS_transform(32), 2)

# Tutorial transform - imshow compatible
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


#%% def func
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#%% load a batch of images

dataiter = iter(testloader)

images, labels = next(dataiter)

#print(images.size)
#print(labels)

for im in images:
    imshow(torchvision.utils.make_grid(im))


#%% get latents
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# Printing the available layers
train_nodes, eval_nodes = get_graph_node_names(model)

return_nodes = {
    # node_name: user-specified key for output dict
    'backbone.layer1.2.relu_2': 'layer1',
    'backbone.layer2.3.relu_2': 'layer2',
    'backbone.layer3.5.relu_2': 'layer3',
    'backbone.layer4.2.relu_2': 'layer4',
    'backbone.fc.0': 'fc0',
    'backbone.fc.1': 'fc1',
    'backbone.fc.2': 'fc2' # The final layer
}

# Create feature extractor and apply to model
model = create_feature_extractor(model, return_nodes)

#%%

out = model(images[0])

#%%
print(out)

# Visualize these TODO:: (improve)
#npimg = img.numpy()
plt.imshow(out['layer1'].detach()[1,80,:,:])
plt.show()


#%%


# result should be something like - https://www.researchgate.net/figure/Representative-convolutional-kernels-of-the-final-convolutional-layer-within-the-ResNet_fig3_334813175
