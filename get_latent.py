import torch
import torchvision
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator
from models.resnet_simclr import ResNetSimCLR
import matplotlib.pyplot as plt
import numpy as np
from OADSDataset import OADSDataset

# cifar10 checkpoint
#checkpoint_path = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Practice\checkpoints (runs)\Nov19_23-33-24_node404\checkpoint_1000.pth.tar"

# OADS checkpoint
checkpoint_path = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Model srcs\checkpoints\2023.12.12\checkpoint_0500.pth.tar"

data_path = r"C:\Users\YO\OneDrive - UvA\ARW"
csv_path = r"C:\Users\YO\OneDrive - UvA\ARW\OADS_file_names.csv"


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
#transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_pipeline_transform(32), 2)
transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_OADS_transform(400), 2)

# Tutorial transform - imshow compatible
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
#                                       download=True, transform=transform)

testset = OADSDataset(root_dir=data_path, csv_file=csv_path, transform=transform, split='test')



testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


#%% def func
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# load a batch of images

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

out = model(images)

#%%
#print(out)

# Visualize these TODO:: (improve)
#npimg = img.numpy()

# First dimension - batch size
# Second dimension - the number of channels / filters / kernels / activation map
kernel = out['layer4'].detach()[3,1000,:,:]
kernel = out['layer4'].detach()[3,999,:,:]



plt.imshow(kernel)
plt.show()


#%% find interesting kernels (using math)

layer_name = 'layer4'

k_count = out[layer_name].detach().size(dim=1)

k_dict = {}

for k in range(k_count):
    kernel = out['layer4'].detach()[1, k, :, :]
    nonz_count = torch.count_nonzero(kernel).item()

    k_dict[k] = nonz_count

    #if(nonz_count > 0):
    # Can print here or whatever


# A dictionary of all the kernels which aren't all zero, their index and the number of nonzero values (out of 13*13=169)
k_dict_nonzero = {x:y for x,y in k_dict.items() if y!=0}
len(k_dict_nonzero)

#%%

#flatten

l4 = out['layer4']
l4_flat = torch.flatten(l4, start_dim=1)


#%% PCA

from sklearn.decomposition import PCA


x = l4_flat.detach().numpy() #.reshape(-1, 1)

# TODO: make it 100
pca = PCA(4)
pca.fit(x)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)


#%%

# create the linear model SGDclassifier
from sklearn.linear_model import SGDClassifier
linear_clf = SGDClassifier()

# result should be something like - https://www.researchgate.net/figure/Representative-convolutional-kernels-of-the-final-convolutional-layer-within-the-ResNet_fig3_334813175
