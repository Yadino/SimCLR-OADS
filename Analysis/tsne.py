#classification regression
#%% V1.1

# imports
import os
import datetime
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.resnet_simclr import ResNetSimCLR
from models.alexnet_simclr import AlexNetSimCLR
from datasets.SubsetImageFolder import SubsetImageFolder
from torchvision.models import alexnet, resnet18
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#from sklearn.exceptions import ConvergenceWarning
#warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.manifold import TSNE

############################################ Params

gpu_name = 'cuda:0'
device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')
torch.cuda.empty_cache()

batch_size = 32  # 512

# TODO: change when running on DAS4
n_workers = 1

# Output dir for processed data
output_dir = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\object_class_reg_outputs"

#dataset_root_dir = r"C:\Users\YO\OneDrive - UvA\ML_new"
dataset_root_dir = r"C:\Users\YO\UvA\imagenet-16"



# TODO: save this info
#checkpoint_path = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\checkpoints\runs resnet18 6x crop\Jan29_19-06-56_node436\checkpoint_0200.pth.tar"
#checkpoint_path = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\checkpoints\from niklas\best_model_05-06-23-164521.pth"
checkpoint_path = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\checkpoints\runs resnet18 30x crop\Apr15_23-58-43_node414\checkpoint_0050.pth.tar"

model_type = 'resnet18'
is_from_niklas = False  # Whether it is a non-simclr, supervised checkpoint obtained from Niklas

out_dim = 128

# The size of the subset out of the full dataset, in percents (this is to save time and RAM)
subset_size_percentage = 100
################################ Define util funcs


# noinspection PyShadowingNames
def record_activations(data_loader, model, return_nodes, device):
    """Extract features for all layers and all images/batches and return a dictionary"""

    model = create_feature_extractor(model, return_nodes)
    model.eval()  # Set the feature extractor to evaluation mode

    all_activations = {}

    # Iterate through the data loader
    img_idx = 0
    for batch_idx, inputs in enumerate(tqdm(data_loader)):
        labels = inputs[1]
        inputs = inputs[0].to(device)
        with torch.no_grad():
            activations = model(inputs)

            for layer_name, activation_tensor in activations.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = {}

                #all_activations[layer_name][batch_idx] = activation_tensor.detach().cpu().numpy()
                # iterate over the batch
                for img in activation_tensor.detach().cpu().numpy():
                    all_activations[layer_name][img_idx] = img
                    img_idx += 1
    return all_activations

def flatten_layer(layer):
    """flatten a layer of a dictionary returned from record_activations"""
    flattened_arrays = []

    for arr in layer:
        # Flatten the ndarray
        flattened_arr = arr.flatten()
        # Append the flattened array to the list
        flattened_arrays.append(flattened_arr)

    # Convert the list of flattened arrays into a 2D array
    # Stack the flattened arrays vertically
    x = np.vstack(flattened_arrays)
    return x


def creat_output_dir(output_dir):
    """Create a directory with the current datetime and return it"""

    # Get current date and time
    current_datetime = datetime.datetime.now()

    # Create a directory with current date and time inside the output directory
    datetime_dir = os.path.join(output_dir, current_datetime.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(datetime_dir, exist_ok=True)

    return datetime_dir


################### Load model

def load_model_nm():
    """Load Niklas' supervised, non-SimCLR model checkpoints"""

    # TODO: not sure what values are right
    output_channels = 21
    use_rgbedges = False
    use_cocedges = False

    if model_type == 'resnet18':
        return_nodes = {
            # node_name: user-specified key for output dict
            'layer1.1.relu_1': 'layer1',
            'layer2.1.relu_1': 'layer2',
            'layer3.1.relu_1': 'layer3',
            'layer4.1.relu_1': 'layer4',
            # 'fc': 'fc'  # For this PCA 100 is too big...
            #'flatten': 'feature',
        }

        model = resnet18()
        model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        model.fc = torch.nn.Linear(in_features=512, out_features=output_channels, bias=True)

    elif model_type == 'alexnet':
        return_nodes = {
            'features.2': 'layer1',
            'features.5': 'layer2',
            'features.12': 'layer3',
            # 'classifier.5': 'feature',
        }

        model = alexnet()
        model.features[0] = torch.nn.Conv2d(6 if use_rgbedges or use_cocedges else 3, 64, kernel_size=11, stride=4, padding=2)
        model.classifier[6] = torch.nn.Linear(4096, output_channels, bias=True)

    else:
        print(f'Model {model_type} not implemented (is_from_niklas is True)')
        exit(1)

    #gpu_name = 'cuda'
    try:
        model.load_state_dict(torch.load(checkpoint_path))#, map_location=gpu_name))
    except RuntimeError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(checkpoint_path))#, map_location=gpu_name))
        model = model.module

    model = model.to(device)

    return model, return_nodes

def load_model():
    """Load a simclr model"""

    if model_type == 'resnet50':
        return_nodes = {
            # node_name: user-specified key for output dict
            'backbone.layer1.2.relu_2': 'layer1',
            'backbone.layer2.3.relu_2': 'layer2',
            'backbone.layer3.5.relu_2': 'layer3',
            'backbone.layer4.2.relu_2': 'layer4',
            'backbone.fc.2': 'fc2',
        }
        model = ResNetSimCLR('resnet50', out_dim)

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

    elif model_type == 'resnet18':
        return_nodes = {
            # node_name: user-specified key for output dict
            'backbone.layer1.1.relu_1': 'layer1',
            'backbone.layer2.1.relu_1': 'layer2',
            'backbone.layer3.1.relu_1': 'layer3',
            'backbone.layer4.1.relu_1': 'layer4',
            #'backbone.fc.0': 'fc0',
            #'backbone.fc.1': 'fc1',
            #'backbone.fc.2': 'fc2', # The final layer
        }

        model = ResNetSimCLR('resnet18', out_dim)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

    elif model_type == 'alexnet':
        return_nodes = {
            'backbone.features.2': 'layer1',
            'backbone.features.5': 'layer2',
            'backbone.features.12': 'layer3',
            'backbone.classifier.5': 'feature',
        }

        model = AlexNetSimCLR(out_dim)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

    else:
        print(f'Model {model_type} not implemented')
        exit(1)

    model = model.to(device)

    return model, return_nodes


if is_from_niklas:
    model, return_nodes = load_model_nm()
else:
    model, return_nodes = load_model()


######################## Create feature extractor to retrieve above specified activations per layer
# Normalize values
mean = [0.3410, 0.3123, 0.2787]
std = [0.2362, 0.2252, 0.2162]

# Resize values
desired_height = 400 #50
desired_width = 400 #50

transform_list = [transforms.Resize((desired_height, desired_width)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)]
transform = transforms.Compose(transform_list)


# %%
##################### Loop over all subjects
def main():
    # Create an output dir for current run
    datetime_dir = creat_output_dir(output_dir)

    # Get test features - returns a dict with chosen layers, for each layer an array with indices
    #   corresponding to the images presented to a subject
    print("Extracting features")
    # Dataset & dataloaders
    #dataset = torchvision.datasets.ImageFolder(root=dataset_root_dir, transform=transform)
    dataset = SubsetImageFolder(root=dataset_root_dir, transform=transform, percentage=subset_size_percentage)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    all_activations = record_activations(dataloader, model, return_nodes, device)

    # prep fig
    idx = 0
    num_cols, num_rows, num_layers = 2, 2, 4
    fig, axs = plt.subplots(2, 2)

    # Loop over the activations layer by layer
    for layer_name in all_activations:
        # Make dir
        layer_dir = os.path.join(datetime_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        print(f"Layer: {layer_name}")

        activations_layer = all_activations[layer_name]

        # Flatten the activations to create the design matrix
        X = flatten_layer(list(activations_layer.values()))

        # y is the labels / label indices
        y = dataset.targets

        # Split to test and train data
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ## PCA
        pca = PCA(n_components=100)  # You can adjust the number of components as needed
        # Fit to training data
        X = pca.fit_transform(X)
        # transform test data with PCA
        #X_test = pca.transform(X_test)

        # Perform t-SNE to reduce the dimensionality of the features to 2D
        tsne = TSNE(n_components=2)
        tsne_features = tsne.fit_transform(X)

        """
        # Visualize the t-SNE embeddings
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y, cmap='viridis')
        plt.title(f"t-SNE {layer_name}")
        plt.colorbar()
        plt.show()
        """

        # Determine the subplot location
        # Plot t-SNE embeddings
        row_idx = idx // num_cols
        col_idx = idx % num_cols
        idx += 1

        if num_layers > num_cols:
            ax = axs[row_idx, col_idx]
        else:
            ax = axs[col_idx]
        ax.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y, cmap='viridis')
        ax.set_title(f"{layer_name}")
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

    plt.suptitle("t-SNE Visualization of Different Layers")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
