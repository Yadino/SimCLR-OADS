# %%################# IMPORTS
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from sklearn.metrics import r2_score
import tqdm
from scipy.stats import zscore
import pandas as pd
from mne import read_epochs
# YO::
from OADSDataset import OADSDataset
from models.resnet_simclr import ResNetSimCLR
from models.alexnet_simclr import AlexNetSimCLR

############################################


######################## Set paths, number of workers/cores, GPU/CPU access

gpu_name = 'cuda:0'
device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')
torch.cuda.empty_cache()

batch_size = 32  # 512 # 512

# TODO: change when running on DAS4
n_workers = 1

# TODO: what is this?
######################## Set number of output channels of the models (corresponds to number of classes)
output_channels = 19

# Directory with files
eeg_dir = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\EEG data\oads_eeg_rsvp"

dataset_root_dir = r"C:\Users\YO\OneDrive - UvA\ARW"
# dataset_csv_file = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Practice\cropped_images\png\filenames.csv"


checkpoint_path = (r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Model srcs\checkpoints\runs resnet18\Jan29_19-06-56_node436\checkpoint_0200.pth.tar")

model_type = 'resnet18'

out_dim = 128

# EEG channels by name (const)
channel_order = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'left', 'right', 'above', 'below']

# A subset of channels that are visual system related
visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'PO4', 'PO8']
################################ Define util funcs

# noinspection PyShadowingNames
def record_activations(data_loader, model, return_nodes, device):
    """Extract features for all layers and all images/batches and return a dictionary"""

    model = create_feature_extractor(model, return_nodes)
    model.eval()  # Set the feature extractor to evaluation mode

    all_activations = {}

    # Iterate through the data loader
    for batch_idx, inputs in enumerate(data_loader):     # tqdm(data_loader) or tqdm(enumerate(data_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            activations = model(inputs)

            for layer_name, activation_tensor in activations.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = {}
                all_activations[layer_name][batch_idx] = activation_tensor.detach().cpu().numpy()

    return all_activations


def format_filenames(filename_list):
    """Make necessary changes to filenames"""
    # Change the tiff ending to ARW
    return [fn.replace("tiff", "ARW") for fn in filename_list]


########################

# Get a list of all images used
all_filenames = []

for sub in range(5, 36):
    ######################## Set EEG path and load EEG data
    filenames_file = f"{eeg_dir}/subject_info/filenames_oads_eeg_rsvp_sub-{sub:02}.pkl"
    filenames = pickle.load(open(filenames_file, "rb"))
    all_filenames.extend(filenames)

all_filenames = sorted(set(all_filenames))


################### Load model

if model_type == 'resnet50':
    return_nodes = {
        # node_name: user-specified key for output dict
        # 'layer1.2.relu_2': 'layer1',
        # 'layer2.3.relu_2': 'layer2',
        # 'layer3.5.relu_2': 'layer3',
        # 'layer4.2.relu_2': 'layer4',
        'flatten': 'feature',
    }
    model = ResNetSimCLR('resnet50', out_dim)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

elif model_type == 'resnet18':
    return_nodes = {
        # node_name: user-specified key for output dict
        # 'backbone.layer1.1.relu_1': 'layer1',
        # 'backbone.layer2.1.relu_1': 'layer2',
        'backbone.layer3.1.relu_1': 'layer3',
        'backbone.layer4.1.relu_1': 'layer4',
        #'backbone.fc.0': 'fc0',
        #'backbone.fc.1': 'fc1',
        'backbone.fc.2': 'fc2', # The final layer
    }

    model = ResNetSimCLR('resnet18', out_dim)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

elif model_type == 'alexnet':
    return_nodes = {
        'features.2': 'layer1',
        'features.5': 'layer2',
        'features.12': 'layer3',
        'classifier.5': 'feature',
    }

    model = AlexNetSimCLR(out_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=gpu_name))

else:
    print(f'Model {model_type} not implemented')
    exit(1)

model = model.to(device)


######################## Create feature extractor to retrieve above specified activations per layer
# Normalize values
mean = [0.3410, 0.3123, 0.2787]
std = [0.2362, 0.2252, 0.2162]

#feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

print(f"Getting data loaders")
transform_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
transform = transforms.Compose(transform_list)


# %%
##################### Loop over all subjects

for sub in range(5, 6):
    ######################## Set EEG path and load EEG data
    eeg_file = f"{eeg_dir}/oads_eeg_rsvp_sub-{sub:02}.npy"
    filenames_file = f"{eeg_dir}/subject_info/filenames_oads_eeg_rsvp_sub-{sub:02}.pkl"
    is_test_file = f"{eeg_dir}/subject_info/is_test_oads_eeg_rsvp_sub-{sub:02}.pkl"

    eeg_data = np.load(eeg_file)
    filenames = np.array(format_filenames(pickle.load(open(filenames_file, "rb"))))
    is_test = np.array(pickle.load(open(is_test_file, "rb")))

    # Divide to train and test (indexes should still be correct)
    train_data = eeg_data[~is_test]
    test_data = eeg_data[is_test]

    train_filenames = [filename for filename, is_t in zip(filenames, is_test) if not is_t]
    test_filenames = [filename for filename, is_t in zip(filenames, is_test) if is_t]

    _, n_channels, n_timepoints = test_data.shape

    ###################### TRAIN DATA ########################
    train_dataset = OADSDataset(dataset_root_dir, filelist=train_filenames, transform=transform)
    # TODO: replace with the batch_size param for a bigger batch
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=n_workers)

    # Get test features - returns a dict with chosen layers, for each layer an array with indices
    #   corresponding to the images presented to a subject
    all_activations = record_activations(train_loader, model, return_nodes, device)

    # NOTE: could also loop over images first, then layers
    # Loop over the activations layer by layer
    for layer_name in all_activations:

        print(layer_name)

        layer = all_activations[layer_name]

        # TODO: mmake into a func
        ####### flatten
        flattened_arrays = []

        # Iterate over each ndarray in the dictionary
        for key, arr in layer.items():
            # Flatten the ndarray
            flattened_arr = arr.flatten()
            # Append the flattened array to the list
            flattened_arrays.append(flattened_arr)

        # Convert the list of flattened arrays into a 2D array
        # Stack the flattened arrays vertically
        x = np.vstack(flattened_arrays)

        # get PCA matrix
        pca = PCA(100)


        # pca.fit(x)

        # TODO: rename x or whatever
        # new_x = pca.fit_transform(x)

        # do linear regression with the EEG data

        for channel in range(n_channels):
            test_r2s = []
            test_betas = []
            test_lin_regs = []

            ######################## Iterate over timepoints
            for timepoint in range(n_timepoints):
                # The Y array for all images in one timepoint
                y = test_data[:, channel, timepoint]

                # Zscore the Y
                y = zscore(y)

                # NOTE: Make sure that this is ran only on training and not testing...
                lin_reg = LinearRegression()
                lin_reg.fit(x, y)

                predictions = lin_reg.predict(x)
                r2 = lin_reg.score(x, y)
                beta = model.coef_
                r, _ = pearsonr(y, predictions)

                test_r2s.append(r2)
                test_betas.append(beta)
                test_lin_regs.append(lin_reg)
#%%
"""
    ######################## TEST DATA ##################
    # TODO: make sure its correct
    #  Batch size 1 to get individual activation for each image...
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=n_workers)

    # Get test features - returns a dict with chosen layers, for each layer an array with indices
    #   corresponding to the images presented to a subject
    all_activations = record_activations(test_loader, model, return_nodes, device)

    # NOTE: could also loop over images first, then layers
    # Loop over the activations layer by layer
    for layer_name in all_activations:

        print(layer_name)

        layer = all_activations[layer_name]

        # TODO: mmake into a func
        ####### flatten
        flattened_arrays = []

        # Iterate over each ndarray in the dictionary
        for key, arr in layer.items():
            # Flatten the ndarray
            flattened_arr = arr.flatten()
            # Append the flattened array to the list
            flattened_arrays.append(flattened_arr)

        # Convert the list of flattened arrays into a 2D array
        # Stack the flattened arrays vertically
        x = np.vstack(flattened_arrays)

        # Use PCA matrix from training
        pca...
        
        # Dont fit (obv) just transform
        # new_x = pca.transform(x)

        # do linear regression with the EEG data

        for channel in range(n_channels):
            test_r2s = []
            test_betas = []
            test_lin_regs = []

            ######################## Iterate over timepoints
            for timepoint in range(n_timepoints):
                # The Y array for all images in one timepoint
                y = test_data[:, channel, timepoint]

                # TODO: get from Niklas or use another func
                r2, beta, lin_reg = regress(design_matrix=x, y=y, return_regression_object=True)

                # TODO: use the lin_reg from training...
                lin_reg

                test_r2s.append(r2)
                test_betas.append(beta)
                test_lin_regs.append(lin_reg)
"""

#%%
# No need to loop over images for PCA
"""
        # Each layer is a dictionary encapsulating an array of all image activations by index
        for image_idx, image_activation in enumerate(layer):

            # flatten and prepare data

            # get the EEG for the image
            eeg_data_image = test_data[image_idx]

            # Loop over electrodes
            for electrode_idx in range(eeg_data_image.shape[0]):
                pass
"""



