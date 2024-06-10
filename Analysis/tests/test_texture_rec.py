import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from texture_bias_class_reg import record_activations_texture  # Import function
from models.resnet_simclr import ResNetSimCLR
from models.alexnet_simclr import AlexNetSimCLR
from datasets.TextureBiasDataset import TextureBiasDataset

# Define a simple dataset (you can replace this with your actual dataset)
dataset_root_dir = r"C:\Users\YO\UvA\Niklas Müller - Cue-Conflict_Stimuli_1.0"
batch_size = 8

# Normalize values
mean = [0.3410, 0.3123, 0.2787]
std = [0.2362, 0.2252, 0.2162]

# Resize values
desired_height = 400  # 50
desired_width = 400  # 50

transform_list = [#transforms.Resize((desired_height, desired_width)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)]
transform = transforms.Compose(transform_list)

dataset = TextureBiasDataset(dataset_root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a simple model (you can replace this with your actual model)
model = ResNetSimCLR('resnet18', 128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define return nodes (you can replace this with your actual return nodes)
return_nodes = {
    # node_name: user-specified key for output dict
    # 'backbone.layer1.1.relu_1': 'layer1',
    'backbone.layer2.1.relu_1': 'layer2',
    'backbone.layer3.1.relu_1': 'layer3',
    'backbone.layer4.1.relu_1': 'layer4',
    # 'backbone.fc.0': 'fc0',
    # 'backbone.fc.1': 'fc1',
    'backbone.fc.2': 'fc2',  # The final layer
}

# Test the modified function
all_activations, all_shape_classes, all_texture_classes = record_activations_texture(data_loader, model, return_nodes, device)

# Print some information to verify the results
print("Number of layers with activations:", len(all_activations))
print("Number of layers with shape classes:", len(all_shape_classes))
print("Number of layers with texture classes:", len(all_texture_classes))

# Print the number of images and their corresponding activations, shape classes, and texture classes for a specific layer
layer_name = "layer4"  # Change this to the layer you want to inspect
print(f"Number of images in layer '{layer_name}':", len(all_activations[layer_name]))
print(f"Activations for layer '{layer_name}':", all_activations[layer_name])
print(f"Shape classes for layer '{layer_name}':", all_shape_classes[layer_name])
print(f"Texture classes for layer '{layer_name}':", all_texture_classes[layer_name])
