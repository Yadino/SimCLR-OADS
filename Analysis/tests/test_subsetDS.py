import torch
from torchvision.datasets import ImageFolder
from datasets.SubsetImageFolder import SubsetImageFolder

# Define the root directory of the dataset
dataset_root_dir = r"C:\Users\YO\OneDrive - UvA\ML_new"

# Initialize the original ImageFolder dataset
original_dataset = ImageFolder(root=dataset_root_dir)

# Calculate the number of images in each class in the original dataset
original_class_counts = torch.tensor(original_dataset.targets).bincount()

# Initialize the SubsetImageFolder dataset with the specified percentage
subset_dataset = SubsetImageFolder(root=dataset_root_dir, percentage=8)

# Calculate the number of images in each class in the subset dataset
subset_class_counts = torch.tensor(subset_dataset.targets).bincount()

# Create dictionaries to store class counts
original_counts = {}
subset_counts = {}

# Fill dictionaries with class counts
for class_idx, count in enumerate(original_class_counts):
    original_counts[class_idx] = count

for class_idx, count in enumerate(subset_class_counts):
    subset_counts[class_idx] = count

# Calculate percentages based on counts
percentages = {}
for class_idx in range(len(original_counts)):
    original_count = original_counts[class_idx]
    subset_count = subset_counts[class_idx]
    percentage = (subset_count / original_count) * 100
    percentages[class_idx] = percentage

# Print percentages for each class
print("Percentages for each class:")
for class_idx, percentage in percentages.items():
    print(f"Class {class_idx}: {percentage:.2f}%")
