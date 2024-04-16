"""
Subset based on ImageFolder that returns a precentage of the real subset.
It gives the precentage of each class, to keep the balance between the ratios of images in each class.
"""

import torch
from torchvision.datasets import ImageFolder
import numpy as np


class SubsetImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, percentage=5):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.percentage = percentage
        self.select_subset()

    def select_subset(self):
        # Calculate the number of samples to select for each class
        class_counts = torch.bincount(torch.tensor(self.targets))
        num_samples_per_class = (class_counts * self.percentage / 100).long()

        # Create a list of indices for each class
        class_indices = {}
        for idx, (image_path, label) in enumerate(self.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Randomly select a single index from each class
        selected_indices = []
        for label, indices in class_indices.items():
            num_samples_to_select = min(num_samples_per_class[label].item(), len(indices))
            selected_index = np.random.choice(indices, size=num_samples_to_select, replace=False)
            selected_indices.extend(selected_index.tolist())

        # Set the dataset to the subset of selected indices
        self.samples = [self.samples[idx] for idx in selected_indices]
        self.targets = [self.targets[idx] for idx in selected_indices]