import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image
import re

class TextureBiasImagenetDataset(Dataset):
    def __init__(self, root_dir=r"C:\Users\YO\UvA\rgeirhos_github_cue_conflict_512\all", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = []
        self.shape_classes = []
        self.texture_classes = []

        # Iterate over the files to collect file names and extract class labels
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.png'):
                self.file_names.append(os.path.join(root_dir, file_name))
                self.shape_classes.append(re.sub(r'\d+', '', file_name.split('-')[0]))
                self.texture_classes.append(re.sub(r'\d+|\..*', '', file_name.split('-')[1]))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_path = self.file_names[idx]
        shape_class = self.shape_classes[idx]
        texture_class = self.texture_classes[idx]

        # Load the image
        image = io.imread(image_path)

        #with Image.open(image_path) as img:
        #    image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, shape_class, texture_class


#test
#dataset_root_dir = r"C:\Users\YO\UvA\Niklas Müller - Cue-Conflict_Stimuli_1.0"
"""
# Normalize values
mean = [0.3410, 0.3123, 0.2787]
std = [0.2362, 0.2252, 0.2162]
# transform
transform_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
transform = transforms.Compose(transform_list)

texture_dataset = TextureBiasDataset(root_dir=dataset_root_dir)
# Define dataloader
dataloader = DataLoader(texture_dataset, batch_size=1, shuffle=True)
# Iterate through the dataloader
for batch_idx, (images, shape_classes, texture_classes) in enumerate(dataloader):
    # Now you have access to images, shape_classes, and texture_classes for each batch
    print("Batch:", batch_idx)
    print("Images:", images)
    print("Shape Classes:", shape_classes)
    print("Texture Classes:", texture_classes)
"""
"""
filename = "Lamppost_5090a1a1299090d8_7-Balcony door_c607c4c13b9884c6_25.png"
# Split the filename by underscores
shape_class = filename.split('_')[0]
texture_class = filename.split('-')[1].split('_')[0]
"""