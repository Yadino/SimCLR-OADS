import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io


class OADSDataset(Dataset):

    def __init__(self,
                 root_dir,
                 csv_file,
                 split='train',
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            split (string): The name of the group to select,
                            possible values: 'train', 'test', and 'val'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        if split:
            assert split in ['train', 'test', 'val']
            self.file_names = self.file_names.loc[self.file_names['split'] == split]
            self.file_names.reset_index(drop=True)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.file_names.iloc[idx, 0])

        # TODO: add "with" if necessary
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image
