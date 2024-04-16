import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io


class OADSDataset(Dataset):

    def __init__(self,
                 root_dir,
                 csv_file=None,
                 filelist=None,
                 split='train',
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to a csv file with filenames split into groups.
            filelist (string list): list of filenames instead of a csv file. No split needed when used.
            split (string): The name of the group to select,
                            possible values: 'train', 'test', and 'val'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if csv_file:
            self.file_names = pd.read_csv(csv_file)
            if split:
                assert split in ['train', 'test', 'val']
                self.file_names = self.file_names.loc[self.file_names['split'] == split]
                self.file_names.reset_index(drop=True)
            self.using_csv = True
        elif filelist:
            self.file_names = filelist
            self.using_csv = False
        else:
            raise Exception("Either a csv file path or a filelist must be provided")

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.using_csv:
            img_name = os.path.join(self.root_dir, self.file_names.iloc[idx, 0])
        else:
            img_name = os.path.join(self.root_dir, self.file_names[idx])

        # TODO: add "with" if necessary
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image
