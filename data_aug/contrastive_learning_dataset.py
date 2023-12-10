from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from OADSDataset import OADSDataset


class ContrastiveLearningDataset:
    def __init__(self, root_folder, csv_file=None):
        self.root_folder = root_folder
        self.csv_file = csv_file

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_simclr_OADS_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper,
        YO:: slight changes applied (RandomResizedCrop -> RandomCrop)."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),

                          'OADS': lambda: OADSDataset(root_dir=self.root_folder,
                                                      csv_file=self.csv_file,
                                                      transform=ContrastiveLearningViewGenerator(
                                                          self.get_simclr_OADS_transform(400),
                                                          n_views)),
                          # YO: added imagenet
                          # YO:: TODO:: check if this works after downloading imagenet
                          #      https://huggingface.co/datasets/imagenet-1k/tree/main/data
                          'imagenet': lambda: datasets.ImageNet(self.root_folder, split='train',
                                                                transform=ContrastiveLearningViewGenerator(
                                                                    self.get_simclr_pipeline_transform(224),
                                                                    n_views),
                                                                download=True)
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
