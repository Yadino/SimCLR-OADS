import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from models.alexnet_simclr import AlexNetSimCLR
from simclr import SimCLR
from exceptions.exceptions import InvalidBackboneError

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
#parser.add_argument('-data', metavar='DIR', default='./datasets',

# ARW dataset
#parser.add_argument('-data', metavar='DIR', default=r"C:\Users\YO\OneDrive - UvA\ARW",
#                                        help='path to dataset')
#parser.add_argument('-data-csv', metavar='CSV', default=r"C:\Users\YO\OneDrive - UvA\ARW\OADS_file_names.csv",
#                                        help='path to csv file with dataset file names')

# for cropped images
parser.add_argument('--data', metavar='DIR', default=r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Practice\cropped_images\png",
                                        help='path to dataset')

parser.add_argument('--data-csv', metavar='CSV', default=r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Practice\cropped_images\png\filenames.csv",
                                        help='path to csv file with dataset file names')

parser.add_argument('--ckpt', default=None, type=str, metavar='CKPT',
                    help='a checkpoint to resume training from')

# YO:: changed default from stl10
parser.add_argument('--dataset-name', default='OADS',
                    help='dataset name', choices=['stl10', 'cifar10', 'imagenet', 'OADS'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

#parser.add_argument('-b', '--batch-size', default=256, type=int,
# TODO: change default batch size before production
parser.add_argument('-b', '--batch-size', default=64, type=int,
                                        metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=10, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--validation-interval', default=0, type=int,
                    help='How often to perform validation during training')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data, args.data_csv)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # Add validation datasets for OADS
    if args.dataset_name == 'OADS' and args.validation_interval > 0:
        val_dataset = dataset.get_dataset('OADS_val', args.n_views)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        val_loader = None


    # YO:: initialize the right model
    if "resnet" in args.arch:
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    elif args.arch == "alexnet":
        model = AlexNetSimCLR(out_dim=args.out_dim)
    else:
        raise InvalidBackboneError

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
