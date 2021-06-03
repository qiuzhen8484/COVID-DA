import random
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.nn import DataParallel
from torchvision import transforms
from model_compat_CovidDA import COVID_DA_with_resnet18
from utils import log
from dataset import penuDataset, covidDataset_target
import time
import argparse
from train import train_covid_da

######################
# params             #
######################


def arg_parser():
    parser = argparse.ArgumentParser()
    # basic opts
    parser.add_argument('--gpu', default='0,1', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=16, type=int, help='batch size for each domain')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--step_decay_weight', default=0.95, type=float)
    parser.add_argument('--lr_decay_step', default=4000, type=int)
    parser.add_argument('--epoch', default=200, type=int)

    # trade-off parameter
    parser.add_argument('--alpha_weight', default=0.1, type=float, help='trade-off parameter for domain loss')
    parser.add_argument('--gamma_weight', default=0.1, type=float, help='trade-off parameter for entropy loss')
    parser.add_argument('--beta_weight', default=0.1, type=float, help='trade-off parameter for diversity loss')
    parser.add_argument('--image_size', type=int, default=224, help='image resolution')

    # datapath
    parser.add_argument('--path_source', default='all_data_pneumonia', type=str, help='data path of source data')
    parser.add_argument('--label_file_source', default='./data/pneumonia_task_for_python3.pkl', type=str,
                        help='the pkl file that records the source data list and labels')
    parser.add_argument('--path_target', default='all_data_covid', type=str, help='data path of target data')
    parser.add_argument('--label_file_target', default='./data/COVID-19_task_for_python3.pkl', type=str,
                        help='the pkl file that records the target data list and labels')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()

    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # create logger
    logger = log()
    cudnn.benchmark = True
    args.cuda = True
    logger.info('original train process')
    args.time_stamp_launch = time.strftime('%Y%m%d') + '-' + time.strftime('%H%M')
    logger.info('COVID-DA:' + args.time_stamp_launch)

    # record the best F1 score
    args.max_F1 = 0

    # create path for saving the checkpoints
    args.model_root = './model_checkpoint'
    if not os.path.exists(args.model_root):
        os.mkdir(args.model_root)

    args.image_size = (args.image_size, args.image_size)

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    #######################
    #      load data      #
    #######################

    source_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4859, 0.4859, 0.4859), (0.0820, 0.0820, 0.0820)),  # grayscale mean/std
    ])

    target_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5002, 0.5002, 0.5002), (0.0893, 0.0893, 0.0893)),  # grayscale mean/std
    ])

    dataset_source = penuDataset(args.path_source, args.label_file_source, train=True, transform=source_train)

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=2
    )

    # load unlabeled target domain data
    dataset_target = covidDataset_target(args.path_target, args.label_file_target, train=True, transform=target_train)

    dataloader_target_unlabeled = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=2
    )

    # load labeled target domain data
    dataset_target_labeled = covidDataset_target(args.path_target, args.label_file_target, semi=True, train=True,
                                                 transform=target_train)

    dataloader_target_labeled = torch.utils.data.DataLoader(
        dataset=dataset_target_labeled,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=2
    )
    #####################
    #  load model       #
    #####################

    my_net = COVID_DA_with_resnet18(args)
    my_net = DataParallel(my_net, device_ids=[0, 1])

    train_covid_da(my_net, dataloader_source, dataloader_target_unlabeled, dataloader_target_labeled, args, logger)
