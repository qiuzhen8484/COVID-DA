import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torch.nn import DataParallel
from utils import log, analyse
from torchvision import transforms
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
from PIL import Image
import argparse


class covid_target(Dataset):
    """
    COVID-19 class
    """
    def __init__(self, root, label_file, train=True, transform=None):
        super(covid_target, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list_semi'] + train_dict['train_list']
        # load the test list
        val_list = train_dict['test_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                img = train_list[i][0]
                if isinstance(img, bytes):
                    img = img.decode("utf-8")
                self.train_data.append(os.path.join(self.root, 'train', img))
                self.train_labels.append(train_list[i][1])
        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(val_list)):
                img = val_list[i][0]
                if isinstance(img, bytes):
                    img = img.decode("utf-8")
                self.test_data.append(os.path.join(self.root, 'test', img))
                self.test_labels.append(val_list[i][1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def test(my_net, logger, name, args):
    cuda = True
    cudnn.benchmark = True
    image_size = (args.image_size, args.image_size)

    ###################
    # load data       #
    ###################

    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5002, 0.5002, 0.5002), (0.0893, 0.0893, 0.0893)),  # grayscale mean/std
    ])

    if name == 'test':
        test_dataset = covid_target(args.path_target, args.label_file_target, train=False, transform=img_transform)
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    else:
        print('error dataset name')

    ####################
    # load model       #
    ####################

    my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    prob_list = []
    gt_list = []

    while i < len_dataloader:

        data_input = data_iter.next()
        img, label = data_input
        gt_list.append(label.numpy())

        batch_size = len(label)
        input_img = torch.FloatTensor(batch_size, 3, image_size[0], image_size[1])
        class_label = torch.LongTensor(batch_size)

        if cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(input_img).copy_(img)
        class_label.resize_as_(label).copy_(label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        result = my_net(inputv_img, inputv_img)
        pred = result[9].data.max(1, keepdim=True)[1]
        output_prob = result[9].data
        prob_list.append(output_prob[:, 1].detach().cpu().numpy())

        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = np.float(n_correct * 1.0) / np.float(n_total)
    prob_list = np.concatenate(prob_list)
    gt_list = np.concatenate(gt_list)

    analyse(gt_list, prob_list, prob=True, logger=logger)
    logger.info('accuracy of the %s dataset: %f' % (name, accu))
    return accu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0,1', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=16, type=int, help='batch size for each domain')
    parser.add_argument('--model_path', type=str, help='the path of the checkpoint to be tested')

    # the path of test dataset
    parser.add_argument('--path_target', default='all_data_covid', type=str, help='data path of target data')
    parser.add_argument('--label_file_target', default='./data/pneumonia_task_for_python3.pkl',
                        type=str, help='the pkl file that records the data list and labels')
    parser.add_argument('--image_size', type=int, default=224, help='image resolution')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_sharing_strategy('file_system')
    my_net = torch.load(args.model_path)
    my_net = DataParallel(my_net, device_ids=[0, 1])
    my_net = my_net.module
    my_net.cuda()
    logger = log()
    test(my_net, logger, 'test', args)
