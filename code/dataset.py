# coding: utf-8
from __future__ import print_function
from PIL import Image
import os
import os.path
import torch
import pickle as pkl
from torch.utils.data import Dataset


class penuDataset(Dataset):
    """
    Pneumonia class.
    """

    def __init__(self, root, label_file, train=True, transform=None):
        super(penuDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list']
        val_list = train_dict['val_list']

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


class covidDataset_target(Dataset):
    """
    COVID-19 class
    Args:
        root (str): datapath
        label_file (str): the pkl file that restore the list of data
        semi (bool): If true, read the list of labeled target domain data
              If false, read the list of unlabeled target domain data
        train (bool): If true: load the training data
        transform: the transformation which is employed on the data
    """
    def __init__(self, root, label_file, semi=False, train=True, transform=None):
        super(covidDataset_target, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        if semi is True:
            train_list = train_dict['train_list_semi']
        else:
            train_list = train_dict['train_list']
        val_list = train_dict['val_list']

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
                self.test_data.append(os.path.join(self.root, 'val', img))
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

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

