import pdb
import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import collections

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


class TransformFixMatch(object):
    def __init__(self,cropsize, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(size=cropsize,
                                  padding=int(cropsize*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            #transforms.RandomGrayscale(),
            #transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(size=cropsize,
                                  padding=int(cropsize*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=cropsize,
                                  padding=int(cropsize*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(weak2), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, train_dataset, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.data = []
        self.targets = []
        self.correct_cnt = 0
        
        for idx in indexs:
            #img, target = train_dataset(idx)
            self.data.append(train_dataset.train_data[idx])
            self.targets.append(train_dataset.train_noisy_labels[idx])
            #self.targets.append(train_dataset.train_labels[idx].item())
            if train_dataset.train_noisy_labels[idx] == train_dataset.train_labels[idx]:
                self.correct_cnt += 1
        cnt = collections.Counter(np.array(self.targets))
        print("* idx distribution: ", cnt)
        
        #if indexs is not None:
            #self.data = self.data[indexs] #self.data: from superclass
            #self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)   #Unlabeled 경우 weak, strong 모두 return

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, train_dataset, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.data = []
        self.targets = []
        self.correct_cnt = 0

        for idx in indexs:
            #img, target = train_dataset(idx)
            self.data.append(train_dataset.train_data[idx])
            self.targets.append(train_dataset.train_noisy_labels[idx])
            #self.targets.append(train_dataset.train_labels[idx].item())
            if train_dataset.train_noisy_labels[idx] == train_dataset.train_labels[idx]:
                self.correct_cnt += 1
        cnt = collections.Counter(np.array(self.targets))
        print("* idx distribution: ", cnt)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


"""
Not Used
"""
def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    #train dataset -> labeled/unlabeled datasets
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)
    #Labeled dataset
    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)
    #Unlabeled dataset
    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    #base_dataset.targets: (50000,)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    #Labeled dataset
    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)
    #Unlabeled dataset
    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels) #labels: (50000,)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0] #i에 해당하는 index return
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx) #extend: 여러 개 한번에 append
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

    
DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}