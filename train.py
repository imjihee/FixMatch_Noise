import argparse
import logging
import math
import sys
import os
import random
import shutil
import time
import pathlib
import datetime
from pytz import timezone
import pdb
from collections import OrderedDict
import collections

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import datasets

from dataset.cifar import DATASET_GETTERS
from dataset.cifar import CIFAR10SSL, CIFAR100SSL, TransformFixMatch

from utils import AverageMeter, accuracy
from dataset.noisy_cifar import nCIFAR10, nCIFAR100
from models.resnet import ResNet50, ResNet101
from utils import evaluate, adjust_learning_rate, adjust_lambda, evaluate_nepes
import transform_ad

import torchvision.models as torchvision_models
import torch.nn as nn

from nepes_dataset import create_dataset, Nepes_SSL
from models.resnet_g import build_resnet

logger = logging.getLogger(__name__)
best_acc = 0



def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='nepes', type=str,
                        choices=['cifar10', 'cifar100', 'nepes'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision tfshrough NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='data/')
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.6)
    parser.add_argument('--remove_rate', type=float, help='rate of the total dataset to be removed', default=0.8)
    parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')
    parser.add_argument('--mask_epoch', type=int, default=30)
    parser.add_argument('--ema_ensemble', action='store_true')
    parser.add_argument('--ema_train_only', action='store_true')
    parser.add_argument('--ricap', action='store_true')
    parser.add_argument('--ricap-beta', type=float, default=0.3)
    parser.add_argument('--pretrain', action='store_true')

    parser.add_argument('--lr-type', type=str, default='linear')

    parser.add_argument('--path', type=str, default='//')
    parser.add_argument('--augmentation', type=str, default='')
    parser.add_argument('--use-eval', action='store_true')

    parser.add_argument('--save-path', type=str, default='./save/')
    
    parser.add_argument('--uniform-masking', action='store_true')
    parser.add_argument('--epochs', default=200, type=int)


    args = parser.parse_args()
    global best_acc

    output_d = "log/" + args.dataset + "/noise_" + str(args.noise_rate) + "_remove_" + str(args.remove_rate)
    output_dir = pathlib.Path(output_d)
    output_dir.mkdir(exist_ok=True, parents=True)

    td = datetime.datetime.now(timezone('Asia/Seoul'))
    file_name = td.strftime('%m-%d_%H.%M') + ".log"

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(output_d + '/' + file_name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    #sys.stdout = Logger(output_d, args)
    
    print(args)

    def create_model(args):
        if args.dataset=='nepes':
            print("*** build resnet50 nepes model ***")
            print("Num Class:", args.num_classes)
            model = build_resnet('resnet50','fanin')

            pretrained_weights = torchvision_models.resnet50(pretrained=True).state_dict()
            model.load_state_dict(pretrained_weights)

            if model.fc.out_features != args.num_classes:
                    fc_in = model.fc.in_features
                    model.fc = nn.Linear(fc_in, args.num_classes)
                    model.fc.reset_parameters()

        elif args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    
    elif args.dataset == 'nepes':
        args.num_classes = 22
        #model is always resnet50

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    """create dataset
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    
    """
    transformer = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transform_ad.TranslateX(p=0.3),
	transform_ad.Posterize(p=0.2),
	transforms.ToTensor()
	])

    if args.dataset == 'cifar10':
        args.top_bn = False
        train_dataset = nCIFAR10(root=args.result_dir,
                                download=True,
                                train=True,
                                transform=transformer,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

        test_dataset_mask = nCIFAR10(root=args.result_dir,
                               download=True,
                               train=False,
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                               )

    if args.dataset == 'cifar100':
        args.top_bn = False
        train_dataset = nCIFAR100(root=args.result_dir,
                                 download=True,
                                 train=True,
                                 transform=transformer,
                                 noise_type=args.noise_type,
                                 noise_rate=args.noise_rate
                                 )

        test_dataset_mask = nCIFAR100(root=args.result_dir,
                                download=True,
                                train=False,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
        
    if args.dataset == 'nepes':
        args.top_bn = False
        train_dataset, test_dataset_mask = create_dataset(args, args.path, is_train=True)
        print("*** nepes dataset ready ***")

    if args.local_rank == 0:
        torch.distributed.barrier()
        
    remove_rate = args.remove_rate

    if args.dataset=='nepes' and not args.uniform_masking:
        mask = masking_nepes(args, train_dataset, test_dataset_mask, remove_rate)
        print("*** nepes dataset masking ready ***")
    elif args.dataset=='nepes' and args.uniform_masking:
        mask = masking_nepes_uniform(args, train_dataset, test_dataset_mask, remove_rate)
        print("*** nepes dataset masking ready ***")
    else:
        mask = masking(args, train_dataset, test_dataset_mask, remove_rate)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler 
    #distributedsampler: batch dataset을 core만큼 나눔
    clear_idx = np.where(mask)[0]
    logger.info("*** Masking Finished ***")
    logger.info(f"num of labeled samples: {len(clear_idx)}")
    logger.info(f"num of unlabeled samples: {len(mask)-len(clear_idx)}")

    labeled_idx = np.hstack([clear_idx for _ in range(7)])
    unlabeled_idx = np.array(range(len(mask)))

    #print("* Labeled Index Length: ", len(clear_idx), "*Expanded Index Length: ", len(labeled_idx))
    
    if args.dataset=="nepes":
        cropsize=400
        mean_ = (0.485, 0.456, 0.406)
        std_ = (0.229, 0.224, 0.225)
    else:
        cropsize=32
        mean_ = (0.4914, 0.4822, 0.4465)
        std_ = (0.2471, 0.2435, 0.2616)
            
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(cropsize,
                              padding=4,
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_, std=std_)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_, std=std_)
    ])
    if args.dataset == 'cifar10':
        labeled_dataset = CIFAR10SSL('./data', train_dataset, labeled_idx, train=True, transform = transform_labeled)
        unlabeled_dataset = CIFAR10SSL('./data', train_dataset, unlabeled_idx, train=True, transform = TransformFixMatch(cropsize, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)))
        
        test_dataset = datasets.CIFAR10(
            './data', train=False, transform=transform_val, download=False)
        correct_ac = labeled_dataset.correct_cnt / len(labeled_dataset.targets)
        logger.info(f"  Correct_Accuracy = {correct_ac}")

    if args.dataset == 'cifar100':
        labeled_dataset = CIFAR100SSL('./data', train_dataset, labeled_idx, train=True, transform = transform_labeled)
        unlabeled_dataset = CIFAR100SSL('./data', train_dataset, unlabeled_idx, train=True, transform = TransformFixMatch(cropsize, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)))
       
       #torchvision datasets https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py
        test_dataset = datasets.CIFAR100(
            './data', train=False, transform=transform_val, download=False)
        correct_ac = labeled_dataset.correct_cnt / len(labeled_dataset.targets)
        logger.info(f"  Correct_Accuracy = {correct_ac}")
    
    if args.dataset == 'nepes':
        labeled_dataset = Nepes_SSL(args.path, train_dataset, labeled_idx, train=True, transform = transform_labeled, log=logger)
        unlabeled_dataset = Nepes_SSL(args.path, train_dataset, unlabeled_idx, train=True, transform = TransformFixMatch(cropsize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), log=logger)
        test_dataset = test_dataset_mask

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu, #mu:7(default)
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)
    """
    if args.pretrain:
        if args.arch == 'wideresnet':
            pretrain = models.resnet50(pretrained=True)
        else:
            pretrain = models.resnext50_32x4d(pretrained=True)

        fc_in = model.fc.in_features
        pretrain.fc = nn.Linear(fc_in, args.num_classes)
        try:
            model.load_state_dict(pretrain.state_dict(), strict=False)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
    """

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)


    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    #args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    #logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()

    """Start Training"""
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)

def masking(args, train_dataset, test_dataset, remove_rate):
    network = ResNet50(input_channel=3, n_outputs = args.num_classes).cuda()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=64,
                                                    num_workers=args.num_workers,
                                                    shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=128,
        num_workers=args.num_workers, shuffle=False, pin_memory=False)

    optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()

    noise_or_not = train_dataset.noise_or_not
    moving_loss_dic = np.zeros_like(noise_or_not)
    ndata = train_dataset.__len__()
    best_mask_acc = 0

    for epoch in range(1, args.mask_epoch):
        # train models
        globals_loss = 0
        network.train()
        with torch.no_grad():
            accuracy = evaluate(test_loader, network)
        example_loss = np.zeros_like(noise_or_not, dtype=float)

        # Learning Rate Scheduling
        t = (epoch % 10 + 1) / float(10)  # 40: lr frequency
        lr = (1 - t) * 0.01 + t * 0.001  # default _ 0.01: max, 0.001: min lr

        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr

        for i, (images, labels, indexes) in enumerate(train_loader):

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logits = network(images)
            loss_1 = criterion(logits, labels)

            for pi, cl in zip(indexes, loss_1):
                example_loss[pi] = cl.cpu().data.item()

            globals_loss += loss_1.sum().cpu().data.item()

            loss_1 = loss_1.mean()
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()  # training in an epoch finish

        example_loss = example_loss - example_loss.mean()
        moving_loss_dic = moving_loss_dic + example_loss  # moving_loss_dic: ndarray, (50000,0)

        ind_1_sorted = np.argsort(moving_loss_dic)  # moving_loss_dic를 오름차순 정렬하는 인덱스의 array 반환.
        loss_1_sorted = moving_loss_dic[ind_1_sorted]

        remember_rate = 1 - remove_rate  # 남길 데이터 비율
        num_remember = int(remember_rate * len(loss_1_sorted))  # num_remember: 40000 @ remove_rate=0.2

        noise_accuracy = np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(
            len(loss_1_sorted) - num_remember)  # 제거할 데이터 중 노이즈 개수 / 총 노이즈 개수
        mask = np.ones_like(noise_or_not, dtype=np.float32)
        mask[ind_1_sorted[num_remember:]] = 0  # 지워야 할 인덱스에 대해 0 저장. mask[idx]=0

        correct_acc = np.sum(np.logical_and(mask, noise_or_not)) / (np.sum(mask))
        if correct_acc>best_mask_acc:
            best_mask_acc = correct_acc
            best_mask = mask

        top_accuracy_rm = int(0.9 * len(loss_1_sorted))
        top_accuracy = 1 - np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(
            len(loss_1_sorted) - top_accuracy_rm)

        print("Masking - " + "epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata,
              "test_accuarcy:%f" % accuracy, "!!noise_accuracy:%f" % (correct_acc),
              "!! top 0.1 noise accuracy:%f" % top_accuracy)
        logger.info('epoch: {:d}'.format(epoch))
        logger.info('noise accuracy: {:.2f}'.format(correct_acc))
        logger.info('test accuracy: {:.2f}'.format(accuracy))

    return best_mask

def masking_nepes(args, train_dataset, test_dataset, remove_rate):

    #network = ResNet50(input_channel=3, n_outputs = args.num_classes).cuda()
    #모델수정
    network = build_resnet('resnet50','fanin')
    if network.fc.out_features != args.num_classes:
            fc_in = network.fc.in_features
            network.fc = nn.Linear(fc_in, args.num_classes)
            network.fc.reset_parameters()
    network = network.cuda()
    #끝

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=4,
                                                    num_workers=args.num_workers,
                                                    shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=32,
        num_workers=args.num_workers, shuffle=False, pin_memory=False)

    optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()

    noise_or_not = len(train_dataset)
    logger.info(f"train dataset 크기: {noise_or_not}")
    moving_loss_dic = np.zeros(noise_or_not)
    ndata = train_dataset.__len__()
    #best_mask_acc = 0
    

    for epoch in range(1, args.mask_epoch):
        # train models
        globals_loss = 0
        network.train()
        with torch.no_grad():
            accuracy = evaluate_nepes(test_loader, network)
        example_loss = np.zeros([noise_or_not], dtype=float) #수정

        # Learning Rate Scheduling
        t = (epoch % 10 + 1) / float(10)  # 40: lr frequency
        lr = (1 - t) * 0.01 + t * 0.001  # default _ 0.01: max, 0.001: min lr

        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr

        for i, (images, labels, indexes) in enumerate(train_loader):

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logits = network(images)
            loss_1 = criterion(logits, labels)

            for pi, cl in zip(indexes, loss_1):
                
                example_loss[pi] = cl.cpu().data.item()

            globals_loss += loss_1.sum().cpu().data.item()

            loss_1 = loss_1.mean()
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()  # training in an epoch finish

        example_loss = example_loss - example_loss.mean()
        moving_loss_dic = moving_loss_dic + example_loss  # moving_loss_dic: ndarray, (50000,0)
        
        print("Masking - " + "epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata,
              "test_accuarcy:%f" % accuracy)
        logger.info('epoch: {:d}'.format(epoch))
        logger.info('test accuracy: {:.2f}'.format(accuracy))
        
    ind_1_sorted = np.argsort(moving_loss_dic)  # moving_loss_dic를 오름차순 정렬하는 인덱스의 array 반환.
    loss_1_sorted = moving_loss_dic[ind_1_sorted]

    remember_rate = 1 - remove_rate  # 남길 데이터 비율
    num_remember = int(remember_rate * len(loss_1_sorted))  # num_remember: 40000 @ remove_rate=0.2

    mask = np.ones(noise_or_not, dtype=np.float32)
    mask[ind_1_sorted[num_remember:]] = 0  # 지워야 할 인덱스에 대해 0 저장. mask[idx]=0
    
    return mask


def masking_nepes_uniform(args, train_dataset, test_dataset, remove_rate):

    network = build_resnet('resnet50','fanin')
    if network.fc.out_features != args.num_classes:
            fc_in = network.fc.in_features
            network.fc = nn.Linear(fc_in, args.num_classes)
            network.fc.reset_parameters()
    network = network.cuda()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=4,
                                                    num_workers=args.num_workers,
                                                    shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=32,
        num_workers=args.num_workers, shuffle=False, pin_memory=False)

    optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()

    train_len = len(train_dataset)
    logger.info(f"train dataset 크기: {train_len}")
    moving_loss_dic = np.zeros(train_len)
    ndata = train_dataset.__len__()
    
    for epoch in range(1, args.mask_epoch):
        # train models
        globals_loss = 0
        network.train()
        with torch.no_grad():
            accuracy = evaluate_nepes(test_loader, network)
        example_loss = np.zeros([train_len], dtype=float) #수정

        # Learning Rate Scheduling
        t = (epoch % 10 + 1) / float(10)  # 40: lr frequency
        lr = (1 - t) * 0.01 + t * 0.001  # default _ 0.01: max, 0.001: min lr

        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr

        for i, (images, labels, indexes) in enumerate(train_loader):

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logits = network(images)
            loss_1 = criterion(logits, labels)

            for pi, cl in zip(indexes, loss_1):
                
                example_loss[pi] = cl.cpu().data.item()

            globals_loss += loss_1.sum().cpu().data.item()

            loss_1 = loss_1.mean()
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()  # training in an epoch finish

        example_loss = example_loss - example_loss.mean()
        moving_loss_dic = moving_loss_dic + example_loss  # moving_loss_dic: ndarray, (50000,0)
        
        print("Masking - " + "epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata,
              "test_accuarcy:%f" % accuracy)
        logger.info('epoch: {:d}'.format(epoch))
        logger.info('test accuracy: {:.2f}'.format(accuracy))

    ind_1_sorted = np.argsort(moving_loss_dic)  # moving_loss_dic를 *오름차순* 정렬하는 인덱스의 array 반환.

    remember_rate = 1 - remove_rate  # 남길 데이터 비율
    classcnt = train_dataset.classcnt
    class_remember = (np.array(classcnt)*remember_rate).round(0) #클래스별 남길 데이터 개수 저장한 리스트
    indlist = []
    for i in ind_1_sorted:
        _, c, _ = train_dataset[i]
        if class_remember[c]>0:
            class_remember[c]-=1
            indlist.append(i)
        if sum(class_remember)==0:
            break
    
    mask = np.zeros(train_len, dtype=np.float32)
    mask[indlist] = 1  # 남길 인덱스에 대해 1 저장

    #       end of masking_nepes_uniform    
    return mask


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()
    model_ema = ema_model.ema
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    criterion = nn.CrossEntropyLoss().cuda()
    
    model.train()
    for epoch in range(args.start_epoch, args.epochs): #수정
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        
        #with torch.autocast(device_type='cuda', dtype=torch.float16):
        for batch_idx in range(args.eval_step):
            """Training Data Setting"""
            #labeled data
            try:
                inputs_x, targets_x = labeled_iter.next()
            except: #dataloader 모두 순회한 경우, epoch+=1
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            #unlabeled data
            try:
                (inputs_u_w, inputs_u_w2, inputs_u_s), _ = unlabeled_iter.next()
            except: #dataloader 모두 순회한 경우, epoch+=1
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_w2, inputs_u_s), _ = unlabeled_iter.next() #weak, strong - return only image, not target


            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]

            mul=3*args.mu+1 #inputs_u_w2까지 쓰는 경우 *3

            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_w2, inputs_u_s)), mul).to(args.device) #torch.cat(cifar!): torch.Size([960, 3, 32, 32])
                #torch.cat(nepes!): torch.Size([704, 3, 400, 400])

            targets_x = targets_x.to(args.device)

            

            """Start Training"""
            logits = model(inputs)

            logits = de_interleave(logits, mul)
            #logits_x, logits_u_w, logits_u_w2, logits_u_s: 각 labeled, unlabeled_weak, unlabeled_strong에 대한 logits
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_w2, logits_u_s = logits[batch_size:].chunk(3)
            del logits

            """EMA logits - labeled data"""
            logits_ema = model_ema(inputs)
            logits_ema = de_interleave(logits_ema, mul)

            logits_x_ema = logits_ema[:batch_size]
            logits_u_w_ema, logits_u_w2_ema, logits_u_s_ema = logits_ema[batch_size:].chunk(3)
            del logits_ema

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            if args.ema_ensemble:
                logits_u_w = (logits_u_w + logits_u_w2 + logits_u_w_ema + logits_u_w2_ema)/4
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            
            #torch.max: return values, indices
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)

            #ge: check if >=
            mask = max_probs.ge(args.threshold).float() #pseudo-labeling

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                reduction='none') * mask).mean()
            
            """Final loss"""
            lambda_ = args.lambda_u
            loss = Lx + lambda_ * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
        
                p_bar.update()
        #
        #logger.info(trainloss:losses.avg)

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        """test"""
        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("epoch: {:d}".format(epoch+1))
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


class Logger(object):
    def __init__(self, dir, args):
        td = datetime.datetime.now(timezone('Asia/Seoul'))
        time_now = td.strftime('%m-%d_%H.%M')
        
        file_name = time_now + ".log"
        self.terminal = sys.stdout
        self.log = open(dir + "/" + file_name, "a")
        
    def write(self, temp):
        self.terminal.write(temp)
        self.log.write(temp)
	
    def flush(self):
        pass


if __name__ == '__main__':
    main()
