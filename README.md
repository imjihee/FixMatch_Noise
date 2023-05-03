# FixMatch
This is an unofficial PyTorch implementation of [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685).
The official Tensorflow implementation is [here](https://github.com/google-research/fixmatch).

This code is only available in FixMatch (RandAugment).

## Usage

### Train
Train the model by 4000 labeled data of CIFAR-10 dataset:

```
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5
```

Train the model by 10000 labeled data of CIFAR-100 dataset by using DistributedDataParallel:
```
python -m torch.distributed.launch --nproc_per_node 4 ./train.py --dataset cifar100 --num-labeled 10000 --arch wideresnet --batch-size 16 --lr 0.03 --wdecay 0.001 --expand-labels --seed 5 --out results/cifar100@10000
```

(my)Train the model by 4000 labeled data of CIFAR-10 dataset by using DistributedDataParallel with 2GPUS:
```
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node 2 ./train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5

CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset cifar100 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5 --lambda-u 1.2 --T 0.5 --mask_epoch 2 --noise_rate 0.4 --use-ema --ema_ensemble

```

### Monitoring training progress
```
tensorboard --logdir=<your out_dir>
```

## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)

## My other implementations
- [Meta Pseudo Labels](https://github.com/kekmodel/MPL-pytorch)
- [UDA for images](https://github.com/kekmodel/UDA-pytorch)


## References
- [Official TensorFlow implementation of FixMatch](https://github.com/google-research/fixmatch)
- [Unofficial PyTorch implementation of MixMatch](https://github.com/YU1ut/MixMatch-pytorch)
- [Unofficial PyTorch Reimplementation of RandAugment](https://github.com/ildoonet/pytorch-randaugment)
- [PyTorch image models](https://github.com/rwightman/pytorch-image-models)

## Citations
```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```
