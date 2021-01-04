"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler



def fetch_dataloader(trainset,shuffle=False,params=None):
    np.random.seed(230)

    train_sampler=None 
    if hasattr(params,"params") and params.subset_percent>0 :
        trainset_size = len(trainset)
        indices = list(range(trainset_size))
        split = int(np.floor(params.subset_percent * trainset_size))
        np.random.shuffle(indices)
    
        train_sampler = SubsetRandomSampler(indices[:split])


    trainset_dataloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
            sampler=train_sampler,
        shuffle=shuffle, num_workers=params.num_workers, pin_memory=torch.cuda.is_available())

    return trainset_dataloader 

def transform_cifar(params_augmentation=None,is_aug=True):
    
    # using random crops and horizontal flip for train set
    if is_aug:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])


    return train_transformer,dev_transformer

def transform_cifar_non_normalize(is_aug):
    
    # using random crops and horizontal flip for train set
    if is_aug:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            ])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            ])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        ])


    return train_transformer,dev_transformer

def transform_imagenet(is_aug=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    # using random crops and horizontal flip for train set
    if is_aug:
        train_transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            normalize,
            ])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])


    return train_transformer,dev_transformer

class WarpCifar10(torchvision.datasets.cifar.CIFAR10):
    def __getitem__(self,x):
        item=super().__getitem__(x)
        return {
            "A_input":item[0],
            "A_input_lbl":item[1],
            "A_input_ids":x,
            }
#     def __len__(self):
#         return 1000
    
class WarpCifar100(torchvision.datasets.cifar.CIFAR100):
    def __getitem__(self,x):
        item=super().__getitem__(x)
        return {
            "A_input":item[0],
            "A_input_lbl":item[1],
            "A_input_ids":x,
            }

def create_dataset(params=None,is_train=True,is_aug=True,is_shuffle=True ):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    
    transform_name= params.transform_name
    dataset_name = params.dataset_name
    is_train= params.is_train & is_train
    if hasattr(params,"augmentation"):
        is_aug= is_aug & params.augmentation=="yes"
        
    is_aug = is_aug & is_train 
    is_shuffle = is_train & is_shuffle
    '''
    transform_name in [transform_cifar, transform_cifar_v2,transform_imagenet]
    is_train : true/false
    dataset_name in [cifar10,cifar100,imagenet]
    '''
    
    train_tf ,val_tf=None,None,
    if transform_name=="transform_cifar":
        train_tf ,val_tf = transform_cifar(params,is_aug)
    elif transform_name=="transform_cifar_non_normalize":
        train_tf ,val_tf = transform_cifar_non_normalize(is_aug)
    elif transform_name=="transform_imagenet":
        train_tf ,val_tf = transform_imagenet(is_aug)
    else:
        raise Exception(f"unknown transform, transform_name={transform_name}")

    if dataset_name=="cifar10":
        trans = train_tf if is_train else val_tf
        dataset = WarpCifar10(root="~/.torch",transform=trans,train=is_train)
#         dataset = torchvision.datasets.cifar.CIFAR10(
#             root="~/.torch",transform=trans)
    
    if dataset_name=="cifar100":
        trans = train_tf if is_train else val_tf
        dataset = WarpCifar100(root="~/.torch",transform=trans,train=is_train)
#         dataset = torchvision.datasets.cifar.CIFAR100(
#             root="~/.torch",transform=trans)
    
    if dataset_name=="imagenet":
        raise Exception("wait for implementation")
    
    print (dataset_name,"::","is_train",is_train,"is_aug",is_aug,"is_shuffle",is_shuffle)
    dl= fetch_dataloader(trainset=dataset,shuffle=is_shuffle,params=params)

    return dl


if __name__=="__main__":
    val_dic = {
        "batch_size":64,
        "num_workers":4,
        "cuda":True,
        }
    import itertools
    from collections import namedtuple
    
    d1=["cifar10","cifar100"]
    d2=["transform_cifar","transform_cifar_v2"]
    d3=[True,False]
    
    for dd1,dd2,dd3 in list(itertools.product(d1,d2,d3)):
        val_dic.update({
            "dataset_name":dd1,
            "transform_name":dd2,
            "is_train":dd3,
            })
        params = namedtuple('Struct', val_dic.keys())(*val_dic.values())

        print (params)
        dl=create_dataset(params)
        data = next(iter(dl))
        print (type(data))