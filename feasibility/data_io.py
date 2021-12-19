import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms as T

from dataset import SeaShips, Cats
from collate import default_collate as collate_fn

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
normalize = T.Normalize(mean, std)
inv_normalize = T.Normalize(
        mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
        std=[1/std[0], 1/std[1], 1/std[2]]
    )
def get_augmentations(image_size):
    return albu.ReplayCompose([
        albu.Resize(width=image_size, height=image_size),
        albu.RandomBrightnessContrast(p=0.2),
        albu.Normalize(mean=mean,std=std),
        ToTensorV2()
    ], bbox_params=albu.BboxParams(format='pascal_voc'))

def trainval_dataset(dataset, data_dir, batch_size=16, image_size=224):
    augmentations = get_augmentations(image_size)
    dset = get_dataset(dataset, data_dir, augmentations, True)
    
    # Creating data indices for training and validation splits:
    validation_split=0.30
    dataset_size = len(dset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, 
                                               collate_fn=collate_fn,
                                               sampler=train_sampler, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                                   collate_fn=collate_fn,
                                                   sampler=val_sampler, num_workers=0)

    return train_loader, val_loader

def get_dataset(dataset, data_dir, augmentations, train):
    if dataset == "cats":
        dset =  Cats(root=data_dir, 
                     train=False,
                     augmentations=augmentations)
    elif dataset == "cats-draw":
        dset =  CatsDraw(root=data_dir, 
                           train=True,
                           augmentations=augmentations)
    elif dataset == "seaships":
        dset =  SeaShips(root=data_dir, 
                         train=False,
                         augmentations=augmentations)
    else:
        raise NotImplemented(f"`{dataset}` is not implemented")

    return dset

def test_dataset(dataset, data_dir, batch_size=16, image_size=224):
    augmentations = get_augmentations(image_size)
    dset = get_dataset(dataset, data_dir, augmentations, False)

    sampler = SequentialSampler(dset)
    test_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, 
                                               collate_fn=collate_fn,
                                               num_workers=0, sampler=sampler)

    return test_loader
