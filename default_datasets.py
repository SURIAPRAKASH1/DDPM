import torch
import torchvision
from torchvision.transforms import v2

from typing import Literal, Iterable, Union

def normalize_img(t):
    return (t / 255) * 2 - 1

def preprocessing_pipeline(image_height: int, image_width: int):

    "Image preprocessing pipeline"

    transform = v2.Compose([
            v2.ToImage(),                                                     # convert PIL image/array to tensor
            v2.Resize(size = (image_height, image_width), antialias= True),      # v2.Resize() requires input to be tensor
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32),
            v2.Lambda(normalize_img),
            # v2.Lambda(lambda t: (t / 255 ) * 2 - 1)        # this way of nested lambda function will create can't pickle error in DDP env
        ])

    return transform


def reverseprocessing_pipeline():

    " reverse pipline for normalized/transformed images to its original format"

    reverse_transform = v2.Compose([
        v2.ToImage(),
        v2.Lambda(lambda t: (t + 1)/ 2 * 255),                           # convert to original pixel value
        v2.Lambda(lambda t: t.permute(1, 2, 0) if t.size(0) > 4 else t ),
        v2.ToDtype(torch.uint8),                                         # carefull with dtype
        v2.ToPILImage()                                                  # convert to PIL image
    ])

    return reverse_transform


def prepare_dataset(
        dataset_name: Literal["FashionMNIST", "CIFAR10", "CelebA"], 
        transform: v2.Compose, 
        ):

    if dataset_name == "FashionMNIST":
        # Fashion Mnist dataset
        fmnist_root = "github.com/zalandoresearch/fashion-mnist/blob/master/data/"
        dataset = torchvision.datasets.FashionMNIST(fmnist_root, train= True, download= True, transform= transform)
        
    elif dataset_name == "CIFAR10":
        # CIFAR10 dataset
        cifar10_root = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        dataset = torchvision.datasets.CIFAR10(cifar10_root, train= True, download= True, transform= transform)
    
    elif dataset_name == "CelebA":
        # CelebA dataset
        celebA_root = "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing"
        dataset = torchvision.datasets.CelebA(celebA_root, split = 'train', target_type = 'attr' ,download = True)
    else:
        raise NotImplementedError(dataset_name)
        
    return dataset
    
