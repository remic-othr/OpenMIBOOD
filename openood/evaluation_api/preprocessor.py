import torchvision.transforms as tvs_trans
import torch
import torchio as tio
from monai.transforms import CropForeground

from openood.preprocessors import BasePreprocessor
from openood.utils import Config

INTERPOLATION = tvs_trans.InterpolationMode.BILINEAR

default_preprocessing_dict = {
    'midog': {
        'pre_size': 50,
        'img_size': 50,
        'normalization': [[0.712, 0.496, 0.756], [0.167, 0.167, 0.110]],
    },
    'phakir': {
        'pre_size': (360, 640),
        'img_size': (360, 640),
        'normalization': [[0.517, 0.361, 0.336], [0.166, 0.143, 0.137]],
    },
    'oasis3': {
        'pre_size': 128,
        'img_size': 128,
        'normalization': 'z-normalize',
    },
    'cifar10': {
        'pre_size': 32,
        'img_size': 32,
        'normalization': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    },
    'cifar100': {
        'pre_size': 32,
        'img_size': 32,
        'normalization': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    },
    'imagenet': {
        'pre_size': 256,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'imagenet200': {
        'pre_size': 256,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'aircraft': {
        'pre_size': 512,
        'img_size': 448,
        'normalization': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    },
    'cub': {
        'pre_size': 512,
        'img_size': 448,
        'normalization': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    }
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class TestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""
    def __init__(self, config: Config):
        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(config.pre_size, interpolation=INTERPOLATION),
            tvs_trans.CenterCrop(config.img_size),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(*config.normalization),
        ])


class ImageNetCPreProcessor(BasePreprocessor):
    def __init__(self, mean, std):
        self.transform = tvs_trans.Compose([
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean, std),
        ])


def get_default_preprocessor(data_name: str):
    # TODO: include fine-grained datasets proposed in Vaze et al.?

    if data_name not in default_preprocessing_dict:
        raise NotImplementedError(f'The dataset {data_name} is not supported')

    config = Config(**default_preprocessing_dict[data_name])
    if data_name == 'oasis3':
        preprocessor = Oasis3PreProcessor(config.pre_size)
    else:
        preprocessor = TestStandardPreProcessor(config)

    return preprocessor

class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample.get_fdata())
    
class AddChannelDim:
    def __call__(self, sample, dim=0):
        return sample.unsqueeze(dim=dim)

class ToFloat:
    def __call__(self, sample):
        return sample.float()

def empty_threshold(x):
    return x > 0

class Oasis3PreProcessor(BasePreprocessor):
    def __init__(self, crop_dim):
        self.transform = tvs_trans.Compose([
            tio.ToCanonical(),
            ToTensor(),
            AddChannelDim(),
            CropForeground(select_fn=empty_threshold, margin=0,  k_divisible=[crop_dim, crop_dim, crop_dim]),
            tio.CropOrPad((crop_dim,crop_dim,crop_dim)),
            ToFloat(),
            tio.ZNormalization()
    ])
