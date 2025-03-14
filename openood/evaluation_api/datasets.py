import os
import gdown
import zipfile

from torch.utils.data import DataLoader
import torchvision as tvs
if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors import BasePreprocessor

from .preprocessor import get_default_preprocessor, ImageNetCPreProcessor

DATA_INFO = {
<<<<<<< HEAD
    'midog': {
        'num_classes': 3,
        'id': {
            'train': {
                'data_dir': 'midog/',
                'imglist_path': 'benchmark_imglist/midog/train_midog.txt' # 1a
            },
            'val': {
                'data_dir': 'midog/',
                'imglist_path': 'benchmark_imglist/midog/valid_midog.txt' # 1a
            },
            'test': {
                'data_dir': 'midog/',
                'imglist_path': 'benchmark_imglist/midog/test_midog.txt' # 1a
            }
        },
        'csid': {
            'datasets': ['midog_csid_1b', 'midog_csid_1c'],
            'midog_train': {
                'data_dir': 'midog/',
                'imglist_path': 'benchmark_imglist/midog/train_midog.txt' 
            },
            'midog_csid_1b': {
                'data_dir': 'midog/',
                'imglist_path': 'benchmark_imglist/midog/test_midog_1b.txt' 
            },
            'midog_csid_1c': {
                'data_dir': 'midog/',
                'imglist_path': 'benchmark_imglist/midog/test_midog_1c.txt' 
            }
        },
        'ood': {
            'val': {
                'data_dir': 'midog/',
                'imglist_path': 'benchmark_imglist/midog/valid_midog_near.txt'
            },
            'near': {
                'datasets': ['midog_2', 'midog_3', 'midog_4', 'midog_5', 'midog_6a', 'midog_6b', 'midog_7'],
                'midog_2': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_2.txt' 
                },
                'midog_3': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_3.txt' 
                },
                'midog_4': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_4.txt' 
                },
                'midog_5': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_5.txt' 
                },
                'midog_6a': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_6a.txt'
                },
                'midog_6b': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_6b.txt'
                },
                'midog_7': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_7.txt' 
                }
            },
            'far': {
                'datasets': ['midog_ccagt', 'midog_fnac2019'],
                'midog_ccagt': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_ccagt.txt'
                },
                'midog_fnac2019': {
                    'data_dir': 'midog/',
                    'imglist_path': 'benchmark_imglist/midog/test_midog_fnac2019.txt'
                }
            }
        }
    },
    'phakir': {
        'num_classes': 7,
        'id': {
            'train': {
                'data_dir': 'phakir/',
                'imglist_path': 'benchmark_imglist/phakir/train_phakir.txt' # Video_02, Video_03, Video_04, Video_07
            },
            'val': {
                'data_dir': 'phakir/',
                'imglist_path': 'benchmark_imglist/phakir/valid_phakir.txt' # Video_05
            },
            'test': {
                'data_dir': 'phakir/',
                'imglist_path': 'benchmark_imglist/phakir/test_phakir.txt' # Video_01
            }
        },
        'csid': {
            'datasets': ['phakir_medium_smoke', 'phakir_heavy_smoke'],
            'phakir_medium_smoke': {
                'data_dir': 'phakir/',
                'imglist_path': 'benchmark_imglist/phakir/test_phakir_medium_smoke_csid.txt' 
            },
            'phakir_heavy_smoke': {
                'data_dir': 'phakir/',
                'imglist_path': 'benchmark_imglist/phakir/test_phakir_heavy_smoke_csid.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'phakir/',
                'imglist_path': 'benchmark_imglist/phakir/valid_phakir_near.txt'
            },
            'near': {
                'datasets': ['phakir_cholec', 'phakir_endovis2015', 'phakir_endovis2018'],
                'phakir_cholec': {
                    'data_dir': 'phakir/',
                    'imglist_path': 'benchmark_imglist/phakir/test_phakir_cholec_near.txt' # Cholec80 Near
                },
                'phakir_endovis2015': {
                    'data_dir': 'phakir/',
                    'imglist_path': 'benchmark_imglist/phakir/test_phakir_endovis2015_near.txt' # Endovis2015 Near
                },
                'phakir_endovis2018': {
                    'data_dir': 'phakir/',
                    'imglist_path': 'benchmark_imglist/phakir/test_phakir_endovis2018_near.txt' # Endovis2018 Near
                }
            },
            'far': {
                'datasets': ['phakir_kvasir', 'phakir_cataracts'],
                'phakir_kvasir': {
                    'data_dir': 'phakir/',
                    'imglist_path': 'benchmark_imglist/phakir/test_phakir_kvasir_far.txt' # Kvasir-Seg Far
                },
                'phakir_barret': {
                    'data_dir': 'phakir/',
                    'imglist_path': 'benchmark_imglist/phakir/test_phakir_barret_far.txt' # Barret MICCAI2015 Far
                },
                'phakir_cataracts': {
                    'data_dir': 'phakir/',
                    'imglist_path': 'benchmark_imglist/phakir/test_phakir_cataracts_far.txt' # CATARACTS Far
                }
            }
        }
    },
    'oasis3': {
        'num_classes': 2,
        'id': {
            'train': {
                'data_dir': 'oasis/',
                'imglist_path': 'benchmark_imglist/oasis3/train_oasis3.txt' # T1w OASIS3 data
            },
            'val': {
                'data_dir': 'oasis/',
                'imglist_path': 'benchmark_imglist/oasis3/valid_oasis3.txt' # T1w OASIS3 data
            },
            'test': {
                'data_dir': 'oasis/',
                'imglist_path': 'benchmark_imglist/oasis3/test_oasis3.txt' # T1w OASIS3 data
            }
        },
        'csid': {
            'datasets': ['oasis3_scanner', 'oasis3_modality'],
            'oasis3_scanner': {
                'data_dir': 'oasis/',
                'imglist_path': 'benchmark_imglist/oasis3/test_oasis3_scanner_csid.txt' # T1w OASIS3 data from 'Vision' MRI scanner
            },
            'oasis3_modality': {
                'data_dir': 'oasis/',
                'imglist_path': 'benchmark_imglist/oasis3/test_oasis3_t2w_csid.txt' # T2w OASIS3 data
            }
        },
        'ood': {
            'val': {
                'data_dir': 'oasis/',
                'imglist_path': 'benchmark_imglist/oasis3/valid_oasis3_near.txt'
            },
            'near': {
                'datasets': ['oasis3_atlas', 'oasis3_brats', 'oasis3_ct'],
                'oasis3_atlas': {
                    'data_dir': 'oasis/',
                    'imglist_path': 'benchmark_imglist/oasis3/test_oasis3_atlas_near.txt' # ATLAS T1w data
                },
                'oasis3_brats': {
                    'data_dir': 'oasis/',
                    'imglist_path': 'benchmark_imglist/oasis3/test_oasis3_brats_near.txt' # BraTS T1w data
                },
                'oasis3_ct': {
                    'data_dir': 'oasis/',
                    'imglist_path': 'benchmark_imglist/oasis3/test_oasis3_ct_near.txt' # OASIS3 CT data
                }
            },
            'far': {
                'datasets': ['oasis3_heart', 'oasis3_chaos_inPhase'],
                'oasis3_heart': {
                    'data_dir': 'oasis/',
                    'imglist_path': 'benchmark_imglist/oasis3/test_oasis3_heart_far.txt' # Heart MRI data
                },
                'oasis3_chaos_inPhase': {
                    'data_dir': 'oasis/',
                    'imglist_path': 'benchmark_imglist/oasis3/test_oasis3_chaos_inPhase_far.txt' # CHAOS MRI data
                }
            }
        }
    },
=======
>>>>>>> origin/main
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {
            'datasets': ['cifar10c'],
            'cinic10': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cinic10.txt'
            },
            'cifar10c': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10c.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_places365.txt'
                },
            }
        }
    },
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                    'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets':
            ['imagenet_v2', 'imagenet_c', 'imagenet_r', 'imagenet_es'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
            'imagenet_es': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_es.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
}

download_id_dict = {
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'imagenet_es': '1ATz11vKmPqyzfEaEDRaPTF9TXiC244sw',
    'benchmark_imglist': '1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1'
}

dir_dict = {
    'images_classic/': [
        'cifar100', 'tin', 'tin597', 'svhn', 'cinic10', 'imagenet10', 'mnist',
        'fashionmnist', 'cifar10', 'cifar100c', 'places365', 'cifar10c',
        'fractals_and_fvis', 'usps', 'texture', 'notmnist'
    ],
    'images_largescale/': [
        'imagenet_1k',
        'ssb_hard',
        'ninco',
        'inaturalist',
        'places',
        'sun',
        'openimage_o',
        'imagenet_v2',
        'imagenet_c',
        'imagenet_r',
        'imagenet_es',
    ],
    'images_medical/': ['actmed', 'bimcv', 'ct', 'hannover', 'xraybone'],
}

benchmarks_dict = {
<<<<<<< HEAD
    'oasis3': [],
    'phakir': [],
    'midog': [],
=======
>>>>>>> origin/main
    'cifar10':
    ['cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'cifar100':
    ['cifar100', 'cifar10', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'imagenet200': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
    'imagenet': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r', 'imagenet_es'
    ],
}


def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, data_root):
    for key in dir_dict.keys():
        if dataset in dir_dict[key]:
            store_path = os.path.join(data_root, key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print('Invalid dataset detected {}'.format(dataset))
        return

    if require_download(dataset, store_path):
        print(store_path)
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + '.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


def data_setup(data_root, id_data_name):
    if not data_root.endswith('/'):
        data_root = data_root + '/'

    if not os.path.exists(os.path.join(data_root, 'benchmark_imglist')):
        gdown.download(id=download_id_dict['benchmark_imglist'],
                       output=data_root)
        file_path = os.path.join(data_root, 'benchmark_imglist.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(data_root)
        os.remove(file_path)

    for dataset in benchmarks_dict[id_data_name]:
        download_dataset(dataset, data_root)


def get_id_ood_dataloader(id_name, data_root, preprocessor, **loader_kwargs):
    if 'imagenet' in id_name:
        if tvs_new:
            if isinstance(preprocessor,
                          tvs.transforms._presets.ImageClassification):
                mean, std = preprocessor.mean, preprocessor.std
            elif isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        else:
            if isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        imagenet_c_preprocessor = ImageNetCPreProcessor(mean, std)

    # weak augmentation for data_aux
    test_standard_preprocessor = get_default_preprocessor(id_name)

    dataloader_dict = {}
    data_info = DATA_INFO[id_name]

    # id
    sub_dataloader_dict = {}
    for split in data_info['id'].keys():
        dataset = ImglistDataset(
            name='_'.join((id_name, split)),
            imglist_pth=os.path.join(data_root,
                                     data_info['id'][split]['imglist_path']),
            data_dir=os.path.join(data_root,
                                  data_info['id'][split]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[split] = dataloader
    dataloader_dict['id'] = sub_dataloader_dict

    # csid
    sub_dataloader_dict = {}
    for dataset_name in data_info['csid']['datasets']:
        dataset = ImglistDataset(
            name='_'.join((id_name, 'csid', dataset_name)),
            imglist_pth=os.path.join(
                data_root, data_info['csid'][dataset_name]['imglist_path']),
            data_dir=os.path.join(data_root,
                                  data_info['csid'][dataset_name]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor
            if dataset_name != 'imagenet_c' else imagenet_c_preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[dataset_name] = dataloader
    dataloader_dict['csid'] = sub_dataloader_dict

    # ood
    dataloader_dict['ood'] = {}
    for split in data_info['ood'].keys():
        split_config = data_info['ood'][split]

        if split == 'val':
            # validation set
            dataset = ImglistDataset(
                name='_'.join((id_name, 'ood', split)),
                imglist_pth=os.path.join(data_root,
                                         split_config['imglist_path']),
                data_dir=os.path.join(data_root, split_config['data_dir']),
                num_classes=data_info['num_classes'],
                preprocessor=preprocessor,
                data_aux_preprocessor=test_standard_preprocessor)
            dataloader = DataLoader(dataset, **loader_kwargs)
            dataloader_dict['ood'][split] = dataloader
        else:
            # dataloaders for nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config['datasets']:
                dataset_config = split_config[dataset_name]
                dataset = ImglistDataset(
                    name='_'.join((id_name, 'ood', dataset_name)),
                    imglist_pth=os.path.join(data_root,
                                             dataset_config['imglist_path']),
                    data_dir=os.path.join(data_root,
                                          dataset_config['data_dir']),
                    num_classes=data_info['num_classes'],
                    preprocessor=preprocessor,
                    data_aux_preprocessor=test_standard_preprocessor)
                dataloader = DataLoader(dataset, **loader_kwargs)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict['ood'][split] = sub_dataloader_dict

    return dataloader_dict
