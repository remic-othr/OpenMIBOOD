import collections
import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import argparse
import pickle

import torch
from torchvision import transforms as trn
import torchio as tio
from monai.transforms import CropForeground

from openood.evaluation_api import Evaluator
from openood.networks import r2plus1d_18

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


parser = argparse.ArgumentParser()
parser.add_argument('--postprocessor', choices=['ash', 'dice', 'dropout', 'ebo', 'fdbd', 'gen', 'klm', 'knn', 'mls', 'mds_ensemble', 'mds', 'nnguide', 'odin', 'msp', 'openmax', 'rankfeat', 'react', 'relation', 'residual', 'rmds', 'scale', 'she', 'temp_scaling', 'vim', 'msp'], default='msp')
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--batch-size', default=10, type=int)
args = parser.parse_args()

fsood = True

root = os.path.join(
    ROOT_DIR, 'results',
    f'oasis3')
if not os.path.exists(root):
    os.makedirs(root)

ckpt_path = os.path.join(root, 'oasis3_classifier.pth')
if not os.path.exists(ckpt_path):
    print('Please download the model checkpoint from the associated github release first.')
    exit()

# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor
# load pre-setup postprocessor if exists
if os.path.isfile(
        os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl')):
    with open(
            os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl'),
            'rb') as f:
        postprocessor = pickle.load(f)
else:
    postprocessor = None

# assuming the model is either
# 1) torchvision pre-trained; or
# 2) a specified checkpoint    
net = r2plus1d_18(num_classes=2)

ckpt = torch.load(ckpt_path, map_location='cpu')
net.load_state_dict(ckpt)

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

crop_dim = 128
preprocessor = trn.Compose([
    tio.ToCanonical(),
    ToTensor(),
    AddChannelDim(),
    CropForeground(select_fn=empty_threshold, margin=0,  k_divisible=[crop_dim, crop_dim, crop_dim]),
    tio.CropOrPad((crop_dim,crop_dim,crop_dim)),
    ToFloat(),
    tio.ZNormalization()
])
#NAECHSTE SCHRITTE: NEUEN CLEANEN CONTAINER ANLEGEN - IN DEM CONTAINER DANN OPENOOD INSTALLIEREN UND NOCHMAL LAUFEN LASSEN
net.cuda()
net.eval()
# a unified evaluator
evaluator = Evaluator(
    net,
    id_name='oasis3',  # the target ID dataset
    data_root=os.path.join(ROOT_DIR, 'data'),
    config_root=os.path.join(ROOT_DIR, 'configs'),
    preprocessor=preprocessor,  # default preprocessing
    postprocessor_name=postprocessor_name,
    postprocessor=postprocessor,
    batch_size=args.batch_size,  # for certain methods the results can be slightly affected by batch size
    shuffle=True,
    num_workers=8)

# load pre-computed scores if exists
if os.path.isfile(os.path.join(root, 'scores', f'{postprocessor_name}.pkl')):
    with open(os.path.join(root, 'scores', f'{postprocessor_name}.pkl'),
              'rb') as f:
        scores = pickle.load(f)
    update(evaluator.scores, scores)
    print('Loaded pre-computed scores from file.')

# save postprocessor for future reuse
if hasattr(evaluator.postprocessor, 'setup_flag'
           ) or evaluator.postprocessor.hyperparam_search_done is True:
    pp_save_root = os.path.join(root, 'postprocessors')
    if not os.path.exists(pp_save_root):
        os.makedirs(pp_save_root)

    if not os.path.isfile(
            os.path.join(pp_save_root, f'{postprocessor_name}.pkl')):
        with open(os.path.join(pp_save_root, f'{postprocessor_name}.pkl'),
                  'wb') as f:
            pickle.dump(evaluator.postprocessor, f, pickle.HIGHEST_PROTOCOL)

# the metrics is a dataframe
metrics = evaluator.eval_csid_ood()

# saving and recording
if args.save_csv:
    saving_root = os.path.join(root, 'ood' if not fsood else 'fsood')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)

    if not os.path.isfile(
            os.path.join(saving_root, f'{postprocessor_name}.csv')):
        metrics.to_csv(os.path.join(saving_root, f'{postprocessor_name}.csv'),
                       float_format='{:.2f}'.format)

if args.save_score:
    score_save_root = os.path.join(root, 'scores')
    if not os.path.exists(score_save_root):
        os.makedirs(score_save_root)
    with open(os.path.join(score_save_root, f'{postprocessor_name}.pkl'),
              'wb') as f:
        pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)
