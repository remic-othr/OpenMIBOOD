from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm
import random
import numpy as np


class BasePostprocessor:

    def set_seed(self, seed, fast_mode=False):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        print(f'Setting seed to {seed}')
        
        torch.backends.cudnn.deterministic = not fast_mode
        torch.backends.cudnn.benchmark = fast_mode    
        torch.backends.cuda.matmul.allow_tf32 = fast_mode
        torch.backends.cudnn.allow_tf32 = fast_mode

    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
