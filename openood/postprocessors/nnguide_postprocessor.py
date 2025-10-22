from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.special import logsumexp
from copy import deepcopy
from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    

class NNGuidePostprocessor(BasePostprocessor):

    def __init__(self, config):
        super(NNGuidePostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.alpha = self.args.alpha
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            bank_feas = []
            bank_logits = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    logit, feature = net(data, return_feature=True)
                    bank_feas.append(normalizer(feature.data.cpu().numpy()))
                    bank_logits.append(logit.data.cpu().numpy())
                    
                    if len(bank_feas
                           ) * id_loader_dict['train'].batch_size > int(
                               len(id_loader_dict['train'].dataset) *
                               self.alpha):
                        break
            self.all_bank_feas = np.concatenate(bank_feas, axis=0)
            self.all_bank_logits = np.concatenate(bank_logits, axis=0)

            self.update_internals()

            self.setup_flag = True
        else:
            pass

    def update_internals(self):
        num_samples = int(len(self.all_bank_feas) * self.alpha)
        indices = np.random.choice(len(self.all_bank_feas), num_samples, replace=False)
        
        self.bank_confs = logsumexp(self.all_bank_logits[indices], axis=-1)
        bank_guide = self.all_bank_feas[indices] * self.bank_confs[:, None]
        self.index = faiss.IndexFlatIP(bank_guide.shape[-1])
        self.index.add(bank_guide)
    
    def knn_score(self, queryfeas, min=False):
        D, indices = self.index.search(queryfeas, self.K)
        valid_mask = indices != -1
        # Create a mask for valid distances
        valid_distances = np.where(valid_mask, D, np.nan)
        
        if min:
            scores = np.nanmin(valid_distances, axis=1)
        else:
            scores = np.nanmean(valid_distances, axis=1)
        
        return scores

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logit, feature = net(data, return_feature=True)
        feas_norm = normalizer(feature.data.cpu().numpy())
        energy = logsumexp(logit.data.cpu().numpy(), axis=-1)
    
        conf = self.knn_score(feas_norm)
        score = conf * energy

        _, pred = torch.max(torch.softmax(logit, dim=1), dim=1)
        return pred, torch.from_numpy(score)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]
        self.alpha = hyperparam[1]
        self.update_internals()

    def get_hyperparam(self):
        return [self.K, self.alpha]
