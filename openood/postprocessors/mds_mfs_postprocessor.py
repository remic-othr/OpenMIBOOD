from __future__ import annotations

from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sklearn.covariance

from .distribution_utils import wasserstein_marginals_torch_cpu

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class MDS_MFSPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        # Hyperparam: fraction of dimensions to keep
        self.percentage = float(self.args.percentage)
        self.imagenet_subsample = 100000
        self.identify_dims = True

        # Cached selection scores (per dimension)
        self.global_scores: list[float] | None = None

        # Feature caches (CPU)
        self.all_feats: torch.Tensor | None = None      # [Ntrain, D] CPU
        self.all_labels: torch.Tensor | None = None     # [Ntrain] CPU

        self.ood_feats: torch.Tensor | None = None      # [Nproxy, D] CPU (proxy-OOD for selection)

        # Learned MDS parameters (in selected subspace)
        self.global_indices: np.ndarray | None = None
        self.global_class_mean: torch.Tensor | None = None   # [C, K] CPU
        self.global_precision: torch.Tensor | None = None    # [K, K] CPU

        # State
        self.setup_flag = False

    # --------------------------------------------------------
    # Hyperparam interface
    # --------------------------------------------------------

    def set_hyperparam(self, hyperparam: list):
        self.percentage = float(hyperparam[0])
        self.identify_dims = True
        self._update_internals()

    def get_hyperparam(self):
        return self.percentage

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        net.eval()
        print("\nEstimating mean and covariance from training set...")

        self._compute_and_cache_repr(net, id_loader_dict, ood_loader_dict)
        self._update_internals()

        self.setup_flag = True

    def _compute_and_cache_repr(self, net: nn.Module, id_loader_dict, ood_loader_dict) -> None:
        train_feats = []
        train_labels = []

        proxy_feats = []
        miuxp_alpha_pool = [0.25, 0.5, 0.75]

        # Build proxy pool (CPU) for mixup
        pool_loader = ood_loader_dict["val_proxy"]
        print(pool_loader.dataset)
        print(dir(pool_loader.dataset))
        proxy_pool = []
        with torch.no_grad():
            for batch in tqdm(pool_loader, desc="Setup (proxy pool): ", position=0, leave=True):
                proxy_pool.extend(batch["data"].float())
        proxy_pool = torch.stack(proxy_pool)  # CPU

        with torch.no_grad():
            self.set_seed(self.seed) # Main paper results were calculated with seeds 0, 1, 2, 3, 4, 5, 6, 7, 8, and 42.

            for batch in tqdm(id_loader_dict["train"], desc="Setup (ID train): ", position=0, leave=True):
                x = batch["data"].cuda()
                y = batch["label"]
                train_labels.append(deepcopy(y))

                _, feats = net(x, return_feature=True)
                train_feats.append(feats.cpu())

                b = x.size(0)
                idx = torch.randint(0, proxy_pool.size(0), (b,))
                mix = proxy_pool[idx].cuda()
                cur_mixup_alpha = np.random.choice(miuxp_alpha_pool, size=b)
                ref = [xi * alpha + mi * (1.0 - alpha) for xi, mi, alpha in zip(x, mix, cur_mixup_alpha)]
                _, f_ref = net(torch.stack(ref).cuda(), return_feature=True)
                proxy_feats.append(f_ref.cpu())

        self.all_feats = torch.cat(train_feats).contiguous()        # [Ntrain, D] CPU
        self.all_labels = torch.cat(train_labels)                   # [Ntrain] CPU

        self.ood_feats = torch.cat(proxy_feats).contiguous()  # CPU

    # --------------------------------------------------------
    # Marginal feature selection
    # --------------------------------------------------------
    def _compute_global_scores(self, id_n: torch.Tensor, ood_n: torch.Tensor) -> None:
        """
        Compute per-dimension selection scores (higher = "more OOD-shifted").
        """
        # self.set_seed(self.seed)
        scores = wasserstein_marginals_torch_cpu(id_n, ood_n)
        self.global_scores = scores.detach().cpu().numpy().tolist()
        return


    def identify_dimensions(self, id_data: torch.Tensor, ood_data: torch.Tensor, percentage: float) -> np.ndarray:
        id_s = id_data.detach().to(device="cpu", dtype=torch.float32)
        ood_s = ood_data.detach().to(device="cpu", dtype=torch.float32)
        dmin = torch.min(id_s, dim=0)[0]
        dmax = torch.max(id_s, dim=0)[0]
        denom = (dmax - dmin)
        assert torch.all(denom > 0), "Some feature dimensions have zero variance!"

        if self.global_scores is None:
            is_imagenet = "imagenet" in self.config.dataset.name.lower()
            if is_imagenet:
                self.set_seed(self.seed)
                Nx, Ny = id_s.shape[0], ood_s.shape[0]
                n = min(Nx, Ny, self.imagenet_subsample)
                if n > 1:
                    g = torch.Generator(device="cpu")
                    g.manual_seed(self.seed)
                    ix = torch.randperm(Nx, generator=g)[:n]
                    iy = torch.randperm(Ny, generator=g)[:n]
                    id_s = id_s[ix]
                    ood_s = ood_s[iy]


        id_n = (id_s - dmin) / denom
        ood_n = (ood_s - dmin) / denom

        if self.global_scores is None:
            self._compute_global_scores(id_n, ood_n)

        scores = np.asarray(self.global_scores)
        order_desc = np.argsort(scores)[::-1]  # largest first

        keep = max(int(id_n.shape[1] * percentage), 1)
        keep = min(keep, id_n.shape[1])

        selected = order_desc[:keep]
        selected.sort()
        return np.ascontiguousarray(selected)

    # --------------------------------------------------------
    # Internals update 
    # --------------------------------------------------------
    def _update_internals(self):
        assert self.all_feats is not None
        assert self.all_labels is not None
        assert self.ood_feats is not None

        # 1) Select dimensions using proxy-OOD shift (marginal)
        if self.identify_dims or self.global_indices is None:
            self.global_indices = self.identify_dimensions(
                id_data=self.all_feats,
                ood_data=self.ood_feats,
                percentage=self.percentage,
            )
            self.identify_dims = False

        idx = self.global_indices

        # 2) Estimate class means in selected subspace using ALL train samples of each class
        class_means = []
        centered_data = []

        for c in range(self.num_classes):
            class_samples = self.all_feats[self.all_labels.eq(c)][:, idx]  # CPU [Nc, K]
            mu = class_samples.mean(0)
            class_means.append(mu)
            centered_data.append(class_samples - mu.view(1, -1))

        self.global_class_mean = torch.stack(class_means).contiguous()  # [C, K] CPU

        # 3) Fit a shared covariance (EmpiricalCovariance) on all centered samples (selected dims)
        cov_est = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        cov_est.fit(torch.cat(centered_data, dim=0).cpu().numpy().astype(np.float32))
        self.global_precision = torch.from_numpy(cov_est.precision_).float().contiguous()  # [K, K] CPU

    # --------------------------------------------------------
    # Inference (global only, MDS methodology)
    # --------------------------------------------------------

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, feats = net(data, return_feature=True)
        pred = logits.argmax(1)

        assert self.global_indices is not None
        assert self.global_class_mean is not None
        assert self.global_precision is not None   

        x = feats[:, self.global_indices].cpu()  # [B, K] on CPU for mahalanobis

        class_scores = torch.empty((x.size(0), self.num_classes), dtype=torch.float32)
        for c in range(self.num_classes):
            d = x - self.global_class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.sum((torch.matmul(d, self.global_precision) * d), dim=1)

        conf = torch.max(class_scores, dim=1)[0]

        return pred, conf
