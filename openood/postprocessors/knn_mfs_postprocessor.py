from __future__ import annotations

from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .distribution_utils import wasserstein_marginals_torch_cpu
from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

def normalizer(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


class KNN_MFSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.num_classes = num_classes_dict[self.config.dataset.name]

        # Hyperparams
        self.K = int(self.args.K)
        self.percentage = float(self.args.percentage)

        # Selection subsampling for ImageNet feasibility
        self.imagenet_subsample = 100000
        self.identify_dims = True

        # Cached selection scores
        self.global_scores: list[float] | None = None

        # Feature caches. These are full-vector L2-normalized features.
        self.activation_log: np.ndarray | None = None      # [Ntrain, D]
        self.ood_activation_log: np.ndarray | None = None  # [Nproxy, D]

        # Learned WKNN/MFS state
        self.global_indices: np.ndarray | None = None
        self.index: faiss.IndexFlatL2 | None = None

        # State
        self.setup_flag = False

    # --------------------------------------------------------
    # Hyperparam interface
    # --------------------------------------------------------

    def set_hyperparam(self, hyperparam: list):
        self.K = int(hyperparam[0])
        self.percentage = float(hyperparam[1])
        self.identify_dims = True
        self._update_internals()

    def get_hyperparam(self):
        return self.K, self.percentage

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        net.eval()

        self._compute_and_cache_repr(net, id_loader_dict, ood_loader_dict)
        self._update_internals()

        self.setup_flag = True

    def _compute_and_cache_repr(self, net: nn.Module, id_loader_dict, ood_loader_dict) -> None:
        activation_log = []
        ood_activation_log = []
        mixup_alpha_pool = [0.25, 0.5, 0.75]

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
                x = batch["data"].cuda().float()

                _, feat = net(x, return_feature=True)
                activation_log.append(normalizer(feat.detach().cpu().numpy()))

                b = x.size(0)
                idx = torch.randint(0, proxy_pool.size(0), (b,))
                mix = proxy_pool[idx].cuda()
                cur_mixup_alpha = np.random.choice(mixup_alpha_pool, size=b)
                ref = [
                    xi * alpha + mi * (1.0 - alpha)
                    for xi, mi, alpha in zip(x, mix, cur_mixup_alpha)
                ]

                _, feat_ref = net(torch.stack(ref).cuda(), return_feature=True)
                ood_activation_log.append(normalizer(feat_ref.detach().cpu().numpy()))

        self.activation_log = np.concatenate(activation_log, axis=0).astype(np.float32, copy=False)
        self.ood_activation_log = np.concatenate(ood_activation_log, axis=0).astype(np.float32, copy=False)

    # --------------------------------------------------------
    # Marginal feature selection
    # --------------------------------------------------------

    def _compute_global_scores(self, id_n: torch.Tensor, ood_n: torch.Tensor) -> None:
        self.set_seed(self.seed)
        scores = wasserstein_marginals_torch_cpu(id_n, ood_n)
        self.global_scores = scores.detach().cpu().numpy().tolist()

    def identify_dimensions(self, id_data: np.ndarray, ood_data: np.ndarray, percentage: float) -> np.ndarray:
        id_s = torch.from_numpy(np.ascontiguousarray(id_data)).to(torch.float32)
        ood_s = torch.from_numpy(np.ascontiguousarray(ood_data)).to(torch.float32)

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

        id_n = id_s
        ood_n = ood_s

        if self.global_scores is None:
            self._compute_global_scores(id_n, ood_n)

        scores = np.asarray(self.global_scores, dtype=np.float64)
        order_desc = np.argsort(scores)[::-1]  # largest first

        keep = max(int(id_n.shape[1] * percentage), 1)
        keep = min(keep, id_n.shape[1])

        selected = order_desc[:keep]
        selected.sort()
        return np.ascontiguousarray(selected)

    # --------------------------------------------------------
    # kNN helpers
    # --------------------------------------------------------

    @staticmethod
    def _kth_distance_from_faiss(D: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
        valid = indices != -1
        last_valid = np.where(valid, np.arange(k), -1).max(axis=1)
        row = np.arange(D.shape[0])
        return D[row, last_valid]

    # --------------------------------------------------------
    # Internals update
    # --------------------------------------------------------

    def _update_internals(self):
        assert self.activation_log is not None
        assert self.ood_activation_log is not None

        if self.identify_dims or self.global_indices is None:
            if self.ood_activation_log.size == 0:
                raise ValueError("No proxy OOD features available; cannot perform marginal Wasserstein selection.")

            self.global_indices = self.identify_dimensions(
                id_data=self.activation_log,
                ood_data=self.ood_activation_log,
                percentage=self.percentage,
            )
            self.identify_dims = False

        assert self.global_indices is not None
        self.global_indices = np.ascontiguousarray(self.global_indices)

        index_data = np.ascontiguousarray(
            self.activation_log[:, self.global_indices]
        ).astype(np.float32, copy=False)

        self.index = faiss.IndexFlatL2(len(self.global_indices))
        self.index.add(index_data)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, feat = net(data, return_feature=True)
        pred = logits.argmax(1)

        assert self.index is not None
        assert self.global_indices is not None

        feat_np = normalizer(feat.detach().cpu().numpy()).astype(np.float32, copy=False)
        feat_sel = np.ascontiguousarray(feat_np[:, self.global_indices]).astype(np.float32, copy=False)

        D, I = self.index.search(feat_sel, self.K)
        kth = self._kth_distance_from_faiss(D, I, self.K)
        conf = -kth

        return pred, torch.from_numpy(conf)
