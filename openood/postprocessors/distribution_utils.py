import torch

@torch.no_grad()
def wasserstein_marginals_torch_cpu(
    X: torch.Tensor,  # [n, D] CPU float32
    Y: torch.Tensor,  # [n, D] CPU float32
    dim_chunk: int = 128,
) -> torch.Tensor:
    assert X.device.type == "cpu" and Y.device.type == "cpu"
    assert X.shape == Y.shape
    n, D = X.shape

    out = torch.empty(D, dtype=torch.float32)
    for j in range(0, D, dim_chunk):
        jj = min(j + dim_chunk, D)
        Xs = X[:, j:jj].sort(dim=0).values
        Ys = Y[:, j:jj].sort(dim=0).values
        out[j:jj] = (Xs - Ys).abs().mean(dim=0)
    return out