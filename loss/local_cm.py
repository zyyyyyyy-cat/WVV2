"""Six directed cross-modality Local contrastive directions from ideav2.md."""
import torch
import torch.nn.functional as F


def local_cross_modality_loss(local, pids, cameras, modalities, temperature=0.07):
    features = F.normalize(local.float(), dim=1)
    scores = features @ features.t() / temperature
    pids = pids.reshape(-1)
    cameras = cameras.reshape(-1)
    modalities = modalities.reshape(-1)
    directions = []
    coverage = {}
    for source in range(3):
        for target in range(3):
            if source == target:
                continue
            anchor_ids = (modalities == source).nonzero(as_tuple=True)[0]
            anchor_losses = []
            for i in anchor_ids:
                candidates = (modalities == target) & (cameras != cameras[i])
                positives = candidates & (pids == pids[i])
                negatives = candidates & (pids != pids[i])
                if not positives.any() or not negatives.any():
                    continue
                denominator = torch.logsumexp(scores[i, candidates], dim=0)
                anchor_losses.append(denominator - scores[i, positives].mean())
            coverage[f'{source}->{target}'] = (len(anchor_losses), anchor_ids.numel())
            if anchor_losses:
                directions.append(torch.stack(anchor_losses).mean())
    if not directions:
        return features.sum() * 0.0, coverage
    return torch.stack(directions).mean(), coverage
