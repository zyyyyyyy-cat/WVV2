"""Six directed cross-modality Local contrastive directions from ideav2.md."""
import torch
import torch.nn.functional as F


def local_cross_modality_loss(local, pids, cameras, modalities, temperature=0.07):
    pids = pids.reshape(-1)
    cameras = cameras.reshape(-1)
    modalities = modalities.reshape(-1)
    if not (local.shape[0] == pids.numel() == cameras.numel() == modalities.numel()):
        raise ValueError('Local CM requires one PID, camera and modality per feature')
    if temperature <= 0:
        raise ValueError('Local CM temperature must be positive')

    # Direction dimension: R->N, R->T, N->R, N->T, T->R, T->N.
    source_modality = torch.tensor((0, 0, 1, 1, 2, 2), device=local.device)
    target_modality = torch.tensor((1, 2, 0, 2, 0, 1), device=local.device)
    names = ('0->1', '0->2', '1->0', '1->2', '2->0', '2->1')

    with torch.autocast(device_type=local.device.type, enabled=False):
        features = F.normalize(local.float(), dim=1)
        scores = (features @ features.t()) / temperature
        anchors = modalities.unsqueeze(0) == source_modality.unsqueeze(1)  # [6, B]
        candidates = (modalities.view(1, 1, -1) == target_modality.view(6, 1, 1))
        candidates = candidates & (cameras.view(1, -1, 1) != cameras.view(1, 1, -1))
        same_pid = pids.view(1, -1, 1) == pids.view(1, 1, -1)
        positives = candidates & same_pid
        negatives = candidates & ~same_pid
        positive_count = positives.sum(dim=-1)
        negative_count = negatives.sum(dim=-1)
        valid = anchors & (positive_count > 0) & (negative_count > 0)

        # Finite masking keeps all-invalid rows safe; torch.where removes them
        # before either the anchor or direction mean is computed.
        masked_scores = scores.unsqueeze(0).masked_fill(
            ~candidates, torch.finfo(scores.dtype).min)
        denominator = torch.logsumexp(masked_scores, dim=-1)
        positive_mean = (scores.unsqueeze(0) * positives).sum(dim=-1) / positive_count.clamp_min(1)
        anchor_loss = torch.where(valid, denominator - positive_mean,
                                  torch.zeros_like(denominator))
        valid_count = valid.sum(dim=-1)
        direction_loss = anchor_loss.sum(dim=-1) / valid_count.clamp_min(1)
        loss = direction_loss.sum() / (valid_count > 0).sum().clamp_min(1)

    anchor_count = anchors.sum(dim=-1)
    coverage = {name: (valid_count[index], anchor_count[index])
                for index, name in enumerate(names)}
    return loss, coverage
