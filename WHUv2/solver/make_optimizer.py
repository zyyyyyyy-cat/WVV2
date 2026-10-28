import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    seen = set()
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if id(value) in seen:
            raise RuntimeError(f'duplicate optimizer parameter: {key}')
        seen.add(id(value))
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        # V2 old/new base rates are exclusive; do not multiply bias/classifier rates twice.
        is_new = cfg.MODEL.VISA_ADAPT.MODE != 'off' and (
            key.startswith('base.specific_tokens.')
            or key == 'base.specific_pos_embed'
            or key.startswith(('etgm.', 'dlfm.', 'local_classifier'))
        )
        if is_new:
            lr = cfg.SOLVER.BASE_LR * cfg.MODEL.VISA_ADAPT.NEW_LR_FACTOR
            if key.endswith('bias') or 'norm' in key.lower() or 'token' in key or 'slots' in key or 'scale' in key or 'control' in key or key == 'local_classifier':
                weight_decay = 0.0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay,
                    "visa_new": is_new, "name": key}]

    expected = {id(value) for value in model.parameters() if value.requires_grad}
    if seen != expected:
        raise RuntimeError('optimizer parameter coverage mismatch')

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = (torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
                        if center_criterion is not None else None)

    return optimizer, optimizer_center
