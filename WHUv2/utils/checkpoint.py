"""Load checkpoints produced by the trusted UAD/V2 training runs."""

import inspect

import torch


def load_trusted_checkpoint(path, map_location='cpu'):
    # PyTorch 1.x forwards unknown torch.load keywords to pickle.Unpickler.
    # PyTorch 2.6+ defaults to weights_only=True, which cannot read full V2
    # resume checkpoints containing NumPy RNG state. Select the supported
    # signature explicitly rather than catching unrelated unpickling errors.
    kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)
