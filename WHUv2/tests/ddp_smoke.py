"""CPU four-process smoke test for V2 sampler, bank and evaluation collectives.

Run with: torchrun --standalone --nproc_per_node=4 tests/ddp_smoke.py
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.sampler_ddp import PKMSampler_DDP
from processor.processor import PrototypeBank
from utils.metrics import R1_mAP_eval


def main(rank, world_size, init_method):
    assert world_size == 4
    dist.init_process_group('gloo', init_method=init_method,
                            rank=rank, world_size=world_size)
    try:
        modalities = ('RGB', 'IR', 'Thermal')
        data = {
            modality: [(None, pid, instance, index + 1)
                       for pid in range(64) for instance in range(4)]
            for index, modality in enumerate(modalities)
        }
        sampler = PKMSampler_DDP(data, batch_size=256,
                                 num_instances=4, modalities=modalities)
        indices = list(iter(sampler))
        assert len(indices) == 64
        assert len({data['RGB'][item[0]][1] for item in indices}) == 16

        bank = PrototypeBank(64, 4, device='cpu')
        ids = torch.tensor([data['RGB'][item[0]][1] for item in indices[::4]])
        centers = torch.nn.functional.normalize(torch.randn(16, 4), dim=1)
        bank.update(ids, centers, ddp=True)
        gathered = [torch.empty_like(bank.prototypes) for _ in range(world_size)]
        dist.all_gather(gathered, bank.prototypes)
        assert all(torch.equal(gathered[0], part) for part in gathered)

        evaluator = R1_mAP_eval(max_rank=5, feat_norm='yes',
                                reranking=False, top_k=0,
                                logger=logging.getLogger('ddp-smoke'))
        for mode in (1, 2, 3):
            evaluator.set_query_num(mode, 2)
            if rank != mode - 1:
                continue
            feature = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                                    [1., 0., 0.], [0., 1., 0.],
                                    [0., 0., 1.]])
            evaluator.update((feature, [0, 1, 0, 1, 2],
                              [0, 0, 1, 1, 2], [mode] * 5), mode)
        evaluator.split_all()
        cmc, mean_ap, *_ = evaluator.compute()
        if rank == 0:
            assert cmc[0] == mean_ap == evaluator.overall_minp == 1.0
            print('Four-process sampler, bank and evaluation smoke passed')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), 'env://')
