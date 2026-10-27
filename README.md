# WV V2 / WHU-MARS-1000

Implementation follows `../ideav2.md` and `../guidev2.md`. The code base is WV V1.1, while **all original UAD identity parameters** initialize from the original UAD Epoch 120 checkpoint. WV V1.1 checkpoints are not initialization inputs.

The server checkpoint path is set in `configs/UAD_V2.yml`. Override `MODEL.VISA_ADAPT.UAD_CHECKPOINT` when running elsewhere. On the local Windows machine, the original UAD checkpoint is `D:\CODE\Doing\WHU_base\transformer_120.pth`.

Run the targeted mechanism checks before training:

```bash
pytest -q tests/test_v2_mechanisms.py
```

Four-GPU training on the server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/UAD_V2.yml MODEL.DIST_TRAIN True
```

The first run validates UAD CLS/Patch11 parity, rebuilds the GPD Bank from deterministic training images, then trains for 60 continuous epochs. `SOLVER.IMS_PER_BATCH=256` means 64 PKM indices and 192 real images per GPU with three modalities.

To resume from a complete V2 checkpoint, set `MODEL.VISA_ADAPT.RESUME_CHECKPOINT` to the saved path. A saved checkpoint contains model, optimizer, scheduler, GradScaler, GPD Bank, RNG state and full configuration.

For formal evaluation, point `TEST.WEIGHT` at a V2 checkpoint:

```bash
python test.py --config_file configs/UAD_V2.yml \
  TEST.WEIGHT logs/UAD_V2/transformer_60.pth
```

The same query and gallery protocol reports Global, Local and Fusion, including overall mAP, mINP and CMC, nine directed modality pairs, and ground/aerial combinations. Fusion is the 1536-dimensional unit vector `[L2(Global), 0.5 L2(Local)] / sqrt(1.25)`.
