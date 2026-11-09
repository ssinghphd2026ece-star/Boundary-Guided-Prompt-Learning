# SAR-BAP-SAM implementation — dataset-ready

# Requirements

Use Python 3.10+ with a CUDA-capable GPU for neural training. Install:

```bash
pip install -r requirements.txt
```

# Prepared dataset

`SSDD_ready/annotations/` contains `train_789_139.json`, `val_789_139.json`, `test.json`, `test_inshore.json`, `test_offshore.json`, `test_scene_map.json`, and `split_manifest_seed2026.json`.

`SSDD_ready/images/` contains `train/*.jpg` and `test/*.jpg`. Use `--images SSDD_ready/images` for all commands; the program automatically resolves both subfolders.

# Commands

```bash
python reproduce.py baselines --json SSDD_ready/annotations/test.json --images SSDD_ready/images --scene-json SSDD_ready/annotations/test_scene_map.json --out results/baselines

python reproduce.py train --train-json SSDD_ready/annotations/train_789_139.json --val-json SSDD_ready/annotations/val_789_139.json --images SSDD_ready/images --sam-checkpoint /weights/sam_vit_b_01ec64.pth --out results/full

python reproduce.py evaluate --json SSDD_ready/annotations/test.json --images SSDD_ready/images --sam-checkpoint /weights/sam_vit_b_01ec64.pth --checkpoint results/full/best.pt --out results/test
```

Default neural configuration uses 40 epochs, batch size 2, 4 gradient accumulation steps, 100 learned queries, 3-layer transformer decoder, trainable SAR adapter, gated fusion, and instance-conditioned refinement. Try `--batch-size 1 --accum 8 --chunk 1` if your GPU runs out of memory; this changes some numerical characteristics.
