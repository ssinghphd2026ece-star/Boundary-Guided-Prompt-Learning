# SAR-BAP-SAM independent implementation — dataset-ready

This is an independent research implementation based on the uploaded manuscript, **not** the original authors' source code or verified reproduction of their numerical claims. The archive `BBox_RBox_PSeg_SSDD.zip` has been inspected. SSDD annotations use ship category ID **0**, 928 development images and 232 test images, with test inshore/offshore counts of 46/186.

## Requirements

Use Python 3.10+ with a CUDA-capable GPU for neural training. Install:

```bash
pip install -r requirements.txt
```

Obtain the SAM ViT-B pretrained checkpoint separately (`sam_vit_b_01ec64.pth`). It is **not bundled**. The exact original 789/139 split and the manuscript's trained weights are also **not available**.

## Prepared dataset

`SSDD_ready/annotations/` contains `train_789_139.json`, `val_789_139.json`, `test.json`, `test_inshore.json`, `test_offshore.json`, `test_scene_map.json`, and `split_manifest_seed2026.json`.

`SSDD_ready/images/` contains `train/*.jpg` and `test/*.jpg`. Use `--images SSDD_ready/images` for all commands; the program automatically resolves both subfolders.

The generated 789/139 split is deterministic (NumPy seed 2026), but **is not the original manuscript split**.

## Commands

```bash
python reproduce.py baselines --json SSDD_ready/annotations/test.json --images SSDD_ready/images --scene-json SSDD_ready/annotations/test_scene_map.json --out results/baselines

python reproduce.py train --train-json SSDD_ready/annotations/train_789_139.json --val-json SSDD_ready/annotations/val_789_139.json --images SSDD_ready/images --sam-checkpoint /weights/sam_vit_b_01ec64.pth --out results/full

python reproduce.py evaluate --json SSDD_ready/annotations/test.json --images SSDD_ready/images --sam-checkpoint /weights/sam_vit_b_01ec64.pth --checkpoint results/full/best.pt --out results/test
```

Default neural configuration uses 40 epochs, batch size 2, 4 gradient accumulation steps, 100 learned queries, 3-layer transformer decoder, trainable SAR adapter, gated fusion, and instance-conditioned refinement. Try `--batch-size 1 --accum 8 --chunk 1` if your GPU runs out of memory; this changes some numerical characteristics.

## Limitations

1. The first three automatic-box ablations cannot be exactly recreated: their proposal generator is undocumented and is not implemented here.
2. Model checkpoint, scored COCO prediction JSONs, original validation IDs, original SAM weight hash, and test evaluation logs were not supplied.
3. Some architecture-specific and augmentation details are ambiguous in the manuscript; this implementation documents its choices. The 100-query SAM decoder and 256-resolution refinement are expensive; GPU memory use has not been benchmarked.
4. Model training and `pycocotools` COCO evaluation **have not been executed in this environment**; no measured neural AP is asserted. The code passes Python syntax compilation only.
5. Please treat reported manuscript values separately from computed results. A claim of exact reproduction would require comparing evaluated outputs to the originals.
