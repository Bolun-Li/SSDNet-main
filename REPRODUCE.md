# Reproduce SSDNet on RUOD and DUO

The text encoder runs only during the offline embedding step. Training loads the
saved tensors and does not load CLIP. This setup uses OpenCLIP `ViT-H-14` with
the LAION-2B `laion2b_s32b_b79k` weights. The run on 9 October 2026 resolved
the model repository to revision `1c2b8495b28150b8a4922ee1c8edee224c284c0c`.

The paper does not publish the complete prompt list or an exact ViT-H/14 weight
revision. The prompts used here are defined in
`tools/text_embeddings/generate_clip_embeddings.py` and saved with each output
in `prompts.json`. This reproduces the described method, but the numerical
results may differ from Table I.

## Data

- RUOD: `/home/user03/lbl/Dataset/RUOD`; 10 classes, 9,800 train images and
  4,200 test images.
- DUO: `/home/user03/lbl/Dataset/DUO`; 4 classes, 6,671 train images and
  1,111 test images.

The COCO annotation class order is checked by the embedding generator.

## Offline CLIP embeddings

From the project root, after installing `requirements.txt`:

```bash
HF_ENDPOINT=https://hf-mirror.com python tools/text_embeddings/generate_clip_embeddings.py --dataset ruod --output-dir embeddings/ruod
HF_ENDPOINT=https://hf-mirror.com python tools/text_embeddings/generate_clip_embeddings.py --dataset duo --output-dir embeddings/duo
```

Each positive tensor has shape `[classes, 3, 1024]`; each negative tensor has
shape `[15, 1024]`. The negative prompts describe confusing underwater
backgrounds. The generator saves raw CLIP features; DTSC normalizes them when
computing cosine similarity and selects the top three backgrounds for each
class. The classifier receives the resulting semantic residual. The generated
embeddings are ignored by Git because they are reproducible artifacts.

## Training

```bash
python train.py -c configs/dfine/underwater/ssdnet_s_ruod.yml --seed 42
python train.py -c configs/dfine/underwater/ssdnet_s_duo.yml --seed 42
```

Both configurations use D-FINE-S, 640 × 640 images, batch size 16, AMP, EMA,
220 epochs, and no pretrained detector or backbone checkpoint. Results and
checkpoints are written to `output/ssdnet_s_ruod` and `output/ssdnet_s_duo`.
Their `log.txt` files contain COCO AP, AP50, AP75, and small-object AP after
each epoch. When both jobs run on one 32 GB GPU, allow about 27 GB for their
combined training allocations.
