"""Generate the offline CLIP text features consumed by SSDNet.

Example:
  HF_ENDPOINT=https://hf-mirror.com python tools/text_embeddings/generate_clip_embeddings.py \
      --dataset ruod --output-dir embeddings/ruod
"""

import argparse
import json
from pathlib import Path

import torch


POSITIVE_PROMPTS = {
    "holothurian": [
        "a small sea cucumber on the underwater seabed",
        "a holothurian resting on sand under the sea",
        "an elongated sea cucumber in an underwater photograph",
    ],
    "echinus": [
        "a small sea urchin on the seabed",
        "a round spiny echinus underwater",
        "a sea urchin among rocks in the ocean",
    ],
    "scallop": [
        "a small scallop on the sandy seabed",
        "a scallop with a fan-shaped ribbed shell underwater",
        "an underwater scallop shell lying on sand",
    ],
    "starfish": [
        "a tiny underwater starfish",
        "a starfish on the seabed",
        "a sea star with radial arms under water",
    ],
    "fish": [
        "a small fish swimming underwater",
        "a fish in a natural marine scene",
        "an underwater photograph of a fish",
    ],
    "corals": [
        "a small coral colony on the seabed",
        "branching corals growing underwater",
        "an underwater photograph of living coral",
    ],
    "diver": [
        "a scuba diver swimming underwater",
        "a human diver with scuba equipment in the sea",
        "an underwater photograph of a diver",
    ],
    "cuttlefish": [
        "a cuttlefish swimming near the seabed",
        "an underwater cuttlefish with a broad body and fins",
        "a small cuttlefish in a marine photograph",
    ],
    "turtle": [
        "a sea turtle swimming underwater",
        "an underwater turtle with flippers and a shell",
        "a small sea turtle in the ocean",
    ],
    "jellyfish": [
        "a translucent jellyfish floating underwater",
        "a jellyfish with a bell and tentacles in the sea",
        "an underwater photograph of a jellyfish",
    ],
}

NEGATIVE_PROMPTS = [
    "small pieces of marine debris on the seabed",
    "rocks and stones on the seabed",
    "a sandy seabed texture underwater",
    "scattered pebbles and gravel under water",
    "seaweed and kelp in an underwater scene",
    "floating algae in murky water",
    "a dark shadow on the seafloor",
    "a cloud of silt in underwater water",
    "underwater mud and sediment",
    "a discarded plastic bag floating in the sea",
    "driftwood and plant fragments on the seabed",
    "air bubbles and suspended particles underwater",
    "a patch of sea foam in an underwater image",
    "a rough ocean floor with no marine animal",
    "a blurry underwater background with poor visibility",
]


def encode(model, tokenizer, prompts, device, batch_size):
    chunks = []
    for start in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[start : start + batch_size]).to(device)
        with torch.inference_mode():
            chunks.append(model.encode_text(tokens).float().cpu())
    return torch.cat(chunks, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ruod", "duo"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    annotation = Path(f"/home/user03/lbl/Dataset/{args.dataset.upper()}/annotations/instances_train.json")
    categories = sorted(json.loads(annotation.read_text())["categories"], key=lambda item: item["id"])
    ids = [category["id"] for category in categories]
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError(f"Expected contiguous one-based category IDs, got {ids}")
    names = [category["name"].lower() for category in categories]
    missing = set(names) - POSITIVE_PROMPTS.keys()
    if missing:
        raise ValueError(f"No prompts for classes: {sorted(missing)}")

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=args.device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    positive_prompts = [POSITIVE_PROMPTS[name] for name in names]
    positive = encode(model, tokenizer, sum(positive_prompts, []), args.device, args.batch_size)
    positive = positive.reshape(len(names), len(positive_prompts[0]), -1)
    negative = encode(model, tokenizer, NEGATIVE_PROMPTS, args.device, args.batch_size)
    if positive.shape[-1] != negative.shape[-1] or not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("Invalid CLIP embeddings")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(positive, args.output_dir / "positive.pt")
    torch.save(negative, args.output_dir / "negative.pt")
    metadata = {
        "dataset": args.dataset,
        "annotation": str(annotation),
        "category_ids": ids,
        "categories": names,
        "positive_prompts": positive_prompts,
        "negative_prompts": NEGATIVE_PROMPTS,
        "model": args.model,
        "pretrained": args.pretrained,
        "positive_shape": list(positive.shape),
        "negative_shape": list(negative.shape),
        "open_clip_version": open_clip.__version__,
    }
    (args.output_dir / "prompts.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), **{k: metadata[k] for k in ("categories", "positive_shape", "negative_shape")}}, indent=2))


if __name__ == "__main__":
    main()
