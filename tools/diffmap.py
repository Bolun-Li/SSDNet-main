import os
import sys
import json
import glob
import random
import argparse
from pathlib import Path
from collections import defaultdict
from math import ceil

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Basic utils
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in s)


def find_config(repo_root, ckpt_path, config_arg=None):
    if config_arg is not None and os.path.isfile(config_arg):
        print("[Config] Use user specified config:", config_arg)
        return config_arg

    ckpt_dir = os.path.dirname(ckpt_path)

    patterns = [
        os.path.join(ckpt_dir, "*.yml"),
        os.path.join(ckpt_dir, "*.yaml"),
        os.path.join(repo_root, "configs", "dfine", "underwater", "*duo*.yml"),
        os.path.join(repo_root, "configs", "dfine", "underwater", "*duo*.yaml"),
        os.path.join(repo_root, "configs", "**", "*duo*.yml"),
        os.path.join(repo_root, "configs", "**", "*duo*.yaml"),
    ]

    for pat in patterns:
        found = glob.glob(pat, recursive=True)
        found = [
            p for p in found
            if "crowdhuman" not in p.lower()
            and "coco" not in p.lower()
            and "objects365" not in p.lower()
        ]
        found = sorted(list(set(found)))
        if len(found) > 0:
            print("[Auto config] Use:", found[0])
            return found[0]

    raise FileNotFoundError(
        "没有自动找到 DUO 配置文件。请手动加 --config。"
    )


# =========================================================
# Disable pretrained downloading
# =========================================================
def disable_pretrained_in_obj(obj, visited=None):
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    pretrained_keys = {
        "pretrained",
        "pretrain",
        "pretrained_model",
        "pretrain_model",
        "pretrain_weights",
        "pretrained_weights",
        "pretrained_path",
        "pretrain_path",
        "pretrained_url",
        "pretrain_url",
    }

    if isinstance(obj, dict):
        for k in list(obj.keys()):
            kl = str(k).lower()
            if kl in pretrained_keys:
                if obj[k] not in [False, None, "", 0]:
                    print(f"[Patch] Disable {k}: {obj[k]} -> False")
                obj[k] = False
            else:
                disable_pretrained_in_obj(obj[k], visited)

    elif isinstance(obj, list):
        for v in obj:
            disable_pretrained_in_obj(v, visited)

    elif isinstance(obj, tuple):
        for v in obj:
            disable_pretrained_in_obj(v, visited)


def patch_yaml_config_pretrained(cfg):
    patched_any = False

    candidate_attrs = [
        "yaml_cfg",
        "_yaml_cfg",
        "cfg",
        "_cfg",
        "config",
        "_config",
        "global_cfg",
        "_global_cfg",
    ]

    for attr in candidate_attrs:
        if hasattr(cfg, attr):
            val = getattr(cfg, attr)
            if isinstance(val, (dict, list, tuple)):
                disable_pretrained_in_obj(val)
                patched_any = True
                print(f"[Patch] Checked cfg.{attr}")

    for k, v in vars(cfg).items():
        if isinstance(v, (dict, list, tuple)):
            disable_pretrained_in_obj(v)
            patched_any = True
            print(f"[Patch] Checked cfg.__dict__['{k}']")

    if patched_any:
        print("[Patch] Pretrained backbone loading has been disabled.")
    else:
        print("[Patch] Warning: no dict-like config found. If it still downloads, edit yaml manually.")


def load_state_dict_from_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        print("[Load] Use raw non-dict checkpoint.")
        return ckpt

    if "ema" in ckpt and ckpt["ema"] is not None:
        ema = ckpt["ema"]

        if isinstance(ema, dict):
            if "module" in ema:
                print("[Load] Use EMA module weights.")
                return ema["module"]
            if "state_dict" in ema:
                print("[Load] Use EMA state_dict weights.")
                return ema["state_dict"]

        if hasattr(ema, "module"):
            print("[Load] Use EMA.module.state_dict().")
            return ema.module.state_dict()

        if hasattr(ema, "state_dict"):
            print("[Load] Use EMA.state_dict().")
            return ema.state_dict()

    if "model" in ckpt:
        print("[Load] Use model weights.")
        model_part = ckpt["model"]
        if hasattr(model_part, "state_dict"):
            return model_part.state_dict()
        return model_part

    if "state_dict" in ckpt:
        print("[Load] Use checkpoint['state_dict'].")
        return ckpt["state_dict"]

    print("[Load] Use checkpoint dict as state_dict.")
    return ckpt


def load_dfine_model(repo_root, config_path, ckpt_path, device):
    sys.path.insert(0, repo_root)
    from src.core import YAMLConfig

    cfg = YAMLConfig(config_path, resume=None)
    patch_yaml_config_pretrained(cfg)

    model = cfg.model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = load_state_dict_from_checkpoint(ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_state_dict[nk] = v

    model_state = model.state_dict()
    filtered_state_dict = {}
    skipped = []

    for k, v in new_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state_dict[k] = v
        else:
            if k in model_state:
                skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            else:
                skipped.append((k, tuple(v.shape), None))

    print("[Load] matched keys:", len(filtered_state_dict))
    print("[Load] skipped keys because of shape/name mismatch:", len(skipped))

    if len(skipped) > 0:
        print("[Load] skipped sample:")
        for item in skipped[:20]:
            print("   ", item)

    msg = model.load_state_dict(filtered_state_dict, strict=False)
    print("[Load] missing keys:", len(msg.missing_keys))
    print("[Load] unexpected keys:", len(msg.unexpected_keys))

    if len(msg.missing_keys) > 0:
        print("[Load] missing keys sample:", msg.missing_keys[:10])
    if len(msg.unexpected_keys) > 0:
        print("[Load] unexpected keys sample:", msg.unexpected_keys[:10])

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


# =========================================================
# Feature hook
# =========================================================
def collect_4d_tensors(x, out):
    if torch.is_tensor(x) and x.dim() == 4:
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for v in x:
            collect_4d_tensors(v, out)
    elif isinstance(x, dict):
        for v in x.values():
            collect_4d_tensors(v, out)


def forward_model_safely(model, inp):
    return model(inp)


class MultiFeatureCatcher:
    """
    同时 hook 多个层级：
    P2: backbone.stages.0
    P3: backbone.stages.1
    P4: backbone.stages.2
    P5: backbone.stages.3
    Encoder: encoder
    """

    def __init__(self, model, layer_specs):
        self.model = model
        self.layer_specs = layer_specs
        self.features = {}
        self.handles = []
        self._printed = set()

        for spec in self.layer_specs:
            label = spec["label"]
            hook_name = spec["hook_name"]
            module_name, module = self.find_module(model, hook_name)
            print(f"[Hook] {label}: {module_name}")
            handle = module.register_forward_hook(self.make_hook(spec, module_name))
            self.handles.append(handle)

    def find_module(self, model, hook_name):
        named_modules = list(model.named_modules())

        for name, module in named_modules:
            if name == hook_name:
                return name, module

        for name, module in named_modules:
            if name.endswith(hook_name):
                return name, module

        for name, module in named_modules:
            if hook_name in name:
                return name, module

        print("\n[Hook] 找不到指定模块。下面是部分可用模块名：")
        for name, _ in named_modules[:300]:
            print(name)

        raise RuntimeError(f"找不到 hook 模块：{hook_name}")

    def make_hook(self, spec, module_name):
        label = spec["label"]
        feature_index = spec.get("feature_index", -1)

        def hook_fn(module, inputs, output):
            feats = []
            collect_4d_tensors(output, feats)

            if len(feats) == 0:
                raise RuntimeError(
                    f"{label} hook 到的模块 {module_name} 没有输出 4D feature map。"
                )

            if label not in self._printed:
                print(f"[Hook] {label} collected 4D features:")
                for i, f in enumerate(feats):
                    print(f"  index={i}, shape={tuple(f.shape)}")
                self._printed.add(label)

            if feature_index >= 0:
                if feature_index >= len(feats):
                    raise IndexError(
                        f"{label}: feature_index={feature_index} 超出范围，当前只有 {len(feats)} 个 4D 特征。"
                    )
                feat = feats[feature_index]
            else:
                feat = sorted(
                    feats,
                    key=lambda t: t.shape[-1] * t.shape[-2],
                    reverse=True,
                )[0]

            self.features[label] = feat.detach()

        return hook_fn

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def parse_layer_specs(layer_str):
    """
    格式：
    P2:backbone.stages.0:-1,P3:backbone.stages.1:-1

    默认：
    P2/P3/P4/P5/Encoder
    """
    if layer_str is None or layer_str.strip() == "":
        layer_str = (
            "P2:backbone.stages.0:-1,"
            "P3:backbone.stages.1:-1,"
            "P4:backbone.stages.2:-1,"
            "P5:backbone.stages.3:-1,"
            "Encoder:encoder:-1"
        )

    specs = []

    for item in layer_str.split(","):
        item = item.strip()
        if not item:
            continue

        parts = item.split(":")
        if len(parts) == 2:
            label, hook_name = parts
            feature_index = -1
        elif len(parts) == 3:
            label, hook_name, feature_index = parts
            feature_index = int(feature_index)
        else:
            raise ValueError(
                f"错误的 layer 配置：{item}\n"
                "正确格式示例：P2:backbone.stages.0:-1"
            )

        specs.append(
            {
                "label": label,
                "hook_name": hook_name,
                "feature_index": feature_index,
            }
        )

    return specs


def probe_4d_module_outputs(model, inp, max_print=600):
    records = []
    handles = []

    def make_hook(name):
        def hook_fn(module, inputs, output):
            feats = []
            collect_4d_tensors(output, feats)
            for f in feats:
                if torch.is_tensor(f) and f.dim() == 4:
                    records.append((name, tuple(f.shape)))
        return hook_fn

    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = forward_model_safely(model, inp)

    for h in handles:
        h.remove()

    seen = set()
    unique = []
    for name, shape in records:
        key = (name, shape)
        if key not in seen:
            seen.add(key)
            unique.append((name, shape))

    target_sizes = [160, 80, 40, 20]
    filtered = []
    for name, shape in unique:
        if len(shape) == 4 and shape[-1] in target_sizes and shape[-2] in target_sizes:
            filtered.append((name, shape))

    print("\n[Probe] 4D module outputs with pyramid-like sizes:")
    for i, (name, shape) in enumerate(filtered[:max_print]):
        print(f"{i:03d}: {name:90s} -> {shape}")

    if len(filtered) > max_print:
        print(f"[Probe] Only printed first {max_print}, total {len(filtered)}.")

    print("\n[Probe] Candidate layers:")
    print("       P2/C2: (1, C, 160, 160)")
    print("       P3/C3: (1, C, 80, 80)")
    print("       P4/C4: (1, C, 40, 40)")
    print("       P5/C5: (1, C, 20, 20)")


# =========================================================
# COCO dataset
# =========================================================
def load_coco(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_infos = data["images"]

    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    cat_id_to_name = {
        c["id"]: c.get("name", str(c["id"]))
        for c in data.get("categories", [])
    }

    return img_infos, anns_by_img, cat_id_to_name


class DUOImageDataset(Dataset):
    def __init__(self, image_dir, img_infos, img_size=640):
        self.image_dir = Path(image_dir)
        self.img_infos = img_infos
        self.img_size = img_size

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.img_infos)

    def get_path(self, info):
        file_name = info["file_name"]

        p = self.image_dir / file_name
        if p.exists():
            return p

        p = self.image_dir / Path(file_name).name
        if p.exists():
            return p

        raise FileNotFoundError(f"找不到图片：{file_name}")

    def __getitem__(self, idx):
        info = self.img_infos[idx]
        img_path = self.get_path(info)

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"读取失败：{img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            rgb,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )

        target_01 = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        inp = (target_01 - self.mean) / self.std

        return inp, target_01, idx


# =========================================================
# Reconstruction heads
# =========================================================
class AdaptiveReconHead(nn.Module):
    """
    多尺度通用重建头：
    从任意 HxW 特征逐步上采样到 640x640，再输出 RGB 重建图。
    每个层级单独一个 head。
    """

    def __init__(self, in_channels, mid_channels=64, max_upsamples=6):
        super().__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.up_blocks = nn.ModuleList()
        for _ in range(max_upsamples):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.out = nn.Sequential(
            nn.Conv2d(mid_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, target_hw):
        x = self.reduce(feat)
        x = self.refine(x)

        target_h, target_w = target_hw
        step = 0

        while (x.shape[-2] < target_h or x.shape[-1] < target_w) and step < len(self.up_blocks):
            next_h = min(x.shape[-2] * 2, target_h)
            next_w = min(x.shape[-1] * 2, target_w)
            x = F.interpolate(x, size=(next_h, next_w), mode="bilinear", align_corners=False)
            x = self.up_blocks[step](x)
            step += 1

        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

        return self.out(x)


def get_first_features(model, catcher, dataset, device):
    inp, _, _ = dataset[0]
    inp = inp.unsqueeze(0).to(device)

    with torch.no_grad():
        _ = forward_model_safely(model, inp)

    feats = {}
    for spec in catcher.layer_specs:
        label = spec["label"]
        if label not in catcher.features:
            raise RuntimeError(f"没有捕获到 {label} 的特征。")
        feats[label] = catcher.features[label]
        print(f"[Feature] {label}: {tuple(feats[label].shape)}")

    return feats


def build_recon_heads(args, first_feats, device):
    heads = nn.ModuleDict()

    for label, feat in first_feats.items():
        in_channels = feat.shape[1]
        heads[label] = AdaptiveReconHead(
            in_channels=in_channels,
            mid_channels=args.recon_channels,
            max_upsamples=6,
        ).to(device)

    return heads


def train_recon_heads(args, model, catcher, dataset, device, recon_path):
    first_feats = get_first_features(model, catcher, dataset, device)
    recon_heads = build_recon_heads(args, first_feats, device)

    if os.path.isfile(recon_path) and not args.force_train_recon:
        print("[Recon] Load existing:", recon_path)
        ckpt = torch.load(recon_path, map_location=device)
        recon_heads.load_state_dict(ckpt["heads"], strict=True)
        recon_heads.eval()
        return recon_heads

    print("[Recon] Train reconstruction heads...")
    print("[Recon] Detector is frozen. Only reconstruction heads will be trained.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        recon_heads.parameters(),
        lr=args.recon_lr,
        weight_decay=1e-4,
    )

    model.eval()
    recon_heads.train()

    labels = [spec["label"] for spec in catcher.layer_specs]

    for epoch in range(args.recon_epochs):
        total_by_label = {label: 0.0 for label in labels}
        count = 0

        for inp, target_01, _ in loader:
            inp = inp.to(device, non_blocking=True)
            target_01 = target_01.to(device, non_blocking=True)

            with torch.no_grad():
                _ = forward_model_safely(model, inp)

            optimizer.zero_grad()

            bs = inp.size(0)
            for label in labels:
                feat = catcher.features[label]
                pred = recon_heads[label](feat, target_hw=target_01.shape[-2:])
                loss = F.mse_loss(pred, target_01)
                loss.backward()

                total_by_label[label] += loss.item() * bs

            optimizer.step()
            count += bs

        msg = f"[Recon] Epoch {epoch + 1}/{args.recon_epochs}"
        for label in labels:
            msg += f", {label}_MSE={total_by_label[label] / max(count, 1):.6f}"
        print(msg)

    os.makedirs(os.path.dirname(recon_path), exist_ok=True)
    torch.save(
        {
            "heads": recon_heads.state_dict(),
            "layers": catcher.layer_specs,
        },
        recon_path,
    )
    print("[Recon] Saved:", recon_path)

    recon_heads.eval()
    return recon_heads


# =========================================================
# Heatmap and visualization
# =========================================================
def normalize_map(x, p_low=5, p_high=99.5):
    x = np.asarray(x, dtype=np.float32)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    x = (x - lo) / (hi - lo + 1e-6)
    x = np.clip(x, 0, 1)

    return x


def heatmap_color(x):
    x = normalize_map(x)
    x_u8 = (x * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(x_u8, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def gaussian_blur_tensor(x, ksize=11, sigma=2.5):
    device = x.device
    dtype = x.dtype

    ax = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, ksize, ksize)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)

    pad = ksize // 2
    return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])


def diff_to_heatmap(recon, target_01, out_hw, diff_mode="pixel"):
    if diff_mode == "pixel":
        diff = torch.abs(recon - target_01).mean(dim=1, keepdim=True)

    elif diff_mode == "highfreq":
        recon_blur = gaussian_blur_tensor(recon, ksize=11, sigma=2.5)
        target_blur = gaussian_blur_tensor(target_01, ksize=11, sigma=2.5)

        recon_hf = torch.abs(recon - recon_blur)
        target_hf = torch.abs(target_01 - target_blur)

        diff = torch.abs(recon_hf - target_hf).mean(dim=1, keepdim=True)

    else:
        raise ValueError(f"Unknown diff_mode: {diff_mode}")

    diff = F.interpolate(diff, size=out_hw, mode="bilinear", align_corners=False)
    diff = diff[0, 0].detach().cpu().numpy()

    return heatmap_color(diff)


def get_palette(cat_id_to_name):
    base = {
        "holothurian": (128, 0, 128),
        "echinus": (70, 130, 180),
        "echinu": (70, 130, 180),
        "scallop": (218, 165, 32),
        "starfish": (255, 69, 0),
    }

    fallback = [
        (255, 0, 0),
        (0, 200, 255),
        (255, 128, 0),
        (180, 0, 255),
        (0, 128, 255),
        (255, 0, 128),
        (128, 255, 0),
        (255, 255, 0),
    ]

    palette = {}
    for i, cid in enumerate(cat_id_to_name.keys()):
        name = cat_id_to_name[cid]
        palette[cid] = base.get(name, fallback[i % len(fallback)])

    return palette


def draw_title(img, title):
    img = img.copy()

    cv2.rectangle(
        img,
        (0, 0),
        (min(img.shape[1], 520), 36),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        img,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return img


def draw_gt_boxes(
    img,
    anns,
    cat_id_to_name,
    palette,
    scale_x=1.0,
    scale_y=1.0,
    show_label=False,
):
    img = img.copy()

    for ann in anns:
        x, y, w, h = ann["bbox"]

        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + w) * scale_x))
        y2 = int(round((y + h) * scale_y))

        cid = ann.get("category_id", -1)
        color = palette.get(cid, (255, 0, 0))

        thickness = max(1, int(round(min(img.shape[:2]) / 350)))

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        if show_label:
            name = cat_id_to_name.get(cid, str(cid))
            cv2.putText(
                img,
                name,
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return img


def make_grid_mosaic(panels, anns, cat_id_to_name, palette, show_label=False, cols=3):
    """
    panels: list of (title, rgb_image), each image H,W,3
    输出网格拼图。每个 panel 都画 GT。
    """
    assert len(panels) > 0

    h, w = panels[0][1].shape[:2]
    rows = ceil(len(panels) / cols)

    canvas = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255

    for idx, (title, img) in enumerate(panels):
        r = idx // cols
        c = idx % cols

        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        panel = draw_gt_boxes(
            img,
            anns,
            cat_id_to_name,
            palette,
            scale_x=1.0,
            scale_y=1.0,
            show_label=show_label,
        )
        panel = draw_title(panel, title)

        y1 = r * h
        x1 = c * w
        canvas[y1:y1 + h, x1:x1 + w] = panel

    return canvas


def visualize_all(
    args,
    model,
    catcher,
    recon_heads,
    dataset,
    img_infos,
    anns_by_img,
    cat_id_to_name,
    device,
):
    os.makedirs(args.out_dir, exist_ok=True)
    palette = get_palette(cat_id_to_name)

    model.eval()
    recon_heads.eval()

    labels = [spec["label"] for spec in catcher.layer_specs]
    total = len(img_infos)

    for idx, info in enumerate(img_infos):
        img_path = dataset.get_path(info)

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"读取失败：{img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        inp, target_01, _ = dataset[idx]
        inp = inp.unsqueeze(0).to(device)
        target_01 = target_01.unsqueeze(0).to(device)

        panels = [("Original + GT", rgb)]

        with torch.no_grad():
            _ = forward_model_safely(model, inp)

            for label in labels:
                feat = catcher.features[label]
                recon = recon_heads[label](feat, target_hw=target_01.shape[-2:])

                diff_heat = diff_to_heatmap(
                    recon,
                    target_01,
                    out_hw=(h, w),
                    diff_mode=args.diff_mode,
                )

                panels.append((f"{label} Diff", diff_heat))

        anns = anns_by_img.get(info["id"], [])

        mosaic = make_grid_mosaic(
            panels,
            anns,
            cat_id_to_name,
            palette,
            show_label=args.show_label,
            cols=args.grid_cols,
        )

        stem = Path(info["file_name"]).stem
        out_name = f"{int(info['id']):06d}_{stem}_multi_diff.jpg"
        out_path = os.path.join(args.out_dir, out_name)

        cv2.imwrite(out_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"[Vis] {idx + 1}/{total} saved.")

    print("[Done] Results saved to:", args.out_dir)


# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default="/home/user03/lbl/D-FINE-master",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="D-FINE config path. 建议手动指定 DUO 配置。",
    )

    parser.add_argument(
        "--checkpoint",
        default="/home/user03/lbl/D-FINE-master/output/dfine_s_duo/checkpoint0189.pth",
    )

    parser.add_argument(
        "--image-dir",
        default="/home/user03/lbl/Dataset/DUO/images/test",
    )

    parser.add_argument(
        "--ann-file",
        default="/home/user03/lbl/Dataset/DUO/annotations/instances_test.json",
    )

    parser.add_argument(
        "--out-dir",
        default="/home/user03/lbl/Dataset/DUO/chayi_multi",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--layers",
        default=(
            "P2:backbone.stages.0:-1,"
            "P3:backbone.stages.1:-1,"
            "P4:backbone.stages.2:-1,"
            "P5:backbone.stages.3:-1,"
            "Encoder:encoder:-1"
        ),
        help=(
            "多层 hook 配置。格式："
            "P2:backbone.stages.0:-1,P3:backbone.stages.1:-1"
        ),
    )

    parser.add_argument(
        "--probe-hooks",
        action="store_true",
        help="打印所有 4D 中间特征尺寸，用来寻找 P2/C2 层。",
    )

    parser.add_argument(
        "--max-probe-print",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--diff-mode",
        default="highfreq",
        choices=["pixel", "highfreq"],
        help="pixel 是普通重建差异图；highfreq 是高频差异图。",
    )

    parser.add_argument(
        "--recon-epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--recon-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--recon-channels",
        type=int,
        default=64,
        help="如果显存不够，可以改成 32。",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--force-train-recon",
        action="store_true",
        help="如果已有 recon heads，强制重新训练。",
    )

    parser.add_argument(
        "--show-label",
        action="store_true",
        help="是否在 GT 框旁边显示类别名。",
    )

    parser.add_argument(
        "--grid-cols",
        type=int,
        default=3,
        help="拼图列数，默认 3。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    config_path = find_config(args.repo_root, args.checkpoint, args.config)
    print("[Config]", config_path)
    print("[Checkpoint]", args.checkpoint)

    img_infos, anns_by_img, cat_id_to_name = load_coco(args.ann_file)
    print("[Dataset] images:", len(img_infos))
    print("[Dataset] categories:", cat_id_to_name)

    dataset = DUOImageDataset(
        args.image_dir,
        img_infos,
        img_size=args.img_size,
    )

    model = load_dfine_model(
        repo_root=args.repo_root,
        config_path=config_path,
        ckpt_path=args.checkpoint,
        device=device,
    )

    if args.probe_hooks:
        inp, _, _ = dataset[0]
        inp = inp.unsqueeze(0).to(device)
        probe_4d_module_outputs(
            model,
            inp,
            max_print=args.max_probe_print,
        )
        return

    layer_specs = parse_layer_specs(args.layers)

    print("[Layers]")
    for spec in layer_specs:
        print(
            f"  {spec['label']}: hook={spec['hook_name']}, "
            f"feature_index={spec['feature_index']}"
        )

    catcher = MultiFeatureCatcher(
        model,
        layer_specs=layer_specs,
    )

    layer_name_for_file = safe_name(args.layers)
    recon_path = os.path.join(
        args.out_dir,
        f"recon_heads_{layer_name_for_file}.pth",
    )

    recon_heads = train_recon_heads(
        args=args,
        model=model,
        catcher=catcher,
        dataset=dataset,
        device=device,
        recon_path=recon_path,
    )

    visualize_all(
        args=args,
        model=model,
        catcher=catcher,
        recon_heads=recon_heads,
        dataset=dataset,
        img_infos=img_infos,
        anns_by_img=anns_by_img,
        cat_id_to_name=cat_id_to_name,
        device=device,
    )

    catcher.close()


if __name__ == "__main__":
    main()