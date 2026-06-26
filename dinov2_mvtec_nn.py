import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MVTecDataset(Dataset):
    def __init__(self, root, category, split, input_size):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.input_size = input_size

        self.category_dir = self.root / category
        assert self.category_dir.exists(), f"Category dir not found: {self.category_dir}"

        self.image_paths = []
        self.labels = []
        self.defect_types = []

        split_dir = self.category_dir / split
        assert split_dir.exists(), f"Split dir not found: {split_dir}"

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            for p in sorted(defect_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    self.image_paths.append(p)
                    self.defect_types.append(defect_type)
                    self.labels.append(0 if defect_type == "good" else 1)

        self.img_tf = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.mask_resize = transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.image_paths)

    def _load_mask(self, img_path, defect_type):
        if self.split == "train" or defect_type == "good":
            return torch.zeros(1, self.input_size, self.input_size, dtype=torch.float32)

        mask_dir = self.category_dir / "ground_truth" / defect_type
        mask_name = img_path.stem + "_mask.png"
        mask_path = mask_dir / mask_name

        if not mask_path.exists():
            return torch.zeros(1, self.input_size, self.input_size, dtype=torch.float32)

        mask = Image.open(mask_path).convert("L")
        mask = self.mask_resize(mask)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()
        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        defect_type = self.defect_types[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.img_tf(img)

        mask = self._load_mask(img_path, defect_type)

        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path),
            "defect_type": defect_type,
        }


def list_mvtec_categories(data_root):
    data_root = Path(data_root)
    categories = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and (p / "train").exists() and (p / "test").exists():
            categories.append(p.name)
    return categories


def load_dinov2(model_name, device, hub_dir=None):
    if hub_dir is not None:
        torch.hub.set_dir(hub_dir)

    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        trust_repo=True,
    )
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_patch_tokens(model, images, device):
    images = images.to(device, non_blocking=True)

    if device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model.forward_features(images)
    else:
        out = model.forward_features(images)

    if isinstance(out, dict):
        tokens = out["x_norm_patchtokens"]
    else:
        tokens = out

    tokens = F.normalize(tokens.float(), dim=-1)

    n_patches = tokens.shape[1]
    grid_size = int(math.sqrt(n_patches))
    assert grid_size * grid_size == n_patches, f"Patch token number is not square: {n_patches}"

    return tokens, grid_size


def build_memory_bank(model, loader, device, max_bank_size):
    all_feats = []

    for batch in tqdm(loader, desc="Build memory bank"):
        images = batch["image"]
        feats, _ = extract_patch_tokens(model, images, device)
        feats = feats.reshape(-1, feats.shape[-1]).cpu()
        all_feats.append(feats)

    memory_bank = torch.cat(all_feats, dim=0)

    if max_bank_size > 0 and memory_bank.shape[0] > max_bank_size:
        perm = torch.randperm(memory_bank.shape[0])[:max_bank_size]
        memory_bank = memory_bank[perm]

    memory_bank = F.normalize(memory_bank, dim=-1)
    return memory_bank.to(device)


@torch.no_grad()
def nearest_cosine_distance(query_feats, memory_bank, query_chunk_size=4096):
    """
    query_feats: [B, N, C], normalized
    memory_bank: [M, C], normalized
    return: [B, N], cosine distance = 1 - max cosine similarity
    """
    b, n, c = query_feats.shape
    flat = query_feats.reshape(-1, c)

    scores = []
    for start in range(0, flat.shape[0], query_chunk_size):
        q = flat[start:start + query_chunk_size]
        sim = q @ memory_bank.T
        max_sim = sim.max(dim=1).values
        dist = 1.0 - max_sim
        scores.append(dist.cpu())

    scores = torch.cat(scores, dim=0)
    return scores.reshape(b, n)


def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if len(np.unique(labels)) < 2:
        return float("nan")

    return float(roc_auc_score(labels, scores))


def run_category(args, category, device):
    train_set = MVTecDataset(args.data_root, category, "train", args.input_size)
    test_set = MVTecDataset(args.data_root, category, "test", args.input_size)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"\n========== Category: {category} ==========")
    print(f"Train images: {len(train_set)}, Test images: {len(test_set)}")

    model = load_dinov2(args.model, device, args.hub_dir)

    memory_bank = build_memory_bank(
        model=model,
        loader=train_loader,
        device=device,
        max_bank_size=args.max_bank_size,
    )

    image_labels = []
    image_scores = []

    pixel_labels = []
    pixel_scores = []

    per_image_rows = []

    for batch in tqdm(test_loader, desc=f"Eval {category}"):
        images = batch["image"]
        masks = batch["mask"]
        labels = batch["label"].numpy()
        paths = batch["path"]
        defect_types = batch["defect_type"]

        feats, grid_size = extract_patch_tokens(model, images, device)
        patch_scores = nearest_cosine_distance(
            feats,
            memory_bank,
            query_chunk_size=args.query_chunk_size,
        )

        b = patch_scores.shape[0]
        patch_maps = patch_scores.reshape(b, 1, grid_size, grid_size)

        heatmaps = F.interpolate(
            patch_maps,
            size=(args.input_size, args.input_size),
            mode="bilinear",
            align_corners=False,
        )

        img_scores = patch_scores.max(dim=1).values.numpy()

        image_labels.extend(labels.tolist())
        image_scores.extend(img_scores.tolist())

        pixel_labels.extend(masks.reshape(-1).numpy().astype(np.uint8).tolist())
        pixel_scores.extend(heatmaps.cpu().reshape(-1).numpy().tolist())

        for p, dt, lab, score in zip(paths, defect_types, labels, img_scores):
            per_image_rows.append({
                "category": category,
                "path": p,
                "defect_type": dt,
                "label": int(lab),
                "image_score": float(score),
            })

    image_auroc = safe_auroc(image_labels, image_scores)
    pixel_auroc = safe_auroc(pixel_labels, pixel_scores)

    print(f"[{category}] image_AUROC = {image_auroc:.6f}")
    print(f"[{category}] pixel_AUROC = {pixel_auroc:.6f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "category": category,
        "model": args.model,
        "input_size": args.input_size,
        "max_bank_size": args.max_bank_size,
        "image_AUROC": image_auroc,
        "pixel_AUROC": pixel_auroc,
    }, per_image_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--model", type=str, default="dinov2_vits14")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-bank-size", type=int, default=30000)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="./dinov2_mvtec_results")
    parser.add_argument("--hub-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Input size: {args.input_size}")
    print(f"Max memory bank size: {args.max_bank_size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.category == "all":
        categories = list_mvtec_categories(args.data_root)
    else:
        categories = [args.category]

    summary_rows = []
    all_per_image_rows = []

    for category in categories:
        summary, per_image_rows = run_category(args, category, device)
        summary_rows.append(summary)
        all_per_image_rows.extend(per_image_rows)

        pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
        pd.DataFrame(all_per_image_rows).to_csv(output_dir / "per_image_scores.csv", index=False)

    print("\n========== Final Summary ==========")
    df = pd.DataFrame(summary_rows)
    print(df)

    print(f"\nSaved summary to: {output_dir / 'summary.csv'}")
    print(f"Saved per-image scores to: {output_dir / 'per_image_scores.csv'}")


if __name__ == "__main__":
    main()