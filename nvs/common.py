from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from dinov2_mvtec_nn import (  # noqa: E402
    extract_patch_tokens,
    load_dinov2,
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    defect_type: str
    mask_path: Path | None = None


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )


def build_mvtec_records(root: str | Path, category: str) -> tuple[list[ImageRecord], list[ImageRecord]]:
    category_root = Path(root) / category
    train_good = category_root / "train" / "good"
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"
    if not train_good.is_dir():
        raise FileNotFoundError(f"Missing train/good directory: {train_good}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_root}")
    train_records = [
        ImageRecord(path=path, label=0, defect_type="good")
        for path in list_images(train_good)
    ]
    test_records: list[ImageRecord] = []
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        defect_type = defect_dir.name
        for image_path in list_images(defect_dir):
            if defect_type == "good":
                test_records.append(ImageRecord(image_path, 0, "good", None))
            else:
                mask_path = ground_truth_root / defect_type / f"{image_path.stem}_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
                test_records.append(ImageRecord(image_path, 1, defect_type, mask_path))
    return train_records, test_records


def split_normal_records(
    records: list[ImageRecord],
    calibration_fraction: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    generator = np.random.default_rng(int(seed))
    indices = generator.permutation(len(records))
    calibration_count = max(2, int(round(len(records) * float(calibration_fraction))))
    calibration_count = min(calibration_count, len(records) - 1)
    calibration_indices = set(indices[:calibration_count].tolist())
    memory = [record for index, record in enumerate(records) if index not in calibration_indices]
    calibration = [record for index, record in enumerate(records) if index in calibration_indices]
    return memory, calibration


def split_records(
    records: list[ImageRecord],
    fraction: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if len(records) < 2:
        raise ValueError("At least two records are required")
    generator = np.random.default_rng(int(seed))
    indices = generator.permutation(len(records))
    first_count = int(round(len(records) * float(fraction)))
    first_count = min(max(1, first_count), len(records) - 1)
    first = set(indices[:first_count].tolist())
    return (
        [record for index, record in enumerate(records) if index in first],
        [record for index, record in enumerate(records) if index not in first],
    )


def image_to_dino_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - DINO_MEAN) / DINO_STD


def resize_pil(image: Image.Image, size: int, is_mask: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    return image.convert("L" if is_mask else "RGB").resize((size, size), resample)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return torch.from_numpy((array > 0).astype(np.uint8))


class RecordDataset(Dataset):
    def __init__(
        self,
        records: Iterable[ImageRecord],
        input_size: int,
        include_mask: bool = False,
        transform_spec: dict[str, Any] | None = None,
    ) -> None:
        from .transforms import apply_transform

        self.records = list(records)
        self.input_size = int(input_size)
        self.include_mask = bool(include_mask)
        self.transform_spec = transform_spec
        self._apply_transform = apply_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.path) as handle:
            image = resize_pil(handle, self.input_size, is_mask=False)
        if self.transform_spec is not None:
            image = self._apply_transform(image, self.transform_spec, seed=index)
        item: dict[str, Any] = {
            "image": image_to_dino_tensor(image),
            "label": int(record.label),
            "path": str(record.path),
            "defect_type": str(record.defect_type),
        }
        if self.include_mask:
            if record.mask_path is None:
                mask = Image.new("L", (self.input_size, self.input_size), color=0)
            else:
                with Image.open(record.mask_path) as handle:
                    mask = resize_pil(handle, self.input_size, is_mask=True)
            item["mask"] = mask_to_tensor(mask)
        return item


def collate_records(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.tensor([int(item["label"]) for item in batch], dtype=torch.int64),
        "path": [str(item["path"]) for item in batch],
        "defect_type": [str(item["defect_type"]) for item in batch],
    }
    if "mask" in batch[0]:
        output["mask"] = torch.stack([item["mask"] for item in batch])
    return output


@torch.inference_mode()
def extract_features(
    model,
    loader,
    device: torch.device,
    keep_on_device: bool = False,
) -> tuple[torch.Tensor, list[str], int]:
    chunks: list[torch.Tensor] = []
    paths: list[str] = []
    grid_side = 0
    for batch in loader:
        features, grid_side = extract_patch_tokens(model, batch["image"], device)
        chunks.append(features if keep_on_device else features.cpu())
        paths.extend(batch["path"])
    return torch.cat(chunks, dim=0), paths, int(grid_side)


def build_memory_bank(features: torch.Tensor, max_bank_size: int, seed: int) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1]).float().cpu()
    if int(max_bank_size) > 0 and flat.shape[0] > int(max_bank_size):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.randperm(flat.shape[0], generator=generator)[: int(max_bank_size)]
        flat = flat[indices]
    return F.normalize(flat, dim=-1).contiguous()


@torch.inference_mode()
def nearest_with_indices(
    query_features: torch.Tensor,
    memory_bank: torch.Tensor,
    query_chunk_size: int,
    bank_chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_features.ndim != 3:
        raise ValueError("query_features must have shape [B, P, C]")
    b, p, c = query_features.shape
    flat = query_features.reshape(-1, c).float()
    memory_cpu = F.normalize(memory_bank.float().cpu(), dim=-1)
    memory_gpu: torch.Tensor | None = None
    if torch.device(device).type == "cuda":
        try:
            # High-VRAM machines are much faster if the bank stays resident on
            # GPU. The old path copied every bank chunk from CPU to GPU for
            # every query chunk, which is painful on CPU-bound servers.
            memory_gpu = memory_cpu.to(device, non_blocking=True)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            memory_gpu = None
            torch.cuda.empty_cache()
    all_distances: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for query_start in range(0, flat.shape[0], int(query_chunk_size)):
        query = flat[query_start : query_start + int(query_chunk_size)].to(
            device,
            non_blocking=True,
        )
        best_sim = torch.full((query.shape[0],), -float("inf"), device=device)
        best_idx = torch.full((query.shape[0],), -1, dtype=torch.long, device=device)
        memory_rows = (
            memory_gpu.shape[0] if memory_gpu is not None else memory_cpu.shape[0]
        )
        for bank_start in range(0, memory_rows, int(bank_chunk_size)):
            if memory_gpu is None:
                bank = memory_cpu[bank_start : bank_start + int(bank_chunk_size)].to(
                    device,
                    non_blocking=True,
                )
            else:
                bank = memory_gpu[bank_start : bank_start + int(bank_chunk_size)]
            similarity = query @ bank.T
            local_sim, local_idx = similarity.max(dim=1)
            update = local_sim > best_sim
            best_sim[update] = local_sim[update]
            best_idx[update] = local_idx[update] + bank_start
        all_distances.append((1.0 - best_sim).cpu())
        all_indices.append(best_idx.cpu())
    return (
        torch.cat(all_distances).reshape(b, p),
        torch.cat(all_indices).reshape(b, p),
    )


def patch_scores_to_maps(scores: torch.Tensor, grid_side: int, output_size: int) -> torch.Tensor:
    maps = scores.reshape(scores.shape[0], 1, int(grid_side), int(grid_side))
    return torch.nn.functional.interpolate(
        maps,
        size=(int(output_size), int(output_size)),
        mode="bilinear",
        align_corners=False,
    )[:, 0].cpu()

