"""Lightweight dataset wrappers used by GAPNet training scripts.

This module recreates the expected ``Dataset`` class that the original
training code imports.  The implementation focuses on the SOD setting used
on Kaggle but keeps the VSOD hooks (`use_flow`) so downstream scripts keep
working.  The loader accepts either explicit `.txt`/`.lst` split files or
raw directory structures containing image/mask pairs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

__all__ = ["Dataset"]


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_MASK_EXTENSIONS = _IMAGE_EXTENSIONS | {".gif"}


@dataclass(frozen=True)
class _SODSample:
    """Container describing a static image sample."""

    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class _VSODSample:
    """Container describing a video sample with optional optical flow."""

    image_path: Path
    flow_path: Path
    mask_path: Path


class Dataset(TorchDataset):
    """Dataset wrapper matching the original GAPNet training code expectations."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        dataset_name: str,
        transform: Optional[Callable[..., Sequence[torch.Tensor]]] = None,
        process_label: bool = False,
        ignore_index: bool = False,
        use_flow: bool = False,
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.transform = transform
        self.process_label = process_label
        self.ignore_index = ignore_index
        self.use_flow = use_flow

        if self.use_flow:
            self.samples: List[_VSODSample] = self._load_vsod_samples(dataset_name)
        else:
            self.samples = self._load_sod_samples(dataset_name)

        if not self.samples:
            raise RuntimeError(
                f"Dataset '{dataset_name}' resolved zero usable samples under {self.root}."
            )

    # ------------------------------------------------------------------
    # Sample discovery helpers
    # ------------------------------------------------------------------
    def _load_sod_samples(self, dataset_name: str) -> List[_SODSample]:
        """Load static image samples (SOD training/eval)."""

        list_file = self.root / f"{dataset_name}.txt"
        if list_file.exists():
            return self._parse_sod_split(list_file)

        dataset_dir = self.root / dataset_name
        if dataset_dir.exists():
            return self._scan_sod_directory(dataset_dir)

        raise FileNotFoundError(
            f"Could not locate '{dataset_name}.txt' or directory '{dataset_name}' "
            f"inside {self.root}."
        )

    def _load_vsod_samples(self, dataset_name: str) -> List[_VSODSample]:
        """Load video samples using `.lst` split files."""

        list_file = self.root / f"{dataset_name}.lst"
        if not list_file.exists():
            raise FileNotFoundError(
                f"VSOD dataset '{dataset_name}' expects list file {list_file}."
            )

        samples: List[_VSODSample] = []
        for line in list_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rgb_rel, flow_rel, mask_rel = parts[:3]
            rgb_path = (self.root / rgb_rel).resolve()
            flow_path = (self.root / flow_rel).resolve()
            mask_path = (self.root / mask_rel).resolve()
            if rgb_path.exists() and mask_path.exists():
                samples.append(_VSODSample(rgb_path, flow_path, mask_path))
        return samples

    def _parse_sod_split(self, list_file: Path) -> List[_SODSample]:
        samples: List[_SODSample] = []
        for line in list_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rgb_rel, mask_rel = parts[:2]
            rgb_path = (self.root / rgb_rel).resolve()
            mask_path = (self.root / mask_rel).resolve()
            if rgb_path.exists() and mask_path.exists():
                samples.append(_SODSample(rgb_path, mask_path))
        return samples

    def _scan_sod_directory(self, dataset_dir: Path) -> List[_SODSample]:
        image_dirs: List[Path] = []
        mask_dirs: List[Path] = []

        for child in dataset_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            if any(token in name for token in ("image", "img", "rgb", "frame")):
                image_dirs.append(child)
            if any(token in name for token in ("mask", "gt", "ann", "label")):
                mask_dirs.append(child)

        if not image_dirs:
            image_dirs = [dataset_dir]
        if not mask_dirs:
            mask_dirs = [dataset_dir]

        masks = self._build_mask_index(mask_dirs)
        samples: List[_SODSample] = []
        for image_dir in image_dirs:
            for path in image_dir.rglob("*"):
                if path.suffix.lower() not in _IMAGE_EXTENSIONS or not path.is_file():
                    continue
                key = self._canonical_key(path)
                mask_path = masks.get(key)
                if mask_path is not None:
                    samples.append(_SODSample(path.resolve(), mask_path.resolve()))
        return samples

    def _build_mask_index(self, mask_dirs: Iterable[Path]) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for mask_dir in mask_dirs:
            for path in mask_dir.rglob("*"):
                if path.suffix.lower() not in _MASK_EXTENSIONS or not path.is_file():
                    continue
                index[self._canonical_key(path)] = path
        return index

    @staticmethod
    def _canonical_key(path: Path) -> str:
        key = path.stem.lower()
        for suffix in ("_mask", "-mask", "_gt", "-gt", "_fix", "-fix"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key

    # ------------------------------------------------------------------
    # PyTorch dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int):  # type: ignore[override]
        if self.use_flow:
            sample = self.samples[index]
            image = self._imread(sample.image_path)
            mask = self._imread(sample.mask_path, grayscale=True)
            flow = self._load_flow(sample.flow_path)
        else:
            sample = self.samples[index]
            image = self._imread(sample.image_path)
            mask = self._imread(sample.mask_path, grayscale=True)
            flow = None

        if self.transform is not None:
            transformed = self.transform(image, mask, flow)
            if len(transformed) == 3:
                image_tensor, mask_tensor, flow_tensor = transformed
            else:
                image_tensor, mask_tensor = transformed
                flow_tensor = None
        else:
            image_tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            flow_tensor = (
                torch.from_numpy(flow[:, :, ::-1].copy()).permute(2, 0, 1).float()
                if flow is not None
                else None
            )

        mask_tensor = mask_tensor.float()
        if self.process_label:
            mask_tensor = mask_tensor.clone()
            base = mask_tensor
            if base.dim() == 3 and base.size(0) == 1:
                base = base.squeeze(0)
            base = base.float()
            # Replicate the primary annotation into the six supervision slots.
            stacked = torch.stack([base] * 6, dim=0)
            mask_tensor = stacked
        else:
            if mask_tensor.dim() == 3 and mask_tensor.size(0) == 1:
                mask_tensor = mask_tensor.squeeze(0)

        if self.use_flow:
            if flow_tensor is None:
                flow_tensor = torch.zeros_like(image_tensor)
            return image_tensor.float(), flow_tensor.float(), mask_tensor.float()
        return image_tensor.float(), mask_tensor.float()

    # ------------------------------------------------------------------
    # Image utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _imread(path: Path, grayscale: bool = False) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        data = cv2.imread(str(path), flag)
        if data is None:
            raise FileNotFoundError(f"Failed to read image at {path}")
        return data

    @staticmethod
    def _load_flow(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            data = np.load(str(path))
            if data.ndim == 2:
                data = np.stack([data, data, data], axis=-1)
            return data.astype(np.float32)
        return Dataset._imread(path, grayscale=False)
