# src/dataset.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


# Fixed label mapping (document this in README and never change)
LABEL_MAP: Dict[str, int] = {"N": 0, "A": 1, "O": 2, "~": 3}
INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

TARGET_FS = 300  # CinC 2017 is already 300 Hz
WINDOW_SEC = 10
WINDOW_SAMPLES = TARGET_FS * WINDOW_SEC  # 3000
STRIDE_SAMPLES = WINDOW_SAMPLES          # no overlap
EPS = 1e-8


@dataclass(frozen=True)
class Paths:
    project_root: Path
    raw_dir: Path
    training_dir: Path
    labels_csv: Path
    splits_json: Path


def resolve_paths(project_root: str | Path) -> Paths:
    """
    Expected data layout:
      <project_root>/
        data/raw/
          training2017/
            A00001.mat, A00001.hea, ...
          REFERENCE-V3.csv   (or your renamed labels.csv)
    """
    project_root = Path(project_root).resolve()
    raw_dir = project_root / "data" / "raw"

    # Adjust these if your filenames differ
    training_dir = raw_dir / "training2017"

    # If you renamed REFERENCE-V3.csv to labels.csv, keep it as labels.csv
    # Otherwise, set this to raw_dir / "REFERENCE-V3.csv"
    labels_csv = raw_dir / "labels.csv"

    splits_json = raw_dir / "splits.json"
    return Paths(project_root, raw_dir, training_dir, labels_csv, splits_json)


def load_label_index(labels_csv: Path) -> List[Tuple[str, int]]:
    """
    Reads CSV with rows like: record_id,label_char
    Example: A00001,N
    Returns list of (record_id, label_int).
    """
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    records: List[Tuple[str, int]] = []
    with labels_csv.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            record_id = row[0].strip()

            # Handle possible header row
            if record_id.lower() in {"record", "record_id", "recordid"}:
                continue

            label_char = row[1].strip()
            if label_char not in LABEL_MAP:
                raise ValueError(f"Unknown label '{label_char}' in row: {row}")

            records.append((record_id, LABEL_MAP[label_char]))

    if len(records) == 0:
        raise ValueError("No records found in labels CSV.")

    return records


def make_splits(
    records: List[Tuple[str, int]],
    splits_json: Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, List[str]]:
    """
    Split by record_id (prevents leakage). Saves splits to disk for reproducibility.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    record_ids = [rid for rid, _ in records]

    rng = np.random.default_rng(seed)
    rng.shuffle(record_ids)

    n = len(record_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = record_ids[:n_train]
    val_ids = record_ids[n_train : n_train + n_val]
    test_ids = record_ids[n_train + n_val :]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    splits_json.parent.mkdir(parents=True, exist_ok=True)
    with splits_json.open("w") as f:
        json.dump(splits, f, indent=2)

    return splits


def load_or_create_splits(
    records: List[Tuple[str, int]],
    splits_json: Path,
    seed: int = 42,
) -> Dict[str, List[str]]:
    if splits_json.exists():
        with splits_json.open("r") as f:
            splits = json.load(f)
        # Basic sanity
        for k in ("train", "val", "test"):
            if k not in splits:
                raise ValueError(f"Invalid splits file, missing key: {k}")
        return splits

    return make_splits(records, splits_json, seed=seed)


def _extract_signal_from_mat(mat_path: Path) -> np.ndarray:
    """
    CinC 2017 .mat usually has a variable named 'val' shaped (1, N).
    This function returns a 1D float32 array of shape (N,).
    """
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    m = loadmat(mat_path)

    if "val" not in m:
        # Fall back: find the first ndarray that looks like a signal
        candidates = [v for v in m.values() if isinstance(v, np.ndarray)]
        if not candidates:
            raise ValueError(f"No ndarray found in {mat_path.name}")
        arr = candidates[0]
    else:
        arr = m["val"]

    arr = np.asarray(arr)

    # Expect (1, N) or (N, 1) or (N,)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            sig = arr[0]
        elif arr.shape[1] == 1:
            sig = arr[:, 0]
        else:
            # If it's unexpectedly multi-dim, flatten
            sig = arr.reshape(-1)
    elif arr.ndim == 1:
        sig = arr
    else:
        sig = arr.reshape(-1)

    return sig.astype(np.float32)


def normalize_per_record(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / (sigma + EPS)


def windows_from_record(x: np.ndarray) -> List[np.ndarray]:
    """
    Windowing rules:
      - window size = 3000 samples (10 sec at 300 Hz)
      - stride = 3000 (no overlap)
      - drop tail if < window (Option 1)
      - if record shorter than window, pad to window and keep one window
    """
    n = x.shape[0]

    if n < WINDOW_SAMPLES:
        padded = np.zeros((WINDOW_SAMPLES,), dtype=np.float32)
        padded[:n] = x
        return [padded]

    out: List[np.ndarray] = []
    start = 0
    while start + WINDOW_SAMPLES <= n:
        out.append(x[start : start + WINDOW_SAMPLES])
        start += STRIDE_SAMPLES

    return out


class CinC2017WindowedDataset(Dataset):
    """
    Dataset that returns (x, y) where:
      x: torch.FloatTensor [1, 3000]
      y: torch.LongTensor  scalar (0..3)
    """

    def __init__(
        self,
        records: List[Tuple[str, int]],
        training_dir: Path,
        record_ids_subset: List[str],
        precompute_index: bool = True,
    ):
        self.training_dir = training_dir
        self.id_to_label: Dict[str, int] = {rid: y for rid, y in records}

        # Only keep records in this split
        self.record_ids = [rid for rid in record_ids_subset if rid in self.id_to_label]
        if len(self.record_ids) == 0:
            raise ValueError("No record IDs found for this split.")

        # Build an index of (record_id, window_idx) so __len__ is known
        self.index: List[Tuple[str, int]] = []
        self._cached_windows: Dict[str, List[np.ndarray]] = {}

        if precompute_index:
            for rid in self.record_ids:
                windows = self._load_and_prepare_windows(rid)
                self._cached_windows[rid] = windows
                for w_i in range(len(windows)):
                    self.index.append((rid, w_i))
        else:
            # If not precomputing, we only store record ids and generate on the fly.
            # __len__ will be approximate unless you precompute lengths; keep precompute_index=True for now.
            raise ValueError("Set precompute_index=True for the first version (recommended).")

    def _load_and_prepare_windows(self, record_id: str) -> List[np.ndarray]:
        mat_path = self.training_dir / f"{record_id}.mat"
        x = _extract_signal_from_mat(mat_path)
        x = normalize_per_record(x)          # per-record normalization
        return windows_from_record(x)        # then windowing

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rid, w_i = self.index[idx]
        y = self.id_to_label[rid]

        windows = self._cached_windows.get(rid)
        if windows is None:
            windows = self._load_and_prepare_windows(rid)

        xw = windows[w_i]  # np.ndarray (3000,)
        x_tensor = torch.from_numpy(xw).unsqueeze(0)  # [1, 3000]
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor, rid


def build_dataloaders(
    project_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function:
      - loads labels
      - loads/creates deterministic record-level splits
      - creates windowed datasets and dataloaders
    """
    p = resolve_paths(project_root)

    records = load_label_index(p.labels_csv)
    if len(records) != 8528:

        print(f"Warning: expected 8528 records; found {len(records)} in {p.labels_csv.name}")

    splits = load_or_create_splits(records, p.splits_json, seed=seed)

    train_ds = CinC2017WindowedDataset(records, p.training_dir, splits["train"], precompute_index=True)
    val_ds = CinC2017WindowedDataset(records, p.training_dir, splits["val"], precompute_index=True)
    test_ds = CinC2017WindowedDataset(records, p.training_dir, splits["test"], precompute_index=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    # Quick sanity run (edit project_root if needed)
    project_root = Path(__file__).resolve().parents[1]  # assumes src/ is one level under project root
    train_dl, val_dl, test_dl = build_dataloaders(project_root, batch_size=32, num_workers=0)

    xb, yb = next(iter(train_dl))
    print("Batch x shape:", xb.shape)  # [B, 1, 3000]
    print("Batch y shape:", yb.shape)  # [B]
    print("Unique labels in batch:", sorted(set(yb.tolist())))
    print("Train windows:", len(train_dl.dataset))
    print("Val windows:", len(val_dl.dataset))
    print("Test windows:", len(test_dl.dataset))