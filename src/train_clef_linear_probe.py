# src/train_clef_linear_probe.py
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataset import build_dataloaders
from evaluate import evaluate
from utils import set_seed
from models.clef_head import CLEFClassifierHead


def load_clef_backbone(project_root: Path, ckpt_path: Path, device: torch.device):
    """
    Loads CLEF backbone from external repo without installing it as a package.
    Returns a model that maps (B, 1, T) -> (B, 256).
    """
    clef_repo_dir = project_root / "external" / "clef" / "ecg-foundation-model"
    if not clef_repo_dir.exists():
        raise FileNotFoundError(f"CLEF repo not found at: {clef_repo_dir}")

    sys.path.insert(0, str(clef_repo_dir))

    from clef.model_config import ModelConfig  # type: ignore

    cfg = {
        "checkpoint_path": str(ckpt_path.parent),
        "statekey_file": ckpt_path.name,
        "model_size": "small",
        "lead_config": "1lead",
        "num_classes": 4,
    }

    backbone = ModelConfig(name="clef").model_class(cfg).to(device)
    backbone.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    loaded = False

    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "ecg_model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                backbone.load_state_dict(ckpt[k], strict=False)
                loaded = True
                break

    if not loaded:
        backbone.load_state_dict(ckpt, strict=False)

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    return backbone


@torch.no_grad()
def eval_window_level(backbone, head, dataloader, device):
    head.eval()
    all_preds, all_targets = [], []

    for xb, yb, _rids in tqdm(dataloader, desc="[Eval windows]"):
        xb = xb.to(device)
        yb = yb.to(device)

        feats = backbone(xb)           # (B, 256)
        logits = head(feats)           # (B, 4)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(yb.cpu().numpy().tolist())

    return evaluate(all_targets, all_preds, verbose=True)


@torch.no_grad()
def eval_record_level(backbone, head, dataloader, device, *, verbose=True):
    """
    Aggregates window logits per record_id by averaging logits.
    Then predicts one label per record.
    Also returns per-record correctness/confidence for variance analysis.
    """
    head.eval()

    rec_logits_sum = defaultdict(lambda: None)  # record_id -> tensor(4)
    rec_counts = defaultdict(int)
    rec_true = {}  # record_id -> true label int

    for xb, yb, rids in tqdm(dataloader, desc="[Eval records]"):
        xb = xb.to(device)
        yb = yb.to(device)

        feats = backbone(xb)
        logits = head(feats)  # (B, 4)

        for i, rid in enumerate(rids):
            rid = str(rid)
            log_i = logits[i].detach().cpu()

            if rec_logits_sum[rid] is None:
                rec_logits_sum[rid] = log_i.clone()
            else:
                rec_logits_sum[rid] += log_i

            rec_counts[rid] += 1
            if rid not in rec_true:
                rec_true[rid] = int(yb[i].detach().cpu().item())

    # Build per-record predictions + stats
    y_true, y_pred = [], []
    record_ids = []
    correct_list = []
    confidence_list = []
    true_by_class = defaultdict(list)     # class_id -> list of 0/1 correctness
    conf_by_class = defaultdict(list)     # class_id -> list of confidences

    for rid, sum_logits in rec_logits_sum.items():
        avg_logits = sum_logits / rec_counts[rid]
        probs = torch.softmax(avg_logits, dim=0)
        pred = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())
        true = rec_true[rid]
        correct = 1 if pred == true else 0

        record_ids.append(rid)
        y_true.append(true)
        y_pred.append(pred)
        correct_list.append(correct)
        confidence_list.append(conf)

        true_by_class[true].append(correct)
        conf_by_class[true].append(conf)

    print(f"Num records evaluated: {len(y_true)}")
    metrics = evaluate(y_true, y_pred, verbose=verbose)

    record_stats = {
        "record_ids": record_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": correct_list,
        "confidence": confidence_list,
        "correct_by_class": {int(k): v for k, v in true_by_class.items()},
        "confidence_by_class": {int(k): v for k, v in conf_by_class.items()},
    }

    return metrics, record_stats


def train():
    set_seed(42)


    # Config
    batch_size = 32
    lr = 1e-3
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # CLEF checkpoint location
    clef_ckpt = project_root / "models" / "clef" / "clef_small.ckpt"
    if not clef_ckpt.exists():
        raise FileNotFoundError(f"Missing CLEF checkpoint at {clef_ckpt}")

    # Data
    train_dl, val_dl, test_dl = build_dataloaders(
        project_root=project_root,
        batch_size=batch_size,
        num_workers=0,  # Windows-friendly
        seed=42,
    )

    # Models
    backbone = load_clef_backbone(project_root, clef_ckpt, device)
    head = CLEFClassifierHead(in_dim=256, num_classes=4, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(head.parameters(), lr=lr)

    best_val_macro_f1 = -1.0
    best_ckpt_path = models_dir / "clef_linear_probe_best.pt"

    for epoch in range(1, epochs + 1):
        head.train()
        running_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [Train head]")
        for xb, yb, _rids in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.no_grad():
                feats = backbone(xb)  # (B, 256)

            logits = head(feats)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = running_loss / len(train_dl.dataset)
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

        # Validation (record-level is closer to your professor's request; we compute both)
        print("\nValidation (window-level):")
        val_metrics_win = eval_window_level(backbone, head, val_dl, device)

        print("\nValidation (record-level, avg logits across windows):")
        val_metrics_rec, _ = eval_record_level(backbone, head, val_dl, device)

        val_macro_f1 = val_metrics_rec["macro_f1"]
        print(f"Epoch {epoch} Val Record Macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save({"head_state_dict": head.state_dict()}, best_ckpt_path)
            print(f"Saved best head to {best_ckpt_path}")

        print("-" * 60)

    print("\nTest (record-level):")
    test_metrics, record_stats = eval_record_level(backbone, head, test_dl, device, verbose=True)

    # --- Variance across "patients" (record IDs) ---
    correct = record_stats["correct"]  # list of 0/1 per record
    conf = record_stats["confidence"]  # list of max softmax prob per record

    mean_correct = sum(correct) / len(correct)
    var_correct = sum((c - mean_correct) ** 2 for c in correct) / len(correct)
    std_correct = var_correct ** 0.5

    mean_conf = sum(conf) / len(conf)
    var_conf = sum((p - mean_conf) ** 2 for p in conf) / len(conf)
    std_conf = var_conf ** 0.5

    print("\nRecord-level variance summary (record_id as patient proxy):")
    print(f"Num records: {len(correct)}")
    print(f"Mean correctness (record accuracy): {mean_correct:.4f}")
    print(f"Std correctness: {std_correct:.4f}")
    print(f"Mean confidence (max prob): {mean_conf:.4f}")
    print(f"Std confidence: {std_conf:.4f}")

    # --- Per-class variance summary ---
    print("\nPer-class record-level summary:")
    for cls_id in [0, 1, 2, 3]:
        cls_correct = record_stats["correct_by_class"].get(cls_id, [])
        cls_conf = record_stats["confidence_by_class"].get(cls_id, [])
        if len(cls_correct) == 0:
            print(f"Class {cls_id}: no records")
            continue

        m_c = sum(cls_correct) / len(cls_correct)
        m_p = sum(cls_conf) / len(cls_conf)
        print(f"Class {cls_id} | n={len(cls_correct)} | mean_correct={m_c:.4f} | mean_conf={m_p:.4f}")


if __name__ == "__main__":
    train()