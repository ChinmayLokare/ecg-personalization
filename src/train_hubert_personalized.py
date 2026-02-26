# src/train_hubert_personalized.py
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModel

from dataset import build_dataloaders
from evaluate import evaluate
from utils import set_seed
from models.hubert_personalized_head import HuBERTPersonalizedHead


def load_hubert_backbone(model_id: str, device: torch.device):
    backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


@torch.no_grad()
def eval_window_level(backbone, head, dataloader, device):
    head.eval()
    all_preds, all_targets = [], []

    for xb, yb, _rids in tqdm(dataloader, desc="[Eval windows]"):
        xb = xb.to(device)  # (B, 1, 3000)
        yb = yb.to(device)

        x_bt = xb.squeeze(1)  # (B, 3000)
        out = backbone(x_bt)
        hs = out.last_hidden_state
        pooled = hs.mean(dim=1)  # (B, 512)

        logits = head(pooled)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(yb.cpu().numpy().tolist())

    return evaluate(all_targets, all_preds, verbose=True)


@torch.no_grad()
def eval_record_level(backbone, head, dataloader, device, *, verbose=True):
    head.eval()

    rec_logits_sum = defaultdict(lambda: None)
    rec_counts = defaultdict(int)
    rec_true = {}

    for xb, yb, rids in tqdm(dataloader, desc="[Eval records]"):
        xb = xb.to(device)
        yb = yb.to(device)

        x_bt = xb.squeeze(1)
        out = backbone(x_bt)
        hs = out.last_hidden_state
        pooled = hs.mean(dim=1)      # (B, 512)

        logits = head(pooled)        # (B, 4)

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

    y_true, y_pred = [], []
    record_ids = []
    correct_list, confidence_list = [], []
    true_by_class = defaultdict(list)
    conf_by_class = defaultdict(list)

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


def print_variance_summary(record_stats):
    correct = record_stats["correct"]
    conf = record_stats["confidence"]

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


def train():
    set_seed(42)


    batch_size = 32
    lr = 1e-3
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_id = "Edoardo-BS/hubert-ecg-small"

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_dir / "hubert_personalized_best.pt"

    train_dl, val_dl, test_dl = build_dataloaders(
        project_root=project_root,
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )

    backbone = load_hubert_backbone(model_id, device)
    head = HuBERTPersonalizedHead(
        in_dim=512,
        num_classes=4,
        adapter_bottleneck=128,
        adapter_dropout=0.1,
        head_dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(head.parameters(), lr=lr)

    best_val_macro_f1 = -1.0

    for epoch in range(1, epochs + 1):
        head.train()
        running_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [Train adapter+head]")
        for xb, yb, _rids in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            x_bt = xb.squeeze(1)
            with torch.no_grad():
                out = backbone(x_bt)
                hs = out.last_hidden_state
                pooled = hs.mean(dim=1)  # (B, 512)

            logits = head(pooled)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = running_loss / len(train_dl.dataset)
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

        print("\nValidation (record-level):")
        val_metrics_rec, _ = eval_record_level(backbone, head, val_dl, device, verbose=True)
        val_macro_f1 = val_metrics_rec["macro_f1"]
        print(f"Epoch {epoch} Val Record Macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save({"head_state_dict": head.state_dict(), "model_id": model_id}, best_ckpt_path)
            print(f"Saved best personalized head to {best_ckpt_path}")

        print("-" * 60)

    print("\nLoading best personalized head for test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    print("\nTest (window-level):")
    eval_window_level(backbone, head, test_dl, device)

    print("\nTest (record-level):")
    _, record_stats = eval_record_level(backbone, head, test_dl, device, verbose=True)
    print_variance_summary(record_stats)


if __name__ == "__main__":
    train()