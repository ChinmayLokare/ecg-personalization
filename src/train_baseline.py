# src/train_baseline.py
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path

from dataset import build_dataloaders
from models.baseline_cnn import BaselineECGCNN
from evaluate import evaluate
from utils import set_seed





def train():
    set_seed(42)

    # Config
    batch_size = 32
    lr = 1e-3
    epochs = 10
    device = torch.device("cpu")

    print(f"Using device: {device}")

    # Paths
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_dir / "baseline_best.pt"

    # Data
    train_dl, val_dl, test_dl = build_dataloaders(
        project_root=project_root,
        batch_size=batch_size,
        num_workers=0,  # safer for Windows + CPU
        seed=42,
    )

    # Model
    model = BaselineECGCNN(num_classes=4, in_channels=1).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_macro_f1 = -1.0

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [Train]")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_dl.dataset)
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in tqdm(val_dl, desc=f"Epoch {epoch}/{epochs} [Val]"):
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(yb.cpu().numpy().tolist())

        metrics = evaluate(all_targets, all_preds, verbose=True)
        val_macro_f1 = metrics["macro_f1"]
        print(f"Epoch {epoch} Val Macro-F1: {val_macro_f1:.4f}")

        # ---- Checkpoint ----
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Saved new best model to {best_ckpt_path}")

        print("-" * 50)

    # ---- Final Test Evaluation ----
    print("Loading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in tqdm(test_dl, desc="[Test]"):
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(yb.cpu().numpy().tolist())

    print("\nFinal Test Metrics:")
    evaluate(all_targets, all_preds, verbose=True)


if __name__ == "__main__":
    train()