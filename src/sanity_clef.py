
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import sys
from pathlib import Path

import torch


def main():
    # ---- Project paths ----
    project_root = Path(__file__).resolve().parents[1]

    # If you cloned CLEF into: <project_root>/external/clef/
    clef_repo_dir = project_root / "external" / "clef" / "ecg-foundation-model"
    if not clef_repo_dir.exists():
        raise FileNotFoundError(
            f"Could not find CLEF repo at {clef_repo_dir}. "
            f"Clone the CLEF repo into external/clef/ first."
        )

    # Add CLEF repo to Python path so we can import it
    sys.path.insert(0, str(clef_repo_dir))

    # ---- Checkpoint path ----
    # Put your downloaded checkpoint here:
    # <project_root>/models/clef/clef_small.ckpt
    ckpt_path = project_root / "models" / "clef" / "clef_small.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Download a CLEF checkpoint and place it there (or edit ckpt_path)."
        )

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- CLEF imports (from their repo) ----
    from clef.model_config import ModelConfig  # type: ignore

    # ---- Config (match their quickstart) ----
    cfg = {
        "checkpoint_path": str(ckpt_path.parent),
        "statekey_file": ckpt_path.name,
        "model_size": "small",
        "lead_config": "1lead",
        "num_classes": 4,
        # Many CLEF configs default to input_len=5000; we test both 5000 and 3000
        # "input_len": 5000,
    }

    # ---- Instantiate backbone ----
    backbone = ModelConfig(name="clef").model_class(cfg).to(device)
    backbone.eval()
    print("Backbone created:", getattr(backbone, "name", type(backbone)))

    # ---- Load checkpoint (best-effort like their notebook) ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    loaded = False

    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "ecg_model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                backbone.load_state_dict(ckpt[k], strict=False)
                loaded = True
                break

    if not loaded:
        try:
            backbone.load_state_dict(ckpt, strict=False)
            loaded = True
        except Exception:
            loaded = False

    print("Checkpoint loaded:", loaded)

    # ---- Forward pass tests ----
    def forward_test(seq_len: int):
        x = torch.randn(1, 1, seq_len, device=device)
        with torch.no_grad():
            out = backbone(x)
        print(f"Input shape: {tuple(x.shape)} -> Output shape: {tuple(out.shape)}")

    # Test the notebook default length
    forward_test(5000)

    # Also test your dataset window length (10s @ 300Hz = 3000)
    forward_test(3000)

    print("Sanity check complete.")


if __name__ == "__main__":
    # Avoid noisy OpenMP duplication issues on some Windows setups
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()

