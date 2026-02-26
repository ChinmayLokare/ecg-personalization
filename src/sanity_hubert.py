# src/sanity_hubert.py
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import traceback
import torch


def try_forward(model, x, desc: str):
    print(f"\n--- Trying: {desc} ---")
    print("Input shape:", tuple(x.shape), "dtype:", x.dtype, "device:", x.device)
    try:
        with torch.no_grad():
            out = model(x)
        # HF models usually return a ModelOutput with .last_hidden_state
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            print("Success. last_hidden_state shape:", tuple(out.last_hidden_state.shape))
        else:
            # Fallback: print keys or type
            if isinstance(out, dict):
                print("Success. Output keys:", list(out.keys()))
            else:
                print("Success. Output type:", type(out))
                # sometimes out is a tuple
                if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    print("First tensor shape:", tuple(out[0].shape))
        return True
    except Exception as e:
        print("Failed with exception:", repr(e))
        print("Trace (last lines):")
        tb = traceback.format_exc().splitlines()
        for line in tb[-8:]:
            print(line)
        return False


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # If transformers isn't installed yet:
    # pip install transformers huggingface_hub safetensors
    from transformers import AutoModel  # noqa: E402

    size = "small"  # small/base/large
    model_id = f"Edoardo-BS/hubert-ecg-{size}"
    print("Loading:", model_id)

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    print("Model class:", type(model))
    print("Config class:", type(model.config))
    print("Config.model_type:", getattr(model.config, "model_type", None))
    print("Config.hidden_size:", getattr(model.config, "hidden_size", None))
    print("Config.conv_dim:", getattr(model.config, "conv_dim", None))
    print("Config.num_hidden_layers:", getattr(model.config, "num_hidden_layers", None))

    # HuBERT-ECG papers often use ~5s windows and mention 100Hz inputs after preprocessing.
    # We'll just probe a few shapes:
    #
    # A) (B, T)    like audio HuBERT expects
    # B) (B, 1, T) sometimes used for 1D conv frontends
    # C) (B, 12, T) intuitive 12-lead layout
    #
    # Use small T first so it runs fast.
    T = 500  # a "5s @ 100Hz" proxy length; this is just a probe
    x_bt = torch.randn(1, T, device=device)          # (B, T)
    x_b1t = torch.randn(1, 1, T, device=device)      # (B, 1, T)
    x_b12t = torch.randn(1, 12, T, device=device)    # (B, 12, T)

    ok = False
    ok = try_forward(model, x_bt, "shape (B, T)") or ok
    ok = try_forward(model, x_b1t, "shape (B, 1, T)") or ok
    ok = try_forward(model, x_b12t, "shape (B, 12, T)") or ok

    # Also probe your current window length for later compatibility (10s @ 300Hz = 3000)
    T2 = 3000
    x_bt2 = torch.randn(1, T2, device=device)
    x_b12t2 = torch.randn(1, 12, T2, device=device)
    ok = try_forward(model, x_bt2, "shape (B, T=3000)") or ok
    ok = try_forward(model, x_b12t2, "shape (B, 12, T=3000)") or ok

    if not ok:
        print("\nNo forward pass succeeded with the probed shapes.")
        print("Next step: we will inspect the model's feature encoder input expectations via model internals.")


if __name__ == "__main__":
    main()