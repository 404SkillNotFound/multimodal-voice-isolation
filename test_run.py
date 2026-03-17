"""
test_run.py
-----------
Quick sanity check. Run before training to confirm the model
forward pass works and shapes are correct.

    python test_run.py
"""

import torch
from models.fusion_net import CocktailNet


def test_architecture():
    print("--- ARCHITECTURE TEST ---")

    model = CocktailNet()
    print(f"Model created.")

    # dummy batch: B=2, matching training tensor shapes
    dummy_video = torch.randn(2, 3, 50, 112, 112)   # [B, C, T_v, H, W]
    dummy_audio = torch.randn(2, 257, 200)           # [B, freq_bins, T_a]

    print(f"Video input : {dummy_video.shape}")
    print(f"Audio input : {dummy_audio.shape}")

    output = model(dummy_video, dummy_audio)
    print(f"Mask output : {output.shape}")

    expected = (2, 257, 200)
    assert output.shape == expected, f"Shape mismatch: got {output.shape}, expected {expected}"
    assert output.min() >= 0.0 and output.max() <= 1.0, "Mask values out of [0, 1]"

    print("\n✅ PASSED: shapes correct, mask values in [0, 1]")


def test_with_cuda():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping GPU test.")
        return

    device = torch.device("cuda")
    model = CocktailNet().to(device)

    dummy_video = torch.randn(1, 3, 50, 112, 112).to(device)
    dummy_audio = torch.randn(1, 257, 200).to(device)

    with torch.amp.autocast("cuda"):
        output = model(dummy_video, dummy_audio)

    print(f"GPU test — output shape: {output.shape}")
    print("✅ PASSED: forward pass on CUDA with autocast")


if __name__ == "__main__":
    test_architecture()
    test_with_cuda()
