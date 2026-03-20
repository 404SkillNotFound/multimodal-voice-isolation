"""
separate.py — Uses Meta's Demucs for real voice separation.
"""
import os
import numpy as np
import librosa
import soundfile as sf
import torch

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def separate_audio():
    print("Loading Demucs model (downloads ~300MB first time)...")
    
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # htdemucs_ft is the best model for speech separation
    model = get_model("htdemucs_ft")
    model.eval()
    print("✅ Model loaded")

    print("Loading mixed audio...")
    mix, sr = librosa.load(
        f"{OUTPUT_DIR}/mixed_input.wav", 
        sr=model.samplerate,   # Demucs needs its own sample rate (44100)
        mono=False
    )

    # Demucs expects stereo [2, T]
    if mix.ndim == 1:
        mix = np.stack([mix, mix], axis=0)

    mix_tensor = torch.tensor(mix).unsqueeze(0).float()  # [1, 2, T]

    print("Separating voices (this takes ~30s)...")
    with torch.no_grad():
        sources = apply_model(model, mix_tensor, device="cpu")
    # sources shape: [1, num_sources, 2, T]
    # Demucs splits into: drums, bass, other, vocals
    # For speech: vocals = target speaker, other = background/second speaker

    vocals = sources[0, model.sources.index("vocals")].mean(0).numpy()
    other  = sources[0, model.sources.index("other")].mean(0).numpy()

    def normalize(a):
        p = np.max(np.abs(a))
        return a / p if p > 0 else a

    sf.write(f"{OUTPUT_DIR}/separated_voice1.wav", normalize(vocals), model.samplerate)
    sf.write(f"{OUTPUT_DIR}/separated_voice2.wav", normalize(other),  model.samplerate)

    print("✅ Done!")
    print("   Voice 1 (vocals)  → outputs/separated_voice1.wav")
    print("   Voice 2 (other)   → outputs/separated_voice2.wav")

if __name__ == "__main__":
    separate_audio()