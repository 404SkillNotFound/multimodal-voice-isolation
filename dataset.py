import os
import numpy as np
import cv2
import librosa
import torch
from torch.utils.data import Dataset


# these constants are shared with utils/audio_processor.py
SR = 16000
N_FFT = 512
HOP_LENGTH = 160
NUM_VIDEO_FRAMES = 50    # frames per clip (2 seconds at 25 fps)
NUM_AUDIO_FRAMES = 200   # STFT columns per clip (2 seconds at sr=16000, hop=160)
IMG_SIZE = 112            # r3d_18 expects 112x112

# ImageNet-style normalisation stats used when r3d_18 was pretrained on Kinetics
VIDEO_MEAN = np.array([0.43216, 0.39466, 0.37645], dtype=np.float32)
VIDEO_STD  = np.array([0.22803, 0.22145, 0.21699], dtype=np.float32)


class CocktailDataset(Dataset):
    """
    Scans a directory of processed segments.
    Each segment is a sub-folder containing:
        frames/0000.jpg, 0001.jpg, ...   — extracted video frames
        audio.wav                         — clean audio for that clip

    At training time, two clips are picked: the target (whose video we use) and an
    interferer.  Their audio magnitudes are added to simulate a noisy mixture.
    The IRM (Ideal Ratio Mask) is computed as target_mag / (mixture_mag + eps).
    The model learns to recover that mask given the video of the target speaker.
    """

    def __init__(self, data_dir):
        self.segments = sorted([
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        if len(self.segments) == 0:
            raise ValueError(f"No segment folders found in {data_dir}")

        print(f"CocktailDataset: {len(self.segments)} segments in {data_dir}")

    def __len__(self):
        return len(self.segments)

    # ------------------------------------------------------------------ #
    # internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_video(self, seg_dir):
        frames_dir = os.path.join(seg_dir, "frames")
        files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

        frames = []
        for fname in files[:NUM_VIDEO_FRAMES]:
            img = cv2.imread(os.path.join(frames_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            # per-channel normalisation matching Kinetics pretraining
            img = (img - VIDEO_MEAN) / VIDEO_STD
            frames.append(img)  # [H, W, C]

        # pad with black frames if the clip is shorter than expected
        while len(frames) < NUM_VIDEO_FRAMES:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

        video = np.stack(frames, axis=0)                # [T, H, W, C]
        video = torch.FloatTensor(video).permute(3, 0, 1, 2)  # [C, T, H, W]
        return video

    def _load_audio_mag(self, seg_dir):
        audio_path = os.path.join(seg_dir, "audio.wav")
        y, _ = librosa.load(audio_path, sr=SR)

        target_samples = SR * 2
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]

        stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag = np.abs(stft)  # [257, T]

        if mag.shape[1] < NUM_AUDIO_FRAMES:
            mag = np.pad(mag, ((0, 0), (0, NUM_AUDIO_FRAMES - mag.shape[1])))
        else:
            mag = mag[:, :NUM_AUDIO_FRAMES]

        return mag  # [257, NUM_AUDIO_FRAMES]

    # ------------------------------------------------------------------ #
    # main item getter                                                     #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx):
        target_dir = self.segments[idx]

        # pick a different segment as the interfering speaker
        interferer_idx = idx
        if len(self.segments) > 1:
            while interferer_idx == idx:
                interferer_idx = np.random.randint(0, len(self.segments))
        interferer_dir = self.segments[interferer_idx]

        video = self._load_video(target_dir)
        target_mag = self._load_audio_mag(target_dir)
        interferer_mag = self._load_audio_mag(interferer_dir)

        # random SNR between -5 dB and +5 dB so model is robust to volume imbalance
        snr_db = np.random.uniform(-5, 5)
        scale = 10 ** (-snr_db / 20.0)
        mixture_mag = target_mag + scale * interferer_mag

        # Ideal Ratio Mask: fraction of energy that belongs to the target speaker
        irm = target_mag / (mixture_mag + 1e-8)
        irm = np.clip(irm, 0.0, 1.0)

        return (
            video,                           # [3, 50, 112, 112]
            torch.FloatTensor(mixture_mag),  # [257, 200]
            torch.FloatTensor(irm),          # [257, 200]  — training target
        )
