# 🎙️ Multimodal Voice Isolation (The Cocktail Party Problem)


This project implements a **Multimodal Late Fusion Pipeline** to isolate a target speaker's voice from a noisy, overlapping audio environment. 

## The Approach
Unlike traditional "blind" source separation, our system utilizes **Visual Activity Detection (VAD)** to anchor the audio isolation process:

1.  **Visual Frontend:** Uses MediaPipe FaceMesh (Landmarks 13 & 14) to track lip-gap distance.
2.  **Audio Backend:** Employs STFT-based Spectral Masking to separate overlapping speech signals.
3.  **Cross-Modal Fusion:** Uses the **Pearson Correlation Coefficient** to mathematically match the isolated audio track with the physical lip movements of the speaker.

## Tech Stack
- **Vision:** MediaPipe (FaceMesh Landmark Tracking)
- **Audio:** Librosa, Soundfile, MoviePy (DSP & Signal Processing)
- **Decision Logic:** NumPy-based Pearson Correlation matching
- **Backend:** FFmpeg 8.1

## Performance
The system calculates a synchronization score ($r$) between the visual "rhythm" and audio energy. A high correlation (typically $r > 0.7$) indicates a successful match between the video feed and the isolated voice.