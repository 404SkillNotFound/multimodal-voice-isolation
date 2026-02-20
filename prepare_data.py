"""
prepare_data.py
---------------
Run this once in Colab before training.

Two modes:
  A) You already have .mp4 files on Drive  →  use prepare_from_local_videos()
  B) You have a list of YouTube Shorts URLs →  use prepare_from_urls()

Output structure (one folder per 2-second segment):
  processed_segments/
    seg_00000/
      frames/
        0000.jpg
        0001.jpg
        ...  (50 frames)
      audio.wav  (2 seconds, 16 kHz mono)
    seg_00001/
      ...

Usage example at the bottom of this file.
"""

import os
import shutil
import subprocess
import numpy as np
import cv2
import librosa
import soundfile as sf


# ------------------------------------------------------------------ #
# constants                                                            #
# ------------------------------------------------------------------ #

SR = 16000           # audio sample rate
FPS_OUT = 25         # video frames per second we want to save
IMG_SIZE = 112       # frame size expected by r3d_18
SEG_DURATION = 2.0   # seconds per segment
FRAMES_PER_SEG = int(SEG_DURATION * FPS_OUT)  # 50


# ------------------------------------------------------------------ #
# low-level helpers                                                    #
# ------------------------------------------------------------------ #

def _ensure_yt_dlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Installing yt-dlp...")
        subprocess.run(["pip", "install", "yt-dlp", "-q"], check=True)


def _download_video(url, out_path):
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", out_path,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(out_path)


def _next_seg_index(output_dir):
    existing = [d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seg_")]
    return len(existing)


def _extract_segment(video_path, output_dir, seg_index, start_sec):
    """
    Extract one 2-second segment from video_path starting at start_sec.
    Saves FRAMES_PER_SEG jpg frames and one audio.wav.
    Returns True if segment was saved, False if skipped (too short / no audio).
    """
    seg_name = f"seg_{seg_index:05d}"
    seg_dir = os.path.join(output_dir, seg_name)
    frames_dir = os.path.join(seg_dir, "frames")

    # ---- video frames ----
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_sec * src_fps)
    src_frames_needed = int(SEG_DURATION * src_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw = []
    for _ in range(src_frames_needed):
        ok, frame = cap.read()
        if not ok:
            break
        raw.append(frame)
    cap.release()

    if len(raw) < src_frames_needed // 2:
        return False  # video too short at this offset

    # uniformly sample FRAMES_PER_SEG frames from raw
    indices = np.linspace(0, len(raw) - 1, FRAMES_PER_SEG, dtype=int)

    os.makedirs(frames_dir, exist_ok=True)
    for out_idx, src_idx in enumerate(indices):
        frame = cv2.resize(raw[src_idx], (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(frames_dir, f"{out_idx:04d}.jpg"), frame)

    # ---- audio ----
    y, _ = librosa.load(video_path, sr=SR, offset=start_sec, duration=SEG_DURATION)

    if len(y) < SR * 0.5:
        shutil.rmtree(seg_dir)
        return False

    # pad to exactly 2 seconds
    target_len = int(SR * SEG_DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # normalise loudness
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9

    sf.write(os.path.join(seg_dir, "audio.wav"), y, SR)
    return True


def _process_video_file(video_path, output_dir):
    """Split one video file into non-overlapping 2-second segments."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    total_duration = total_frames / src_fps
    num_segments = int(total_duration / SEG_DURATION)

    base_index = _next_seg_index(output_dir)
    created = 0

    for i in range(num_segments):
        start = i * SEG_DURATION
        ok = _extract_segment(video_path, output_dir, base_index + created, start)
        if ok:
            created += 1

    return created


# ------------------------------------------------------------------ #
# public API                                                           #
# ------------------------------------------------------------------ #

def prepare_from_local_videos(video_dir, output_dir):
    """
    Process all .mp4 / .avi / .mov files in video_dir.
    Call this if your YouTube Shorts are already downloaded on Drive.

    Example:
        from prepare_data import prepare_from_local_videos
        prepare_from_local_videos(
            video_dir  = "/content/drive/MyDrive/PE2_Project/raw_videos",
            output_dir = "/content/drive/MyDrive/PE2_Project/processed_segments",
        )
    """
    os.makedirs(output_dir, exist_ok=True)

    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = [
        os.path.join(video_dir, f)
        for f in sorted(os.listdir(video_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not video_files:
        print(f"No video files found in {video_dir}")
        return 0

    total = 0
    for i, vf in enumerate(video_files):
        print(f"[{i+1}/{len(video_files)}] {os.path.basename(vf)}", end=" ... ")
        n = _process_video_file(vf, output_dir)
        print(f"{n} segments")
        total += n

    print(f"\nDone. Total segments saved: {total}")
    return total


def prepare_from_urls(url_list, output_dir, download_dir="/tmp/yt_downloads"):
    """
    Download YouTube Shorts and process them into segments.

    Example:
        from prepare_data import prepare_from_urls
        urls = [
            "https://www.youtube.com/shorts/XXXXXXXXXX",
            ...
        ]
        prepare_from_urls(urls, "/content/drive/MyDrive/PE2_Project/processed_segments")
    """
    _ensure_yt_dlp()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    total = 0
    for i, url in enumerate(url_list):
        print(f"\n[{i+1}/{len(url_list)}] {url}")
        out_path = os.path.join(download_dir, f"video_{i:04d}.mp4")

        if not os.path.exists(out_path):
            print("  Downloading...")
            ok = _download_video(url, out_path)
            if not ok:
                print("  Download failed — skipping.")
                continue
        else:
            print("  Already downloaded.")

        n = _process_video_file(out_path, output_dir)
        print(f"  Extracted {n} segments.")
        total += n

    print(f"\nFinished. Total segments: {total}")
    return total


# ------------------------------------------------------------------ #
# quick sanity check                                                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # change these two paths to match your Drive layout
    VIDEO_DIR  = "/content/drive/MyDrive/PE2_Project/raw_videos"
    OUTPUT_DIR = "/content/drive/MyDrive/PE2_Project/processed_segments"

    prepare_from_local_videos(VIDEO_DIR, OUTPUT_DIR)
