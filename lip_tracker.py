"""
lip_tracker.py — Lip movement detection using MediaPipe.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

UPPER_LIP = 13
LOWER_LIP = 14

def get_lip_movement(video_path, max_seconds=5):
    # Import inside function to avoid module-level failures
    import mediapipe as mp

    cap        = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * max_seconds)
    movements  = []
    timestamps = []

    # Try both import styles — covers mediapipe 0.10.x and older
    try:
        face_mesh_module = mp.solutions.face_mesh
        FaceMesh = face_mesh_module.FaceMesh
    except AttributeError:
        # Newer mediapipe API
        from mediapipe.tasks.python import vision
        print("⚠️ Falling back to tasks API — upgrade mediapipe if issues persist")
        FaceMesh = None

    if FaceMesh is not None:
        with FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.3
        ) as face_mesh:
            frame_idx = 0
            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm  = results.multi_face_landmarks[0].landmark
                    gap = abs(lm[LOWER_LIP].y - lm[UPPER_LIP].y)
                    movements.append(gap)
                else:
                    movements.append(0.0)
                timestamps.append(frame_idx / fps)
                frame_idx += 1
    else:
        # Fallback: use basic face detection + estimate lip gap from face height
        print("Using fallback face detector...")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        frame_idx = 0
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Approximate lip region (bottom third of face)
                lip_region = gray[y + 2*h//3 : y + h, x : x + w]
                # Variance in lip region correlates with mouth open/closed
                movements.append(float(np.std(lip_region)) / 255.0)
            else:
                movements.append(0.0)
            timestamps.append(frame_idx / fps)
            frame_idx += 1

    cap.release()
    return np.array(timestamps), np.array(movements)


def plot_lip_movement(times, lips):
    plt.figure(figsize=(12, 3))
    plt.plot(times, lips, color='royalblue', linewidth=1.5)
    plt.axhline(np.mean(lips), color='red',
                linestyle='--', label='Mean (speaking threshold)')
    plt.fill_between(times, lips, alpha=0.2, color='royalblue')
    plt.title('Lip Movement — Speaker Activity Detection')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Lip Gap')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lip_movement.png", dpi=150)
    plt.show()
    print(f"✅ Chart saved → outputs/lip_movement.png")


if __name__ == "__main__":
    RAW_DIR = "./raw_videos"
    videos  = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.mp4')])
    target_path = os.path.join(RAW_DIR, videos[0])
    print(f"Analyzing: {videos[0]}")
    times, lips = get_lip_movement(target_path)
    print(f"✅ {len(times)} frames analyzed")
    plot_lip_movement(times, lips)