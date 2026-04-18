import os
import cv2
import json
import numpy as np
import mediapipe as mp

# =========================
# CONFIG
# =========================
WLASL_JSON = "./wlasl/WLASL_v0.3.json"
VIDEOS_PATH = "./wlasl/videos"
SAVE_PATH = "filtered_data"
SEQUENCE_LENGTH = 30

TARGET_WORDS = [
    "hello", "yes", "no", "please", "thank you",
    "food", "water", "eat", "want", "help",
    "drink", "hungry", "more", "finish", "none"
]

os.makedirs(SAVE_PATH, exist_ok=True)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# =========================
# LOAD JSON
# =========================
with open(WLASL_JSON, "r") as f:
    data = json.load(f)

print(f"Total entries in WLASL: {len(data)}")

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(results):
    left = [0.0] * 63
    right = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = []
            for p in lm.landmark:
                coords.extend([p.x, p.y, p.z])

            label = handedness.classification[0].label
            if label == "Left":
                left = coords
            else:
                right = coords

    return left + right


def sample_sequence(frames):
    n = len(frames)
    if n == 0:
        return None

    if n < SEQUENCE_LENGTH:
        frames = frames + [frames[-1]] * (SEQUENCE_LENGTH - n)
        return np.array(frames)

    idx = np.linspace(0, n - 1, SEQUENCE_LENGTH, dtype=int)
    return np.array([frames[i] for i in idx])


# =========================
# PROCESS
# =========================
total_saved = 0

for entry in data:

    gloss = entry["gloss"].lower()

    # 🔥 FILTER HERE
    if gloss not in TARGET_WORDS:
        continue

    print(f"\nProcessing word: {gloss}")

    save_dir = os.path.join(SAVE_PATH, gloss)
    os.makedirs(save_dir, exist_ok=True)

    count = 0

    for instance in entry["instances"]:
        video_id = instance["video_id"]
        video_file = os.path.join(VIDEOS_PATH, f"{video_id}.mp4")

        if not os.path.exists(video_file):
            continue

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            continue

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            frames.append(extract_features(results))

        cap.release()

        if not frames:
            continue

        seq = sample_sequence(frames)
        if seq is None:
            continue

        np.save(os.path.join(save_dir, f"{video_id}.npy"), seq)
        count += 1
        total_saved += 1

    print(f"Saved {count} samples for {gloss}")

hands.close()

print("\n==============================")
print(f"✅ TOTAL SAMPLES SAVED: {total_saved}")
print("==============================")