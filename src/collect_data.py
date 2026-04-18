import cv2
import mediapipe as mp
import numpy as np
import os

# =========================
# CONFIG
# =========================
DATA_PATH = "data"
SEQUENCE_LENGTH = 30
NUM_SAMPLES = 110 # per gesture

# =========================
# INPUT LABEL
# =========================
label = input("Enter gesture label: ").lower()

save_path = os.path.join(DATA_PATH, label)
os.makedirs(save_path, exist_ok=True)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

print("\n🎯 Instructions:")
print("Press SPACE → start recording one sample")
print("Perform FULL gesture (1–2 sec)")
print("ESC → exit\n")

sample_count = 51

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        continue

    cv2.putText(frame, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Samples: {sample_count}/{NUM_SAMPLES}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)

    # =========================
    # START RECORDING
    # =========================
    if key == ord(" "):

        print(f"\nRecording sample {sample_count+1}/{NUM_SAMPLES}")

        sequence = []

        for frame_num in range(SEQUENCE_LENGTH):

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # =========================
            # FEATURE EXTRACTION
            # =========================
            left_lm = [0.0]*63
            right_lm = [0.0]*63

            if results.multi_hand_landmarks and results.multi_handedness:

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    lm_flat = []
                    for lm in hand_landmarks.landmark:
                        lm_flat.extend([lm.x, lm.y, lm.z])

                    label_handed = handedness.classification[0].label

                    if label_handed == "Left":
                        left_lm = lm_flat
                    else:
                        right_lm = lm_flat

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            features = left_lm + right_lm
            sequence.append(features)

            cv2.putText(frame, f"Recording {frame_num+1}/{SEQUENCE_LENGTH}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)

        # =========================
        # SAVE
        # =========================
        np.save(
            os.path.join(save_path, f"{sample_count}.npy"),
            np.array(sequence)
        )

        sample_count += 1
        print("✅ Saved")

    # =========================
    # EXIT
    # =========================
    if key == 27:
        break

    # =========================
    # AUTO STOP
    # =========================
    if sample_count >= NUM_SAMPLES:
        print("\n✅ Collection complete!")
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
hands.close()