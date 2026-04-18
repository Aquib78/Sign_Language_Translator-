"""
inference.py  -  Sign Language AI System
Professional split-panel UI: camera left, control panel right.
Canvas: 1280 x 720
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import pickle
import time
import threading
import math

from src.sentence_engine import generate_sentences
from src.tts_engine import speak, is_available as tts_available

# =======================================================================
# DEVICE
# =======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"TTS    : {'ON' if tts_available() else 'OFF (pip install pyttsx3)'}\n")

# =======================================================================
# CONFIG
# =======================================================================
CONF_HIGH            = 0.98
CONF_MEDIUM          = 0.90
MOTION_THRESHOLD     = 0.02
STABLE_FRAMES_NEEDED = 30
MIN_SIGN_FRAMES      = 25
MAX_SIGN_FRAMES      = 60
TARGET_FRAMES        = 30
PREDICTION_COOLDOWN  = 2.0

# Canvas
CW, CH  = 1280, 720
CAM_W   = 760
PANEL_X = CAM_W
PANEL_W = CW - CAM_W   # 520 px

# =======================================================================
# COLOUR PALETTE  (BGR)
# =======================================================================
BG        = ( 18,  14,  24)
CAM_BG    = ( 10,   8,  16)
PANEL_BG  = ( 30,  24,  44)
CARD_BG   = ( 42,  34,  60)
CARD_SEL  = ( 58,  46,  88)
DIVIDER   = ( 60,  50,  80)

ACCENT    = ( 60, 200, 255)   # gold-yellow
GREEN     = ( 80, 210,  90)
CYAN_W    = (230, 215,  60)   # word display colour
RED_S     = ( 80,  80, 200)
WHITE     = (240, 235, 228)
SUBTEXT   = (155, 145, 170)
HDR_LINE  = (100,  80, 140)   # visible header separator

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_B    = cv2.FONT_HERSHEY_DUPLEX

# =======================================================================
# DRAWING PRIMITIVES
# =======================================================================
def fill_rect(img, x, y, w, h, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, (x, y), (x+w, y+h), color, -1)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def rr(img, x, y, w, h, r, color, border_col=None, bw=1):
    """Filled rounded rectangle with optional border."""
    r = min(r, w//2, h//2)
    cv2.rectangle(img, (x+r, y),   (x+w-r, y+h),   color, -1)
    cv2.rectangle(img, (x,   y+r), (x+w,   y+h-r), color, -1)
    for cx, cy, a in [(x+r, y+r, 180), (x+w-r, y+r, 270),
                      (x+r, y+h-r, 90), (x+w-r, y+h-r, 0)]:
        cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, -1)
    if border_col:
        cv2.rectangle(img, (x+r, y),   (x+w-r, y+h),   border_col, bw)
        cv2.rectangle(img, (x,   y+r), (x+w,   y+h-r), border_col, bw)
        for cx, cy, a in [(x+r, y+r, 180), (x+w-r, y+r, 270),
                          (x+r, y+h-r, 90), (x+w-r, y+h-r, 0)]:
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, border_col, bw)


def t(img, text, x, y, sc=0.60, col=WHITE, th=1, fn=FONT):
    cv2.putText(img, str(text), (x, y), fn, sc, col, th, cv2.LINE_AA)


def tc(img, text, cx, y, sc=0.60, col=WHITE, th=1, fn=FONT):
    (tw, _), _ = cv2.getTextSize(str(text), fn, sc, th)
    cv2.putText(img, str(text), (cx-tw//2, y), fn, sc, col, th, cv2.LINE_AA)


def wrap(img, text, x, y, mw, sc=0.58, col=WHITE, lh=24):
    words, line = str(text).split(), ""
    for word in words:
        test = line + word + " "
        (tw, _), _ = cv2.getTextSize(test, FONT, sc, 1)
        if tw > mw and line:
            t(img, line.strip(), x, y, sc, col)
            y += lh; line = word + " "
        else:
            line = test
    if line.strip():
        t(img, line.strip(), x, y, sc, col)
    return y + lh


def prog_bar(img, x, y, w, h, val, mx, fg=ACCENT, bg=CARD_BG, r=4):
    rr(img, x, y, w, h, r, bg)
    filled = int((val/max(mx,1)) * w)
    if filled > r*2:
        rr(img, x, y, filled, h, r, fg)


def conf_bar(img, x, y, w, h, conf):
    rr(img, x, y, w, h, 4, CARD_BG)
    col = GREEN if conf >= CONF_HIGH else ACCENT if conf >= CONF_MEDIUM else RED_S
    filled = int(conf * w)
    if filled > 8:
        rr(img, x, y, filled, h, 4, col)
    # threshold tick marks
    for th in (CONF_MEDIUM, CONF_HIGH):
        tx = x + int(th * w)
        cv2.line(img, (tx, y-3), (tx, y+h+3), DIVIDER, 1)


# =======================================================================
# MODEL
# =======================================================================
def normalize_landmarks(sequence):
    norm_seq = []
    for frame in sequence:
        frame = np.array(frame).reshape(2, 21, 3)
        new_frame = []
        for hand in frame:
            if np.sum(hand) == 0:
                new_frame.extend(hand.flatten()); continue
            hand = hand - hand[0]
            mv = np.max(np.abs(hand))
            if mv != 0: hand = hand / mv
            new_frame.extend(hand.flatten())
        norm_seq.append(new_frame)
    return np.array(norm_seq)


def sample_fixed(frames, target):
    n = len(frames)
    if n == target: return frames
    if n < target: return frames + [frames[-1]] * (target - n)
    return [frames[i] for i in np.linspace(0, n-1, target, dtype=int)]


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.attn    = nn.Linear(hidden_size*2, 1)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        c = torch.sum(out * w, dim=1)
        return self.fc(self.dropout(c))


with open("models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

model = AttentionLSTM(126, 128, len(encoder.classes_)).to(device)
model.load_state_dict(torch.load("models/model.pth", map_location=device, weights_only=True))
model.eval()
print(f"Loaded model — {len(encoder.classes_)} classes")
print(f"Vocabulary: {[str(c) for c in encoder.classes_]}\n")

# =======================================================================
# MEDIAPIPE + CAMERA
# =======================================================================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

LM_SPEC   = mp_draw.DrawingSpec(color=ACCENT,      thickness=2, circle_radius=3)
CONN_SPEC = mp_draw.DrawingSpec(color=(80, 60, 120), thickness=1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# =======================================================================
# GLOBAL STATE
# =======================================================================
sequence         = []
recording        = False
stable_count     = 0
prev_features    = None
hand_was_present = False

last_pred_label  = ""
last_pred_time   = 0.0
last_conf        = 0.0

word_buffer      = []
sentences        = []
selected         = ""

candidate_word   = ""
candidate_conf   = 0.0
retry_prompt     = False
retry_timer      = 0.0
retry_msg        = ""

speaking         = False
pulse_t          = 0.0

# =======================================================================
# CONFIDENCE HANDLER
# =======================================================================
def handle_prediction(label, conf):
    global word_buffer, sentences, selected
    global retry_prompt, retry_timer, retry_msg
    global candidate_word, candidate_conf
    global last_pred_label, last_pred_time, last_conf

    if str(label) == "none":
        return
    now = time.time()
    if label == last_pred_label and (now - last_pred_time) < PREDICTION_COOLDOWN:
        return

    last_conf = conf

    if conf >= CONF_HIGH:
        word_buffer.append(str(label))
        last_pred_label = label; last_pred_time = now
        candidate_word  = ""; retry_prompt = False
        sentences = generate_sentences(word_buffer)
        selected  = ""
        print(f"  HIGH   [{label}] {conf:.3f}  buffer={word_buffer}")

    elif conf >= CONF_MEDIUM:
        candidate_word = str(label); candidate_conf = conf
        retry_prompt   = False
        print(f"  MEDIUM [{label}] {conf:.3f}")

    else:
        retry_prompt = True; retry_timer = now
        retry_msg    = f"Low confidence ({conf:.0%}) - sign more clearly"
        candidate_word = ""
        print(f"  LOW    [{label}] {conf:.3f} (rejected)")


# =======================================================================
# RENDER: HEADER
# =======================================================================
def render_header(canvas, is_recording):
    # Background strip
    fill_rect(canvas, 0, 0, CW, 62, (22, 17, 35)) # will be overwritten below
    cv2.rectangle(canvas, (0, 0), (CW, 62), (22, 17, 35), -1)

    # Bottom border — bright enough to see
    cv2.line(canvas, (0, 62), (CW, 62), HDR_LINE, 2)

    # Logo accent dot
    cv2.circle(canvas, (28, 31), 9, ACCENT, -1)
    cv2.circle(canvas, (28, 31), 5, (20, 15, 30), -1)

    # Title
    t(canvas, "SIGN LANGUAGE AI SYSTEM", 50, 24, 0.70, WHITE, 1, FONT_B)
    t(canvas, "Real-time gesture recognition & sentence generation",
      50, 46, 0.42, SUBTEXT)

    # Status pills — right side
    pills = [
        (f"DEVICE: {str(device).upper()}", ACCENT),
        (f"TTS: {'ON' if tts_available() else 'OFF'}",
         GREEN if tts_available() else SUBTEXT),
    ]
    px = CW - 14
    for label_str, col in reversed(pills):
        (tw, _), _ = cv2.getTextSize(label_str, FONT, 0.48, 1)
        pw = tw + 20
        px -= pw + 10
        rr(canvas, px, 17, pw, 28, 6, (42, 34, 60))
        t(canvas, label_str, px+10, 36, 0.48, col)

    # Recording badge
    if is_recording:
        pulse = int(180 + 60 * math.sin(pulse_t * 8))
        rr(canvas, 410, 17, 130, 28, 6, (40, 40, min(pulse, 220)))
        cv2.circle(canvas, (426, 31), 5, (80, 80, 255), -1)
        t(canvas, "RECORDING", 436, 36, 0.50, WHITE, 1)


# =======================================================================
# RENDER: CAMERA
# =======================================================================
def render_camera(canvas, cam_frame, is_recording, seq_len):
    # Background
    cv2.rectangle(canvas, (0, 63), (CAM_W, CH), CAM_BG, -1)

    if cam_frame is None:
        return

    h, w = cam_frame.shape[:2]
    avail_h = CH - 63 - 48
    avail_w = CAM_W - 20

    scale = min(avail_w / w, avail_h / h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(cam_frame, (nw, nh))

    ox = (CAM_W - nw) // 2
    oy = 63 + (avail_h - nh) // 2

    # Border around camera — ACCENT when recording, subtle when idle
    border_col = ACCENT if is_recording else (50, 42, 68)
    border_w   = 3 if is_recording else 1
    cv2.rectangle(canvas, (ox-border_w, oy-border_w),
                  (ox+nw+border_w, oy+nh+border_w), border_col, border_w)

    canvas[oy:oy+nh, ox:ox+nw] = resized

    # Status text over camera bottom
    status_y = oy + nh - 12
    if is_recording:
        status_str = f"Capturing gesture...  {seq_len}/{MAX_SIGN_FRAMES} frames"
        status_col = ACCENT
    else:
        status_str = "Waiting for gesture..."
        status_col = SUBTEXT

    # Subtle background behind text
    (sw, _), _ = cv2.getTextSize(status_str, FONT, 0.52, 1)
    fill_rect(canvas, ox, status_y-16, sw+16, 22, (10,8,16), 0.65)
    t(canvas, status_str, ox+8, status_y, 0.52, status_col)

    # Progress bar below camera
    if is_recording:
        pb_y = oy + nh + 10
        prog_bar(canvas, ox, pb_y, nw, 8, seq_len, MAX_SIGN_FRAMES,
                 fg=ACCENT, bg=(38,30,55))


# =======================================================================
# RENDER: RIGHT PANEL
# =======================================================================
def render_panel(canvas):
    # FIX: declare all globals that this function reads OR writes
    global retry_prompt

    px = PANEL_X + 12
    pw = PANEL_W - 24
    y  = 74

    # Panel background
    cv2.rectangle(canvas, (PANEL_X, 0), (CW, CH), PANEL_BG, -1)
    cv2.line(canvas, (PANEL_X, 0), (PANEL_X, CH), HDR_LINE, 2)

    # ── Detected Words ─────────────────────────────────────────────
    t(canvas, "DETECTED WORDS", px, y, 0.43, SUBTEXT)
    y += 14

    rr(canvas, px, y, pw, 52, 8, CARD_BG)
    if word_buffer:
        ws = "  >  ".join(word_buffer)
        if len(ws) > 30: ws = "..." + ws[-27:]
        t(canvas, ws, px+14, y+34, 0.75, CYAN_W, 2, FONT_B)
        t(canvas, "u = undo", px+pw-75, y+34, 0.40, SUBTEXT)
    else:
        t(canvas, "No words detected yet", px+14, y+33, 0.55, SUBTEXT)
    y += 66

    # ── Confidence ────────────────────────────────────────────────
    t(canvas, "LAST CONFIDENCE", px, y, 0.43, SUBTEXT)
    y += 14

    rr(canvas, px, y, pw, 54, 8, CARD_BG)

    bar_x = px+14; bar_y = y+14; bar_w = pw-28; bar_h = 12
    conf_bar(canvas, bar_x, bar_y, bar_w, bar_h, last_conf)

    # Labels below bar
    t(canvas, "LOW",  bar_x,                         bar_y+28, 0.38, SUBTEXT)
    t(canvas, "0.90", bar_x+int(0.87*bar_w)-10,      bar_y+28, 0.38, SUBTEXT)
    t(canvas, "0.98", bar_x+int(0.96*bar_w)-10,      bar_y+28, 0.38, SUBTEXT)

    # Percentage value — safe ASCII only (no em-dash)
    conf_str = f"{last_conf:.1%}" if last_conf > 0 else "- -"
    conf_col = GREEN if last_conf >= CONF_HIGH else \
               ACCENT if last_conf >= CONF_MEDIUM else SUBTEXT
    t(canvas, conf_str, bar_x+bar_w-40, bar_y-2, 0.60, conf_col, 1)

    y += 70

    # ── Candidate banner ──────────────────────────────────────────
    if candidate_word:
        rr(canvas, px, y, pw, 48, 8, (28, 55, 75), border_col=(50,100,140), bw=1)
        t(canvas, f"CANDIDATE: {candidate_word.upper()}  ({candidate_conf:.0%})",
          px+14, y+20, 0.58, CYAN_W)
        t(canvas, "Press  'a'  to accept", px+14, y+38, 0.44, SUBTEXT)
        y += 64

    # ── Retry banner ──────────────────────────────────────────────
    elif retry_prompt:
        if (time.time() - retry_timer) < 2.5:
            rr(canvas, px, y, pw, 40, 8, (28, 28, 80), border_col=(80,80,180), bw=1)
            t(canvas, retry_msg, px+14, y+26, 0.48, (130, 130, 230))
            y += 56
        else:
            retry_prompt = False   # auto-dismiss (safe: global declared above)
            y += 4
    else:
        y += 4

    # ── Sentence Suggestions ──────────────────────────────────────
    t(canvas, "SENTENCE SUGGESTIONS", px, y, 0.43, SUBTEXT)
    y += 14

    if sentences:
        for i, sent in enumerate(sentences[:3]):
            is_sel  = (selected == sent)
            bg_col  = CARD_SEL if is_sel else CARD_BG
            bdr_col = ACCENT   if is_sel else None
            bdr_w   = 2        if is_sel else 1

            card_h = 58
            rr(canvas, px, y, pw, card_h, 8, bg_col,
               border_col=bdr_col, bw=bdr_w)

            # Key badge
            badge_col = ACCENT if is_sel else (68, 56, 95)
            rr(canvas, px+8, y+15, 28, 28, 6, badge_col)
            tc(canvas, str(i+1), px+22, y+34, 0.56, WHITE, 1, FONT_B)

            # Sentence text
            wrap(canvas, sent, px+46, y+26, pw-60, 0.59,
                 WHITE if is_sel else (205, 198, 220), lh=22)

            y += card_h + 8
    else:
        rr(canvas, px, y, pw, 54, 8, CARD_BG)
        t(canvas, "Sign gestures to generate", px+14, y+22, 0.53, SUBTEXT)
        t(canvas, "sentence suggestions.", px+14, y+40, 0.53, SUBTEXT)
        y += 62

    # ── Selected Output ───────────────────────────────────────────
    if selected:
        sel_y = CH - 110
        cv2.line(canvas, (px, sel_y-6), (px+pw, sel_y-6), DIVIDER, 1)
        t(canvas, "SELECTED OUTPUT", px, sel_y, 0.43, SUBTEXT)
        sel_y += 14
        rr(canvas, px, sel_y, pw, 62, 8, (30, 52, 30),
           border_col=GREEN, bw=1)
        wrap(canvas, f'"{selected}"', px+14, sel_y+22, pw-28,
             0.60, GREEN, lh=22)
        if speaking:
            t(canvas, "Speaking...", px+pw-100, sel_y+52, 0.42, CYAN_W)


# =======================================================================
# RENDER: FOOTER
# =======================================================================
def render_footer(canvas):
    fy = CH - 40
    cv2.rectangle(canvas, (0, fy), (CW, CH), (14, 10, 22), -1)
    cv2.line(canvas, (0, fy), (CAM_W, fy), HDR_LINE, 1)

    keys = [
        ("1 / 2 / 3", "select sentence"),
        ("a", "accept candidate"),
        ("u", "undo word"),
        ("c", "clear all"),
        ("s", "speak selected"),
        ("ESC", "quit"),
    ]
    kx = 18
    for k, v in keys:
        (kw, _), _ = cv2.getTextSize(k, FONT_B, 0.45, 1)
        rr(canvas, kx, fy+8, kw+14, 24, 4, CARD_BG)
        t(canvas, k, kx+7, fy+25, 0.45, ACCENT, 1, FONT_B)
        kx += kw + 20
        t(canvas, v, kx-6, fy+25, 0.43, SUBTEXT)
        (vw, _), _ = cv2.getTextSize(v, FONT, 0.43, 1)
        kx += vw + 16
        cv2.line(canvas, (kx-6, fy+10), (kx-6, fy+32), DIVIDER, 1)


# =======================================================================
# WINDOW SETUP
# =======================================================================
WIN = "Sign Language AI System"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, CW, CH)

print("Ready — sign gestures and press 1/2/3 to select a sentence.\n")

# =======================================================================
# MAIN LOOP
# =======================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    # ── Landmark extraction ───────────────────────────────────────
    left_lm = [0.0]*63; right_lm = [0.0]*63
    hand_present = False

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_present = True
        for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            lm_flat = [v for lm in hl.landmark for v in (lm.x, lm.y, lm.z)]
            if hd.classification[0].label == "Left": left_lm = lm_flat
            else: right_lm = lm_flat
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                   LM_SPEC, CONN_SPEC)

    features = left_lm + right_lm

    # ── Motion detection ──────────────────────────────────────────
    motion = 0.0
    if hand_present and prev_features is not None and hand_was_present:
        motion = float(np.mean(np.abs(np.array(features) - np.array(prev_features))))

    if hand_present:
        if motion > MOTION_THRESHOLD:
            if not recording: print("Recording...")
            recording = True; stable_count = 0
        elif recording:
            stable_count += 1
    elif recording:
        stable_count += 1

    if recording:
        sequence.append(features)
        if len(sequence) > MAX_SIGN_FRAMES:
            sequence = sequence[-MAX_SIGN_FRAMES:]
        pulse_t += 0.05

    # ── Predict on stable end-of-sign ────────────────────────────
    if recording and stable_count >= STABLE_FRAMES_NEEDED:
        if len(sequence) >= MIN_SIGN_FRAMES:
            sam = sample_fixed(sequence, TARGET_FRAMES)
            sq  = normalize_landmarks(sam)
            xt  = torch.tensor(sq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out   = model(xt)
                probs = torch.softmax(out, dim=1)
                conf  = torch.max(probs).item()
                pred  = torch.argmax(probs).item()
            handle_prediction(encoder.inverse_transform([pred])[0], conf)
        else:
            print(f"  Too short ({len(sequence)}) - ignored.")
        sequence = []; recording = False; stable_count = 0

    prev_features    = features if hand_present else None
    hand_was_present = hand_present

    # ── Compose canvas ────────────────────────────────────────────
    canvas = np.zeros((CH, CW, 3), dtype=np.uint8)
    canvas[:] = BG

    render_header(canvas, recording)
    render_camera(canvas, frame, recording, len(sequence))
    render_panel(canvas)
    render_footer(canvas)

    cv2.imshow(WIN, canvas)

    # ── Key handling ──────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key in (ord('1'), ord('2'), ord('3')):
        idx = key - ord('1')
        if idx < len(sentences):
            selected = sentences[idx]
            print(f"Selected: {selected}")
    elif key == ord('s') and selected and not speaking:
        def _bg(text):
            global speaking
            speaking = True; speak(text); speaking = False
        threading.Thread(target=_bg, args=(selected,), daemon=True).start()
    elif key == ord('a') and candidate_word:
        word_buffer.append(candidate_word)
        last_pred_label = candidate_word
        last_pred_time  = time.time()
        sentences = generate_sentences(word_buffer)
        selected  = ""
        print(f"  Accepted [{candidate_word}]  buffer={word_buffer}")
        candidate_word = ""; candidate_conf = 0.0
    elif key == ord('u') and word_buffer:
        r = word_buffer.pop()
        sentences = generate_sentences(word_buffer) if word_buffer else []
        selected  = ""
        print(f"  Undo [{r}]  buffer={word_buffer}")
    elif key == ord('c'):
        word_buffer = []; sentences = []; selected = ""
        candidate_word = ""; candidate_conf = 0.0
        retry_prompt = False; last_pred_label = ""; last_conf = 0.0
        print("  Cleared.")

# =======================================================================
# CLEANUP
# =======================================================================
cap.release()
cv2.destroyAllWindows()
hands.close()
