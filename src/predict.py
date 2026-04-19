"""
inference.py  -  Sign Language AI System
Minimal professional dark UI. Canvas: 1280 x 720
"""

import cv2, mediapipe as mp, torch, torch.nn as nn
import numpy as np, pickle, time, threading, math
from sentence_engine import generate_sentences
from tts_engine import speak, is_available as tts_available

# ═══════════════════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
CONF_HIGH            = 0.98
CONF_MEDIUM          = 0.90
MOTION_THRESHOLD     = 0.02
STABLE_FRAMES_NEEDED = 30
MIN_SIGN_FRAMES      = 25
MAX_SIGN_FRAMES      = 60
TARGET_FRAMES        = 30
PREDICTION_COOLDOWN  = 2.0

CW, CH  = 1280, 720
CAM_W   = 780
PANEL_X = CAM_W
PANEL_W = CW - CAM_W      # 500 px

# ═══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM  (BGR — much more refined palette)
# ═══════════════════════════════════════════════════════════════════════
# Backgrounds
C_BASE    = (  8,   6,  12)   # near-black
C_SURFACE = ( 16,  13,  22)   # camera bg
C_PANEL   = ( 20,  16,  30)   # right panel
C_CARD    = ( 28,  22,  42)   # default card
C_CARD_HV = ( 38,  30,  58)   # hovered/selected card
C_CARD_HI = ( 30,  42,  30)   # selected output card (green tint)

# Borders
C_BORDER  = ( 45,  36,  68)   # subtle card border
C_BORDER_A = (100,  80, 150)  # accent border (panel edge, header)
C_BORDER_G = ( 55,  90,  55)  # green border

# Text
C_TEXT    = (225, 220, 215)   # primary text
C_MUTED   = (120, 110, 135)   # labels / subtext
C_DIM     = ( 70,  62,  85)   # very faint

# Accents
C_ACCENT  = ( 80, 180, 255)   # blue-white (main accent)
C_GOLD    = ( 50, 185, 255)   # same for words
C_GREEN   = ( 70, 200,  90)   # confirmed
C_AMBER   = ( 50, 150, 255)   # medium confidence
C_RED     = ( 70,  70, 200)   # low confidence / rejected
C_CYAN    = (200, 210,  50)   # word buffer text

# ═══════════════════════════════════════════════════════════════════════
# DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════
FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX

def rect(img, x, y, w, h, col, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, (x,y), (x+w,y+h), col, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, (x,y), (x+w,y+h), col, -1)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

def card(img, x, y, w, h, col=C_CARD, border=C_BORDER, bw=1, r=6):
    """Sleek card with thin border and optional radius."""
    r = min(r, w//2-1, h//2-1, 10)
    # Fill
    cv2.rectangle(img, (x+r,y),   (x+w-r,y+h),   col, -1)
    cv2.rectangle(img, (x,y+r),   (x+w,y+h-r),   col, -1)
    for cx,cy,a in [(x+r,y+r,180),(x+w-r,y+r,270),(x+r,y+h-r,90),(x+w-r,y+h-r,0)]:
        cv2.ellipse(img,(cx,cy),(r,r),a,0,90,col,-1)
    # Border
    if border and bw:
        cv2.rectangle(img, (x+r,y),   (x+w-r,y+h),   border, bw)
        cv2.rectangle(img, (x,y+r),   (x+w,y+h-r),   border, bw)
        for cx,cy,a in [(x+r,y+r,180),(x+w-r,y+r,270),(x+r,y+h-r,90),(x+w-r,y+h-r,0)]:
            cv2.ellipse(img,(cx,cy),(r,r),a,0,90,border,bw)

def txt(img, s, x, y, sc=0.56, col=C_TEXT, th=1, fn=FONT):
    cv2.putText(img, str(s), (x,y), fn, sc, col, th, cv2.LINE_AA)

def txtc(img, s, cx, y, sc=0.56, col=C_TEXT, th=1, fn=FONT):
    (w,_),_ = cv2.getTextSize(str(s),fn,sc,th)
    cv2.putText(img,str(s),(cx-w//2,y),fn,sc,col,th,cv2.LINE_AA)

def txtr(img, s, rx, y, sc=0.56, col=C_TEXT, th=1, fn=FONT):
    (w,_),_ = cv2.getTextSize(str(s),fn,sc,th)
    cv2.putText(img,str(s),(rx-w,y),fn,sc,col,th,cv2.LINE_AA)

def wraptext(img, s, x, y, mw, sc=0.56, col=C_TEXT, lh=22):
    words, line = str(s).split(), ""
    for word in words:
        test = line + word + " "
        (tw,_),_ = cv2.getTextSize(test,FONT,sc,1)
        if tw > mw and line:
            txt(img,line.strip(),x,y,sc,col); y+=lh; line=word+" "
        else: line=test
    if line.strip(): txt(img,line.strip(),x,y,sc,col)
    return y+lh

def hline(img, x, y, w, col=C_BORDER, th=1):
    cv2.line(img,(x,y),(x+w,y),col,th)

def vline(img, x, y, h, col=C_BORDER, th=1):
    cv2.line(img,(x,y),(x,y+h),col,th)

def dot(img, cx, cy, r, col):
    cv2.circle(img,(cx,cy),r,col,-1)

def pill(img, x, y, label, col=C_ACCENT, bg=C_CARD):
    """Small rounded pill badge."""
    (tw,_),_ = cv2.getTextSize(label,FONT,0.42,1)
    pw = tw+16
    card(img,x,y,pw,20,bg,C_BORDER,1,10)
    txt(img,label,x+8,y+14,0.42,col)
    return pw

def section_label(img, x, y, label):
    txt(img, label, x, y, 0.38, C_MUTED)
    return y+16

# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════
def norm_lm(seq):
    out=[]
    for frame in seq:
        frame=np.array(frame).reshape(2,21,3); nf=[]
        for hand in frame:
            if np.sum(hand)==0: nf.extend(hand.flatten()); continue
            hand=hand-hand[0]; mv=np.max(np.abs(hand))
            if mv: hand/=mv
            nf.extend(hand.flatten())
        out.append(nf)
    return np.array(out)

def sample(frames,n):
    k=len(frames)
    if k==n: return frames
    if k<n:  return frames+[frames[-1]]*(n-k)
    return [frames[i] for i in np.linspace(0,k-1,n,dtype=int)]

class AttentionLSTM(nn.Module):
    def __init__(self,i,h,c):
        super().__init__()
        self.lstm=nn.LSTM(i,h,2,batch_first=True,bidirectional=True,dropout=0.3)
        self.attn=nn.Linear(h*2,1); self.drop=nn.Dropout(0.4); self.fc=nn.Linear(h*2,c)
    def forward(self,x):
        o,_=self.lstm(x); w=torch.softmax(self.attn(o),1)
        return self.fc(self.drop((o*w).sum(1)))

with open("models/encoder.pkl","rb") as f: encoder=pickle.load(f)
model=AttentionLSTM(126,128,len(encoder.classes_)).to(device)
model.load_state_dict(torch.load("models/model.pth",map_location=device,weights_only=True))
model.eval()
print(f"Model: {len(encoder.classes_)} classes | Device: {device}")

# ═══════════════════════════════════════════════════════════════════════
# MEDIAPIPE
# ═══════════════════════════════════════════════════════════════════════
mp_h=mp.solutions.hands; mp_d=mp.solutions.drawing_utils
hands=mp_h.Hands(max_num_hands=2,min_detection_confidence=0.7)
LM =mp_d.DrawingSpec(color=C_ACCENT,  thickness=2,circle_radius=2)
CON=mp_d.DrawingSpec(color=C_BORDER_A,thickness=1)

cap=cv2.VideoCapture(0)
if not cap.isOpened(): raise RuntimeError("Webcam not found.")

# ═══════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════
seq=[]; recording=False; stable=0
prev_f=None; prev_hand=False
last_label=""; last_time=0.0; last_conf=0.0
words=[]; sents=[]; selected=""
cand=""; cand_c=0.0
retry=False; retry_t=0.0; retry_msg=""
speaking=False; pulse=0.0; ai_load=False

# ═══════════════════════════════════════════════════════════════════════
# BACKGROUND AI FETCH
# ═══════════════════════════════════════════════════════════════════════
def _ai_bg(snap):
    global sents,ai_load
    ai_load=True
    try: sents=generate_sentences(snap)
    except Exception as e: print(f"[AI] {e}")
    finally: ai_load=False

def ask_ai(w): threading.Thread(target=_ai_bg,args=(list(w),),daemon=True).start()

# ═══════════════════════════════════════════════════════════════════════
# PREDICTION HANDLER
# ═══════════════════════════════════════════════════════════════════════
def predict(label,conf):
    global words,sents,selected,retry,retry_t,retry_msg
    global cand,cand_c,last_label,last_time,last_conf,ai_load
    if str(label)=="none": return
    now=time.time()
    if label==last_label and now-last_time<PREDICTION_COOLDOWN: return
    last_conf=conf
    if conf>=CONF_HIGH:
        words.append(str(label)); last_label=label; last_time=now
        cand=""; retry=False; selected=""; sents=[]; ask_ai(words)
        print(f"  HIGH [{label}] {conf:.3f}  {words}")
    elif conf>=CONF_MEDIUM:
        cand=str(label); cand_c=conf; retry=False
        print(f"  MED  [{label}] {conf:.3f}")
    else:
        retry=True; retry_t=now
        retry_msg=f"Low confidence ({conf:.0%})  —  try again"; cand=""
        print(f"  LOW  [{label}] {conf:.3f}")

# ═══════════════════════════════════════════════════════════════════════
# RENDER: HEADER
# ═══════════════════════════════════════════════════════════════════════
def draw_header(cv, is_rec):
    rect(cv,0,0,CW,56,C_SURFACE)
    hline(cv,0,56,CW,C_BORDER_A,1)

    # Logo mark
    cv2.rectangle(cv,(14,14),(34,42),C_ACCENT,-1)
    cv2.rectangle(cv,(16,16),(32,40),C_SURFACE,-1)
    dot(cv,24,29,4,C_ACCENT)

    # Title block
    txt(cv,"SIGN LANGUAGE AI SYSTEM",46,24,0.65,C_TEXT,1,FONTB)
    txt(cv,"Real-time gesture recognition  |  AI sentence generation",46,44,0.38,C_MUTED)

    # Right-side status
    rx=CW-12
    for label_s,col in [
        (f"TTS {'ON' if tts_available() else 'OFF'}", C_GREEN if tts_available() else C_MUTED),
        (str(device).upper(), C_ACCENT),
    ]:
        (tw,_),_=cv2.getTextSize(label_s,FONT,0.40,1)
        pw=tw+20; rx-=pw+8
        card(cv,rx,16,pw,24,C_CARD,C_BORDER,1,12)
        txt(cv,label_s,rx+10,33,0.40,col)

    # Recording badge
    if is_rec:
        p=int(160+80*math.sin(pulse*6))
        card(cv,rx-140,16,110,24,(30,30,min(p,220)),None,0,12)
        dot(cv,rx-130+14,28,4,(90,90,255))
        txt(cv,"RECORDING",rx-130+24,33,0.40,C_TEXT)

# ═══════════════════════════════════════════════════════════════════════
# RENDER: CAMERA
# ═══════════════════════════════════════════════════════════════════════
def draw_camera(cv, frame, is_rec, seq_n):
    rect(cv,0,57,CAM_W,CH-57,C_SURFACE)

    if frame is None: return
    h,w=frame.shape[:2]
    ah=CH-57-44; aw=CAM_W-16
    sc=min(aw/w,ah/h); nw,nh=int(w*sc),int(h*sc)
    res=cv2.resize(frame,(nw,nh))
    ox=(CAM_W-nw)//2; oy=57+(ah-nh)//2

    # Camera border
    bc=C_ACCENT if is_rec else C_BORDER
    bw=2 if is_rec else 1
    cv2.rectangle(cv,(ox-bw,oy-bw),(ox+nw+bw,oy+nh+bw),bc,bw)
    cv[oy:oy+nh,ox:ox+nw]=res

    # Status overlay (bottom of camera)
    sy=oy+nh-10
    st="Capturing gesture..." if is_rec else "Waiting for gesture..."
    sc2=C_ACCENT if is_rec else C_MUTED
    (sw,_),_=cv2.getTextSize(st,FONT,0.48,1)
    rect(cv,ox,sy-18,sw+18,22,(8,6,12),0.7)
    txt(cv,st,ox+9,sy,0.48,sc2)

    # Progress bar
    if is_rec:
        py=oy+nh+8
        cv2.rectangle(cv,(ox,py),(ox+nw,py+5),C_CARD,-1)
        fw=int(seq_n/MAX_SIGN_FRAMES*nw)
        if fw>2: cv2.rectangle(cv,(ox,py),(ox+fw,py+5),C_ACCENT,-1)

# ═══════════════════════════════════════════════════════════════════════
# RENDER: PANEL
# ═══════════════════════════════════════════════════════════════════════
def draw_panel(cv):
    global retry, ai_load, pulse

    px=PANEL_X; pw=PANEL_W
    rect(cv,px,0,pw,CH,C_PANEL)
    vline(cv,px,0,CH,C_BORDER_A,1)

    mx=px+14; mw=pw-28   # inner margins
    y=68

    # ── Detected Words ──────────────────────────────────────────────
    y=section_label(cv,mx,y,"DETECTED WORDS")
    card(cv,mx,y,mw,50,C_CARD,C_BORDER,1,6)

    if words:
        # Word chips
        cx2=mx+12; cy=y+16
        for i,w in enumerate(words):
            (tw,_),_=cv2.getTextSize(w,FONTB,0.52,1)
            cw2=tw+18
            if cx2+cw2>mx+mw-10: break   # overflow guard
            card(cv,cx2,cy,cw2,22,C_CARD_HV,C_BORDER_A,1,11)
            txt(cv,w,cx2+9,cy+15,0.52,C_CYAN,1,FONTB)
            if i<len(words)-1:
                txt(cv,">",(cx2+cw2+4),cy+15,0.44,C_DIM)
                cx2+=cw2+18
            else:
                cx2+=cw2+6
    else:
        txt(cv,"No words detected yet",mx+12,y+31,0.50,C_MUTED)

    y+=60; hline(cv,mx,y,mw,C_BORDER)

    # ── Confidence ─────────────────────────────────────────────────
    y+=10
    y=section_label(cv,mx,y,"LAST CONFIDENCE")
    card(cv,mx,y,mw,48,C_CARD,C_BORDER,1,6)

    bx=mx+12; by=y+14; bw2=mw-24; bh=8
    # Track background
    cv2.rectangle(cv,(bx,by),(bx+bw2,by+bh),C_DIM,-1)
    # Filled portion
    col=C_GREEN if last_conf>=CONF_HIGH else C_AMBER if last_conf>=CONF_MEDIUM else C_RED
    fw=int(last_conf*bw2)
    if fw>4: cv2.rectangle(cv,(bx,by),(bx+fw,by+bh),col,-1)
    # Threshold markers
    for th,(tc2) in [(CONF_MEDIUM,C_MUTED),(CONF_HIGH,C_MUTED)]:
        mx2=bx+int(th*bw2)
        cv2.line(cv,(mx2,by-3),(mx2,by+bh+3),C_DIM,1)
    # Labels
    txt(cv,"LOW",  bx,          by+24,0.34,C_DIM)
    txt(cv,"0.90", bx+int(0.88*bw2)-14,by+24,0.34,C_DIM)
    txt(cv,"0.98", bx+int(0.96*bw2)-14,by+24,0.34,C_DIM)
    cs=f"{last_conf:.1%}" if last_conf>0 else "- -"
    txtr(cv,cs,bx+bw2,by-2,0.54,col)

    y+=64; hline(cv,mx,y,mw,C_BORDER)

    # ── Banner (candidate / retry) ──────────────────────────────────
    y+=10
    if cand:
        card(cv,mx,y,mw,44,(22,44,62),(50,90,130),1,6)
        txt(cv,f"CANDIDATE  [{cand.upper()}]  {cand_c:.0%}",mx+12,y+18,0.50,C_CYAN)
        txt(cv,"Press  'a'  to accept",mx+12,y+34,0.40,C_MUTED)
        y+=54
    elif retry and (time.time()-retry_t)<2.5:
        card(cv,mx,y,mw,36,(25,25,72),(70,70,180),1,6)
        txt(cv,retry_msg,mx+12,y+22,0.46,(140,140,230))
        y+=46
    else:
        if retry: retry=False
        y+=4

    # ── Sentences ──────────────────────────────────────────────────
    hline(cv,mx,y,mw,C_BORDER); y+=10
    y=section_label(cv,mx,y,"SENTENCE SUGGESTIONS")

    if ai_load:
        card(cv,mx,y,mw,46,(18,18,36),(80,60,120),1,6)
        dots="."*(int(pulse*4)%4)
        txt(cv,f"Generating sentences{dots}",mx+12,y+18,0.50,C_MUTED)
        txt(cv,"AI is processing your words...",mx+12,y+34,0.40,C_DIM)
        y+=56
    elif sents:
        for i,s in enumerate(sents[:3]):
            is_sel=(selected==s)
            bg=C_CARD_HV if is_sel else C_CARD
            bd=C_ACCENT   if is_sel else C_BORDER
            bw3=2         if is_sel else 1
            ch=54
            card(cv,mx,y,mw,ch,bg,bd,bw3,6)

            # Number badge (minimal — just a small circle)
            nc=(C_ACCENT if is_sel else C_BORDER_A)
            dot(cv,mx+18,y+ch//2,10,nc)
            txtc(cv,str(i+1),mx+18,y+ch//2+5,0.44,C_BASE if is_sel else C_TEXT,1,FONTB)

            # Sentence
            tc3=C_TEXT if is_sel else (180,175,190)
            wraptext(cv,s,mx+36,y+18,mw-48,0.55,tc3,20)

            y+=ch+6
    else:
        card(cv,mx,y,mw,46,C_CARD,C_BORDER,1,6)
        txt(cv,"Sign gestures to generate suggestions.",mx+12,y+20,0.47,C_MUTED)
        txt(cv,"Words will appear above as you sign.",  mx+12,y+36,0.42,C_DIM)
        y+=56

    # ── Selected Output ─────────────────────────────────────────────
    if selected:
        oy2=CH-88
        hline(cv,mx,oy2-6,mw,C_BORDER)
        oy2=section_label(cv,mx,oy2,  "SELECTED OUTPUT")
        card(cv,mx,oy2,mw,54,C_CARD_HI,C_BORDER_G,1,6)
        wraptext(cv,f'"{selected}"',mx+12,oy2+16,mw-24,0.56,C_GREEN,20)
        if speaking:
            txt(cv,"Speaking...",mx+mw-90,oy2+44,0.38,C_MUTED)

# ═══════════════════════════════════════════════════════════════════════
# RENDER: FOOTER
# ═══════════════════════════════════════════════════════════════════════
def draw_footer(cv):
    fy=CH-36
    rect(cv,0,fy,CW,36,C_SURFACE)
    hline(cv,0,fy,CAM_W,C_BORDER_A)

    keys=[("1/2/3","select"),("a","accept"),("u","undo"),
          ("c","clear"),("s","speak"),("ESC","quit")]
    kx=16
    for k,v in keys:
        (kw,_),_=cv2.getTextSize(k,FONTB,0.42,1)
        card(cv,kx,fy+8,kw+14,20,C_CARD_HV,C_BORDER,1,10)
        txt(cv,k,kx+7,fy+22,0.42,C_ACCENT,1,FONTB)
        kx+=kw+20
        txt(cv,v,kx-6,fy+22,0.40,C_MUTED)
        (vw,_),_=cv2.getTextSize(v,FONT,0.40,1)
        kx+=vw+14
        vline(cv,kx-6,fy+10,16,C_DIM)

# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
WIN="Sign Language AI System"
cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN,CW,CH)
print("Ready.\n")

while True:
    ret,frame=cap.read()
    if not ret: continue
    frame=cv2.flip(frame,1)

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgb.flags.writeable=False; res=hands.process(rgb); rgb.flags.writeable=True

    lf=[0.]*63; rf=[0.]*63; hp=False
    if res.multi_hand_landmarks and res.multi_handedness:
        hp=True
        for hl,hd in zip(res.multi_hand_landmarks,res.multi_handedness):
            flat=[v for lm in hl.landmark for v in (lm.x,lm.y,lm.z)]
            if hd.classification[0].label=="Left": lf=flat
            else: rf=flat
            mp_d.draw_landmarks(frame,hl,mp_h.HAND_CONNECTIONS,LM,CON)
    feat=lf+rf

    motion=0.
    if hp and prev_f is not None and prev_hand:
        motion=float(np.mean(np.abs(np.array(feat)-np.array(prev_f))))

    if hp:
        if motion>MOTION_THRESHOLD:
            if not recording: print("Recording...")
            recording=True; stable=0
        elif recording: stable+=1
    elif recording: stable+=1

    if recording:
        seq.append(feat)
        if len(seq)>MAX_SIGN_FRAMES: seq=seq[-MAX_SIGN_FRAMES:]
        pulse+=0.05

    if recording and stable>=STABLE_FRAMES_NEEDED:
        if len(seq)>=MIN_SIGN_FRAMES:
            sm=sample(seq,TARGET_FRAMES); sq=norm_lm(sm)
            xt=torch.tensor(sq,dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out=model(xt); pr=torch.softmax(out,1)
                cf=torch.max(pr).item(); pd=torch.argmax(pr).item()
            predict(encoder.inverse_transform([pd])[0],cf)
        else: print(f"  Short ({len(seq)}) ignored.")
        seq=[]; recording=False; stable=0

    prev_f=feat if hp else None; prev_hand=hp

    # Compose
    canvas=np.zeros((CH,CW,3),dtype=np.uint8)
    canvas[:]=C_BASE
    draw_header(canvas,recording)
    draw_camera(canvas,frame,recording,len(seq))
    draw_panel(canvas)
    draw_footer(canvas)
    cv2.imshow(WIN,canvas)

    key=cv2.waitKey(1)&0xFF
    if key==27: break
    elif key in (ord('1'),ord('2'),ord('3')):
        i=key-ord('1')
        if i<len(sents): selected=sents[i]; print(f"Selected: {selected}")
    elif key==ord('s') and selected and not speaking:
        def _sp(t):
            global speaking; speaking=True; speak(t); speaking=False
        threading.Thread(target=_sp,args=(selected,),daemon=True).start()
    elif key==ord('a') and cand:
        words.append(cand); last_label=cand; last_time=time.time()
        selected=""; sents=[]; ask_ai(words)
        print(f"  Accepted [{cand}]"); cand=""; cand_c=0.
    elif key==ord('u') and words:
        r=words.pop(); selected=""; sents=[]
        if words: ask_ai(words)
        print(f"  Undo [{r}]")
    elif key==ord('c'):
        words=[]; sents=[]; selected=""; cand=""; cand_c=0.
        retry=False; last_label=""; last_conf=0.; pulse=0.
        print("  Cleared.")

cap.release(); cv2.destroyAllWindows(); hands.close()
