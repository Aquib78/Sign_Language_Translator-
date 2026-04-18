# 🧠 AI-Powered Sign Language to Sentence Converter

A real-time AI system that converts sign language gestures into meaningful sentences using deep learning and contextual sentence generation.

---

## 🚀 Overview

This project goes beyond simple gesture recognition.
It captures hand movements using a webcam, predicts sign language words using an LSTM model, and intelligently converts them into human-like sentences with multiple suggestions.

---

## ✨ Features

* 🎥 Real-time gesture detection using webcam
* 🖐️ Hand landmark extraction using MediaPipe
* 🧠 LSTM-based sequence model for gesture recognition
* 📊 Confidence-based prediction filtering
* 🧩 AI-powered sentence generation from multiple words
* 🔄 Motion-based gesture capture (prevents partial detection)
* 🗣️ Optional Text-to-Speech output
* 🎯 Multi-option sentence suggestions with user selection

---

## 🧠 How It Works

```
Gesture → Landmarks → Sequence → LSTM Model → Word → Sentence AI → Suggestions → Final Output
```

1. Detect hand using MediaPipe
2. Extract 126 landmark features per frame
3. Capture gesture sequence (~30 frames)
4. Predict word using trained LSTM model
5. Apply confidence filtering
6. Combine words into meaningful sentence suggestions
7. User selects best sentence

---

## 📊 Model Details

* Model: LSTM (2 layers)
* Input: 30 frames × 126 features
* Dataset:

  * Custom collected data (~100 samples per class)
  * Filtered WLASL dataset (~132 samples)

### ✅ Final Accuracy

* **Overall Test Accuracy: 98.86%**
* **Best Validation Accuracy: 98.86%**

### 📈 Per-Class Performance (Core Classes)

* food → 100%
* hello → 100%
* water → 100%
* please → 100%
* thank you → 100%
* yes → 100%
* none → 100%
* no → 90.9%

---

## 🧾 Supported Vocabulary

### 🔥 Strong Classes

* hello
* yes
* no
* please
* thank you
* food
* water
* none

### ⚠️ Additional (lower data)

* drink, eat, finish, help, hungry, more, want

---

## 🧠 Sentence Intelligence (Core Feature)

Instead of raw output:

```
hello food
```

System generates:

```
1. Hello, I had food  
2. Hello, do you want food?  
3. Hello, I want food  
```

Example:

```
Input: ["thank you", "water"]

Output:
- Thank you for the water  
- Thank you, I needed that water  
- Thank you, the water was refreshing  
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Sign_Language_Translator.git
cd Sign_Language_Translator
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python src/predict.py
```

Controls:

* `SPACE` → Start gesture capture
* `1 / 2 / 3` → Select sentence
* `c` → Clear sentence
* `ESC` → Exit

---

## 📁 Project Structure

```
src/
├── predict.py              # Main real-time system
├── train_wlasl_lstm.py    # Training script
├── collect_data.py        # Data collection
├── extract_wlasl.py       # Dataset preprocessing
├── sentence_engine.py     # Sentence generation logic
├── tts_engine.py          # Text-to-speech

models/
├── model.pth              # Trained model
├── encoder.pkl            # Label encoder
```

---

## 📌 Notes

* Dataset is not included due to size
* Model trained on custom + WLASL filtered dataset
* Works best with controlled gestures and good lighting

---

## 🚀 Future Improvements

* UI upgrade (desktop/web interface)
* Transformer-based model (instead of LSTM)
* Larger vocabulary with better generalization
* Auto sentence ranking using NLP models
* Deployment via web or HuggingFace

---

## 👨‍💻 Author

**Aquib Hussain**
B.Tech Robotics & Automation
REVA University

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!
