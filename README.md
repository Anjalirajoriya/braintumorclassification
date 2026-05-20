# Brain Tumor Classification Web App

A deep learning web app that classifies brain MRI scans into four categories — built with Flask and a ResNet50V2 model trained on 3,000+ MRI images.

---

## What it does

Upload any brain MRI scan and the app will tell you:
- What type of tumor is present (if any)
- Confidence percentage
- Probability breakdown for all four classes

### Classes it can detect
| Class | Description |
|---|---|
| Glioma Tumor | A tumor that occurs in the brain and spinal cord |
| Meningioma Tumor | A tumor that forms on membranes covering the brain |
| Pituitary Tumor | A tumor that forms in the pituitary gland |
| No Tumor | No tumor detected |

---

## How I built this

This wasn't straightforward — here's the honest journey:

**The model problem:** I started with a downloaded `.h5` model that always predicted the same class no matter what image I uploaded. Turns out it had two bugs — wrong preprocessing and a class index mismatch. Rather than keep patching it, I retrained from scratch.

**The training journey:** I went through 5 different model architectures before finding one that worked:
- Custom CNN — always predicted same class
- MobileNetV2 — catastrophic forgetting in fine tuning
- EfficientNetB4 — too large for the dataset size
- EfficientNetB0 — BatchNormalization freezing bug
- **ResNet50V2 — finally worked! 89% accuracy ✅**

**The dataset problem:** The original Kaggle train/test split was intentionally hard, causing 85% validation accuracy but only 64% test accuracy. Fixed by merging everything and doing a proper random 80/10/10 split.

---

## Tech Stack

- **Model:** ResNet50V2 (pretrained on ImageNet, fine tuned on brain MRI data)
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Training:** TensorFlow / Keras on Google Colab (T4 GPU)
- **Dataset:** [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) — 3,000+ images

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 89% |
| Glioma F1 | 0.85 |
| Meningioma F1 | 0.85 |
| No Tumor F1 | 0.94 |
| Pituitary F1 | 0.94 |

---

## Run it locally

**1. Clone the repo**
```bash
git clone https://github.com/Anjalirajoriya/braintumorclassification.git
cd braintumorclassification
```

**2. Create virtual environment**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the model**

Download `best_model.keras` and place it in the root folder.
*(Model not included in repo due to file size)*

**5. Run**
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

---

## Disclaimer

This app is for **educational purposes only** and should not be used for actual medical diagnosis. Always consult a qualified medical professional for medical advice.

---

Made by Anjali