import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.keras")
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        print("Loading model...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded!")
    return _model

def preprocess_img(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def is_brain_mri(img_path):
    """
    Basic check — brain MRIs are:
    1. Mostly grayscale (R≈G≈B channels)
    2. Dark background with bright center
    3. Low color saturation
    """
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Check 1 — grayscale check (channels should be similar)
    rg_diff = float(np.mean(np.abs(r - g)))
    rb_diff = float(np.mean(np.abs(r - b)))
    gb_diff = float(np.mean(np.abs(g - b)))
    avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3

    # Check 2 — brightness check (MRIs have dark background)
    brightness = float(np.mean(arr))

    # Check 3 — contrast check (MRIs have high contrast)
    contrast = float(np.std(arr))

    # MRI characteristics:
    # - Very low color difference (grayscale-like)
    # - Low-medium brightness (dark background)
    # - Decent contrast
    is_grayscale = avg_color_diff < 0.08
    has_contrast = contrast > 0.1
    not_too_bright = brightness < 0.6

    return is_grayscale and has_contrast and not_too_bright, {
        "color_diff": round(avg_color_diff, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4)
    }

def predict_image(img_path):
    # First check if it's a brain MRI
    is_mri, stats = is_brain_mri(img_path)

    if not is_mri:
        return "not_mri", 0.0, np.zeros(4), stats

    model = load_model()
    x = preprocess_img(img_path)
    preds = model.predict(x, verbose=0)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    # If confidence is too low — model is unsure
    if confidence < 0.4:
        return "uncertain", confidence, preds[0], stats

    return CLASS_NAMES[pred_idx], confidence, preds[0], stats