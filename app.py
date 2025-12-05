#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import json
import os
from pathlib import Path
import urllib.request
import zipfile
import tempfile
import time
from collections import Counter
import warnings

# تجاهل أخطاء غير مهمة
warnings.filterwarnings("ignore", category=UserWarning, module="rich")

# ================================
# 1. المسارات
# ================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DROWSINESS_ZIP_ID = "1m-6tjfX46a82wxmrMTclBXrVAf_XmJlj"
DISTRACTION_MODEL_ID = "1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z"

DROWSINESS_ZIP_URL = f"https://drive.google.com/uc?id={DROWSINESS_ZIP_ID}&export=download"
DISTRACTION_URL = f"https://drive.google.com/uc?id={DISTRACTION_MODEL_ID}&export=download"

DROWSINESS_MODEL_PATH = MODELS_DIR / "eye_model"
DISTRACTION_MODEL_PATH = MODELS_DIR / "driver_distraction_model.keras"
CLASS_JSON_PATH = BASE_DIR / "class_indices.json"

# ================================
# 2. تحميل الموديلات
# ================================
def download_file(url, path):
    if path.exists():
        return
    with st.spinner(f"تحميل {path.name}..."):
        urllib.request.urlretrieve(url, path)

def download_models():
    # موديل النعاس (SavedModel)
    zip_path = MODELS_DIR / "eye_model.zip"
    if not DROWSINESS_MODEL_PATH.exists():
        download_file(DROWSINESS_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(MODELS_DIR)
        zip_path.unlink()
        st.success("تم تحميل موديل النعاس")

    # موديل التشتت (.keras)
    download_file(DISTRACTION_URL, DISTRACTION_MODEL_PATH)
    st.success("تم تحميل موديل التشتت")

# ================================
# 3. موديل التشتت (مثل كودك)
# ================================
@st.cache_resource
def load_distraction():
    download_models()
    model = tf.keras.models.load_model(str(DISTRACTION_MODEL_PATH))
    with open(CLASS_JSON_PATH) as f:
        idx = json.load(f)
    idx_to_class = {v: k for k, v in idx.items()}
    predict_fn = tf.function(lambda x: model(x, training=False))
    return model, idx_to_class, predict_fn

model, idx_to_class, predict_fn = load_distraction()

# ================================
# 4. تصنيف التشتت (مثل كودك)
# ================================
def get_final_label(cls, conf):
    if cls == 'c6' and conf > 0.30: return 'drinking'
    if cls in ['c1','c2','c3','c4','c9'] and conf > 0.28: return 'using_phone'
    if cls == 'c0' and conf > 0.5: return 'safe_driving'
    if cls == 'c7' and conf > 0.7: return 'turning'
    if cls == 'c8' and conf > 0.8: return 'hair_makeup'
    if cls == 'c5' and conf > 0.6: return 'radio'
    return 'others_activities'

history = []
frame_count = 0

def predict_smooth_fast(frame):
    global history, frame_count
    frame_count += 1

    if frame_count % 2 != 0:
        return history[-1] if history else 'safe_driving'

    input_tensor = tf.convert_to_tensor(np.expand_dims(cv2.resize(frame, (224,224)).astype(np.float32)/255.0, 0))
    pred = predict_fn(input_tensor)[0].numpy()
    idx = np.argmax(pred)
    cls = idx_to_class[idx]
    conf = pred[idx]

    label = get_final_label(cls, conf)

    history.append(label)
    if len(history) > 8:
        history.pop(0)

    if len(history) >= 3:
        return Counter(history).most_common(1)[0][0]
    return label

# ================================
# 5. موديل النعاس (مثل كودك)
# ================================
@st.cache_resource
def load_drowsiness():
    download_models()
    model = tf.saved_model.load(str(DROWSINESS_MODEL_PATH))
    return model.signatures["serving_default"]

drowsiness_fn = load_drowsiness()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
closed_counter = 0
THRESH = 5
BASE_SCALE = 0.7
MIN_FACE_SIZE = 80

def detect_drowsiness(frame):
    global closed_counter
    h_orig, w_orig = frame.shape[:2]
    scale = BASE_SCALE
    if min(w_orig, h_orig) < 400: scale = 1.0
    small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
    eyes_batch = []
    eye_boxes = []
    eyes_closed = False
    eyes_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+int(h*0.65), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20), maxSize=(80, 80))
        for (ex, ey, ew, eh) in eyes:
            if ey > roi_gray.shape[0] * 0.55: continue
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            if eye_img.size == 0 or min(ew, eh) < 18: continue
            eyes_detected = True
            eyes_batch.append(cv2.resize(eye_img, (48,48)).astype(np.float32) / 255.0)
            sx, sy = w_orig / small_frame.shape[1], h_orig / small_frame.shape[0]
            ex_full = int((x + ex) * sx)
            ey_full = int((y + ey) * sy)
            ew_full = int(ew * sx)
            eh_full = int(eh * sy)
            eye_boxes.append((ex_full, ey_full, ew_full, eh_full))

    preds = np.array([])
    if eyes_batch:
        batch = np.stack([np.expand_dims(e, -1) for e in eyes_batch])
        out = drowsiness_fn(tf.constant(batch))
        pred_key = list(out.keys())[0]
        preds = out[pred_key].numpy().flatten()

    current_state = 'unknown'
    for i, (pred, (ex, ey, ew, eh)) in enumerate(zip(preds, eye_boxes)):
        is_open = pred > 0.5
        conf = pred if is_open else 1 - pred
        color = (0, 255, 0) if is_open else (0, 0, 255)
        label = f"{'OPEN' if is_open else 'CLOSED'} {conf:.2f}"
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), color, 2)
        cv2.putText(frame, label, (ex, ey-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if not is_open:
            eyes_closed = True
            current_state = 'closed'
        else:
            current_state = 'open'

    if not eyes_detected and faces:
        cv2.putText(frame, "Eyes not visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Drowsiness Logic
    if eyes_closed:
        closed_counter += 1
    elif eyes_detected:
        closed_counter = max(0, closed_counter - 1)

    if closed_counter >= THRESH:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
        cv2.putText(frame, "DROWSINESS ALERT!", (60, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.putText(frame, "WAKE UP!", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    status = "CLOSED" if eyes_closed else "OPEN"
    color = (0, 0, 255) if eyes_closed else (0, 255, 0)
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if closed_counter > 0:
        cv2.putText(frame, f"Closed: {closed_counter}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return frame, closed_counter >= THRESH

# ================================
# 6. الواجهة
# ================================
st.set_page_config(page_title="نظام سلامة السائق", layout="wide")
st.title("نظام كشف النعاس والتشتت")

tab1, tab2 = st.tabs(["كاميرا مباشرة", "رفع فيديو"])

# ---------- Live ----------
with tab1:
    col1, col2 = st.columns(2)
    with col1: cam1 = st.checkbox("أمامي (نعاس)", True)
    with col2: cam2 = st.checkbox("جانبي (تشتت)", True)
    start = st.button("ابدأ", type="primary")
    stop = st.button("إيقاف")
    ph1 = st.empty(); ph2 = st.empty(); alert = st.empty()

    if start:
        cap1 = cv2.VideoCapture(0) if cam1 else None
        cap2 = cv2.VideoCapture(1) if cam2 else None
        st.session_state.run = True

    if stop:
        st.session_state.run = False
        st.rerun()

    if st.session_state.get("run"):
        while True:
            f1 = f2 = None
            if cap1 and cap1.isOpened():
                r, f = cap1.read()
                if r:
                    f1, a = detect_drowsiness(f.copy())
                    if a:
                        alert.markdown("<h2 style='color:red;text-align:center;'>نعسان!</h2>", unsafe_allow_html=True)
            if cap2 and cap2.isOpened():
                r, f = cap2.read()
                if r:
                    l = predict_distraction(f.copy())
                    cv2.putText(f, l, (10,70), 0, 2, (0,0,255), 4)
                    f2 = f
            if cam1 and f1 is not None: ph1.image(f1, channels="BGR")
            if cam2 and f2 is not None: ph2.image(f2, channels="BGR")
            time.sleep(0.03)

# ---------- Upload ----------
with tab2:
    f1 = st.file_uploader("فيديو أمامي (نعاس)", ["mp4"])
    f2 = st.file_uploader("فيديو جانبي (تشتت)", ["mp4"])
    if st.button("تحليل") and (f1 or f2):
        t1 = t2 = None
        if f1:
            t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t1.write(f1.read())
            t1.close()
        if f2:
            t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t2.write(f2.read())
            t2.close()
        cap1 = cv2.VideoCapture(t1.name) if t1 else None
        cap2 = cv2.VideoCapture(t2.name) if t2 else None
        ph1 = st.empty(); ph2 = st.empty()
        dc = 0; ev = Counter()
        while (cap1 and cap1.isOpened()) or (cap2 and cap2.isOpened()):
            if cap1:
                r, f = cap1.read()
                if r:
                    f, a = detect_drowsiness(f.copy())
                    if a: dc += 1
                    ph1.image(f, channels="BGR")
            if cap2:
                r, f = cap2.read()
                if r:
                    l = predict_distraction(f.copy())
                    ev[l] += 1
                    cv2.putText(f, l, (10,70), 0, 2, (0,0,255), 4)
                    ph2.image(f, channels="BGR")
            time.sleep(0.03)
        st.success(f"النعاس: {dc} | الأحداث: {dict(ev)}")
        # تنظيف الملفات
        for t in (t1, t2):
            if t:
                os.unlink(t.name)
        for c in (cap1, cap2):
            if c:
                c.release()
