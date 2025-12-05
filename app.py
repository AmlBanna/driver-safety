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
from collections import Counter, deque

# ================================
# 1. مسارات الموديلات
# ================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Google Drive IDs
DROWSINESS_ZIP_ID = "1m-6tjfX46a82wxmrMTclBXrVAf_XmJlj"           # eye_model (مجلد)
DISTRACTION_MODEL_ID = "1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z"      # driver_distraction_model.keras
CLASS_JSON_ID = "1zDv7V4iQri7lC-e8BLsUJPLMuLlA8Ylg"              # class_indices.json (من GitHub)

# URLs
DROWSINESS_ZIP_URL = f"https://drive.google.com/uc?id={DROWSINESS_ZIP_ID}&export=download"
DISTRACTION_URL = f"https://drive.google.com/uc?id={DISTRACTION_MODEL_ID}&export=download"

# Paths
DROWSINESS_MODEL_PATH = MODELS_DIR / "eye_model"
DISTRACTION_MODEL_PATH = MODELS_DIR / "driver_distraction_model.keras"
CLASS_JSON_PATH = BASE_DIR / "class_indices.json"  # من GitHub

# ================================
# 2. تحميل الموديلات من Drive
# ================================
def download_file(url, path):
    if path.exists():
        return
    with st.spinner(f"جاري تحميل {path.name}..."):
        urllib.request.urlretrieve(url, path)

def download_and_extract_drowsiness():
    zip_path = MODELS_DIR / "eye_model.zip"
    if not DROWSINESS_MODEL_PATH.exists():
        download_file(DROWSINESS_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(MODELS_DIR)
        zip_path.unlink()
        st.success("تم تحميل موديل النعاس")

# ================================
# 3. موديل التشتت (مثل الكود بتاعك بالظبط)
# ================================
@st.cache_resource
def load_distraction_model():
    download_file(DISTRACTION_URL, DISTRACTION_MODEL_PATH)
    model = tf.keras.models.load_model(str(DISTRACTION_MODEL_PATH))
    with open(CLASS_JSON_PATH) as f:
        idx = json.load(f)
    idx_to_class = {v: k for k, v in idx.items()}
    
    @tf.function
    def predict_fn(x):
        return model(x, training=False)
    
    return model, idx_to_class, predict_fn

distraction_model, idx_to_class, predict_fn = load_distraction_model()

# ================================
# 4. تصنيف دقيق (مثل الكود بتاعك)
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

def predict_distraction(frame):
    global history, frame_count
    frame_count += 1
    if frame_count % 2 != 0:
        return history[-1] if history else 'safe_driving'

    img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
    x = tf.convert_to_tensor(np.expand_dims(img, 0))
    pred = predict_fn(x)[0].numpy()
    idx = np.argmax(pred)
    cls = idx_to_class[idx]
    conf = pred[idx]
    label = get_final_label(cls, conf)
    history.append(label)
    if len(history) > 8: history.pop(0)
    return Counter(history).most_common(1)[0][0] if len(history) >= 3 else label

# ================================
# 5. موديل النعاس (مثل الكود التاني – سريع ودقيق)
# ================================
@st.cache_resource
def load_drowsiness_model():
    download_and_extract_drowsiness()
    model = tf.saved_model.load(str(DROWSINESS_MODEL_PATH))
    predict_fn = model.signatures["serving_default"]
    return predict_fn

drowsiness_fn = load_drowsiness_model()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
closed_counter = 0
CLOSED_THRESHOLD = 5
BASE_SCALE = 0.7

def detect_drowsiness(frame):
    global closed_counter
    h_orig, w_orig = frame.shape[:2]
    scale = BASE_SCALE
    if min(w_orig, h_orig) < 400: scale = 1.0
    small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80,80))
    eyes_batch = []
    eye_boxes = []

    for (x, y, fw, fh) in faces:
        roi = gray[y:y+int(fh*0.65), x:x+fw]
        eyes = eye_cascade.detectMultiScale(roi, 1.05, 4, minSize=(20,20), maxSize=(80,80))
        for (ex, ey, ew, eh) in eyes:
            if ey > roi.shape[0]*0.55: continue
            eye_img = roi[ey:ey+eh, ex:ex+ew]
            if min(ew, eh) < 18: continue
            eyes_batch.append(cv2.resize(eye_img, (48,48)) / 255.0)
            sx, sy = w_orig / small.shape[1], h_orig / small.shape[0]
            eye_boxes.append((int((x+ex)*sx), int((y+ey)*sy), int(ew*sx), int(eh*sy)))

    drowsy = False
    if eyes_batch:
        batch = np.stack([np.expand_dims(e, -1) for e in eyes_batch]).astype(np.float32)
        out = drowsiness_fn(tf.constant(batch))
        key = list(out.keys())[0]
        scores = out[key].numpy().flatten()
        for i, (x,y,w,h) in enumerate(eye_boxes):
            is_open = scores[i] > 0.5
            col = (0,255,0) if is_open else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
            if not is_open:
                closed_counter += 1
                drowsy = True
            else:
                closed_counter = max(0, closed_counter - 1)

    if not eyes_batch and faces:
        cv2.putText(frame, "Eyes not visible", (50,50), 0, 0.8, (0,165,255), 2)

    if closed_counter >= CLOSED_THRESHOLD:
        cv2.rectangle(frame, (0,0), (frame.shape[1],100), (0,0,255), -1)
        cv2.putText(frame, "DROWSINESS ALERT!", (60,55), 0, 1.3, (255,255,255), 3)

    return frame, closed_counter >= CLOSED_THRESHOLD

# ================================
# 6. الواجهة (Live + Upload)
# ================================
st.set_page_config(page_title="نظام سلامة السائق", layout="wide")
st.title("نظام كشف النعاس والتشتت")
st.markdown("**موديلين من Drive + `class_indices.json` من GitHub**")

tab1, tab2 = st.tabs(["كاميرا مباشرة", "رفع فيديو"])

# ========= LIVE =========
with tab1:
    col1, col2 = st.columns(2)
    with col1: cam_d = st.checkbox("أمامي (النعاس)", True)
    with col2: cam_c = st.checkbox("جانبي (التشتت)", True)

    start = st.button("ابدأ", type="primary")
    stop = st.button("إيقاف")

    ph_d = st.empty()
    ph_c = st.empty()
    alert_ph = st.empty()

    if start:
        cap_d = cv2.VideoCapture(0) if cam_d else None
        cap_c = cv2.VideoCapture(1) if cam_c else None
        st.session_state.live = True

    if stop:
        st.session_state.live = False
        st.rerun()

    if st.session_state.get("live"):
        while True:
            f_d = f_c = None
            if cap_d and cap_d.isOpened():
                ret, f = cap_d.read()
                if ret:
                    f, alert = detect_drowsiness(f.copy())
                    if alert:
                        alert_ph.markdown("<h2 style='color:red;text-align:center;'>السائق نعسان!</h2>", unsafe_allow_html=True)
                    f_d = f
            if cap_c and cap_c.isOpened():
                ret, f = cap_c.read()
                if ret:
                    label = predict_distraction(f.copy())
                    color = (0,0,255) if label != 'safe_driving' else (0,255,0)
                    cv2.putText(f, label, (10,70), 0, 2, color, 4)
                    f_c = f

            if cam_d and f_d is not None: ph_d.image(f_d, channels="BGR")
            if cam_c and f_c is not None: ph_c.image(f_c, channels="BGR")
            time.sleep(0.03)

# ========= UPLOAD =========
with tab2:
    f_d = st.file_uploader("فيديو أمامي (النعاس)", ["mp4","avi"])
    f_c = st.file_uploader("فيديو جانبي (التشتت)", ["mp4","avi"])
    if st.button("تحليل") and (f_d or f_c):
        t_d = t_c = None
        if f_d:
            t_d = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_d.write(f_d.read()); t_d.close()
        if f_c:
            t_c = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_c.write(f_c.read()); t_c.close()

        cap_d = cv2.VideoCapture(t_d.name) if t_d else None
        cap_c = cv2.VideoCapture(t_c.name) if t_c else None
        ph_d = st.empty(); ph_c = st.empty()
        drowsy_count = 0; events = Counter()

        while (cap_d and cap_d.isOpened()) or (cap_c and cap_c.isOpened()):
            if cap_d:
                r, f = cap_d.read()
                if r:
                    f, a = detect_drowsiness(f.copy())
                    if a: drowsy_count += 1
                    ph_d.image(f, channels="BGR")
            if cap_c:
                r, f = cap_c.read()
                if r:
                    l = predict_distraction(f.copy())
                    events[l] += 1
                    cv2.putText(f, l, (10,70), 0, 2, (0,0,255), 4)
                    ph_c.image(f, channels="BGR")
            time.sleep(0.03)

        st.success(f"النعاس: {drowsy_count} | الأحداث: {dict(events)}")
        for t in (t_d, t_c): if t: os.unlink(t.name)
        for c in (cap_d, cap_c): if c: c.release()
