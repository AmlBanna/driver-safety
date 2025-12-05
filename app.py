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
import tempfile
import time
from collections import Counter, deque

# ================================
# 1. مسارات الموديلات (من Drive)
# ================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Google Drive IDs
DISTRACTION_MODEL_ID = "1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z"  # driver_distraction_model.keras
CLASS_JSON_ID = "1zDv7V4iQri7lC-e8BLsUJPLMuLlA8Ylg"          # class_indices.json
DROWSINESS_MODEL_ID = "1m-6tjfX46a82wxmrMTclBXrVAf_XmJlj"   # eye_model (SavedModel folder)

# URLs
DISTRACTION_URL = f"https://drive.google.com/uc?id={DISTRACTION_MODEL_ID}&export=download"
CLASS_JSON_URL = f"https://drive.google.com/uc?id={CLASS_JSON_ID}&export=download"
DROWSINESS_ZIP_URL = f"https://drive.google.com/uc?id={DROWSINESS_MODEL_ID}&export=download"

# Paths
DISTRACTION_MODEL_PATH = MODELS_DIR / "driver_distraction_model.keras"
CLASS_JSON_PATH = MODELS_DIR / "class_indices.json"
DROWSINESS_MODEL_PATH = MODELS_DIR / "eye_model"

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
        import zipfile
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
    download_file(CLASS_JSON_URL, CLASS_JSON_PATH)
    
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
# 5. موديل النعاس (SavedModel)
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
closed_count = 0
THRESH = 5

def detect_drowsiness(frame):
    global closed_count
    gray = cv2.cvtColor(cv2.resize(frame, (0,0), fx=0.7, fy=0.7), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
    eyes, boxes = [], []
    h, w = frame.shape[:2]
    for (x,y,fw,fh) in faces:
        roi = gray[y:y+int(fh*0.65), x:x+fw]
        es = eye_cascade.detectMultiScale(roi, 1.05, 4, minSize=(20,20))
        for (ex,ey,ew,eh) in es:
            if ey > roi.shape[0]*0.55: continue
            eye_img = cv2.resize(roi[ey:ey+eh, ex:ex+ew], (48,48)) / 255.0
            eyes.append(np.expand_dims(eye_img, -1).astype(np.float32))
            sx = w / gray.shape[1]; sy = h / gray.shape[0]
            boxes.append((int((x+ex)*sx), int((y+ey)*sy), int(ew*sx), int(eh*sy)))
    drowsy = False
    if eyes:
        batch = np.stack(eyes)
        out = drowsiness_fn(tf.constant(batch))
        key = list(out.keys())[0]
        scores = out[key].numpy().flatten()
        for i, (x,y,w,h) in enumerate(boxes):
            open_eye = scores[i] > 0.5
            col = (0,255,0) if open_eye else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
            if not open_eye:
                closed_count += 1
                drowsy = True
            else:
                closed_count = max(0, closed_count - 1)
    return frame, closed_count >= THRESH

# ================================
# 6. الواجهة
# ================================
st.set_page_config(page_title="نظام سلامة السائق", layout="wide")
st.title("نظام كشف النعاس والتشتت")
st.markdown("**موديلين من Drive + لا رفع على GitHub**")

tab1, tab2 = st.tabs(["كاميرا مباشرة", "رفع فيديو"])

with tab1:
    col1, col2 = st.columns(2)
    with col1: cam1 = st.checkbox("كاميرا أمامية (النعاس)", True)
    with col2: cam2 = st.checkbox("كاميرا جانبية (التشتت)", True)

    start = st.button("ابدأ", type="primary")
    stop = st.button("إيقاف")

    ph1 = st.empty()
    ph2 = st.empty()
    alert = st.empty()

    if start:
        cap1 = cv2.VideoCapture(0) if cam1 else None
        cap2 = cv2.VideoCapture(1) if cam2 else None
        st.session_state.running = True

    if stop:
        st.session_state.running = False
        st.rerun()

    if st.session_state.get("running"):
        while True:
            f1 = f2 = None
            if cap1 and cap1.isOpened():
                ret, f1 = cap1.read()
                if ret:
                    f1, d_alert = detect_drowsiness(f1.copy())
                    if d_alert:
                        alert.markdown("<h2 style='color:red;text-align:center;'>السائق نعسان!</h2>", unsafe_allow_html=True)
            if cap2 and cap2.isOpened():
                ret, f2 = cap2.read()
                if ret:
                    label = predict_distraction(f2.copy())
                    color = (0,0,255) if label != 'safe_driving' else (0,255,0)
                    cv2.putText(f2, label, (10,70), 0, 2, color, 4)

            if cam1 and f1 is not None: ph1.image(f1, channels="BGR")
            if cam2 and f2 is not None: ph2.image(f2, channels="BGR")
            time.sleep(0.03)

with tab2:
    f1 = st.file_uploader("فيديو أمامي (النعاس)", ["mp4","avi"])
    f2 = st.file_uploader("فيديو جانبي (التشتت)", ["mp4","avi"])
    if st.button("تحليل") and (f1 or f2):
        t1 = t2 = None
        if f1:
            t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t1.write(f1.read()); t1.close()
        if f2:
            t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t2.write(f2.read()); t2.close()

        cap1 = cv2.VideoCapture(t1.name) if t1 else None
        cap2 = cv2.VideoCapture(t2.name) if t2 else None
        ph1 = st.empty(); ph2 = st.empty()
        drowsy_count = 0; events = Counter()

        while (cap1 and cap1.isOpened()) or (cap2 and cap2.isOpened()):
            if cap1:
                r, f = cap1.read()
                if r:
                    f, a = detect_drowsiness(f.copy())
                    if a: drowsy_count += 1
                    ph1.image(f, channels="BGR")
            if cap2:
                r, f = cap2.read()
                if r:
                    l = predict_distraction(f.copy())
                    events[l] += 1
                    cv2.putText(f, l, (10,70), 0, 2, (0,0,255), 4)
                    ph2.image(f, channels="BGR")
            time.sleep(0.03)

        st.success(f"النعاس: {drowsy_count} | الأحداث: {dict(events)}")
        for t in (t1,t2): if t: os.unlink(t.name)
        for c in (cap1,cap2): if c: c.release()
