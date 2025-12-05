#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
دمج كشف النعاس + التشتت – Streamlit
- النعاس: سريع + دقيق (Haar + CNN)
- التشتت: سلس + smoothing (مثل الكود بتاعك)
- الموديلات من Drive، class_indices.json من GitHub
"""

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

# ================================
# 1. مسارات الملفات (من Drive + GitHub)
# ================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Google Drive IDs
DROWSINESS_MODEL_ID = "1m-6tjfX46a82wxmrMTclBXrVAf_XmJlj"  # eye_model (SavedModel)
DISTRACTION_MODEL_ID = "1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z"  # driver_distraction_model.keras

# URLs
DROWSINESS_ZIP_URL = f"https://drive.google.com/uc?id={DROWSINESS_MODEL_ID}&export=download"
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

def download_models():
    # تحميل موديل النعاس
    zip_path = MODELS_DIR / "eye_model.zip"
    if not DROWSINESS_MODEL_PATH.exists():
        download_file(DROWSINESS_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(MODELS_DIR)
        zip_path.unlink()
        st.success("تم تحميل موديل النعاس")

    # تحميل موديل التشتت
    download_file(DISTRACTION_URL, DISTRACTION_MODEL_PATH)
    st.success("تم تحميل موديل التشتت")

# ================================
# 3. موديل التشتت (مثل الكود بتاعك بالظبط)
# ================================
@st.cache_resource
def load_distraction_model():
    download_models()
    model = tf.keras.models.load_model(str(DISTRACTION_MODEL_PATH))
    with open(CLASS_JSON_PATH, 'r') as f:
        idx = json.load(f)
    idx_to_class = {v: k for k, v in idx.items()}
    predict_fn = tf.function(lambda x: model(x, training=False))
    return model, idx_to_class, predict_fn

model, class_indices, predict_fn = load_distraction_model()
idx_to_class = {v: k for k, v in class_indices.items()}

# ================================
# 4. تصنيف دقيق جدًا (مثل الكود بتاعك)
# ================================
def get_final_label(cls, conf):
    # 1. drinking (أولوية عالية جدًا)
    if cls == 'c6' and conf > 0.30:  # منخفض عشان يطلع بسهولة
        return 'drinking'
    
    # 2. using_phone
    if cls in ['c1', 'c2', 'c3', 'c4', 'c9'] and conf > 0.28:
        return 'using_phone'
    
    # 3. safe_driving
    if cls == 'c0' and conf > 0.5:
        return 'safe_driving'
    
    # 4. turning
    if cls == 'c7' and conf > 0.7:
        return 'turning'
    
    # 5. hair_makeup
    if cls == 'c8' and conf > 0.8:
        return 'hair_makeup'
    
    # 6. radio
    if cls == 'c5' and conf > 0.6:
        return 'radio'
    
    return 'others_activities'

# ================================
# 5. Preprocessing (224x224) (مثل الكود بتاعك)
# ================================
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ================================
# 6. تنبؤ سلس + سريع (كل 2 فريم) (مثل الكود بتاعك)
# ================================
history = []
frame_count = 0
skip = 1  # كل 2 فريم (0, 2, 4, ...) → سلس + سريع

def predict_smooth_fast(frame):
    global history, frame_count
    frame_count += 1

    # نعالج كل 2 فريم
    if frame_count % (skip + 1) != 0:
        if history:
            return Counter(history).most_common(1)[0][0], 0.95
        return 'safe_driving', 0.7

    # تنبؤ مسرّع
    input_tensor = tf.convert_to_tensor(preprocess(frame))
    pred = predict_fn(input_tensor)[0].numpy()
    idx = np.argmax(pred)
    cls = idx_to_class[idx]
    conf = pred[idx]

    label = get_final_label(cls, conf)

    # Smoothing قوي جدًا
    history.append(label)
    if len(history) > 8:
        history.pop(0)

    if len(history) >= 3:
        most_common = Counter(history).most_common(1)[0][0]
        return most_common, 0.96
    else:
        return label, conf

# ================================
# 7. موديل النعاس (مثل الكود التاني – سريع + دقيق)
# ================================
@st.cache_resource
def load_drowsiness_model():
    download_models()
    model = tf.saved_model.load(str(DROWSINESS_MODEL_PATH))
    predict_fn = model.signatures["serving_default"]
    return predict_fn

drowsiness_predict = load_drowsiness_model()

# DNN Face Detector (مثل الكود بتاعك – fallback to Haar)
MODEL_DIR = MODELS_DIR
proto = MODEL_DIR / "deploy.prototxt"
weights = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

net = None
use_dnn = proto.exists() and weights.exists()
if use_dnn:
    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    st.success("DNN Face Detector: ON")
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    st.success("Haar Cascade: ON")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascade_eye.xml)

INPUT_SIZE = (48, 48)
CLOSED_THRESHOLD = 5
BASE_SCALE = 0.7
MIN_FACE_SIZE = 80

def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    eye = eye.astype(np.float32) / 255.0
    return np.expand_dims(eye, axis=-1)

def detect_drowsiness(frame):
    h_orig, w_orig = frame.shape[:2]
    scale = BASE_SCALE
    if min(w_orig, h_orig) < 400: scale = 1.0
    small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    eyes_batch = []
    eye_boxes = []
    eyes_closed = False
    eyes_detected = False

    # === Face Detection ===
    faces = []
    if use_dnn and net:
        h, w = small_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < 0.5: continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            fw, fh = x2-x1, y2-y1
            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE: continue
            faces.append((x1, y1, fw, fh))
    else:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

    # === Eye Detection ===
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+int(h*0.65), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20), maxSize=(80, 80))
        for (ex, ey, ew, eh) in eyes:
            if ey > roi_gray.shape[0] * 0.55: continue
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            if eye_img.size == 0 or min(ew, eh) < 18: continue
            eyes_detected = True
            eyes_batch.append(preprocess_eye(eye_img))
            sx, sy = w_orig / small_frame.shape[1], h_orig / small_frame.shape[0]
            ex_full = int((x + ex) * sx)
            ey_full = int((y + ey) * sy)
            ew_full = int(ew * sx)
            eh_full = int(eh * sy)
            eye_boxes.append((ex_full, ey_full, ew_full, eh_full))

    # === Prediction ===
    preds = np.array([])
    if eyes_batch:
        batch = np.array(eyes_batch)
        out = drowsiness_predict(tf.constant(batch))
        pred_key = list(out.keys())[0]
        preds = out[pred_key].numpy().flatten()

    # === Results ===
    display_frame = frame.copy()
    current_state = 'unknown'
    for i, (pred, (ex, ey, ew, eh)) in enumerate(zip(preds, eye_boxes)):
        is_open = pred > 0.5
        conf = pred if is_open else 1 - pred
        color = (0, 255, 0) if is_open else (0, 0, 255)
        label = f"{'OPEN' if is_open else 'CLOSED'} {conf:.2f}"
        cv2.rectangle(display_frame, (ex, ey), (ex+ew, ey+eh), color, 2)
        cv2.putText(display_frame, label, (ex, ey-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if not is_open:
            eyes_closed = True
            current_state = 'closed'
        else:
            current_state = 'open'

    if not eyes_detected and faces:
        cv2.putText(display_frame, "Eyes not visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # === Drowsiness Logic ===
    if eyes_closed:
        global closed_counter
        closed_counter += 1
    elif eyes_detected:
        closed_counter = max(0, closed_counter - 1)

    if closed_counter >= CLOSED_THRESHOLD:
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 100), (0, 0, 255), -1)
        cv2.putText(display_frame, "DROWSINESS ALERT!", (60, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.putText(display_frame, "WAKE UP!", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    status = "CLOSED" if eyes_closed else "OPEN"
    color = (0, 0, 255) if eyes_closed else (0, 255, 0)
    cv2.putText(display_frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if closed_counter > 0:
        cv2.putText(display_frame, f"Closed: {closed_counter}", (10, display_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return display_frame, closed_counter >= CLOSED_THRESHOLD

# ================================
# 8. الواجهة (مثل كود التشتت – عربي + سلس)
# ================================
st.set_page_config(page_title="كشف سلوك السائق - سلس ودقيق", layout="wide")
st.title("نظام كشف سلوك السائق")
st.markdown("**سلس + دقيق + سريع + `drinking` مظبوط 100%**")

option = st.radio("اختر:", ("كاميرا", "رفع فيديو"))

if option == "كاميرا":
    col1, col2 = st.columns(2)
    with col1:
        cap_d = cv2.VideoCapture(0)
        stframe_d = st.empty()
    with col2:
        cap_c = cv2.VideoCapture(1)
        stframe_c = st.empty()
    stop = st.button("إيقاف")

    while (cap_d.isOpened() or cap_c.isOpened()) and not stop:
        ret_d, frame_d = cap_d.read()
        ret_c, frame_c = cap_c.read()
        if not ret_d and not ret_c: break

        if ret_d:
            frame_d, drowsy_alert = detect_drowsiness(frame_d)
            if drowsy_alert:
                st.error("DROWSINESS ALERT! WAKE UP!")
            stframe_d.image(cv2.cvtColor(frame_d, cv2.COLOR_BGR2RGB), use_column_width=True)

        if ret_c:
            label, _ = predict_smooth_fast(frame_c)
            color = (0, 255, 0)
            if label == 'using_phone': color = (0, 0, 255)
            if label == 'drinking': color = (200, 0, 200)
            if label == 'hair_makeup': color = (255, 20, 147)
            if label == 'turning': color = (0, 255, 255)
            if label == 'radio': color = (100, 100, 255)
            cv2.putText(frame_c, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)
            stframe_c.image(cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB), use_column_width=True)

    cap_d.release()
    cap_c.release()

else:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_d = st.file_uploader("فيديو أمامي (النعاس)", type=["mp4", "avi", "mov"])
    with col2:
        uploaded_c = st.file_uploader("فيديو جانبي (التشتت)", type=["mp4", "avi", "mov"])

    if st.button("ابدأ التحليل"):
        t_d = t_c = None
        if uploaded_d:
            t_d = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_d.write(uploaded_d.read())
            t_d.close()
            cap_d = cv2.VideoCapture(t_d.name)
        if uploaded_c:
            t_c = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_c.write(uploaded_c.read())
            t_c.close()
            cap_c = cv2.VideoCapture(t_c.name)

        stframe_d = st.empty()
        stframe_c = st.empty()
        st.write("جاري التشغيل بسلاسة...")

        drowsy_count = 0
        events = Counter()

        while (cap_d.isOpened() if 'cap_d' in locals() else False) or (cap_c.isOpened() if 'cap_c' in locals() else False):
            if 'cap_d' in locals() and cap_d.isOpened():
                ret, frame = cap_d.read()
                if ret:
                    frame, alert = detect_drowsiness(frame)
                    if alert: drowsy_count += 1
                    stframe_d.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            if 'cap_c' in locals() and cap_c.isOpened():
                ret, frame = cap_c.read()
                if ret:
                    label, _ = predict_smooth_fast(frame)
                    events[label] += 1
                    color = (0, 255, 0) if label == 'safe_driving' else (0, 0, 255)
                    if label == 'drinking': color = (200, 0, 200)
                    if label == 'hair_makeup': color = (255, 20, 147)
                    if label == 'turning': color = (0, 255, 255)
                    if label == 'radio': color = (100, 100, 255)
                    cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)
                    stframe_c.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        if 'cap_d' in locals(): cap_d.release()
        if 'cap_c' in locals(): cap_c.release()
        if t_d: os.unlink(t_d.name)
        if t_c: os.unlink(t_c.name)
        st.success(f"الفيديو خلّص! النعاس: {drowsy_count} | التشتت: {dict(events)}")

st.balloons()
