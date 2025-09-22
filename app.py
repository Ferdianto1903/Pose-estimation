# app.py (Versi Final dengan UI Proporsional dan Terpusat)

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from collections import deque
import threading
import time
import tensorflow as tf
from PIL import Image
import tempfile
import os

# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="VisionAI - Deteksi Aksi",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Kustom untuk UI yang Lebih Baik ---
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #111827;
        font-family: 'Segoe UI', sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: #1f2937 !important;
        color: #ffffff !important;
    }

    [data-testid="stHeader"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background-color: #f9fafb;
        color: #111827;
        border-right: 1px solid #e5e7eb;
    }

    .main-container {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .bordered-container {
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        background: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #111827;
    }

    p, span, label, li {
        color: #374151;
    }

    .upload-area h4 {
        color: #1d4ed8;
    }

    .upload-area p {
        color: #1e40af;
    }

    .status-success {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #10b981;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }

    .status-info {
        background-color: #e0f2fe;
        color: #075985;
        border: 1px solid #38bdf8;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }

    .image-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
    }

    .centered-content {
        text-align: center;
    }
</style>

""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'action_classifier' not in st.session_state:
    from action_classifier import classify_action
    st.session_state.action_classifier = {
        'HISTORY_SIZE': 16,
        'TRANSITION_SMOOTHING': 5,
        'classify_action': classify_action
    }

# --- HEADER YANG TERPUSAT ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="centered-content">
        <h1 style="margin: 0; color: #1f2937; font-size: 3rem;">🏃‍♂️ VisionAI</h1>
        <h3 style="margin: 10px 0; color: #374151; font-weight: 600;">Deteksi & Klasifikasi Aksi Manusia</h3>
        <p style="color: #6b7280; font-size: 1.1rem;">Teknologi AI canggih untuk mendeteksi aktivitas manusia secara real-time</p>
    </div>
    """, unsafe_allow_html=True)

# --- INISIALISASI MODEL (CACHE) ---
@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        model.predict(np.zeros((360, 480, 3)), verbose=False)
        return model
    except Exception as e:
        st.error(f"Error memuat model YOLO: {e}")
        st.stop()

@st.cache_resource
def load_h5_model(uploaded_file):
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        model = tf.keras.models.load_model(temp_file_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model .h5: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- LOGIKA INTI PEMROSESAN FRAME (DIREFAKTORISASI) ---
def process_frame(frame, yolo_model, person_data, history_size, smoothing_threshold, confidence):
    """Memproses satu frame untuk deteksi dan klasifikasi aksi."""
    results = yolo_model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        conf=confidence, verbose=False
    )
    annotated_frame = results[0].plot(boxes=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data.cpu().numpy()

        for box, track_id, kpts in zip(boxes, track_ids, keypoints_data):
            if track_id not in person_data:
                person_data[track_id] = {
                    'keypoints_history': deque(maxlen=history_size),
                    'action_history': deque(maxlen=smoothing_threshold * 2),
                    'final_action': 'Standing'
                }
            
            p_data = person_data[track_id]
            classify_func = st.session_state.action_classifier['classify_action']
            
            p_data['keypoints_history'].append(kpts)
            current_action = classify_func(p_data['keypoints_history'], p_data['final_action'])
            p_data['action_history'].append(current_action)
            
            if p_data['action_history'].count(current_action) > smoothing_threshold:
                p_data['final_action'] = current_action
            
            x1, y1, _, _ = box
            action_text = f"ID:{track_id} {p_data['final_action']}"
            cvzone.putTextRect(annotated_frame, action_text, (x1, y1 - 10), scale=1.5, thickness=2, colorR=(0, 255, 0), colorT=(255,255,255), colorB=(0,165,0))
            
    return annotated_frame, person_data

# --- UI TABS YANG TERPUSAT ---
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🎥 Deteksi Video", "🖼️ Klasifikasi Gambar"])

# ==============================================================================
# |                            TAB 1: DETEKSI VIDEO                            |
# ==============================================================================
with tab1:
    st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="centered-content">
        <h2 style="color: #1f2937; margin-bottom: 10px;">🎥 Deteksi Aksi dari Video</h2>
        <p style="color: #6b7280;">Pilih sumber video untuk memulai deteksi aksi secara real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Opsi pilihan sumber video yang terpusat ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        source_option = st.radio(
            "",
            ["📷 Webcam Real-time", "⬆️ Unggah Video"],
            horizontal=True,
            label_visibility="collapsed"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Muat model YOLO sekali untuk tab ini
    yolo_model = load_yolo_model("yolov8m-pose.pt")

    # --- MODE 1: WEBCAM REAL-TIME ---
    if source_option == "📷 Webcam Real-time":
        with st.sidebar:
            st.markdown("### ⚙️ Pengaturan Webcam")
            confidence_threshold = st.slider("Threshold Keypoint YOLO", 0.3, 1.0, 0.5, 0.05, key="webcam_conf")
            st.markdown("---")
            st.markdown("### 🎛️ Pengaturan Akurasi")
            history_size_val = st.slider("Ukuran Histori Frame", 5, 50, st.session_state.action_classifier['HISTORY_SIZE'], 1, key="webcam_hist")
            smoothing_val = st.slider("Threshold Smoothing", 1, 20, st.session_state.action_classifier['TRANSITION_SMOOTHING'], 1, key="webcam_smooth")
            st.session_state.action_classifier['HISTORY_SIZE'] = history_size_val
            st.session_state.action_classifier['TRANSITION_SMOOTHING'] = smoothing_val
            st.markdown("---")
            
            if 'run_webcam' not in st.session_state:
                st.session_state.run_webcam = False
            
            def toggle_webcam():
                st.session_state.run_webcam = not st.session_state.run_webcam

            button_text = "⏹️ Hentikan Kamera" if st.session_state.run_webcam else "▶️ Mulai Kamera"
            st.button(button_text, on_click=toggle_webcam, use_container_width=True, type="primary" if st.session_state.run_webcam else "secondary")

        # Container video yang terpusat
        st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.5, 2, 0.5])
        with col2:
            image_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.run_webcam:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            person_data_webcam = {}
            
            while st.session_state.run_webcam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Gagal membaca frame dari kamera. Menghentikan...")
                    st.session_state.run_webcam = False
                    break

                annotated_frame, person_data_webcam = process_frame(
                    frame, yolo_model, person_data_webcam, 
                    st.session_state.action_classifier['HISTORY_SIZE'],
                    st.session_state.action_classifier['TRANSITION_SMOOTHING'],
                    confidence_threshold
                )
                
                with col2:
                    image_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            cap.release()
            if not st.session_state.run_webcam:
                with col2:
                    image_placeholder.markdown("""
                    <div class="status-info">
                        📷 Kamera telah dihentikan. Klik 'Mulai Kamera' untuk memulai lagi.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            with col2:
                image_placeholder.markdown("""
                <div class="video-container">
                    <div class="centered-content">
                        <h3 style="color: #374151;">📷 Webcam Siap</h3>
                        <p style="color: #6b7280;">Klik 'Mulai Kamera' di sidebar untuk mengaktifkan deteksi</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- MODE 2: UNGGAH VIDEO ---
    elif source_option == "⬆️ Unggah Video":
        with st.sidebar:
            st.markdown("### ⚙️ Pengaturan Video")
            confidence_vid = st.slider("Threshold Keypoint YOLO", 0.3, 1.0, 0.5, 0.05, key="vid_conf")
            st.markdown("---")
            st.markdown("### 🎛️ Pengaturan Akurasi")
            history_vid = st.slider("Ukuran Histori Frame", 5, 50, 16, 1, key="vid_hist")
            smoothing_vid = st.slider("Threshold Smoothing", 1, 20, 5, 1, key="vid_smooth")
            st.markdown("---")
            st.info("💡 **Tips:** Sesuaikan pengaturan sebelum mengunggah video untuk hasil optimal")
        
        st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="upload-area">
                <h3 style="margin-bottom: 10px;">📤 Unggah Video Anda</h3>
                <p style="color: #666;">Mendukung format: MP4, MOV, AVI, MKV</p>
            </div>
            """, unsafe_allow_html=True)
            uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_video is not None:
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as temp_file:
                    temp_file.write(uploaded_video.read())
                    temp_file_path = temp_file.name

                cap = cv2.VideoCapture(temp_file_path)
                if not cap.isOpened():
                    st.error("❌ Error: Gagal membuka file video.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="status-success">
                        ✅ Video berhasil dimuat! Total frame: {total_frames}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([0.5, 2, 0.5])
                    with col2:
                        video_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    person_data_vid = {}
                    frame_count = 0

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        annotated_frame, person_data_vid = process_frame(
                            frame, yolo_model, person_data_vid, 
                            history_vid, smoothing_vid, confidence_vid
                        )
                        
                        with col2:
                            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                            
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            status_text.markdown(f"""
                            <div style="text-align: center; padding: 10px;">
                                <strong>Memproses Frame: {frame_count}/{total_frames}</strong>
                                <br><small>Progress: {progress*100:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        status_text.markdown("""
                        <div class="status-success">
                            🎉 Pemrosesan video selesai!
                        </div>
                        """, unsafe_allow_html=True)
                    st.balloons()

            finally:
                if cap: cap.release()
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# ==============================================================================
# |                        TAB 2: KLASIFIKASI GAMBAR                           |
# ==============================================================================
with tab2:
    st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="centered-content">
        <h2 style="color: #1f2937; margin-bottom: 10px;">🖼️ Klasifikasi Aksi dari Gambar</h2>
        <p style="color: #6b7280;">Analisis aktivitas manusia dari gambar statis menggunakan model AI</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Pengaturan Model
    st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
    st.markdown("### 1. ⚙️ Pengaturan Model & Pra-pemrosesan", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            img_size = st.number_input("Ukuran Input (Pixel)", min_value=32, max_value=512, value=224, step=16)
        with col_b:
            norm_method = st.selectbox("Metode Normalisasi", 
                                     ["Bagi dengan 255 (0-1)", "Skala -1 sampai 1", "Tidak ada normalisasi"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload Section
    st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
    st.markdown("### 2. 📤 Unggah Aset", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="upload-area">
            <h4 style="color: #065f46;">🤖 Model Klasifikasi</h4>
            <p style="color: #047857; margin-bottom: 15px;">Format yang didukung: .h5, .hdf5</p>
        </div>
        """, unsafe_allow_html=True)
        h5_file = st.file_uploader("", type=["h5", "hdf5"], key="model_upload", label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="upload-area">
            <h4 style="color: #065f46;">🖼️ Gambar untuk Dianalisis</h4>
            <p style="color: #047857; margin-bottom: 15px;">Format yang didukung: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        image_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_upload", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Hasil Klasifikasi
    if h5_file and image_file:
        model_h5 = load_h5_model(h5_file)
        if model_h5:
            st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
            st.markdown("### 3. 📊 Hasil Klasifikasi")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                col_img, col_result = st.columns([1, 1])
                
                with col_img:
                    st.markdown('<div class="image-card">', unsafe_allow_html=True)
                    image = Image.open(image_file).convert('RGB')
                    st.image(image, caption="📷 Gambar Input", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_result:
                    st.markdown('<div class="image-card">', unsafe_allow_html=True)
                    with st.spinner("🔍 Menganalisis aksi..."):
                        img_resized = image.resize((img_size, img_size))
                        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                        
                        if norm_method == "Bagi dengan 255 (0-1)": 
                            img_array /= 255.0
                        elif norm_method == "Skala -1 sampai 1": 
                            img_array = (img_array / 127.5) - 1
                        
                        img_array = np.expand_dims(img_array, axis=0)
                        predictions = model_h5.predict(img_array)
                        score = tf.nn.softmax(predictions[0])
                        
                        CLASS_NAMES = ['sitting','using_laptop','hugging','sleeping','drinking',
                                     'clapping','dancing','cycling','calling','laughing','eating',
                                     'fighting','listening_to_music','running','texting']
                        
                        class_id = np.argmax(score)
                        if class_id < len(CLASS_NAMES):
                            predicted_action = CLASS_NAMES[class_id].replace("_", " ").title()
                            confidence = 100 * np.max(score)
                            
                            st.markdown("#### 🎯 Hasil Deteksi", unsafe_allow_html=True)
                            st.metric(label="Aksi Terdeteksi", value=predicted_action)
                            st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%")
                            
                            # Progress bar untuk confidence dengan warna yang jelas
                            st.markdown(f"""
                            <div style="background-color: #e5e7eb; border-radius: 10px; overflow: hidden; margin: 10px 0;">
                                <div style="background-color: #10b981; height: 20px; width: {confidence}%; 
                                           transition: width 0.3s ease; display: flex; align-items: center; 
                                           justify-content: center; color: white; font-weight: bold; font-size: 12px;">
                                    {confidence:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Error: Indeks kelas di luar jangkauan.")
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    elif not h5_file and not image_file:
        st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
        st.markdown("""
        <div class="centered-content">
            <h4 style="color: #374151;">🚀 Siap untuk Memulai!</h4>
            <p style="color: #6b7280;">Silakan unggah model (.h5) dan gambar untuk memulai klasifikasi aksi</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif not h5_file:
        st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
        st.info("🤖 Harap unggah model klasifikasi (.h5) terlebih dahulu")
        st.markdown('</div>', unsafe_allow_html=True)
    elif not image_file:
        st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
        st.info("🖼️ Harap unggah gambar untuk dianalisis")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Tutup main-container