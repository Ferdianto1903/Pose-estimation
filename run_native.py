# run_native.py

import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from collections import deque
import time

# Impor fungsi klasifikasi aksi dari file lain
from action_classifier import classify_action, HISTORY_SIZE, TRANSITION_SMOOTHING

# --- PENGATURAN ---
MODEL_NAME = "yolov8n-pose.pt"  # Model paling ringan
CONFIDENCE_THRESHOLD = 0.5  # Threshold kepercayaan
CAMERA_ID = 0  # 0 untuk webcam bawaan, bisa diubah jika ada >1 kamera
FRAME_WIDTH = 640  # Resolusi tangkapan kamera
FRAME_HEIGHT = 480

print("Mempersiapkan model, mohon tunggu...")

# --- INISIALISASI ---
try:
    model = YOLO(MODEL_NAME)
except Exception as e:
    print(f"Error memuat model: {e}")
    exit()

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"Error: Tidak dapat membuka kamera dengan ID {CAMERA_ID}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

person_data = {}
fps_start_time = 0
frame_count = 0

print("Model siap. Menjalankan deteksi real-time...")
print("Tekan tombol 'q' pada jendela video untuk keluar.")

# --- LOOP UTAMA ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera. Keluar...")
        break

    # --- Logika Deteksi Inti (dijalankan langsung di sini) ---
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        conf=CONFIDENCE_THRESHOLD, verbose=False
    )
    
    # Gunakan frame asli untuk menggambar anotasi
    annotated_frame = results[0].plot(boxes=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data.cpu().numpy()

        for box, track_id, kpts in zip(boxes, track_ids, keypoints_data):
            if track_id not in person_data:
                person_data[track_id] = {
                    'keypoints_history': deque(maxlen=HISTORY_SIZE),
                    'action_history': deque(maxlen=TRANSITION_SMOOTHING * 2), 'final_action': 'Standing'
                }
            p_data = person_data[track_id]
            p_data['keypoints_history'].append(kpts)
            current_action = classify_action(p_data['keypoints_history'], p_data['final_action'])
            p_data['action_history'].append(current_action)
            if p_data['action_history'].count(current_action) > TRANSITION_SMOOTHING:
                p_data['final_action'] = current_action
            
            x1, y1, _, _ = box
            action_text = f"ID:{track_id} {p_data['final_action']}"
            color = (0, 255, 0) # Warna hijau untuk teks
            cvzone.putTextRect(annotated_frame, action_text, (x1, y1 - 10), scale=1.5, thickness=2, colorR=color)
            
    # --- Kalkulasi FPS ---
    frame_count += 1
    if fps_start_time == 0:
        fps_start_time = time.time()
        
    elapsed_time = time.time() - fps_start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        # Tulis FPS di sudut frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_count = 0
        fps_start_time = time.time()

    # --- Tampilkan Hasil ---
    cv2.imshow('Deteksi Aksi Real-time Native', annotated_frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Bersih-bersih ---
cap.release()
cv2.destroyAllWindows()
print("Aplikasi ditutup.")