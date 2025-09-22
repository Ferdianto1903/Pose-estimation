import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque
from tensorflow.keras.models import load_model
from action_classifier import classify_action

# Load model-model
pose_model = YOLO("yolov8m-pose.pt")  # Deteksi pose
cnn_model = load_model("cnn_model.h5")  # Model tambahan jika diperlukan

# Riwayat pose untuk klasifikasi aksi
pose_history = deque(maxlen=30)
last_action = "Unknown"

# Setup Tkinter window
root = tk.Tk()
root.title("Aplikasi Deteksi Aksi Manusia")
root.geometry("800x600")

# Tampilan video
video_label = tk.Label(root)
video_label.pack()

# Label hasil aksi
action_label = tk.Label(root, text="Aksi: -", font=("Helvetica", 18))
action_label.pack(pady=10)

# Stop flag dan capture
stop_event = threading.Event()

def preprocess_keypoints(keypoints):
    """Konversi keypoints dari YOLOv8 ke format (x, y, confidence)"""
    return [[float(k[0]), float(k[1]), float(k[2])] for k in keypoints]

def video_loop():
    global last_action
    cap = cv2.VideoCapture(0)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model.predict(frame, verbose=False)
        if results and results[0].keypoints:
            kpts = results[0].keypoints.xy.cpu().numpy()[0]
            conf = results[0].keypoints.conf.cpu().numpy()[0]
            combined = np.hstack((kpts, conf.reshape(-1, 1)))
            pose_history.append(combined)

            last_action = classify_action(pose_history, last_action)

        # Tampilkan teks aksi
        frame = cv2.putText(frame, f"Aksi: {last_action}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Tampilkan frame di Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    cap.release()

def start_video():
    stop_event.clear()
    threading.Thread(target=video_loop, daemon=True).start()

def stop_video():
    stop_event.set()
    messagebox.showinfo("Info", "Video dihentikan.")

def on_close():
    stop_event.set()
    root.destroy()

# Tombol kontrol
btn_start = tk.Button(root, text="Mulai Deteksi", command=start_video, width=20)
btn_start.pack(pady=5)

btn_stop = tk.Button(root, text="Hentikan", command=stop_video, width=20)
btn_stop.pack(pady=5)

# Exit
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
