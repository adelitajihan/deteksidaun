import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Scanner Penyakit Daun")
model = YOLO('runs/detect/train/weights/best.pt')

# Fitur kamera HP (bisa langsung pakai kamera belakang)
img_file = st.camera_input("Ambil foto daun")

if img_file:
    # Ubah gambar jadi format yang bisa dibaca YOLO
    image = Image.open(img_file)
    img_array = np.array(image)
    
    # Deteksi
    results = model(img_array)
    
    # Tampilkan hasil
    st.image(results[0].plot(), caption="Hasil Deteksi", use_column_width=True)
    
    # Tampilkan nama penyakit
    names = results[0].names
    for box in results[0].boxes:
        cls = int(box.cls[0])
        st.success(f"Terdeteksi: {names[cls]}")