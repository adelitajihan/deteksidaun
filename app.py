import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# Setup Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load Model
model = YOLO('best.pt')

st.title("Scanner Penyakit Daun")

# Fitur kamera Streamlit (otomatis handle HP/Desktop)
img_file = st.camera_input("Ambil foto daun")

if img_file:
    image = Image.open(img_file)
    img_array = np.array(image)
    
    # Deteksi
    results = model(img_array)
    
    # Tampilkan hasil
    st.image(results[0].plot(), caption="Hasil Deteksi", use_column_width=True)
    
    # Simpan ke Firebase
    names = results[0].names
    if len(results[0].boxes) > 0:
        cls = int(results[0].boxes.cls[0])
        penyakit = names[cls]
        st.success(f"Terdeteksi: {penyakit}")
        db.collection('hasil_deteksi').add({'penyakit': penyakit, 'waktu': firestore.SERVER_TIMESTAMP})
