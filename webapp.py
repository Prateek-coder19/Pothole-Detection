import streamlit as st
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os
import io
from PIL import Image

def estimate_repair_cost(number_of_potholes, average_pothole_area, cost_per_unit_area=816):
    total_area = number_of_potholes * average_pothole_area
    return total_area * cost_per_unit_area

model = YOLO("yolov8n-seg.pt")

model = YOLO("C:/Users/admin/Desktop/python ds/cvprojects/pothole detection/runs/segment/train3/weights/best.pt")

st.set_page_config(page_title="Pothole Detector", page_icon="traffic light", layout="centered", initial_sidebar_state="expanded")
st.title("Pothole Detection")
st.write("upload Image to detect potholes on road")

uploaded_files = st.file_uploader("Upload Images...", type=["jpg", "jpeg", "png"])

if uploaded_files:
    image = Image.open(uploaded_files)
    st.image(image, caption="Uploading image...", use_column_width=True)
    st.write("Detecting Potholes..")

    results = model(image, save=True)

    img_array = results[0].plot() 
    img_pil = Image.fromarray(img_array) 

    st.image(img_pil, caption="Detected Potholes", use_column_width=True)
    

    number_of_potholes = len(results[0].boxes)  
    average_pothole_area = 0.9  
    repair_cost = estimate_repair_cost(number_of_potholes, average_pothole_area)

    st.write(f"Number of detected potholes: {number_of_potholes}")
    st.write(f"Estimated repair cost: ₹{repair_cost:.2f}")

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    
    st.download_button(
        label="Download Detected Image",
        data=buf,
        file_name="detected_potholes.png",
        mime="image/png"
        )