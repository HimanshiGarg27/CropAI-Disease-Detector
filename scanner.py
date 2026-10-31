import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json

@st.cache_data
def load_labels():
    with open("labels.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return ort.InferenceSession("mobilenetv3_cropcare.onnx")

labels = load_labels()
session = load_model()

st.title("🌱 CropCare Edge AI")
st.write("Take a picture of the crop leaf to detect diseases.")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    img = Image.open(camera_image).convert('RGB')
    img = img.resize((224, 224))
    
    img_array = np.array(img).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    img_array = np.transpose(img_array, (2, 0, 1))
    input_tensor = np.expand_dims(img_array, axis=0)

    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    output = session.run(None, ort_inputs)[0]
    
    pred_index = np.argmax(output)
    disease_name = labels[str(pred_index)]
    
    st.success(f"**Prediction:** {disease_name}")