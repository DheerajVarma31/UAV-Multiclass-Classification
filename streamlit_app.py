import os
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from PIL import Image
from io import BytesIO

# Load model and encoder with cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("uav_multiclass_cnn_model.h5")

@st.cache_data
def load_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
le = load_encoder()

# Extract spectrogram image
def extract_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    S = librosa.feature.melspectrogram(y=y_trimmed, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.27, 2.27), dpi=100)
    ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.axis('off')
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    plt.close(fig)

    return np.array(image.resize((227, 227))) / 255.0

# UI
st.title("🚁 UAV Multiclass Sound Classification")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    try:
        st.write("📈 Extracting features...")
        img = extract_spectrogram(uploaded_file)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        st.write("🧠 Predicting...")
        pred = model.predict(img)
        pred_class = np.argmax(pred)
        class_label = le.inverse_transform([pred_class])[0]
        confidence = np.max(pred)
        if class_label in ["bebop_1", "membo_1"]:
          st.success(f"🛩️ Predicted: **UAV** ({class_label})")
        else:
          st.warning(f"🔊 Predicted: **Non-UAV** ({class_label})")

        st.info(f"🎯 Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
