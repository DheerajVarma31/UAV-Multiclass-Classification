import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow as tf
import os

# Load model & label encoder
model = tf.keras.models.load_model("uav_multiclass_model.h5")
import joblib
le = joblib.load("label_encoder.pkl") if os.path.exists("label_encoder.pkl") else None

@st.cache_resource
def load_encoder():
    return joblib.load("label_encoder.pkl")

le = load_encoder()

def preprocess_audio(fp, sr=16000):
    y, _ = librosa.load(fp, sr=sr)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y, sr

def extract_spec(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(2.27,2.27), dpi=100)
    canvas = FigureCanvas(fig); ax = fig.add_subplot(111)
    librosa.display.specshow(S, sr=sr, ax=ax)
    ax.axis('off'); canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w,h = fig.canvas.get_width_height()
    plt.close(fig)
    img = img.reshape((h,w,4))[:,:,:3] / 255.0
    return img

st.title("🛸 UAV Sound Classifier — Multi‑Class")
st.write("Upload a `.wav` to classify.")

file = st.file_uploader("Audio file", type=["wav"])
if file:
    y, sr = preprocess_audio(file)
    img = extract_spec(y, sr)
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch)[0]
    idx = np.argmax(pred)
    label = le.inverse_transform([idx])[0] if le is not None else idx
    conf = pred[idx]

    st.success(f"**{label}** — Confidence: {conf*100:.1f}%")
    st.image(img, caption="Spectrogram", use_column_width=True)
    st.write({cls: round(float(p)*100,1) for cls,p in zip(le.classes_,pred)})
