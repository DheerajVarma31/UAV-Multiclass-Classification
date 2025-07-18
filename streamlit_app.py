import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from PIL import Image
import io

# Load trained model and label encoder
MODEL_PATH = "uav_multiclass_model.h5"
ENCODER_PATH = "label_encoder.pkl"

model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Helper function to convert audio to spectrogram
def extract_spectrogram(file, sr=16000):
    y, _ = librosa.load(file, sr=sr)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    S = librosa.feature.melspectrogram(y=y_trimmed, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.27, 2.27), dpi=100)
    ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = image.resize((227, 227))
    image = np.array(image) / 255.0
    plt.close(fig)

    return image

# Streamlit app
st.title("🛩️ UAV Sound Classifier")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    spectrogram = extract_spectrogram(uploaded_file)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension

    pred_probs = model.predict(spectrogram)
    pred_index = np.argmax(pred_probs)
    pred_class = label_encoder.inverse_transform([pred_index])[0]

    # Final binary label
    if pred_class.lower() in ["bebop", "mambo"]:
        result = "✅ UAV Detected"
    else:
        result = "❌ Non-UAV Sound"

    st.markdown(f"### Prediction: **{result}**")
    st.markdown(f"*Model Class: `{pred_class}`*")
