import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras.models import load_model
import joblib
import os

# Paths
MODEL_PATH = "uav_multiclass_model.h5"
ENCODER_PATH = "label_encoder.pkl"

# Load model and label encoder
@st.cache_resource
def load_model_and_encoder():
    model = load_model(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le

model, le = load_model_and_encoder()

# Preprocess audio
def preprocess_audio(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed, sr

# Extract spectrogram image
def extract_spectrogram_image(y, sr):
    import matplotlib
    matplotlib.use("Agg")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.27, 2.27), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.axis('off')
    canvas.draw()

    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return image[:, :, :3] / 255.0

# Streamlit UI
st.set_page_config(page_title="UAV Sound Classifier", layout="centered")
st.title("🛩️ UAV Sound Classification App")

uploaded_file = st.file_uploader("📤 Upload a .wav audio file", type=["wav"])

if uploaded_file:
    with st.spinner("⏳ Processing..."):
        try:
            y, sr = preprocess_audio(uploaded_file)
            img = extract_spectrogram_image(y, sr)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            pred_class = np.argmax(prediction, axis=1)[0]
            class_label = le.inverse_transform([pred_class])[0]
            confidence = np.max(prediction)

            # Final binary classification
            is_uav = class_label in ["bebop_1", "membo_1"]
            uav_label = "UAV" if is_uav else "Non-UAV"

            st.success(f"🎯 Predicted Class: **{class_label}**")
            st.info(f"📈 Confidence: `{confidence:.2f}`")
            st.markdown(f"🚁 **Final Decision: `{uav_label}`**", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
