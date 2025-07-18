import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from keras import layers, models
from glob import glob

# 📁 Dataset Path
DATA_DIR = "C:/Users/dheer/UAV-Multiclass-Classification/dataset/Multiclass_Drone_Audio"

# 1️⃣ Preprocess Audio

def preprocess_audio(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed, sr

# 2️⃣ Extract Spectrogram Image

def extract_spectrogram_image(y, sr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.27, 2.27), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
    ax.axis('off')

    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    return image[:, :, :3] / 255.0

# 3️⃣ Load Dataset

def load_dataset():
    X, y = [], []
    for label_dir in os.listdir(DATA_DIR):
       label_path = os.path.join(DATA_DIR, label_dir)
       if not os.path.isdir(label_path):
            continue
       if "bebop_1" in label_dir:
            actual_label = "bebop_1"
       elif "membo_1" in label_dir:
            actual_label = "membo_1"
       elif "unknown" in label_dir:
            actual_label = "unknown"
       else:
        print(f"❌ Skipping unrecognized folder: {label_dir}")
        continue

    print(f"📁 Loading 300 samples of '{label_dir}' (as '{actual_label}')")
    files = glob(os.path.join(label_path, "*.wav"))[:300]

    for f in files:
        try:
            y_audio, sr = preprocess_audio(f)
            img = extract_spectrogram_image(y_audio, sr)
            X.append(img)
            y.append(actual_label)
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")

    print(f"✅ Total loaded samples: {len(X)}")
    return np.array(X), np.array(y)

# Load Data
X, y = load_dataset()
X = X / 255.0

# Label Encoding and Save Encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "label_encoder.pkl")

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Class Weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("📊 Class weights:", class_weights_dict)
print("Train class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))

# 4️⃣ Build Model

def build_multiclass_cnn(input_shape=(227, 227, 3), num_classes=5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_multiclass_cnn(num_classes=len(np.unique(y_encoded)))

# 5️⃣ Train Model
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=16,
    class_weight=class_weights_dict
)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoded classes:", le.classes_)

# 6️⃣ Evaluate and Save
model.save("uav_multiclass_cnn_model.h5")
y_pred = np.argmax(model.predict(X_test), axis=1)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
