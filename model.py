import os
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras import layers, models, callbacks
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Paths to your multiclass folders
DATA_DIR = "C:\Users\dheer\UAV-Multiclass-Classification\dataset\Multiclass_Drone_Audio"  # Adjust if needed

def preprocess_audio(fp, sr=16000):
    y, _ = librosa.load(fp, sr=sr)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y, sr

def extract_spec(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(2.27,2.27), dpi=100)
    canvas = FigureCanvas(fig); ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.axis('off'); canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w,h = fig.canvas.get_width_height()
    plt.close(fig)
    img = img.reshape((h,w,4))[:,:,:3] / 255.0
    return img

def load_dataset(limit_per_class=200):
    X, y = [], []
    for cls in os.listdir(DATA_DIR):
        files = glob(os.path.join(DATA_DIR, cls,"*.wav"))[:limit_per_class]
        print(f"Loading {len(files)} samples of '{cls}'")
        for f in files:
            try:
                y_a, sr = preprocess_audio(f)
                X.append(extract_spec(y_a, sr))
                y.append(cls)
            except Exception as e:
                print("Error:", e)
    return np.array(X), np.array(y)

# Load & preprocess
X, y = load_dataset(limit_per_class=300)
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc,
    test_size=0.2, stratify=y_enc, random_state=42)
print("Classes:", le.classes_)
print("Train distribution:", np.bincount(y_train))
print("Test distribution:", np.bincount(y_test))

# Class weights to balance
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = dict(enumerate(weights))
print("Class weights:", cw)

# Build model
def build_model(input_shape=X.shape[1:], num_classes=len(le.classes_)):
    m = models.Sequential([
        layers.Conv2D(32,3,activation='relu', input_shape=input_shape),
        layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Conv2D(128,3,activation='relu'),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

model = build_model()
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2,
          epochs=30, batch_size=16, class_weight=cw, callbacks=[es])

model.save("uav_multiclass_model.h5")
print("🔖 Saved uav_multiclass_model.h5")

# Evaluate
pred = np.argmax(model.predict(X_test), axis=1)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred, target_names=le.classes_, zero_division=0))
