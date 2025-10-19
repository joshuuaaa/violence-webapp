# Local script: MobileNetV2 + Bi-LSTM Violence Detection
# Adapted for Ubuntu/VS Code from Colab notebook
# Compatible with TensorFlow 2.x (tested with 2.16+)

import os, zipfile, glob, shutil, subprocess, math
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------------- Config ----------------
DATASET = "inzilele/rlvs-hockey-fight-dataset-mp4-files"
BASE_DIR = os.path.expanduser("~/violence_detection")  # Change this to your preferred path
DOWNLOAD_DIR = os.path.join(BASE_DIR, "data")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
MODEL_OUTPUT = os.path.join(BASE_DIR, "violence_model.keras")
SEQ_LEN = 16
FPS_EXTRACT = 3
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 10
SEED = 42
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
SEQ_STRIDE = SEQ_LEN
# ---------------------------------------

print(f"Base directory: {BASE_DIR}")
print(f"Using TensorFlow version: {tf.__version__}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"GPU(s) available: {gpus}")
    except Exception as e:
        print(f"GPU memory growth setup failed: {e}")
else:
    print("No GPU detected; training will run on CPU.")

# Create directories
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# 1) Download and extract dataset using Kaggle API
print(f"\nDownloading dataset: {DATASET}")
download_cmd = f"kaggle datasets download -d {DATASET} -p {DOWNLOAD_DIR} --force"
try:
    subprocess.run(download_cmd, shell=True, check=True)
    print("Dataset downloaded successfully")
except subprocess.CalledProcessError as e:
    print(f"Error downloading dataset: {e}")
    print("Make sure Kaggle API is configured properly (~/.kaggle/kaggle.json)")
    exit(1)

# Extract zip files
print("Extracting dataset...")
zips = glob.glob(os.path.join(DOWNLOAD_DIR, "*.zip"))
for z in zips:
    print(f"Extracting {z}...")
    with zipfile.ZipFile(z, 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR)

VIDEO_ROOTS = [p for p in Path(DOWNLOAD_DIR).iterdir() if p.is_dir()]
print(f"Video roots: {[str(p) for p in VIDEO_ROOTS]}")

def extract_frames_opencv(video_path, out_dir, fps=FPS_EXTRACT):
    """Extract frames using OpenCV"""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    interval = max(int(round(video_fps / fps)), 1)
    idx = 0
    save_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            cv2.imwrite(os.path.join(out_dir, f"{save_idx:06d}.jpg"), frame)
            save_idx += 1
        idx += 1
    cap.release()
    return save_idx

def extract_frames_ffmpeg(video_path, out_dir, fps=FPS_EXTRACT):
    """Extract frames using ffmpeg as fallback"""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-err_detect", "ignore_err",
        "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(out_dir, "%06d.jpg")
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return 0
    return len(glob.glob(os.path.join(out_dir, "*.jpg")))

def safe_extract_frames(video_path, out_dir, fps=FPS_EXTRACT):
    """Try OpenCV first, fallback to ffmpeg"""
    n = extract_frames_opencv(video_path, out_dir, fps=fps)
    if n == 0:
        n = extract_frames_ffmpeg(video_path, out_dir, fps=fps)
    return n

# 2) Discover classes and extract frames
print("\nExtracting frames from videos...")
CLASSES = []
total_videos = 0
total_frames = 0

for root in VIDEO_ROOTS:
    for c in os.listdir(root):
        class_dir = root / c
        if not class_dir.is_dir():
            continue
        if c not in CLASSES:
            CLASSES.append(c)
        matches = [f for f in class_dir.rglob('*') if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
        print(f"{c}: found {len(matches)} videos under {class_dir}")
        for v in matches:
            rel = os.path.relpath(str(v), str(root))
            out = os.path.join(FRAMES_DIR, os.path.splitext(rel)[0])
            n = safe_extract_frames(str(v), out, fps=FPS_EXTRACT)
            total_videos += 1
            total_frames += n
            if total_videos % 10 == 0:
                print(f"  Processed {total_videos} videos...")

print(f"\nClasses: {CLASSES}")
print(f"Total videos processed: {total_videos}")
print(f"Total extracted frames: {total_frames}")

# 3) Build sequence windows
print("\nBuilding sequence windows...")

def list_sequence_windows(frames_dir, classes, seq_len=SEQ_LEN, stride=SEQ_STRIDE):
    class_to_idx = {c:i for i,c in enumerate(sorted(classes))}
    windows = []
    candidate_dirs = [d for d in glob.glob(os.path.join(frames_dir, '**'), recursive=True) if os.path.isdir(d)]
    for c in classes:
        cls_idx = class_to_idx[c]
        for seq_root in candidate_dirs:
            norm = seq_root.replace('\\', '/')
            if f"/{c}/" not in norm and not norm.endswith(f"/{c}"):
                continue
            frames = sorted(glob.glob(os.path.join(seq_root, '**', '*.jpg'), recursive=True))
            if len(frames) < seq_len:
                continue
            for i in range(0, len(frames) - seq_len + 1, stride):
                win = frames[i:i+seq_len]
                windows.append((win, cls_idx))
    return windows, class_to_idx

windows, class_to_idx = list_sequence_windows(FRAMES_DIR, CLASSES, SEQ_LEN, SEQ_STRIDE)
print(f"Total windows: {len(windows)}")
print(f"Class mapping: {class_to_idx}")

if len(windows) == 0:
    raise RuntimeError("No sequence windows built. Verify class names and extraction produced frames.")

# 4) Train/Val split
print("\nSplitting data into train and validation sets...")
indices = np.arange(len(windows))
labels = np.array([w[1] for w in windows])
X_train_idx, X_val_idx, y_train, y_val = train_test_split(
    indices, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# Determine positive (violence) class index
violence_class_name = None
for cname in CLASSES:
    if "violence" in cname.lower():
        violence_class_name = cname
        break
if violence_class_name is None:
    pos_idx = max(class_to_idx.values())
else:
    pos_idx = class_to_idx[violence_class_name]
print(f"Positive class (violence) index: {pos_idx}")

# 5) Keras Sequence streaming loader
class VideoSequence(tf.keras.utils.Sequence):
    def __init__(self, index_array, windows, pos_idx, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
        self.index_array = index_array
        self.windows = windows
        self.pos_idx = pos_idx
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return math.ceil(len(self.index_array) / self.batch_size)

    def on_epoch_end(self):
        rng = np.random.default_rng(SEED)
        rng.shuffle(self.index_array)

    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 127.5 - 1.0
        return img

    def __getitem__(self, idx):
        batch_ids = self.index_array[idx*self.batch_size:(idx+1)*self.batch_size]
        X_batch = []
        y_batch = []
        for bi in batch_ids:
            frame_paths, cls_idx = self.windows[bi]
            seq_imgs = []
            ok = True
            for fp in frame_paths:
                img = cv2.imread(fp)
                if img is None:
                    ok = False
                    break
                seq_imgs.append(self._preprocess(img))
            if not ok:
                seq_imgs = [np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)] * SEQ_LEN
            X_batch.append(np.stack(seq_imgs, axis=0))
            y_batch.append(1.0 if cls_idx == self.pos_idx else 0.0)
        X_batch = np.stack(X_batch, axis=0)
        y_batch = np.array(y_batch, dtype=np.float32)
        return X_batch, y_batch

train_loader = VideoSequence(X_train_idx.copy(), windows, pos_idx, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
val_loader = VideoSequence(X_val_idx.copy(), windows, pos_idx, batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# 6) Build model
print("\nBuilding model...")
base = tf.keras.applications.MobileNetV2(
    include_top=False, 
    weights="imagenet", 
    input_shape=(IMG_SIZE, IMG_SIZE, 3), 
    pooling="avg"
)
base.trainable = False

inputs = tf.keras.Input(shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
td = tf.keras.layers.TimeDistributed(base)(inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))(td)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3), 
    loss="binary_crossentropy", 
    metrics=["AUC"]
)

model.summary()

# 7) Train
print(f"\nStarting training for {EPOCHS} epochs...")
print(f"Steps per epoch: {len(train_loader)}, Validation steps: {len(val_loader)}")

history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    verbose=1
)

# 8) Save model
print(f"\nSaving model to {MODEL_OUTPUT}")
model.save(MODEL_OUTPUT)
print("Training complete!")
print(f"Model saved at: {MODEL_OUTPUT}")
print(f"\nTraining history:")
print(f"  Final training AUC: {history.history['auc'][-1]:.4f}")
print(f"  Final validation AUC: {history.history['val_auc'][-1]:.4f}")