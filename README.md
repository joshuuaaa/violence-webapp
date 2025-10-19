# Smart CCTV — Violence + Fire Detection

End-to-end web app with:
- Real-time webcam streaming from browser
- Violence detection (MobileNetV2 + BiLSTM, Keras)
- Fire detection (YOLOv5, PyTorch)
- Alerts with visual/audio cues
- Incident recording with pre/post roll
- Dashboard with history and stats
- Live configuration of sensitivity and FPS

## Project layout

- `frontend/` — Single-page UI served by FastAPI
- `backend/app/` — FastAPI app, inference, DB, recorder
- `models/` — Place your weights here
  - `best.pt` — Your YOLOv5 fire detector (custom). The app points to this by default.
  - `violence_model.keras` — Violence classifier (optional unless you install TensorFlow).
- `config/config.yaml` — Runtime configuration (auto-created)
- `recordings/`, `snapshots/` — Auto-created output folders
- `notebooks/violence_training_colab.py` — Local/Colab script to train/export the violence model

## Quick start (Linux)

1) Create a virtual environment (Python 3.13 shown). If you want TensorFlow, use Python 3.10–3.11 instead (see notes below).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the server

```bash
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

3) Open the app

- Visit http://127.0.0.1:8000/
- Click “Start” and grant camera permission
- Use the Configuration panel to tune thresholds and FPS
- Alerts appear in the status pill and beep; recordings and snapshots are saved when alerts occur

## Enabling Fire Detection (YOLOv5)

By default the backend points to `models/best.pt`.

- Install PyTorch for your environment. For CPU-only wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- Make sure `models/best.pt` exists (you already have it). On first run, YOLOv5 code is fetched via `torch.hub` — network access required. If you need fully offline, vendor the YOLOv5 repo and adjust the loader.

## Enabling Violence Detection (Keras)

TensorFlow wheels for Python 3.13 are not yet generally available. Use Python 3.10 or 3.11 in a separate venv if you want violence detection now.

Option A: Switch the app venv to Python 3.10/3.11

```bash
# Example with pyenv (or use conda/miniconda)
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tensorflow==2.16.1
```

Option B: Run violence in a separate microservice (advanced), or export to ONNX/TFLite and use onnxruntime — not wired here.

Place your trained `violence_model.keras` into `models/`. The server auto-loads it if TensorFlow is present.

## Configuration

- GET/POST `/api/config` — live config. Keys of interest:
  - `fps`: stream send rate
  - `violence_threshold`, `violence_sequence_length`
  - `fire_threshold`
  - `violence_model_path` (default `models/violence_model.keras`)
  - `fire_model_weights` (default `models/best.pt`)
  - `recording_enabled`, `pre_roll_seconds`, `post_roll_seconds`, `recordings_dir`

## API/WS

- WS `/ws/video` — send base64 JPEG frames, receive overlay + results
- GET `/api/events` — last 1000 events
- GET `/api/stats` — totals by type and by day
- Static:
  - `/static/*` frontend assets
  - `/snapshots/*` saved images
  - `/recordings/*` MP4 clips

## Notes and troubleshooting

- If the server logs "PyTorch not available; fire detection disabled.", install PyTorch as above.
- If violence score stays `-`, TensorFlow isn’t installed or sequence isn’t long enough yet.
- First YOLOv5 load via `torch.hub` downloads the model code; allow internet on first run.
- Webcam must be accessible from your browser; HTTPS is required by some browsers for camera on remote hosts.

## Training the violence model

Use `notebooks/violence_training_colab.py` locally or in Colab. It downloads a Kaggle dataset, extracts frames, trains MobileNetV2 + BiLSTM and exports `violence_model.keras`. After training, copy the file to `models/`.

