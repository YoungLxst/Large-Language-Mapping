from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from typing import Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env (if present)
load_dotenv()

# Import model class
from model.LLM import LargeLanguageMappingModel
from model.Dataset import WAV2VEC2_MODEL

app = FastAPI(title="Language Detection API")
print(os.path.dirname(__file__))

# Allow CORS for local development (the web UI will be served separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure model paths and devices
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "trained")
MODEL_PATH = os.environ.get("LLM_MODEL_PATH")
LABEL_MAP_PATH_JSON = os.path.join(os.path.dirname(__file__), "data", "label_map.json")
LABELS_TXT = os.path.join(os.path.dirname(__file__), "data", "labels.txt")
print(f"MODEL_DIR: {MODEL_DIR}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"LABEL_MAP_PATH_JSON: {LABEL_MAP_PATH_JSON}")
print(f"LABELS_TXT: {LABELS_TXT}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classifier model
llm = LargeLanguageMappingModel()
# find model path
if MODEL_PATH is None:
    # pick newest file in MODEL_DIR
    if os.path.isdir(MODEL_DIR):
        files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
        if files:
            MODEL_PATH = sorted(files, key=os.path.getmtime)[-1]

if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No model found. Set LLM_MODEL_PATH or add a .pt file into {MODEL_DIR}")

state = torch.load(MODEL_PATH, map_location='cpu')
llm.load_state_dict(state)
llm.to(device)
llm.eval()

# Try to load label map
label_map = None
if os.path.exists(LABEL_MAP_PATH_JSON):
    try:
        with open(LABEL_MAP_PATH_JSON, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
    except Exception:
        label_map = None
elif os.path.exists(LABELS_TXT):
    try:
        with open(LABELS_TXT, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            label_map = {i: name for i, name in enumerate(lines)}
    except Exception:
        label_map = None

# Ensure WAV2VEC model device is known
try:
    wav2vec_device = next(WAV2VEC2_MODEL.parameters()).device
except Exception:
    wav2vec_device = torch.device('cpu')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """Accept an audio file upload and return predicted language index and optional name."""
    # Read upload into buffer
    data = await file.read()
    bio = io.BytesIO(data)
    try:
        waveform, sr = torchaudio.load(bio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio file: {e}")

    # Preprocess: mono, resample, pad/truncate to 3s
    SAMPLE_RATE = 16000
    TARGET_LEN = SAMPLE_RATE * 3
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    T = waveform.shape[1]
    if T < TARGET_LEN:
        waveform = F.pad(waveform, (0, TARGET_LEN - T))
    else:
        start = (T - TARGET_LEN) // 2
        waveform = waveform[:, start:start + TARGET_LEN]

    # Move waveform to wav2vec device for feature extraction
    try:
        waveform = waveform.to(wav2vec_device)
    except Exception:
        pass

    # Extract features
    with torch.inference_mode():
        try:
            features, _ = WAV2VEC2_MODEL.extract_features(waveform)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    x = torch.stack(features, dim=0).squeeze(1)  # (24, T', F)
    # Convert to batch shape expected by LLM: (1, 24, T', F)
    x = x.unsqueeze(0)

    # Ensure dtype and device match llm
    try:
        param_dtype = next(llm.parameters()).dtype
        if x.dtype != param_dtype:
            x = x.to(dtype=param_dtype)
    except Exception:
        pass
    x = x.to(device)

    # Forward
    with torch.no_grad():
        logits = llm(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_idx = int(pred.item())
        conf_val = float(conf.item())

    result = {"predicted_index": pred_idx, "confidence": conf_val}
    if label_map is not None:
        result["label"] = label_map.get(str(pred_idx), label_map.get(pred_idx, None))

    return JSONResponse(content=result)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
