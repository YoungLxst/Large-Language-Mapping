# Large-Language-Mapping

A real time séquentiel model for language classification

## Dataset

For this framework the dataset used is the uging face [common_laguage dataset](https://huggingface.co/datasets/speechbrain/common_language). It contain voice records of 45 diffenrents language.

## Warning

On running audio files. Torchcodec need ffmpeg DLL. Verify if your computer have them

## Quickstart

These notes get you from a fresh checkout to running the API locally and testing a prediction.

Prerequisites:
- Python 3.10+ (or your project's supported Python version)
- A virtual environment is strongly recommended (venv / conda)
- ffmpeg installed system-wide (for torchcodec/torchaudio) if you plan to process compressed audio

1) Create & activate a venv

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements
```

3) Prepare dataset files

The repository contains helper utilities in `utils/`. To build the combined CSVs and preprocess the Common Voice folders run:

```bash
python ./utils/load_data.py
```

4) Generate label mapping

If you haven't already generated the label mapping (index -> language name), run:

```bash
python ./scripts/generate_label_map.py
```

This writes `data/label_map.json` and `data/labels.txt`.

5) (Optional) Precompute wav2vec2 features

If you want to speed up training, extract and save features to disk using the extractor in `utils/features_extractor.py`.

```bash
python ./utils/features_extractor.py --input-csv data/train_clean.csv --out-dir data/precomputed_features/train
```

6) Train the model

Run the training script (it will pick sensible defaults but you can edit `model/train_model.py` flags):

```bash
python ./model/train_model.py
```

Checkpoints are saved to `model/trained/`.

7) Serve the model (API)

The FastAPI app is `app.py`. Place a trained `.pt` in `model/trained/` or set `LLM_MODEL_PATH` in a `.env` file at the project root.

Example `.env` (project root):

```
LLM_MODEL_PATH=/home/you/path/to/model/trained/LLM_latest.pt
```

Start the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

8) Test with curl (example)

This repository includes sample audio files under `data/common_voice_kpd/`. To send the Arabic test clip you mentioned, run (from project root):

```bash
curl -X POST "http://localhost:8000/predict" \
	-F "file=@data/common_voice_kpd/Arabic/test/ara_tst_sp_1/common_voice_ar_19224861.wav"
```

For pretty JSON output (requires `jq`):

```bash
curl -sS -X POST "http://localhost:8000/predict" \
	-F "file=@data/common_voice_kpd/Arabic/test/ara_tst_sp_1/common_voice_ar_19224861.wav" \
| jq .
```

Start the web page:

the command line has to be done in the web folder

```bash
uvicorn main_frontend:app --port 5500
```

Troubleshooting
- If `uvicorn` or your editor reports `Import "dotenv" could not be resolved`: install `python-dotenv` into the active venv (`pip install python-dotenv`) and restart your editor language server.
- If torchaudio/torchcodec raises errors about ffmpeg, install system ffmpeg (Debian/Ubuntu: `sudo apt install ffmpeg`).
- CUDA errors like "no kernel image is available" mean your PyTorch binary isn't compatible with your GPU; either install a matching wheel or run on CPU.
- OOM during training/eval: reduce batch size, set `dtype=float32`, or use the precomputed features pipeline.

Contributing
- VANDENBERGHE ilian

License
- None