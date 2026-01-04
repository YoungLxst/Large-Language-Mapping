import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F
import os
import random

BUNDLE = torchaudio.pipelines.WAV2VEC2_XLSR_300M
WAV2VEC2_MODEL = BUNDLE.get_model()
WAV2VEC2_MODEL.eval()

# Move model to GPU if available to speed up feature extraction when used interactively
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    WAV2VEC2_MODEL.to(_device)
except Exception:
    # Some models may not support .to() at import time; ignore if it fails
    pass

SAMPLE_RATE = 16000
TARGET_LEN = SAMPLE_RATE * 3

class FeaturesDataset(Dataset):
    def __init__(self, df, root=""):
        self.paths = df["path"]
        self.labels = df["label"]
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.root + self.paths[idx])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sr, SAMPLE_RATE
            )

        T = waveform.shape[1]

        if T < TARGET_LEN:
            waveform = F.pad(waveform, (0, TARGET_LEN - T))
        else:
            start = (T - TARGET_LEN) // 2
            waveform = waveform[:, start:start + TARGET_LEN]

        # Ensure waveform is on the same device as the model for faster extraction
        try:
            waveform = waveform.to(next(WAV2VEC2_MODEL.parameters()).device)
        except Exception:
            pass

        with torch.inference_mode():
            features, _ = WAV2VEC2_MODEL.extract_features(waveform)

        x = torch.stack(features, dim=0)
        x = x.squeeze(1)

        label = self.labels[idx]

        return x, label
    
class PrecomputedFeaturesDataset(Dataset):
    def __init__(self, feature_dir, max_files: int | None = None, fraction: float | None = None,
                 shuffle: bool = False, seed: int = 0, dtype: torch.dtype = torch.float32):
        """Dataset that reads precomputed .pt feature files from disk.

        Args:
            feature_dir: directory containing .pt files saved as {'features': tensor, 'label': ...}
            max_files: if set, limits dataset to at most this many files (useful to debug / reduce memory)
            fraction: if set (0.0-1.0), keep only this fraction of files (overrides max_files calculation)
            shuffle: whether to randomly sample files when limiting
            seed: RNG seed used when shuffle=True
            dtype: torch dtype to cast loaded features to (use torch.float16 to reduce memory)
        """
        files = sorted([
            os.path.join(feature_dir, f)
            for f in os.listdir(feature_dir)
            if f.endswith(".pt")
        ])

        # Determine subset size if requested
        if fraction is not None:
            if not (0.0 < fraction <= 1.0):
                raise ValueError("fraction must be in (0, 1]")
            max_keep = int(len(files) * fraction)
        else:
            max_keep = max_files if max_files is not None else len(files)

        if max_keep is not None and max_keep < len(files):
            if shuffle:
                rng = random.Random(seed)
                files = rng.sample(files, max_keep)
            else:
                files = files[:max_keep]

        self.files = files
        self.dtype = dtype

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load tensors on CPU to avoid unnecessary GPU allocations during IO
        data = torch.load(self.files[idx], map_location="cpu")
        features = data["features"].to(dtype=self.dtype)
        label = data.get("label", None)
        return features, label
