import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F

BUNDLE = torchaudio.pipelines.WAV2VEC2_XLSR_300M
WAV2VEC2_MODEL = BUNDLE.get_model()
WAV2VEC2_MODEL.eval()

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

        with torch.inference_mode():
            features, _ = WAV2VEC2_MODEL.extract_features(
                waveform
            )

        x = torch.stack(features, dim=0)
        x = x.squeeze(1)

        label = self.labels[idx]

        return x, label