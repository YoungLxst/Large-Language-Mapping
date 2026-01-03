import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F
import argparse
import polars as pl
from tqdm import tqdm

BUNDLE = torchaudio.pipelines.WAV2VEC2_XLSR_300M
WAV2VEC2_MODEL = BUNDLE.get_model()
WAV2VEC2_MODEL.eval()

SAMPLE_RATE = 16000

class RawAudioDataset(Dataset):
    def __init__(self, paths, labels, root=""):
        self.paths = paths
        self.labels = labels
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.root + self.paths[idx])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        return waveform, int(self.labels[idx])


def extract_and_save(dataset, out_dir, time_length=3):
    os.makedirs(out_dir, exist_ok=True)
    target_len = SAMPLE_RATE * time_length

    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        waveform, label = dataset[i]

        T = waveform.shape[1]
        if T < target_len:
            waveform = F.pad(waveform, (0, target_len - T))
        else:
            start = (T - target_len) // 2
            waveform = waveform[:, start:start + target_len]

        with torch.inference_mode():
            features, _ = WAV2VEC2_MODEL.extract_features(waveform)

        x = torch.stack(features, dim=0).squeeze(1)  # (24, T', F)
        torch.save(
            {
                "features": x.cpu(),
                "label": label,
            },
            os.path.join(out_dir, f"sample_{i}.pt")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to the csv file")
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--time_length", type=int, default=3)
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="fraction of data to sample from csv")

    args = parser.parse_args()

    df = pl.read_csv(args.csv).sample(fraction=args.sample_fraction, shuffle=True)

    paths = df["path"]
    labels = df["label"]

    dataset = RawAudioDataset(paths, labels, root=args.root)
    extract_and_save(dataset, args.out_dir, args.time_length)


if __name__ == "__main__":
    main()

    # class PrecomputedFeaturesDataset(Dataset):
    #     def __init__(self, feature_dir):
    #         self.files = sorted([
    #             os.path.join(feature_dir, f)
    #             for f in os.listdir(feature_dir)
    #             if f.endswith(".pt")
    #         ])

    #     def __len__(self):
    #         return len(self.files)

    #     def __getitem__(self, idx):
    #         data = torch.load(self.files[idx])
    #         return data["features"], data["label"]

    # dataset = PrecomputedFeaturesDataset("data/samples")
    # for i in range(len(dataset)):
    #     x, label = dataset[i]
    #     print(f"Sample {i}: features shape {x.shape}, label {label}")