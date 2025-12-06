import requests
from tqdm import tqdm
import os
import zipfile
import numpy as np
import polars as pl

def download():
    url = "https://huggingface.co/datasets/speechbrain/common_language/resolve/main/data/CommonLanguage.zip"
    local_filename = url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length",0))
        with open(local_filename, "wb") as f, tqdm(
            total=total
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"file {local_filename} download")

    repo_root = os.getcwd()
    extract_dir = os.path.join(repo_root, "data")
    print(f"Extracting to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(local_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(local_filename)
    print(f"File {local_filename} extracted to {extract_dir}")

def clean_csv(dataset_path="data/common_voice_kpd/"):
    languages = os.listdir(dataset_path)

    file_type = ["train","test","dev"]
    for lang in languages:
        for file in file_type:
            with open(f"{dataset_path}{lang}/{file}.csv", "r", encoding="utf-16") as f:
                lines = f.readline()
                client_id = np.array([], dtype=np.int16)
                path = np.array([])
                language = np.array([])
                while True:
                    lines = f.readline()
                    if not lines:
                        break
                    lines = lines.split()
                    client_id = np.append(client_id, lines[0])
                    path = np.append(path, f"/{lang}/{file}/{lines[1]}/{lines[2]}")
                    language = np.append(language, lang)

                df = (
                    pl.DataFrame({
                        "client_id": client_id,
                        "path":path,
                        "language": language,
                    })
                )

            df.write_csv(f"{dataset_path}{lang}/{file}_clean.csv")

    file_type = ["train_clean","test_clean","dev_clean"]
    for file in file_type:
        combined_df = pl.DataFrame({
            "client_id": np.array([]),
            "path": np.array([]),
            "language": np.array([]),
        }).with_columns(
            pl.col("*").cast(pl.String)
        )
        for lang in languages:
            df = pl.read_csv(
                f"{dataset_path}{lang}/{file}.csv"
                ).with_columns(
                    pl.col("*").cast(pl.String)
                )
            combined_df = pl.concat([combined_df, df])
        combined_df.write_csv(f"./data/{file}.csv")

def main():
    download()
    clean_csv()

if __name__ == "__main__":
    main()