if __name__ == "__main__":
    import polars as pl
    import torch
    from torch.utils.data import DataLoader
    from Dataset import PrecomputedFeaturesDataset
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import numpy as np
    from tqdm import tqdm

    print("--- Charger le dataset complet ---")
    dataset = PrecomputedFeaturesDataset(
        "data/precomputed_features/train",
        fraction=0.1,
        shuffle=True,
        seed=42,
        dtype=torch.float32
    )

    # --- Extraire toutes les features et labels ---
    all_features = []
    all_labels = []

    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        x, y = dataset[i]
        # x: (T, 1024) = (149, 1024)
        # On applique mean pooling pour réduire à un vecteur unique
        x_reduced = x[12].mean(axis=0)  # (1024,)
        all_features.append(x_reduced.numpy())
        all_labels.append(y)

    X = np.stack(all_features)  # (n_samples, 1024)
    y = np.array(all_labels)    # (n_samples,)

    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)

    print("--- Split train / test ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- Créer et entraîner le modèle sklearn ---")
    clf = LogisticRegression(
        max_iter=2000,
        solver='saga',
    )
    clf.fit(X_train, y_train)

    print("-- Évaluer ---")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Sklearn LogisticRegression Accuracy: {acc*100:.2f}%")
