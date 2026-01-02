if __name__ == "__main__":
    print("------\n    import lib\n\n")
    import polars as pl
    from torch.utils.data import DataLoader
    from LLM import LargeLanguageMappingModel
    from Dataset import FeaturesDataset

    print("------\n    create dataloader\n\n")
    train_loader = DataLoader(
        FeaturesDataset(pl.read_csv("data/train_clean.csv").sample(fraction=0.005, shuffle=True), root="data/common_voice_kpd"),
        batch_size=8,
        shuffle=True,
    )

    test_loader = DataLoader(
        FeaturesDataset(pl.read_csv("data/test_clean.csv").sample(fraction=1.0, shuffle=True), root="data/common_voice_kpd"),
        batch_size=8,
        shuffle=False,
    )

    print("------\n    training\n\n")
    model = LargeLanguageMappingModel()
    model.fit(train_loader, lossFunc="cel", opt="adam", nepochs=1)
    model.save()
