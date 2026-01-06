if __name__ == "__main__":
    print("------\n    import lib\n------\n")
    import polars as pl
    import torch
    from torch.utils.data import DataLoader
    from LLM import LargeLanguageMappingModel
    from Dataset import PrecomputedFeaturesDataset
    import os

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Quick runtime check: try a tiny op on CUDA to ensure kernels are compatible with your GPU
    # If this fails (e.g. 'no kernel image is available'), fall back to CPU automatically.
    if device.type == "cuda":
        try:
            # perform a very small op to trigger any CUDA kernel loading
            _ = torch.zeros(1, device=device) + 1
        except Exception as e:
            print("CUDA runtime test failed, falling back to CPU. Reason:", repr(e))
            device = torch.device("cpu")

    # Performance flags for GPU
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("------\n    create dataloader\n------\n")
    # Increase num_workers to overlap IO and CPU work; tune this number on your machine
    num_workers = min(8, (os.cpu_count() or 4) // 2)

    train_loader = DataLoader(
        PrecomputedFeaturesDataset("data/precomputed_features/train", fraction=0.8, shuffle=True, seed=42, dtype=torch.float16),
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Use smaller batch and float32 features for evaluation to avoid OOM during inference
    # test_loader = DataLoader(
    #     PrecomputedFeaturesDataset("data/precomputed_features/test", fraction=0.25, dtype=torch.float32),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=(device.type == "cuda"),
    #     persistent_workers=(num_workers > 0),
    # )

    model = LargeLanguageMappingModel()
    try:
        model.to(device)
    except Exception:
        # If model doesn't expose .to(), the fit function should handle device placement.
        pass

    # Prefer larger batch sizes when using precomputed features and GPU memory allows it.
    # Try gradient accumulation to reduce memory pressure (e.g., accumulate 4 steps)
    print("------\n    training\n------\n")
    model.fit(train_loader, lossFunc="cel", opt="adam", nepochs=5, device=device, grad_accum_steps=2, overfit_one_batch=False)
    model.save()

    # print("------\n    testing\n------\n")
    # model.load("model/trained/LLM_20260106_223450_nro7.pt")
    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs = inputs.to(device, non_blocking=True)
    #         labels = labels.to(device, non_blocking=True)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")
