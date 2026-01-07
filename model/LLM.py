import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import torch
import random
import string
from datetime import datetime

CONV_LAYERS = [
    [1, 4, 3, 1, 1],
    [4, 8, 3, 1, 1],
    [8, 16, 3, 1, 1]
]
N_CLASSES = 45

class LargeLanguageMappingModel(nn.Module):
    def __init__(self, conv_layers=CONV_LAYERS, input_dim=(24,149,1024)):
        super(LargeLanguageMappingModel, self).__init__()

        self.model_name = "LLM"

        self.attn = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, N_CLASSES)
        )  

    def forward(self, x):
        """
        x: (B, 24, 149, 1024)
        """
        x = x[:, 12]
        w = self.attn(x)        
        w = torch.softmax(w, dim=1)  
        x = (x*w).sum(dim=1)

        return self.classifier(x)

    
    def fit(self, dataLoader:DataLoader, lossFunc:str="cel",
        opt:str ="adam", nepochs:int=20,
        device: torch.device | None = None,
        amp: bool = True,
        val_loader: DataLoader | None = None,
        lr: float = 1e-3,
        grad_accum_steps: int = 1,
        overfit_one_batch: bool = False):

        # Device handling
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.to(device)
        except Exception:
            pass

        crit_methods = {
            "mseloss": nn.MSELoss,
            "l1loss": nn.L1Loss,
            "cel": nn.CrossEntropyLoss,
            "bcel": nn.BCELoss,
            "smoothl1loss": nn.SmoothL1Loss,
        }
        if lossFunc not in crit_methods:
            lossFunc = "mseloss"
        criterion = crit_methods[lossFunc]()

        optim_methods = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
        }
        if opt not in optim_methods:
            opt = "adam"
        optimizer = optim_methods[opt](self.parameters(), lr=lr)

        use_amp = amp and (device.type == "cuda") and hasattr(torch.cuda, "amp")
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # ==========================
        # 🔥 OVERFIT SUR 1 BATCH
        # ==========================
        if overfit_one_batch:
            self.train()
            batch = next(iter(dataLoader))

            if isinstance(batch, (list, tuple)):
                inputs, labels = batch
            else:
                inputs, labels = batch["inputs"], batch["labels"]

            inputs = inputs.to(device)
            labels = labels.to(device)

            if isinstance(criterion, nn.CrossEntropyLoss):
                labels = labels.long()

            print("🔥 Overfitting on one batch...")

            for i in range(300):
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self(inputs.float())
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                if i % 20 == 0:
                    preds = outputs.argmax(dim=1)
                    acc = (preds == labels).float().mean().item()
                    print(f"[{i:03d}] loss={loss.item():.4f} acc={acc:.4f}")

            print("✅ Overfit test finished")
            return


        history = []
        for epoch in range(nepochs):
            self.train()
            epoch_loss = 0.0
            batches = 0

            pbar = tqdm(dataLoader, desc=f"Epoch {epoch+1}/{nepochs}", leave=False)
            for batch in pbar:
                # Support dataloaders that return either (inputs, labels) or a dict
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                elif isinstance(batch, dict) and "inputs" in batch and "labels" in batch:
                    inputs, labels = batch["inputs"], batch["labels"]
                else:
                    raise ValueError("DataLoader must return (inputs, labels) tuples or dict with 'inputs' and 'labels'.")

                inputs = inputs.to(device, non_blocking=True) if isinstance(inputs, torch.Tensor) else inputs
                labels = labels.to(device, non_blocking=True) if isinstance(labels, torch.Tensor) else labels

                # Ensure input dtype matches model parameter dtype to avoid type mismatch
                try:
                    param_dtype = next(self.parameters()).dtype
                    if isinstance(inputs, torch.Tensor) and inputs.dtype != param_dtype:
                        inputs = inputs.to(dtype=param_dtype)
                except StopIteration:
                    # model has no parameters (unlikely); skip
                    pass

                # For CrossEntropyLoss ensure labels are long
                if isinstance(criterion, nn.CrossEntropyLoss):
                    labels = labels.long()

                # Gradient accumulation support: divide loss by grad_accum_steps
                with torch.set_grad_enabled(True):
                    try:
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self(inputs)
                                loss = criterion(outputs, labels)
                            loss = loss / grad_accum_steps
                            scaler.scale(loss).backward()
                            if (batches + 1) % grad_accum_steps == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                        else:
                            outputs = self(inputs)
                            loss = criterion(outputs, labels)
                            loss = loss / grad_accum_steps
                            loss.backward()
                            if (batches + 1) % grad_accum_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            raise RuntimeError(f"CUDA OOM during training step. Consider reducing batch_size, using smaller dtype or increasing grad_accum_steps. Original error: {e}") from e
                        else:
                            raise

                # batch_loss: scale back to per-batch value for logging
                try:
                    batch_loss = (loss * grad_accum_steps).item()
                except Exception:
                    batch_loss = float(loss)
                epoch_loss += batch_loss
                batches += 1
                pbar.set_postfix({'batch_loss': f"{batch_loss:.6f}"})

            avg_train_loss = epoch_loss / max(1, batches)
            tqdm.write(f"Epoch {epoch+1}/{nepochs} - avg_train_loss: {avg_train_loss:.6f}")

            # # Optional validation
            # avg_val_loss = None
            # if val_loader is not None:
            #     self.eval()
            #     val_loss = 0.0
            #     val_batches = 0
            #     with torch.no_grad():
            #         for vbatch in val_loader:
            #             if isinstance(vbatch, (list, tuple)) and len(vbatch) >= 2:
            #                 vinputs, vlabels = vbatch[0], vbatch[1]
            #             elif isinstance(vbatch, dict) and "inputs" in vbatch and "labels" in vbatch:
            #                 vinputs, vlabels = vbatch["inputs"], vbatch["labels"]
            #             else:
            #                 raise ValueError("Validation DataLoader must return (inputs, labels) tuples or dict with 'inputs' and 'labels'.")

            #             vinputs = vinputs.to(device, non_blocking=True) if isinstance(vinputs, torch.Tensor) else vinputs
            #             vlabels = vlabels.to(device, non_blocking=True) if isinstance(vlabels, torch.Tensor) else vlabels
            #             if isinstance(criterion, nn.CrossEntropyLoss):
            #                 vlabels = vlabels.long()

            #             if use_amp:
            #                 with torch.cuda.amp.autocast():
            #                     voutputs = self(vinputs)
            #                     vloss = criterion(voutputs, vlabels)
            #             else:
            #                 voutputs = self(vinputs)
            #                 vloss = criterion(voutputs, vlabels)

            #             val_loss += vloss.item()
            #             val_batches += 1

            #     avg_val_loss = val_loss / max(1, val_batches)

            # print(f"Epoch {epoch+1}/{nepochs} - train_loss: {avg_train_loss:.6f}" + (f", val_loss: {avg_val_loss:.6f}" if avg_val_loss is not None else ""))
            # history.append((epoch + 1, avg_train_loss, avg_val_loss))

            # Clear cache to reduce fragmentation between epochs
            if device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return avg_train_loss

    def save(self, folder_name="trained"):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_dir, folder_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{self.model_name}_{timestamp}_{rand_suffix}.pt")
            
            torch.save(self.state_dict(), path)
            print(f"Model saved at: {path}")
            return path

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from: {path}")
        return self