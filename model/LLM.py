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

class LargeLanguageMappingModel(nn.Module):
    def __init__(self, conv_layers=CONV_LAYERS, input_dim=(24,149,1024)):
        super(LargeLanguageMappingModel, self).__init__()

        self.model_name = "LLM" 

        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 4)),  # -> (32, 12, 74, 256)

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 4)),  # -> (64, 6, 37, 64)

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),  # -> (128, 3, 18, 32)

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # -> (256, 1, 1, 1)
        )
        self.fc = nn.Linear(256, 45)

    def forward(self, x):
        x = x.unsqueeze(1)  # Ajouter une dimension de canal
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def fit(self, dataLoader:DataLoader, lossFunc:str="mseloss",
            opt:str ="adam", nepochs:int=20):

        crit_methods={
            "mseloss":nn.MSELoss,
            "l1loss":nn.L1Loss,
            "cel":nn.CrossEntropyLoss,
            "bcel":nn.BCELoss,
            "smoothl1loss":nn.SmoothL1Loss
        }
        if lossFunc not in crit_methods:
            lossFunc = "mseloss"
        criterion = crit_methods[lossFunc]()

        optim_methods = {
            "adam":optim.Adam,
            "sgd":optim.SGD
        }
        if opt not in optim_methods:
            opt = "adam"
        optimizer = optim_methods[opt](self.parameters(), lr=1e-3)
        
        progress =tqdm(range(nepochs*len(dataLoader)))
        result=[]
        for epoch in range(nepochs):
            self.train()
            total_loss = 0

            for batch, labels in dataLoader:

                outputs = self(batch)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress.update()
                progress.refresh()

            tqdm.write(f"Epoch [{epoch+1}/{nepochs}], Loss: {total_loss/len(dataLoader):.6f}")
            result.append([f"Epoch [{epoch+1}/{nepochs}]", total_loss/len(dataLoader)])
        progress.close()
        return result

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
