# plugins/starter_train_plugin.py
from plugin_base import PluginBase
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.datasets import make_classification

class SimpleNet(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_classes=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class StarterTrainPlugin(PluginBase):
    def run(self, params: dict) -> dict:
        epochs = params.get("epochs", 5)
        batch_size = params.get("batch_size", 32)

        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, random_state=42
        )
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = SimpleNet(input_size=X.shape[1], hidden_size=64, num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"[StarterTrain] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model
        output_path = params.get("output", "models/model.pth")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(model.state_dict(), output_path)
        return {"status": "success", "details": f"Trained model saved to {output_path}"}