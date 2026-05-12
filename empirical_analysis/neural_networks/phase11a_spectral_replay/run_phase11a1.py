import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.fft import fft
from scipy.stats import pearsonr

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "."

EPSILON = 1e-12
EPOCHS = 20

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
X_train = train_ds.data.float().view(-1, 784).numpy()[:2000] / 255.0
y_train = train_ds.targets.numpy()[:2000]

X_t = torch.FloatTensor(X_train)
y_t = torch.LongTensor(y_train)
train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=128, shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    
    def forward(self, x):
        self.activations = [x]
        x = torch.relu(self.fc1(x))
        self.activations.append(x)
        x = torch.relu(self.fc2(x))
        self.activations.append(x)
        x = self.fc3(x)
        self.activations.append(x)
        return x

def compute_spectral_metrics(activations):
    results = []
    for act in activations[1:-1]:
        act_np = act.numpy()
        act_flat = act_np.mean(axis=0)
        
        X = np.abs(fft(act_flat))**2
        X = X[:len(X)//2]
        X = X / (np.sum(X) + EPSILON)
        
        freqs = np.arange(len(X))
        
        entropy = -np.sum(X * np.log(X + EPSILON))
        centroid = np.sum(freqs * X) / (np.sum(X) + EPSILON)
        
        top_n = max(1, int(len(X) * 0.1))
        top_indices = np.argsort(X)[-top_n:]
        concentration = np.sum(X[top_indices])
        
        flatness = np.exp(np.mean(np.log(X + EPSILON))) / (np.mean(X) + EPSILON)
        
        results.append({
            "entropy": float(entropy),
            "centroid": float(centroid),
            "concentration": float(concentration),
            "flatness": float(flatness)
        })
    
    avg = {
        "entropy": np.mean([r["entropy"] for r in results]),
        "centroid": np.mean([r["centroid"] for r in results]),
        "concentration": np.mean([r["concentration"] for r in results]),
        "flatness": np.mean([r["flatness"] for r in results])
    }
    return avg

print("Replaying Phase 10C training with spectral capture...")

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epoch_results = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        _ = model(X_t)
        spectral = compute_spectral_metrics(model.activations)
        preds = model(X_t).argmax(dim=1).numpy()
        accuracy = float((preds == y_train[:len(preds)]).mean())
    
    epoch_results.append({
        "epoch": epoch,
        "accuracy": accuracy,
        "entropy": spectral["entropy"],
        "centroid": spectral["centroid"],
        "concentration": spectral["concentration"],
        "flatness": spectral["flatness"]
    })
    
    print(f"Epoch {epoch}: acc={accuracy:.3f}, entropy={spectral['entropy']:.3f}")

print("\nLoading Phase 10C recurrence data...")
# Try multiple possible paths
possible_paths = [
    "../phase10a_training_dynamics/results.json",
    "../phase10c_training_dynamics/results.json",
    "../../phase10a_training_dynamics/results.json"
]
phase10c_data = None
for path in possible_paths:
    try:
        with open(path) as f:
            phase10c_data = json.load(f)
            print(f"Loaded from: {path}")
            break
    except:
        continue

if phase10c_data is None:
    print("ERROR: Could not find Phase 10C results")
    exit(1)

F_total_values = [d["F_total"] for d in phase10c_data]

if len(F_total_values) != EPOCHS:
    print(f"MISMATCH: Phase 10C has {len(F_total_values)} epochs, expected {EPOCHS}")

for i, (epoch_data, F_val) in enumerate(zip(epoch_results, F_total_values)):
    epoch_results[i]["F_total"] = F_val

print("\nComputing correlations...")
accuracies = [d["accuracy"] for d in epoch_results]
entropys = [d["entropy"] for d in epoch_results]
centroids = [d["centroid"] for d in epoch_results]
concentrations = [d["concentration"] for d in epoch_results]
flatness_vals = [d["flatness"] for d in epoch_results]
F_totals = [d["F_total"] for d in epoch_results]

results = {
    "n_epochs": EPOCHS,
    "correlations": {
        "entropy_accuracy": pearsonr(entropys, accuracies)[0],
        "centroid_accuracy": pearsonr(centroids, accuracies)[0],
        "concentration_accuracy": pearsonr(concentrations, accuracies)[0],
        "flatness_accuracy": pearsonr(flatness_vals, accuracies)[0],
        "F_total_accuracy": pearsonr(F_totals, accuracies)[0]
    },
    "epoch_data": epoch_results
}

print("\n=== PHASE 11A.1 RESULTS ===")
for k, v in results["correlations"].items():
    print(f"{k}: r = {v:.3f}")

best_metric = max(results["correlations"].keys(), key=lambda k: abs(results["correlations"][k]))
print(f"\nBest predictor: {best_metric}")

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {OUTPUT_DIR}/results.json")