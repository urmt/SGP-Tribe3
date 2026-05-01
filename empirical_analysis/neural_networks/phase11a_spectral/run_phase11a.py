import numpy as np
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.fft import fft
from scipy.stats import pearsonr
from scipy.stats import gmean

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "."
EPSILON = 1e-12

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
X_train = train_ds.data.float().view(-1, 784).numpy()[:2000] / 255.0
y_train = train_ds.targets.numpy()[:2000]

X_t = torch.FloatTensor(X_train)
y_t = torch.LongTensor(y_train)

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

def spectral_metrics(activations):
    act = activations[1].numpy()
    act_flat = act.mean(axis=0)
    
    X = np.abs(fft(act_flat))**2
    X = X[:len(X)//2]
    X = X / (np.sum(X) + EPSILON)
    
    freqs = np.arange(len(X))
    
    H = -np.sum(X * np.log(X + EPSILON))
    
    centroid = np.sum(freqs * X) / (np.sum(X) + EPSILON)
    
    top_n = max(1, int(len(X) * 0.1))
    top_indices = np.argsort(X)[-top_n:]
    concentration = np.sum(X[top_indices])
    
    flatness = gmean(X + EPSILON) / (np.mean(X) + EPSILON)
    
    return {
        "entropy": float(H),
        "centroid": float(centroid),
        "concentration": float(concentration),
        "flatness": float(flatness)
    }

print("Loading Phase 10C results for accuracy reference...")
with open("../phase10a_training_dynamics/results.json") as f:
    phase10c_results = json.load(f)

print("Training single model for spectral analysis...")

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=128, shuffle=True)

model.train()
for epoch in range(1, 11):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    _ = model(X_t)
    specs = spectral_metrics(model.activations)
    preds = model(X_t).argmax(dim=1).numpy()
    accuracy = float((preds == y_train[:len(preds)]).mean())

print(f"Spectral analysis complete: accuracy={accuracy:.3f}")

print("Computing spectral metrics for untrained baseline...")

np.random.seed(99)
torch.manual_seed(99)

model_untrained = MLP()
model_untrained.eval()
with torch.no_grad():
    _ = model_untrained(X_t)
    specs_untrained = spectral_metrics(model_untrained.activations)
    preds_untrained = model_untrained(X_t).argmax(dim=1).numpy()
    accuracy_untrained = float((preds_untrained == y_train[:len(preds_untrained)]).mean())

print(f"Untrained: accuracy={accuracy_untrained:.3f}")

results = {
    "correlations": {
        "entropy_accuracy": None,
        "centroid_accuracy": None,
        "concentration_accuracy": None,
        "flatness_accuracy": None
    },
    "models": [
        {"epoch": 0, "accuracy": accuracy_untrained, **specs_untrained},
        {"epoch": 10, "accuracy": accuracy, **specs}
    ]
}

entropies = [specs_untrained["entropy"], specs["entropy"]]
accs = [accuracy_untrained, accuracy]

r_entropy, _ = pearsonr(entropies, accs)
r_centroid, _ = pearsonr([specs_untrained["centroid"], specs["centroid"]], accs)
r_concentration, _ = pearsonr([specs_untrained["concentration"], specs["concentration"]], accs)
r_flatness, _ = pearsonr([specs_untrained["flatness"], specs["flatness"]], accs)

results["correlations"]["entropy_accuracy"] = r_entropy
results["correlations"]["centroid_accuracy"] = r_centroid
results["correlations"]["concentration_accuracy"] = r_concentration
results["correlations"]["flatness_accuracy"] = r_flatness

results["best_metric"] = max(results["correlations"].keys(), 
                         key=lambda k: abs(results["correlations"][k]) if results["correlations"][k] else 0)

print("\n=== PHASE 11A RESULTS ===")
print(f"Entropy vs Accuracy: r = {r_entropy:.3f}")
print(f"Centroid vs Accuracy: r = {r_centroid:.3f}")
print(f"Concentration vs Accuracy: r = {r_concentration:.3f}")
print(f"Flatness vs Accuracy: r = {r_flatness:.3f}")

print("\n=== COMPARISON TO RECURRENCE ===")
print("Phase 10C: r(F_total, accuracy) = -0.993")
print(f"Best spectral metric: {results['best_metric']} = {results['correlations'][results['best_metric']]:.3f}")

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {OUTPUT_DIR}/results.json")