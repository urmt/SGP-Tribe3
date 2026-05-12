import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.fft import fft, ifft

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "."
EPSILON = 1e-12

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
X_train = train_ds.data.float().view(-1, 784).numpy()[:500] / 255.0
y_train = train_ds.targets.numpy()[:500]

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

print("Training model...")
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=128, shuffle=True)

for epoch in range(1, 11):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

print("Extracting activations (2D matrix)...")
model.eval()
with torch.no_grad():
    _ = model(X_t)
    H_base = model.activations[1].numpy()
    
print(f"H_base shape: {H_base.shape}")

def compute_recurrence_2d(H, eps):
    N, D = H.shape
    distances = np.linalg.norm(H[:, np.newaxis, :] - H[np.newaxis, :, :], axis=2)
    R = distances < eps
    return np.mean(R)

def compute_spectral_concentration(H):
    N, D = H.shape
    concentrations = []
    for d in range(D):
        x = H[:, d]
        X = np.abs(fft(x))**2
        X = X[:len(X)//2]
        X = X / (np.sum(X) + EPSILON)
        top_n = max(1, int(len(X) * 0.1))
        top_indices = np.argsort(X)[-top_n:]
        concentrations.append(np.sum(X[top_indices]))
    return np.mean(concentrations)

print("Applying perturbations...")

print("1. Shuffle...")
np.random.seed(42)
H_shuffle = H_base.copy()
np.random.shuffle(H_shuffle)

print("2. Phase randomization...")
H_phase = np.zeros_like(H_base)
for d in range(H_base.shape[1]):
    x = H_base[:, d]
    MAG = np.abs(fft(x))
    PHASE = np.random.uniform(0, 2*np.pi, len(x))
    H_phase[:, d] = np.real(ifft(MAG * np.exp(1j * PHASE)))

print("3. Smoothing...")
window = 3
H_smooth = np.zeros_like(H_base)
for d in range(H_base.shape[1]):
    x = H_base[:, d]
    H_smooth[:, d] = np.convolve(x, np.ones(window)/window, mode='same')

print("4. Noise...")
H_noise = H_base + np.random.randn(*H_base.shape) * 0.1

print("\nComputing epsilon sweep...")

epsilons = np.logspace(-3, 1, 10)

datasets = {
    "base": H_base,
    "shuffle": H_shuffle,
    "phase": H_phase,
    "smooth": H_smooth,
    "noise": H_noise
}

results = {
    "epsilon": epsilons.tolist()
}

print(f"\n{'Perturbation':<12} {'F_total':>10} {'CONC':>10}")
print("-" * 35)

for name, H in datasets.items():
    F_vals = [compute_recurrence_2d(H, eps) for eps in epsilons]
    CONC = compute_spectral_concentration(H)
    results[name] = {
        "F_total": F_vals,
        "CONC": float(CONC)
    }
    print(f"{name:<12} {np.mean(F_vals):>10.4f} {CONC:>10.4f}")

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {OUTPUT_DIR}/results.json")

print("\n=== KEY CHECK ===")
F_base_vals = results["base"]["F_total"]
F_shuffle_vals = results["shuffle"]["F_total"]
F_phase_vals = results["phase"]["F_total"]
F_smooth_vals = results["smooth"]["F_total"]
F_noise_vals = results["noise"]["F_total"]

all_F = [F_base_vals, F_shuffle_vals, F_phase_vals, F_smooth_vals, F_noise_vals]
ranges = [max(v) - min(v) for v in all_F]

if all(r < 0.001 for r in ranges):
    print("ERROR: F not sensitive to perturbations (all ranges < 0.001)")
else:
    print("SUCCESS: F varies across perturbations")
    for name, r in zip(datasets.keys(), ranges):
        print(f"{name}: range = {r:.6f}")