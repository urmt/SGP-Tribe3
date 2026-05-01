import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "empirical_analysis/neural_networks/phase10e_representation"

EPS_STEPS = 10
L_MIN = 2
LONG_THRESHOLD = 5
EPSILON = 1e-8

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

def compute_det(x, eps):
    n = len(x)
    R = np.array([[1 if abs(x[i]-x[j]) < eps else 0 for j in range(n)] for i in range(n)], dtype=np.uint8)
    total = 0
    short_lines = 0
    long_lines = 0
    for k in range(-n+1, n):
        diag = np.diag(R, k=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= L_MIN:
                    total += length
                    if length < LONG_THRESHOLD:
                        short_lines += length
                    else:
                        long_lines += length
                length = 0
        if length >= L_MIN:
            total += length
            if length < LONG_THRESHOLD:
                short_lines += length
            else:
                long_lines += length
    
    denom = np.sum(R) + 1e-12
    return total / denom, short_lines / denom, long_lines / denom

def compute_metrics(activations):
    act = activations[1].numpy()
    act_flat = act.mean(axis=0)
    
    eps_vals = np.linspace(0.1, 1.0, EPS_STEPS)
    dets = []
    dets_short = []
    dets_long = []
    
    for eps in eps_vals:
        F_tot, F_s, F_l = compute_det(act_flat, eps)
        dets.append(F_tot)
        dets_short.append(F_s)
        dets_long.append(F_l)
    
    dets = np.array(dets)
    dets_short = np.array(dets_short)
    dets_long = np.array(dets_long)
    
    F_total = np.trapezoid(dets, eps_vals)
    F_short = np.trapezoid(dets_short, eps_vals)
    F_long = np.trapezoid(dets_long, eps_vals)
    
    return F_total, F_short, F_long

print("Training model...")
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
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

model.eval()
with torch.no_grad():
    _ = model(X_t)
    base_activations = [a.clone() for a in model.activations]
    F_total, F_short, F_long = compute_metrics(model.activations)
    preds = model(X_t).argmax(dim=1).numpy()
    base_acc = (preds == y_train[:len(preds)]).mean()

print(f"Base model: F_long={F_long:.4f}, accuracy={base_acc:.3f}")

def apply_perturbation(name, transform_fn, act_layer):
    model.eval()
    with torch.no_grad():
        perturbed = transform_fn(act_layer.clone())
        model.activations[1] = perturbed
        F_total, F_short, F_long = compute_metrics(model.activations)
        preds = model(X_t).argmax(dim=1).numpy()
        acc = (preds == y_train[:len(preds)]).mean()
    
    print(f"{name}: F_long={F_long:.4f}, accuracy={acc:.3f}")
    return {"F_long": float(F_long), "accuracy": float(acc)}

def gaussian_noise(x, sigma=0.1):
    noise = torch.randn_like(x) * sigma
    return x + noise

def smoothing(x, window=3):
    x_np = x.numpy()
    result = np.copy(x_np)
    for i in range(x_np.shape[0]):
        for j in range(x_np.shape[1]):
            idx_low = max(0, j - window//2)
            idx_high = min(x_np.shape[1], j + window//2 + 1)
            result[i, j] = np.mean(x_np[i, idx_low:idx_high])
    return torch.from_numpy(result)

def sparsify(x, percentile=50):
    threshold = np.percentile(x.abs().numpy(), percentile)
    return torch.where(x.abs() > threshold, x, torch.zeros_like(x))

results = {}

act_layer = base_activations[1]

results["noise"] = apply_perturbation("Gaussian noise (sigma=0.1)", lambda x: gaussian_noise(x, 0.1), act_layer)
results["smooth"] = apply_perturbation("Smoothing (window=3)", lambda x: smoothing(x, 3), act_layer)
results["sparse"] = apply_perturbation("Sparsify (percentile=50)", lambda x: sparsify(x, 50), act_layer)

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {OUTPUT_DIR}/results.json")

key_measure = []
for name, res in results.items():
    print(f"\n{name}: F_long={res['F_long']:.4f}, accuracy={res['accuracy']:.3f}")

all_flong = [results[k]["F_long"] for k in results]
all_acc = [results[k]["accuracy"] for k in results]

if all(abs(f - all_flong[0]) > 0.01 for f in all_flong):
    print("\nSTRUCTURE CHANGES with perturbation")
    
    if all(abs(a - all_acc[0]) > 0.01 for a in all_acc):
        print("STRUCTURE IS CAUSALLY RELEVANT")
    else:
        print("STRUCTURE CHANGES but accuracy STABLE - epiphenomenal")
else:
    print("\nSTRUCTURE IS EPIPHENOMENAL")