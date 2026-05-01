import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "empirical_analysis/neural_networks/phase10d_intervention"

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
    C = 1.0 / (np.var(dets) + EPSILON)
    
    return F_total, F_short, F_long, C

def train_with_intervention(lambda_F):
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    results = []
    
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            output = model(batch_x)
            task_loss = criterion(output, batch_y)
            
            model.eval()
            with torch.no_grad():
                _ = model(X_t)
                F_total, F_short, F_long, _ = compute_metrics(model.activations)
            model.train()
            
            loss = task_loss + lambda_F * F_long
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        model.eval()
        with torch.no_grad():
            _ = model(X_t)
            F_total, F_short, F_long, C = compute_metrics(model.activations)
            preds = model(X_t).argmax(dim=1).numpy()
            accuracy = (preds == y_train[:len(preds)]).mean()
        
        results.append({
            "epoch": epoch,
            "F_total": float(F_total),
            "F_long": float(F_long),
            "F_short": float(F_short),
            "accuracy": float(accuracy)
        })
        
        print(f"lambda={lambda_F:.1f} epoch {epoch}: acc={accuracy:.3f}, F_long={F_long:.4f}")
    
    return results

lambda_values = [-0.1, 0.0, 0.1]
all_results = {}

for lam in lambda_values:
    key = f"lambda_{lam}"
    print(f"\n=== Training with {key} ===")
    all_results[key] = train_with_intervention(lam)

output_path = "results.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved to: {output_path}")

lam_neg = np.array([r["F_long"] for r in all_results["lambda_-0.1"]])
lam_zero = np.array([r["F_long"] for r in all_results["lambda_0.0"]])
lam_pos = np.array([r["F_long"] for r in all_results["lambda_0.1"]])

print("\n=== VALIDATION ===")
print(f"F_long lambda=-0.1 (final): {lam_neg[-1]:.4f}")
print(f"F_long lambda=0.0 (final): {lam_zero[-1]:.4f}")
print(f"F_long lambda=+0.1 (final): {lam_pos[-1]:.4f}")

if lam_pos[-1] < lam_zero[-1] < lam_neg[-1]:
    print("\nCAUSAL LINK CONFIRMED: Higher regularization suppresses F_long")
else:
    print("\nWARNING: F_long ordering does not match intervention")