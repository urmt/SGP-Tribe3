#!/usr/bin/env python3
"""
SFH-SGP_RECURRENCE_MECHANISM_TEST_01
Test whether learning detectability (ΔΦ) is explained by
increase in recurrence structure induced by operator O.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

epsilon = 1e-2
l_min = 2
eps_list = np.logspace(-3, -1, 8)

def recurrence_matrix(x, epsilon):
    x = x.reshape(-1, 1)
    D = pairwise_distances(x)
    R = (D < epsilon).astype(int)
    np.fill_diagonal(R, 0)
    return R

def compute_det(R, l_min=2):
    N = R.shape[0]
    diag_counts = 0
    total_rec = np.sum(R)
    for k in range(-N+1, N):
        diag = np.diagonal(R, offset=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= l_min:
                    diag_counts += length
                length = 0
        if length >= l_min:
            diag_counts += length
    if total_rec == 0:
        return 0.0
    return diag_counts / total_rec

def recurrence_rate(R):
    N = R.shape[0]
    return np.sum(R) / (N * N - N) if N > 1 else 0

def compute_alpha(x, eps_list):
    R_vals = []
    for eps in eps_list:
        R = recurrence_matrix(x, eps)
        R_vals.append(recurrence_rate(R))
    log_eps = np.log(eps_list)
    log_R = np.log(np.array(R_vals) + 1e-10)
    valid = np.isfinite(log_R)
    if valid.sum() > 2:
        alpha = np.polyfit(log_eps[valid], log_R[valid], 1)[0]
    else:
        alpha = np.nan
    return alpha

def apply_operator(x, op_name, k=1.0):
    if op_name == 'identity':
        return x
    elif op_name == 'tanh':
        return np.tanh(k * x)
    elif op_name == 'clipped':
        return np.clip(x, -k, k)
    elif op_name == 'sin':
        return np.sin(k * x)
    elif op_name == 'power':
        p = k if k != 1 else 2
        return np.sign(x) * np.abs(x) ** p
    return x

def compute_phi_full(x, op_name, k=1.0):
    x_op = apply_operator(x, op_name, k)
    if len(x_op) < 100:
        return np.nan, np.nan, np.nan
    
    alpha = compute_alpha(x_op, eps_list)
    R = recurrence_matrix(x_op, epsilon)
    det = compute_det(R, l_min)
    rr = recurrence_rate(R)
    
    return alpha, det, rr

print("Loading MNIST...")
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

X_train = train_ds.data.float().view(-1, 784).numpy()[:6000] / 255.0
y_train = train_ds.targets.numpy()[:6000]
X_test = test_ds.data.float().view(-1, 784).numpy()[:2000] / 255.0
y_test = test_ds.targets.numpy()[:2000]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        self.activations = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
            self.activations.append(x)
        return x

def train_to_epoch(model, X_train, y_train, target_epoch):
    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(target_epoch):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

model_early = MLP([784, 64, 32, 10])
model_early = train_to_epoch(model_early, X_train, y_train, 1)

model_final = MLP([784, 64, 32, 10])
model_final = train_to_epoch(model_final, X_train, y_train, 30)

X_v = torch.FloatTensor(X_test)
y_v = torch.LongTensor(y_test)

model_early.eval()
model_final.eval()

with torch.no_grad():
    preds_early = model_early(X_v).argmax(dim=1).numpy()
    acc_early = (preds_early == y_v.numpy()).mean()
    
    preds_final = model_final(X_v).argmax(dim=1).numpy()
    acc_final = (preds_final == y_v.numpy()).mean()

print(f"Early epoch accuracy: {acc_early:.3f}")
print(f"Final epoch accuracy: {acc_final:.3f}")

def get_layer_activations(model, X):
    model.eval()
    with torch.no_grad():
        _ = model(X)
    return [act.numpy() for act in model.activations]

act_early = get_layer_activations(model_early, X_v)
act_final = get_layer_activations(model_final, X_v)

operators_config = [
    ('identity', 1.0),
    ('tanh', 0.5),
    ('tanh', 1.0),
    ('tanh', 2.0),
    ('tanh', 5.0),
    ('clipped', 1.0),
    ('clipped', 2.0),
    ('clipped', 3.0),
    ('sin', 0.5),
    ('sin', 1.0),
    ('sin', 2.0),
    ('power', 0.5),
    ('power', 2.0),
]

results = []
layer_names = ['input', 'hidden_0', 'hidden_1', 'output']

print("\n" + "="*60)
print("RECURRENCE MECHANISM TEST")
print("="*60)

for op_name, k in operators_config:
    for layer_idx in [1, 2, 3]:
        act_e = act_early[layer_idx]
        act_f = act_final[layer_idx]
        
        if act_e.ndim > 1:
            act_e_flat = act_e.mean(axis=1)
            act_f_flat = act_f.mean(axis=1)
        else:
            act_e_flat = act_e
            act_f_flat = act_f
        
        alpha_e, det_e, rr_e = compute_phi_full(act_e_flat, op_name, k)
        alpha_f, det_f, rr_f = compute_phi_full(act_f_flat, op_name, k)
        
        delta_alpha = alpha_f - alpha_e
        delta_det = det_f - det_e
        delta_rr = rr_f - rr_e
        delta_phi = np.sqrt(delta_alpha**2 + delta_det**2)
        
        results.append({
            'operator': op_name,
            'k': k,
            'layer': layer_names[layer_idx],
            'alpha_early': alpha_e,
            'alpha_final': alpha_f,
            'det_early': det_e,
            'det_final': det_f,
            'rr_early': rr_e,
            'rr_final': rr_f,
            'delta_alpha': delta_alpha,
            'delta_det': delta_det,
            'delta_rr': delta_rr,
            'delta_phi': delta_phi
        })

df = pd.DataFrame(results)
df.to_csv('recurrence_mechanism_test.csv', index=False)

valid_df = df.dropna()
if len(valid_df) > 3:
    print("\n--- Correlations ---")
    corr_phi_rr = np.corrcoef(valid_df['delta_phi'], valid_df['delta_rr'])[0,1]
    corr_phi_det = np.corrcoef(valid_df['delta_phi'], valid_df['delta_det'])[0,1]
    corr_rr_det = np.corrcoef(valid_df['delta_rr'], valid_df['delta_det'])[0,1]
    
    print(f"ΔΦ vs ΔR: {corr_phi_rr:.3f}")
    print(f"ΔΦ vs ΔDET: {corr_phi_det:.3f}")
    print(f"ΔR vs ΔDET: {corr_rr_det:.3f}")

print("\n--- Results table ---")
print(df[['operator', 'k', 'layer', 'delta_phi', 'delta_rr', 'delta_det']].to_string(index=False))

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

print("\nQUESTION 1: Does ΔΦ correlate with ΔR?")
if len(valid_df) > 3:
    if abs(corr_phi_rr) > 0.5:
        print(f"YES - Strong correlation ({corr_phi_rr:.3f})")
    elif abs(corr_phi_rr) > 0.2:
        print(f"MODERATE correlation ({corr_phi_rr:.3f})")
    else:
        print(f"WEAK correlation ({corr_phi_rr:.3f})")

print("\nQUESTION 2: Does ΔΦ correlate with ΔDET?")
if len(valid_df) > 3:
    if abs(corr_phi_det) > 0.5:
        print(f"YES - Strong correlation ({corr_phi_det:.3f})")
    elif abs(corr_phi_det) > 0.2:
        print(f"MODERATE correlation ({corr_phi_det:.3f})")
    else:
        print(f"WEAK correlation ({corr_phi_det:.3f})")

print("\nQUESTION 3: Which explains ΔΦ better?")
if len(valid_df) > 3:
    if abs(corr_phi_rr) > abs(corr_phi_det):
        print("ΔR (recurrence rate) is a better predictor of ΔΦ")
    else:
        print("ΔDET (determinism) is a better predictor of ΔΦ")

print("\n" + "="*60)
print("MECHANISM VERDICT")
print("="*60)

if len(valid_df) > 3:
    max_corr = max(abs(corr_phi_rr), abs(corr_phi_det))
    if max_corr > 0.5:
        verdict = "MECHANISM CONFIRMED"
        explanation = f"ΔΦ strongly correlates with recurrence metrics (r={max_corr:.3f})"
    elif max_corr > 0.2:
        verdict = "MECHANISM PARTIALLY SUPPORTED"
        explanation = f"ΔΦ moderately correlates with recurrence (r={max_corr:.3f})"
    else:
        verdict = "MECHANISM NOT SUPPORTED"
        explanation = "No significant correlation between ΔΦ and recurrence"
    
    print(f"\nVerdict: {verdict}")
    print(f"Explanation: {explanation}")

print("\nResults saved to: recurrence_mechanism_test.csv")