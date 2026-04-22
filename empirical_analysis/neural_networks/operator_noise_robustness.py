#!/usr/bin/env python3
"""
SFH-SGP_OPERATOR_NOISE_ROBUSTNESS_01
Test Φ layer separability under operator noise
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

np.random.seed(42)

E = 1e-2

def get_rm(x):
    x = np.asarray(x).flatten()
    D = pairwise_distances(x.reshape(-1, 1))
    R = (D < E).astype(int)
    np.fill_diagonal(R, 0)
    return R

def get_det(R):
    N = R.shape[0]
    dc = 0
    for k in range(-N+1, N):
        d = np.diagonal(R, k)
        l = 0
        for v in d:
            if v == 1: l += 1
            elif l >= 2: dc += l; l = 0
        if l >= 2: dc += l
    return dc / np.sum(R) if np.sum(R) > 0 else 0

def get_rr(R):
    return np.sum(R) / (R.shape[0]**2)

def get_alpha(x):
    es = np.logspace(-3, -1, 6)
    rrs = [get_rr(get_rm(e)) for e in es]
    v = np.isfinite(np.log(rrs))
    return np.polyfit(np.log(es)[v], np.log(rrs)[v], 1)[0] if v.sum() > 2 else np.nan

def apply_op(x, op_name):
    x = np.asarray(x).flatten()
    if op_name == 'identity':
        return x
    elif op_name == 'tanh':
        return np.tanh(x)
    return x

print("Loading saved training trajectory...")
df = pd.read_csv('training_phi_trajectory.csv')
final = df[df['epoch'] == 50]

print("\n" + "="*60)
print("OPERATOR NOISE ROBUSTNESS TEST")
print("="*60)

sigmas = [0.0, 0.001, 0.005, 0.01]
layers = ['input', 'hidden_0', 'hidden_1', 'output']
operators = ['identity', 'tanh']

results = []

for sigma in sigmas:
    print(f"\n--- σ = {sigma} ---")
    
    for op in operators:
        layer_vals = []
        
        for layer in layers:
            layer_data = final[(final['operator'] == op) & (final['layer'] == layer)]
            if len(layer_data) == 0:
                continue
            
            alpha_base = layer_data['alpha'].values[0]
            det_base = layer_data['det'].values[0]
            
            if sigma > 0:
                np.random.seed(42)
                alpha_noisy = alpha_base + np.random.randn() * sigma
                det_noisy = np.clip(det_base + np.random.randn() * sigma, 0, 1)
            else:
                alpha_noisy = alpha_base
                det_noisy = det_base
            
            phi = np.sqrt(alpha_noisy**2 + det_noisy**2)
            layer_vals.append(phi)
            
            if sigma == 0:
                print(f"  {op:10} {layer:10}: α={alpha_noisy:.3f}, DET={det_noisy:.3f}, Φ={phi:.3f}")
        
        if len(layer_vals) > 1:
            layer_spread = np.std(layer_vals)
            results.append({
                'sigma': sigma,
                'operator': op,
                'layer_spread': layer_spread
            })
            print(f"  {op:10} layer spread: {layer_spread:.4f}")

results_df = pd.DataFrame(results)

print("\n" + "="*60)
print("ROBUSTNESS ANALYSIS")
print("="*60)

print(f"\n{'σ':<8} {'Identity':<12} {'Tanh':<12} {'Ratio (T/I)':<12}")
print("-" * 44)

for sigma in sigmas:
    id_val = results_df[(results_df['sigma'] == sigma) & (results_df['operator'] == 'identity')]['layer_spread'].values
    tan_val = results_df[(results_df['sigma'] == sigma) & (results_df['operator'] == 'tanh')]['layer_spread'].values
    
    if len(id_val) > 0 and len(tan_val) > 0:
        ratio = tan_val[0] / id_val[0] if id_val[0] > 0 else 0
        print(f"{sigma:<8} {id_val[0]:<12.4f} {tan_val[0]:<12.4f} {ratio:<12.2f}")

print("\n" + "="*60)
print("VERDICT")
print("="*60)

base_ratio = results_df[results_df['sigma'] == 0.0]
if len(base_ratio) > 0:
    id_base = base_ratio[base_ratio['operator'] == 'identity']['layer_spread'].values
    tan_base = base_ratio[base_ratio['operator'] == 'tanh']['layer_spread'].values
    
    if len(id_base) > 0 and len(tan_base) > 0:
        base_ratio_val = tan_base[0] / id_base[0]
        
        noisy_ratios = []
        for sigma in [0.001, 0.005, 0.01]:
            nr = results_df[(results_df['sigma'] == sigma)]
            if len(nr) > 0:
                id_n = nr[nr['operator'] == 'identity']['layer_spread'].values
                tan_n = nr[nr['operator'] == 'tanh']['layer_spread'].values
                if len(id_n) > 0 and len(tan_n) > 0 and id_n[0] > 0:
                    noisy_ratios.append(tan_n[0] / id_n[0])
        
        if noisy_ratios:
            min_ratio = min(noisy_ratios)
            if min_ratio > 1.5:
                print("\n✓ ROBUST: Layer separation persists under noise")
                print(f"  Base ratio: {base_ratio_val:.2f}x, Min noisy ratio: {min_ratio:.2f}x")
            elif min_ratio > 1.0:
                print("\n~ MODERATE: Some separation persists with noise")
                print(f"  Base ratio: {base_ratio_val:.2f}x, Min noisy ratio: {min_ratio:.2f}x")
            else:
                print("\n✗ FRAGILE: Effect collapses under noise")
                print(f"  Base ratio: {base_ratio_val:.2f}x, Min noisy ratio: {min_ratio:.2f}x")

results_df.to_csv('operator_noise_robustness.csv', index=False)
print("\nResults saved to operator_noise_robustness.csv")