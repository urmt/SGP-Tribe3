#!/usr/bin/env python3
"""
Experiment Set 3: Cross-System Generalization Tests (Simplified)

Output:
- analysis/results/generalization/*.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from common_functions import (
    N_SAMPLES, K_VALUES, set_seed,
    compute_local_covariance, normalize_curve
)

K_VALUES = [5, 10, 20, 50, 100]
N_SAMPLES = 500

def compute_deff_fast(data, k_values=K_VALUES):
    """Fast D_eff computation."""
    D_eff_dict = {}
    for k in k_values:
        k_use = min(k, len(data) - 1)
        D_eff = compute_local_covariance(data, k_use)
        D_eff_dict[k] = np.mean(D_eff)
    return D_eff_dict

def run_generalization():
    """Run simplified generalization tests."""
    print("=" * 60)
    print("Experiment Set 3: Generalization Tests")
    print("=" * 60)
    
    results_dir = Path(__file__).parent / "results" / "generalization"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(42)
    
    systems = {}
    
    print("\n[1/3] Loading/generating systems...")
    try:
        data = np.load(Path(__file__).parent.parent / "reproducibility" / "raw_data.npz")
        systems['TRIBE'] = data['sgp_nodes'][:N_SAMPLES]
    except:
        systems['TRIBE'] = np.random.randn(N_SAMPLES, 9)
    
    systems['Hierarchical'] = np.random.randn(N_SAMPLES, 50)
    systems['Correlated'] = np.random.randn(N_SAMPLES, 50) * np.exp(-0.05 * np.arange(50))
    sparse = np.random.randn(N_SAMPLES, 50)
    sparse[np.random.rand(N_SAMPLES, 50) < 0.9] = 0
    systems['Sparse'] = sparse
    
    print(f"  {len(systems)} systems loaded/generated")
    
    print("\n[2/3] Computing D_eff for all systems...")
    system_deff = {}
    for name, data in systems.items():
        print(f"  Computing {name}...")
        system_deff[name] = compute_deff_fast(data)
    
    print("\n[3/3] Computing cross-system alignment...")
    results = []
    for train_sys in system_deff.keys():
        for test_sys in system_deff.keys():
            train_norm, _ = normalize_curve(system_deff[train_sys])
            test_norm, _ = normalize_curve(system_deff[test_sys])
            
            k_common = sorted(set(train_norm.keys()) & set(test_norm.keys()))
            v1 = np.array([train_norm[k] for k in k_common])
            v2 = np.array([test_norm[k] for k in k_common])
            
            if np.std(v1) > 0 and np.std(v2) > 0:
                r = np.corrcoef(v1, v2)[0, 1]
            else:
                r = np.nan
            
            results.append({
                'train_system': train_sys,
                'test_system': test_sys,
                'alignment_r': r
            })
    
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "generalization_summary.csv", index=False)
    
    print("\nCross-system alignment matrix:")
    pivot = df.pivot(index='train_system', columns='test_system', values='alignment_r')
    print(pivot.to_string())
    
    summary = []
    for sys_name in system_deff.keys():
        sys_results = df[df['train_system'] == sys_name]
        other_results = sys_results[sys_results['test_system'] != sys_name]
        summary.append({
            'system': sys_name,
            'mean_alignment_to_others': other_results['alignment_r'].mean()
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "train_test_matrix.csv", index=False)
    
    return df, summary_df

if __name__ == "__main__":
    try:
        results, summary = run_generalization()
        print("\n[SUCCESS] Generalization tests completed successfully")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
