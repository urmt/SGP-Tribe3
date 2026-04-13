#!/usr/bin/env python3
"""
Experiment Set 4: Minimal Condition Analysis (Simplified)

Output:
- analysis/results/minimal_conditions/*.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from common_functions import set_seed, compute_local_covariance

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

def check_growth_saturation(D_eff_dict):
    """Check for growth-saturation pattern."""
    k_vals = sorted(D_eff_dict.keys())
    D_vals = [D_eff_dict[k] for k in k_vals]
    
    early_idx = [i for i, k in enumerate(k_vals) if k <= 20]
    late_idx = [i for i, k in enumerate(k_vals) if k >= 50]
    
    if len(early_idx) < 1 or len(late_idx) < 1:
        return False, {}
    
    D_early = np.mean([D_vals[i] for i in early_idx])
    D_late = np.mean([D_vals[i] for i in late_idx])
    
    growth = D_late > D_early * 1.1
    
    return growth, {'D_early': D_early, 'D_late': D_late}

def run_minimal_conditions():
    """Run minimal condition analysis."""
    print("=" * 60)
    print("Experiment Set 4: Minimal Conditions")
    print("=" * 60)
    
    results_dir = Path(__file__).parent / "results" / "minimal_conditions"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(42)
    
    conditions = [
        ('Isotropic Gaussian (d=9)', lambda: np.random.randn(N_SAMPLES, 9)),
        ('Isotropic Gaussian (d=50)', lambda: np.random.randn(N_SAMPLES, 50)),
        ('Correlated (decay=0.02)', lambda: np.random.randn(N_SAMPLES, 50) * np.exp(-0.02 * np.arange(50))),
        ('Correlated (decay=0.1)', lambda: np.random.randn(N_SAMPLES, 50) * np.exp(-0.1 * np.arange(50))),
        ('Sparse (90%)', lambda: (np.random.rand(N_SAMPLES, 50) < 0.9).astype(float) * np.random.randn(N_SAMPLES, 50)),
        ('Sparse (95%)', lambda: (np.random.rand(N_SAMPLES, 50) < 0.95).astype(float) * np.random.randn(N_SAMPLES, 50)),
    ]
    
    print(f"\nTesting {len(conditions)} conditions...")
    results = []
    
    for name, gen_fn in conditions:
        print(f"\n  {name}...")
        data = gen_fn()
        D_eff = compute_deff_fast(data)
        has_gs, details = check_growth_saturation(D_eff)
        
        results.append({
            'condition': name,
            'D_early': details.get('D_early', np.nan),
            'D_late': details.get('D_late', np.nan),
            'has_growth_saturation': has_gs
        })
        
        status = "YES" if has_gs else "NO"
        print(f"    Growth-saturation: {status}")
    
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "condition_log.csv", index=False)
    
    present = df[df['has_growth_saturation'] == True]
    absent = df[df['has_growth_saturation'] == False]
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total: {len(df)}")
    print(f"  With growth-saturation: {len(present)}")
    print(f"  Without: {len(absent)}")
    
    summary = [{
        'total': len(df),
        'with_gs': len(present),
        'without_gs': len(absent),
        'simplest_with_gs': present.iloc[0]['condition'] if len(present) > 0 else 'None'
    }]
    pd.DataFrame(summary).to_csv(results_dir / "minimal_summary.csv", index=False)
    
    return df

if __name__ == "__main__":
    try:
        results = run_minimal_conditions()
        print("\n[SUCCESS] Minimal condition analysis completed successfully")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
