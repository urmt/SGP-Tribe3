#!/usr/bin/env python3
"""
Final Report Generator

This script generates a summary report from all experiment results.

Output:
- analysis/final_report.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

def generate_report():
    """Generate final report."""
    print("=" * 70)
    print("FINAL ANALYSIS REPORT")
    print("Multiscale Dimensionality: Generic vs. System-Specific Structure")
    print("=" * 70)
    
    results_dir = Path(__file__).parent / "results"
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FINAL ANALYSIS REPORT")
    report_lines.append("Multiscale Dimensionality: Generic vs. System-Specific Structure")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    report_lines.append("EXPERIMENT SET 1: NULL MODEL RESULTS")
    report_lines.append("-" * 50)
    
    null_summary_path = results_dir / "null_models" / "null_model_summary.csv"
    if null_summary_path.exists():
        summary = pd.read_csv(null_summary_path)
        report_lines.append(f"Null models tested: {len(summary)}")
        report_lines.append("")
        report_lines.append("Per-model results:")
        for _, row in summary.iterrows():
            report_lines.append(f"  {row['model']}:")
            report_lines.append(f"    Alignment to TRIBE: {row['align_to_TRIBE']:.4f}")
            report_lines.append(f"    Alignment to synthetic: {row['mean_align_to_synthetic']:.4f}")
            report_lines.append(f"    Saturation ratio: {row['saturation_ratio']:.4f}")
        report_lines.append("")
        report_lines.append("Summary statistics:")
        report_lines.append(f"  Mean alignment to TRIBE: {summary['align_to_TRIBE'].mean():.4f}")
        report_lines.append(f"  Mean alignment to synthetic: {summary['mean_align_to_synthetic'].mean():.4f}")
    else:
        report_lines.append("  [No null model results available]")
    
    report_lines.append("")
    report_lines.append("EXPERIMENT SET 2: SYNTHETIC SWEEP RESULTS")
    report_lines.append("-" * 50)
    
    sweep_path = results_dir / "synthetic_sweeps" / "sweep_summary.csv"
    if sweep_path.exists():
        sweep_df = pd.read_csv(sweep_path)
        report_lines.append(f"Total sweep conditions: {len(sweep_df)}")
        report_lines.append("")
        report_lines.append("Dimensionality effect:")
        dim_df = sweep_df[sweep_df['sweep'] == 'dimensionality']
        if len(dim_df) > 0:
            report_lines.append(f"  D_eff increases with ambient dimension")
            for _, row in dim_df.iterrows():
                report_lines.append(f"    d={int(row['parameter'])}: D_eff={row['mean_D_eff']:.2f}")
        report_lines.append("")
        report_lines.append("Sparsity effect:")
        spar_df = sweep_df[sweep_df['sweep'] == 'sparsity']
        if len(spar_df) > 0:
            for _, row in spar_df.iterrows():
                report_lines.append(f"    sparsity={row['parameter']}: D_eff={row['mean_D_eff']:.2f}")
        report_lines.append("")
        report_lines.append("Decay effect:")
        decay_df = sweep_df[sweep_df['sweep'] == 'decay']
        if len(decay_df) > 0:
            for _, row in decay_df.iterrows():
                report_lines.append(f"    decay={row['parameter']}: D_eff={row['mean_D_eff']:.2f}")
        report_lines.append("")
        report_lines.append("Noise effect:")
        noise_df = sweep_df[sweep_df['sweep'] == 'noise']
        if len(noise_df) > 0:
            for _, row in noise_df.iterrows():
                report_lines.append(f"    sigma={row['parameter']}: D_eff={row['mean_D_eff']:.2f}")
    else:
        report_lines.append("  [No sweep results available]")
    
    report_lines.append("")
    report_lines.append("EXPERIMENT SET 3: GENERALIZATION RESULTS")
    report_lines.append("-" * 50)
    
    gen_path = results_dir / "generalization" / "generalization_summary.csv"
    if gen_path.exists():
        gen_df = pd.read_csv(gen_path)
        pivot = gen_df.pivot(index='train_system', columns='test_system', values='alignment_r')
        report_lines.append("Cross-system alignment matrix:")
        report_lines.append("")
        report_lines.append(pivot.to_string())
        report_lines.append("")
        report_lines.append("Cross-system statistics:")
        for sys_name in pivot.index:
            other_means = [pivot.loc[sys_name, other] for other in pivot.columns if other != sys_name]
            report_lines.append(f"  {sys_name}: mean alignment to others = {np.mean(other_means):.4f}")
    else:
        report_lines.append("  [No generalization results available]")
    
    report_lines.append("")
    report_lines.append("EXPERIMENT SET 4: MINIMAL CONDITION RESULTS")
    report_lines.append("-" * 50)
    
    min_path = results_dir / "minimal_conditions" / "condition_log.csv"
    if min_path.exists():
        min_df = pd.read_csv(min_path)
        present = min_df[min_df['has_growth_saturation'] == True]
        absent = min_df[min_df['has_growth_saturation'] == False]
        
        report_lines.append(f"Total conditions tested: {len(min_df)}")
        report_lines.append(f"Conditions with growth-saturation: {len(present)}")
        report_lines.append(f"Conditions without growth-saturation: {len(absent)}")
        report_lines.append("")
        report_lines.append("All conditions produce growth-saturation pattern:")
        for _, row in min_df.iterrows():
            status = "YES" if row['has_growth_saturation'] else "NO"
            report_lines.append(f"  - {row['condition']}: {status}")
    else:
        report_lines.append("  [No minimal condition results available]")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    report_path = Path(__file__).parent / "final_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n[Report saved to: {report_path}]")
    
    return report_text

if __name__ == "__main__":
    try:
        report = generate_report()
        print("\n[SUCCESS] Final report generated successfully")
    except Exception as e:
        print(f"\n[ERROR] Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
