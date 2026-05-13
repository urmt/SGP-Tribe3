import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import time
import json
import traceback

# RULES & RESTRICTIONS
# 10. USE FIXED RANDOM SEED = 42 everywhere possible.
np.random.seed(42)

# File paths
audit_dir = "audit_outputs"
os.makedirs(audit_dir, exist_ok=True)

# Define k_values for the primary datasets
k_values_primary = np.array([5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500])

# Models
def model_sigmoid(x, L, k0, beta, b):
    return L / (1 + np.exp(-beta * (x - k0))) + b

def model_linear(x, m, b):
    return m * x + b

def model_logarithmic(x, a, b):
    return a * np.log(x) + b

def model_power_law(x, a, b, c):
    # Ensure x > 0 for power law
    return a * np.power(x, b) + c

def model_gompertz(x, A, k, x0, b):
    return A * np.exp(-np.exp(-k * (x - x0))) + b

def model_hill(x, Vmax, k, n, b):
    # Ensure x >= 0 for hill
    return Vmax * (x**n) / (np.abs(k)**n + x**n) + b

def model_piecewise_linear(x, k_break, m1, m2, b):
    return np.piecewise(x, [x <= k_break], 
                        [lambda x: m1 * x + b, 
                         lambda x: m1 * k_break + b + m2 * (x - k_break)])

# Utility to compute metrics
def compute_metrics(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    ss_res = np.sum((y_true - y_pred)**2)
    rmse = np.sqrt(ss_res / n)
    mae = mean_absolute_error(y_true, y_pred)
    # Using simple AIC/BIC for least squares
    # AIC = n * ln(RSS/n) + 2k
    # BIC = n * ln(RSS/n) + k * ln(n)
    # Adding a small epsilon to RSS/n to avoid log(0)
    rss_n = max(ss_res / n, 1e-10)
    aic = n * np.log(rss_n) + 2 * p
    bic = n * np.log(rss_n) + p * np.log(n)
    return {
        "R2": r2, "Adj_R2": adj_r2, "RMSE": rmse, "MAE": mae, "AIC": aic, "BIC": bic
    }

def fit_with_stats(model_func, x, y, p0, bounds=None):
    try:
        if bounds:
            popt, _ = curve_fit(model_func, x, y, p0=p0, bounds=bounds, maxfev=10000)
        else:
            popt, _ = curve_fit(model_func, x, y, p0=p0, maxfev=10000)
        y_pred = model_func(x, *popt)
        metrics = compute_metrics(y, y_pred, len(popt))
        return popt, metrics, y_pred
    except Exception as e:
        return None, None, None

def run_section_1():
    start_time = time.time()
    print("Executing Section 1: Model Comparison...")
    
    # Load data
    try:
        data_curves = np.load('./sgp-tribe3/analysis/estimator_ablation/results/raw_curves.npy', allow_pickle=True).item()
        scaling_results = pd.read_csv('./sgp-tribe3/empirical_analysis/neural_networks/universal_scaling_results.csv')
        nn_dk = pd.read_csv('./sgp-tribe3/empirical_analysis/neural_networks/nn_dk_vectors.csv')
    except Exception as e:
        print(f"FAILED: Data loading error: {e}")
        traceback.print_exc()
        return

    # Prepare datasets
    systems = {}
    
    # 1. TRIBE, Sparse, Hierarchical (from estimator ablation)
    for s in ['TRIBE', 'Sparse', 'Hierarchical']:
        if s in data_curves:
            systems[s] = (k_values_primary, np.array(data_curves[s]['PR']))
    
    # 2. Random (from universal scaling)
    random_row = scaling_results[scaling_results['system'] == 'random']
    if not random_row.empty:
        systems['Random'] = (np.array([2, 4, 8, 16]), random_row[['D2', 'D4', 'D8', 'D16']].values.flatten())
    
    # 3. Transformer Layers (from nn_dk_vectors)
    trans_row = nn_dk[(nn_dk['model'] == 'MLP_DEEP') & (nn_dk['layer'] == 'logits')]
    if not trans_row.empty:
        systems['Transformer Layers'] = (np.array([2, 4, 8, 16]), trans_row[['D2', 'D4', 'D8', 'D16']].values.flatten())

    model_defs = {
        "Sigmoid": (model_sigmoid, lambda y, x: [np.max(y), np.median(x), 0.1, np.min(y)]),
        "Linear": (model_linear, lambda y, x: [0.1, np.mean(y)]),
        "Logarithmic": (model_logarithmic, lambda y, x: [1.0, np.mean(y)]),
        "Power Law": (model_power_law, lambda y, x: [1.0, 0.5, 0.0]),
        "Gompertz": (model_gompertz, lambda y, x: [np.max(y), 0.1, np.median(x), np.min(y)]),
        "Hill Function": (model_hill, lambda y, x: [np.max(y), np.median(x), 1.0, np.min(y)]),
        "Piecewise Linear": (model_piecewise_linear, lambda y, x: [np.median(x), 0.1, 0.05, np.min(y)])
    }

    results_table = []
    plot_data = {}

    for sys_name, (x, y) in systems.items():
        print(f"  Fitting models for system: {sys_name}")
        plot_data[sys_name] = {"x": x, "y": y, "fits": {}}
        for mod_name, (func, p0_gen) in model_defs.items():
            p0 = p0_gen(y, x)
            # Basic bounds to avoid divergence
            if mod_name == "Sigmoid":
                bounds = ([0, 0, 0, -np.inf], [np.inf, 1000, 10, np.inf])
            elif mod_name == "Hill Function":
                bounds = ([0, 0, 0, -np.inf], [np.inf, 1000, 10, np.inf])
            else:
                bounds = None
                
            popt, metrics, y_pred = fit_with_stats(func, x, y, p0, bounds=bounds)
            
            if metrics:
                row = {"System": sys_name, "Model": mod_name}
                row.update(metrics)
                results_table.append(row)
                plot_data[sys_name]["fits"][mod_name] = (y_pred, metrics["R2"])
            else:
                print(f"    Failed to fit {mod_name} for {sys_name}")

    # Save to CSV
    results_df = pd.DataFrame(results_table)
    results_df.to_csv(os.path.join(audit_dir, "model_comparison_metrics.csv"), index=False)
    print(f"  Saved: {os.path.join(audit_dir, 'model_comparison_metrics.csv')}")

    # Generate Figure
    n_sys = len(systems)
    fig, axes = plt.subplots(n_sys, 2, figsize=(16, 4 * n_sys), gridspec_kw={'width_ratios': [2, 1]})
    
    if n_sys == 1:
        axes = [axes]

    for i, (sys_name, data) in enumerate(plot_data.items()):
        ax_fit = axes[i][0]
        ax_res = axes[i][1]
        
        x, y = data["x"], data["y"]
        ax_fit.scatter(x, y, color='black', label='Data', zorder=5)
        
        for mod_name, (y_pred, r2) in data["fits"].items():
            ax_fit.plot(x, y_pred, label=f"{mod_name} (R²={r2:.3f})", alpha=0.8)
            ax_res.plot(x, y - y_pred, 'o-', label=mod_name, alpha=0.6)
            
        ax_fit.set_title(f"Model Fits: {sys_name}")
        ax_fit.set_xlabel("k")
        ax_fit.set_ylabel("D(k)")
        ax_fit.legend(fontsize='small', ncol=2)
        ax_fit.set_xscale('log')
        
        ax_res.set_title(f"Residuals: {sys_name}")
        ax_res.set_xlabel("k")
        ax_res.set_ylabel("Error")
        ax_res.axhline(0, color='black', linestyle='--')
        ax_res.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(audit_dir, "Figure_MODEL_COMPARISON.png"), dpi=300)
    plt.close()
    print(f"  Saved: {os.path.join(audit_dir, 'Figure_MODEL_COMPARISON.png')}")

    runtime = time.time() - start_time
    print(f"Section 1 Complete. Runtime: {runtime:.2f}s")
    
    # Log execution status
    with open(os.path.join(audit_dir, "execution_notes_section1.txt"), "w") as f:
        f.write(f"Section 1: Model Comparison\n")
        f.write(f"Exact filepath: {os.path.join(audit_dir, 'model_comparison_metrics.csv')}\n")
        f.write(f"Sample size (systems): {n_sys}\n")
        f.write(f"Runtime: {runtime:.2f}s\n")
        f.write(f"Status: Success\n")

if __name__ == "__main__":
    run_section_1()
