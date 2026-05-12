# ============================================
# PHASE 13B — HARDENED / AUDITABLE VERSION
# ============================================

import os
import json
import numpy as np
from itertools import product
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import traceback
import sys

# ============================================
# CONFIG (LOCKED)
# ============================================

DATA_PATH = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10c_training_dynamics/activations.npy"
OUTPUT_PATH = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase13b_operator_path_scaling/results.json"

EPSILON = 0.5
L_MIN = 2
DEPTHS = list(range(1, 9))
N_JOBS = -1

# FIXED SUBSET (CRITICAL — DO NOT CHANGE)
SUBSET_SIZE = 300
RANDOM_SEED = 42

# ============================================
# DEBUG LOGGER
# ============================================

def log(msg):
    print(f"[PHASE13B] {msg}", flush=True)

# ============================================
# HARD FAIL FUNCTION
# ============================================

def fail(msg):
    print(f"\n[ERROR] {msg}")
    sys.exit(1)

# ============================================
# STEP 1: PATH CHECK
# ============================================

log("Checking paths...")

if not os.path.exists(DATA_PATH):
    fail(f"Missing activations file: {DATA_PATH}")

# ============================================
# STEP 2: LOAD DATA
# ============================================

log("Loading activations...")

try:
    X_full = np.load(DATA_PATH)
except Exception as e:
    fail(f"Failed to load activations.npy: {e}")

# ============================================
# STEP 3: DATA VALIDATION
# ============================================

log("Validating data...")

if not isinstance(X_full, np.ndarray):
    fail("Data is not a numpy array")

if len(X_full.shape) != 2:
    fail(f"Expected 2D activations, got shape {X_full.shape}")

N, D = X_full.shape
log(f"Data shape: {N} samples, {D} features")

if N < SUBSET_SIZE:
    fail(f"Not enough samples: need {SUBSET_SIZE}, got {N}")

if np.isnan(X_full).any():
    fail("NaN detected in activations")

if np.isinf(X_full).any():
    fail("Inf detected in activations")

# ============================================
# STEP 4: FIXED SUBSET (DETERMINISTIC)
# ============================================

log("Selecting fixed subset...")

np.random.seed(RANDOM_SEED)
indices = np.random.choice(N, SUBSET_SIZE, replace=False)
X_base = X_full[indices].astype(np.float32)

log(f"Subset shape: {X_base.shape}")

# ============================================
# OPERATORS
# ============================================

def op_tanh(x): return np.tanh(x)
def op_relu(x): return np.maximum(0, x)
def op_softplus(x): return np.log1p(np.exp(x))

OPS = {
    "tanh": op_tanh,
    "relu": op_relu,
    "softplus": op_softplus
}

OP_NAMES = list(OPS.keys())

# ============================================
# METRICS (SAFE VERSION)
# ============================================

def compute_metrics(X):
    try:
        D = cdist(X, X)
        R = (D < EPSILON).astype(np.uint8)

        F = np.mean(R)

        N = R.shape[0]
        diag_lengths = []

        for k in range(-N + 1, N):
            diag = np.diag(R, k)
            length = 0
            for v in diag:
                if v:
                    length += 1
                else:
                    if length >= L_MIN:
                        diag_lengths.append(length)
                    length = 0
            if length >= L_MIN:
                diag_lengths.append(length)

        DET = 0.0 if len(diag_lengths) == 0 else np.sum(diag_lengths) / np.sum(R)

        if np.isnan(F) or np.isnan(DET):
            raise ValueError("NaN in metrics")

        return F, DET

    except Exception as e:
        fail(f"Metric computation failed: {e}")

# ============================================
# BASELINE
# ============================================

log("Computing baseline...")

F_base, DET_base = compute_metrics(X_base)

# ============================================
# CACHE SINGLE OPERATORS
# ============================================

log("Caching single operator effects...")

single_cache = {}

for name in OP_NAMES:
    try:
        X_op = OPS[name](X_base)
        single_cache[name] = compute_metrics(X_op)
    except Exception as e:
        fail(f"Single operator failed: {name} → {e}")

# ============================================
# PATH FUNCTION
# ============================================

def evaluate_path(path):
    try:
        X = X_base.copy()

        for op in path:
            X = OPS[op](X)

        F_actual, DET_actual = compute_metrics(X)

        F_expected = F_base
        DET_expected = DET_base

        for op in path:
            F_op, DET_op = single_cache[op]
            F_expected += (F_op - F_base)
            DET_expected += (DET_op - F_base)

        return abs(F_actual - F_expected), abs(DET_actual - DET_expected)

    except Exception:
        traceback.print_exc()
        fail(f"Path failed: {path}")

# ============================================
# MAIN LOOP
# ============================================

results = {}

for depth in DEPTHS:
    log(f"Running depth {depth}")

    paths = list(product(OP_NAMES, repeat=depth))
    log(f"Total paths: {len(paths)}")

    outputs = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_path)(p) for p in paths
    )

    dF = [o[0] for o in outputs]
    dDET = [o[1] for o in outputs]

    results[str(depth)] = {
        "num_paths": len(paths),
        "mean_abs_delta_F": float(np.mean(dF)),
        "mean_abs_delta_DET": float(np.mean(dDET)),
        "max_abs_delta_F": float(np.max(dF)),
        "max_abs_delta_DET": float(np.max(dDET))
    }

# ============================================
# SAVE
# ============================================

log("Saving results...")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

log("SUCCESS — Phase 13B complete")