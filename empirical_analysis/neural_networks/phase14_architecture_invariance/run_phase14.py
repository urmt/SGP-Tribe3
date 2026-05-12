# ============================================
# PHASE 14 — ARCHITECTURE / LEARNING INVARIANCE
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

TRAINED_PATH = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10c_training_dynamics/activations.npy"
OUTPUT_PATH = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase14_architecture_invariance/results.json"

EPSILON = 0.5
L_MIN = 2
DEPTHS = list(range(1, 9))
SUBSET_SIZE = 300
RANDOM_SEED = 42
N_JOBS = -1

# RANDOM NETWORK (FIXED)
INPUT_DIM = 64
HIDDEN_DIM = 64

# ============================================
# LOGGING
# ============================================

def log(msg):
    print(f"[PHASE14] {msg}", flush=True)

def fail(msg):
    print(f"\n[ERROR] {msg}")
    sys.exit(1)

# ============================================
# LOAD TRAINED ACTIVATIONS
# ============================================

log("Loading trained activations...")

if not os.path.exists(TRAINED_PATH):
    fail(f"Missing trained activations: {TRAINED_PATH}")

try:
    X_trained_full = np.load(TRAINED_PATH)
except Exception as e:
    fail(f"Load failed: {e}")

if len(X_trained_full.shape) != 2:
    fail(f"Invalid shape: {X_trained_full.shape}")

N, D = X_trained_full.shape
log(f"Trained data shape: {N}, {D}")

# ============================================
# SUBSET (SHARED)
# ============================================

np.random.seed(RANDOM_SEED)
indices = np.random.choice(N, SUBSET_SIZE, replace=False)

X_trained = X_trained_full[indices].astype(np.float32)

# ============================================
# BUILD RANDOM NETWORK ACTIVATIONS
# ============================================

log("Generating random network activations...")

np.random.seed(RANDOM_SEED)

W = np.random.randn(INPUT_DIM, HIDDEN_DIM) * 0.5
b = np.random.randn(HIDDEN_DIM) * 0.1

# simulate "input" from trained activations
X_random = np.tanh(X_trained @ W + b)

if np.isnan(X_random).any():
    fail("NaN in random activations")

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
# METRICS
# ============================================

def compute_metrics(X):
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

    return F, DET

# ============================================
# CORE FUNCTION
# ============================================

def run_system(X_base):

    F_base, DET_base = compute_metrics(X_base)

    # cache single ops
    single_cache = {}
    for name in OP_NAMES:
        X_op = OPS[name](X_base)
        single_cache[name] = compute_metrics(X_op)

    def evaluate_path(path):
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

    results = {}

    for depth in DEPTHS:
        log(f"Depth {depth}")

        paths = list(product(OP_NAMES, repeat=depth))

        outputs = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_path)(p) for p in paths
        )

        dF = [o[0] for o in outputs]
        dDET = [o[1] for o in outputs]

        results[str(depth)] = {
            "mean_abs_delta_F": float(np.mean(dF)),
            "mean_abs_delta_DET": float(np.mean(dDET))
        }

    return results

# ============================================
# RUN BOTH SYSTEMS
# ============================================

log("Running trained system...")
trained_results = run_system(X_trained)

log("Running random system...")
random_results = run_system(X_random)

# ============================================
# SAVE
# ============================================

final = {
    "trained": trained_results,
    "random": random_results
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(final, f, indent=2)

log("SUCCESS — Phase 14 complete")