# Phase 202 - Metastability Report

## Summary
- **Transition types**: 2 ABRUPT (noise, r), 4 GRADUAL (coupling, burst, coalition, forcing)
- **Stable windows**: 8 total across all parameters
- **Collapse thresholds**: 2 detected (noise, logistic-r)
- **Hysteresis**: NOT DETECTED

## Parameter-Specific Findings

### Coupling (Kuramoto)
- **Type**: GRADUAL_TRANSITION
- **Collapse**: None detected (organization persists across all coupling values)
- **Stable window**: 0.22-0.33

### Noise (Kuramoto) 
- **Type**: ABRUPT_TRANSITION
- **Collapse threshold**: ~0.02 (organization collapses sharply)
- **Stable window**: 0.00-0.02
- **Interpretation**: Noise is the primary destabilizing factor

### Logistic R
- **Type**: ABRUPT_TRANSITION  
- **Collapse threshold**: ~3.53-3.60 (chaos boundary)
- **Stable window**: 3.47-3.53

### Burst Density
- **Type**: GRADUAL_TRANSITION
- **Stable windows**: 2 (0.36-0.49, 0.74-0.87)
- **Interpretation**: Organization tolerates wide burst range

### Coalition Persistence
- **Type**: GRADUAL_TRANSITION
- **Stable window**: 0.71-0.86

### Forcing
- **Type**: GRADUAL_TRANSITION
- **Stable windows**: 2 (0.07-0.14, 0.21-0.36)

## Metastability Assessment

The 5-factor organization exhibits **metastable characteristics**:
- Organization persists in stable windows across multiple parameters
- Abrupt collapse only occurs under extreme noise or chaos
- No hysteresis detected - transitions are reversible
- System returns to organized state when parameters return to stable ranges

## Conclusions

1. **Irreducibility is ROBUST**: Organization survives wide parameter variations
2. **Primary vulnerability**: High noise (>0.02) and chaos onset (r>3.6)
3. **NOT phase transition**: Gradual transitions dominate, suggesting continuous organization
4. **System-specific**: Different dynamical systems have different collapse points
5. **Universal invariants**: O1 and O3 collapse together - eigenvalue/synchronization coupling