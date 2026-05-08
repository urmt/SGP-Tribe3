# SGP-Tribe3 Experiment Inventory
## Neural Observability and Organization Invariants - Complete Chronology

**Repository:** https://github.com/urmt/SGP-Tribe3
**Generated:** May 2026
**Total Experiments:** 115 phases

---

## Overview

This document provides a chronological inventory of all experimental phases conducted in the SGP-Tribe3 research program investigating neural network observability and organizational invariants. Each experiment is summarized with a brief 6th-grade-accessible description.

---

## Experiment Chronology

### Phase 10: Training Dynamics (Early Foundation)
- **Phase 10a**: Tested how neural networks change their internal representations as they learn, finding that simple patterns in initial training evolve into complex structures.
- **Phase 10c**: Measured how quickly networks learn and how their learning speed changes over time.

### Phase 11: Spectral and Frequency Analysis
- **Phase 11a**: Analyzed the frequency patterns (like radio station frequencies) inside trained networks to see what patterns exist.
- **Phase 11b**: Tested how different parts of a network talk to each other by measuring their signals together.
- **Phase 11c**: Made small changes to network inputs and measured how the outputs respond.

### Phase 12-15: Operator and Geometry Studies
- **Phase 12**: Studied how network operations combine together and what mathematical rules they follow.
- **Phase 13**: Traced how information flows through network layers like a path through a maze.
- **Phase 14**: Tested whether neural networks with different architectures share similar underlying mathematical properties.
- **Phase 15a-c**: Mapped the geometric shapes that networks create in their internal space using distance measurements.

### Phase 66-70: Information Geometry and Universality
- **Phase 66**: Used information geometry (measuring shapes in probability space) to analyze network representations.
- **Phase 67**: Stress-tested whether network behaviors are universal across different architectures.
- **Phase 68**: Checked if neural networks form similar geometric patterns regardless of their specific design.
- **Phase 69**: Mapped all the ways neural networks can fail or break down.
- **Phase 70**: Created a universal diagram showing all possible network states and transitions.

### Phase 71-76: Computational Organization and Observer Theory
- **Phase 71**: Investigated how computational processes organize themselves without central control.
- **Phase 72**: Tested whether networks can understand their own structure through recursive self-reference.
- **Phase 73**: Explored how observer-like properties emerge from simple network computations.
- **Phase 74**: Measured critical transitions (sudden state changes) in network observers.
- **Phase 75**: Attempted to falsify (prove wrong) observer-based claims using adversarial conditions.
- **Phase 76**: Tested whether known phenomena can be rediscovered from network data alone.

### Phase 77-82: Leakage, Manifold, and Language Tests
- **Phase 77**: Checked for circularity and data leakage in experimental results.
- **Phase 78**: Tested whether geometric manifolds are necessary for network representations.
- **Phase 79**: Found the minimal set of mathematical rules needed to generate critical phenomena in networks.
- **Phase 80**: Derived universal laws that apply to all recursive critical systems.
- **Phase 81**: Applied high-rigor methods to test recursive criticality claims.
- **Phase 82**: Tested whether language and neural networks have similar or isolated mathematical properties.

### Phase 83-100: Real-World Invariant Falsification Program
- **Phase 83**: Tested whether organizational patterns that work in one domain also work in completely different real-world domains (NO INVARIANTS FOUND).
- **Phase 84**: Built the neuroscience foundation for understanding neural data patterns.
- **Phase 85**: Created a pipeline to work with real neural data instead of artificial data.
- **Phase 86**: Hardened the pipeline against data leakage problems.
- **Phase 87**: Tested network performance on data completely different from training data (out-of-distribution).
- **Phase 88**: Applied harder out-of-distribution tests with more difficult conditions.
- **Phase 89**: Measured how well networks understand their own internal states (self-model coupling).
- **Phase 90**: Tested whether self-model properties work across different neural network architectures.
- **Phase 91**: Ran adversarial tests to falsify self-model claims (ROBUST SIGNAL FOUND).
- **Phase 92**: Tested whether patterns discovered in one dataset transfer to completely new real-world datasets (SIGNAL TRANSFERRED).
- **Phase 92R**: Replicated phase 92 with stricter controls (NO ROBUST SIGNAL).
- **Phase 93**: Applied causal destruction tests to see if temporal patterns are truly causal (NO CAUSAL SIGNAL).
- **Phase 94**: Tested which mathematical properties are necessary vs. optional (MOSTLY REDUCIBLE).
- **Phase 95**: Tested if signals can be reduced to simple linear components (FULLY LINEARLY REDUCIBLE).
- **Phase 96**: Tested intervention stability across domains (PARTIAL INTERVENTION STABILITY).
- **Phase 97**: Attempted to recover physical laws from neural data (CROSS-DOMAIN RECOVERY INVARIANT).
- **Phase 98**: Tested if irreducible complexity exists in model misspecification.
- **Phase 99**: Tested scale-free properties across different domains (DOMAIN-SPECIFIC SCALING).
- **Phase 100**: Final real data reconstruction test (NO ROBUST EFFECT - synthetic patterns don't survive strict validation).

### Phase 101-115: Real EEG Acquisition and Validation
- **Phase 101**: Searched for real EEG/MEG/neural data (DATA NOT FOUND).
- **Phase 103**: Searched PhysioNet, OpenNeuro, TUH for real EEG (INSUFFICIENT REAL DATA).
- **Phase 104**: Exhaustively searched repository for real EEG - found NONE (NO REAL EEG DATA AVAILABLE).
- **Phase 105**: Attempted to download real EEG from PhysioNet (NO REAL DATA ACQUIRED - network timeout).
- **Phase 108-109**: Cross-subject real EEG validation tests (INSUFFICIENT SUBJECTS).
- **Phase 110**: Expanded EEG download attempts (INSUFFICIENT PUBLIC DATA - network issues).
- **Phase 111**: Long-duration EEG acquisition (2 subjects acquired).
- **Phase 112**: Persistent acquisition worker to get 5 subjects (5 UNIQUE SUBJECTS ACQUIRED - READY FOR LOSO).
- **Phase 113**: Strict Leave-One-Subject-Out validation on real EEG (ROBUST REAL STRUCTURE - AUC 1.0).
- **Phase 114**: Causal temporal destruction hierarchy (PHASE DEPENDENT - AUC 1.0).
- **Phase 115**: Hierarchical surrogate preservation test (PHASE SENSITIVE - AUC 0.783).

---

## Key Findings Summary

### Falsification Results (Phases 83-100)
- **Cross-domain invariants**: NOT FOUND - patterns that work in one domain fail in others
- **Causal temporal structure**: NO CAUSAL SIGNAL - signals destroyed by causal controls
- **Universality claims**: MOSTLY REDUCIBLE to simple linear statistics
- **Synthetic artifacts**: All "phenomena" reducible to synthetic data generation artifacts

### Real EEG Results (Phases 113-115)
- **Real EEG validation**: ROBUST REAL STRUCTURE confirmed
- **Phase dependence**: PHASE DEPENDENT - EEG depends on phase relationships
- **Temporal organization**: PHASE SENSITIVE - survives cross-subject validation

---

## Methodology Summary

All experiments follow strict protocols:
- **No synthetic data** after Phase 82 (except where explicitly noted)
- **No embeddings/manifolds/topology** in later phases
- **Strict cross-subject validation** for real EEG tests
- **Leave-One-Subject-Out CV** for generalization testing
- **Null controls** including permutation, temporal shuffle, phase randomization
- **Fixed hyperparameters** (random_state=42, n_estimators=200)

---

## File Structure

Experiments are organized in:
```
empirical_analysis/neural_networks/
├── phase10a_training_dynamics/
├── phase11a_spectral/
├── phase66_information_geometry/
├── ...
├── phase83_real_world_invariant_test/
├── phase100_real_data_rebuild/
├── phase112_persistent_acquisition/   # Contains 5 real EEG subjects
├── phase113_real_loso/
├── phase114_causal_destruction/
└── phase115_hierarchical_surrogates/
```

---

## Dataset Summary

**Final validated data**: 5 unique EEG subjects (chb00-chb04) from CHB-MIT PhysioNet dataset
- Total windows: 2,105 (10-second non-overlapping)
- All subjects passed strict QC (>20 windows, >200 seconds duration)

---

*Document generated from SGP-Tribe3 repository commit history. For detailed methodology and raw results, see individual phase directories.*