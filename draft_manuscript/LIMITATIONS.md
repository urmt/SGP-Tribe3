# Limitations

- **Single-subject/System limitations:** The analysis primarily relies on the TRIBE fMRI predictive model; results may not generalize to other cortical parcellations or modalities.
- **Semantic mapping limitations:** Text embeddings were derived from specific Transformer architectures (e.g., tinyLLaMA); the signature's dependence on the specific encoder architecture requires further testing.
- **Estimator limitations:** Participation Ratio assumes locally linear structures; non-linear manifold features may be obfuscated at certain scales.
- **Curvature-definition limitations:** Estimates based on k-nearest neighbor distance variance are sensitive to local data density, potentially introducing bias in sparsely sampled regions.
- **Benchmark limitations:** The comparison against candidate models (e.g., Gompertz, Hill Function) does not exhaust the space of all possible organizational invariants.
- **Biological interpretation:** The SGP-related potential functions and biological activity mappings are theoretical constructs requiring empirical validation against ground-truth neural recording data.
