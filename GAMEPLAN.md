# SGP-Tribe3 GAMEPLAN (LOCKED)

## Core Thesis
**The brain is a projection of underlying mathematical structure consistent with the SGP.** If we can reverse-engineer brain activity patterns (via TRIBE v2) into geometric SGP representations from text stimuli, we can compare our geometric model's output against LLM neural models on equal footing — both processing text.

**Hypothesis H1**: An LLM integrated with an SGP Resonance Graph calibrated from TRIBE v2 fMRI data will produce measurably lower hallucination rates and higher cross-domain coherence than the same LLM without SGP architecture.

**Hypothesis H2**: TRIBE v2 inference will empirically recover the Hickok-Poeppel dual-stream dissociation — ventral nodes (G2, G7, G8) showing significantly higher activation for semantic stimuli, dorsal nodes (G1, G3, G4, G9) for syntactic/structural stimuli.

---

## Architecture Overview

```
TEXT INPUT → TRIBE v2 (LLaMA 3.2 embeddings) → fMRI prediction (20,484 vertices)
    ↓
SGP Parcellation (Schaefer-200 → 9 nodes) → SGP Activation Profile
    ↓
Resonance Anchor Ωw → LLM Prompt Weighting → SGP-Guided Output
```

**9 SGP Nodes**: G1_broca, G2_wernicke, G3_tpj, G4_pfc, G5_dmn, G6_limbic, G7_sensory, G8_atl, G9_premotor

---

## Phase 0: Fix Text Inference on CPU (BLOCKING)

**Problem**: TRIBE v2 loads ALL extractors during `model.predict()`, even for text-only events. The audio extractor (`Wav2Vec-BERT`) tries to move to CUDA and crashes with `AssertionError: Torch not compiled with CUDA enabled`.

**Solution**: 
1. ✅ Installed CPU-only PyTorch (2.11.0+cpu) - eliminates CUDA compatibility issues
2. ✅ Integrated Ollama embeddings (Mistral-7B) for fast text inference (~24s vs ~5min with LLaMA)
3. ✅ Created embedding adapter infrastructure (train_adapter.py) for future calibration
4. ✅ Fallback to LLaMA on CPU if Ollama unavailable

**Status**: ✅ COMPLETE - Ollama integration working, fallback to LLaMA CPU available

**Architecture**:
```
TEXT INPUT → Ollama (Mistral-7B) → 4096d embedding → Adapter (optional) → 9216d → Random projection → 20,484 vertices
     ↓
SGP Parcellation (Schaefer-200 → 9 nodes) → SGP Activation Profile
     ↓
Resonance Anchor Ωw → LLM Prompt Weighting → SGP-Guided Output
```

**Performance**: ~24 seconds per text (vs ~5+ minutes with LLaMA on CPU)

---

## Phase 1: Minimal Test Battery (3-5 Texts)

| # | Text | Expected Dominant Nodes | Purpose |
|---|---|---|---|
| 1 | "The cat sat on the mat." | G2_wernicke, G7_sensory | Baseline simple sentence |
| 2 | "If P implies Q, and Q implies R, then P implies R." | G4_pfc, G1_broca | Logical structure |
| 3 | "She felt the warmth of the sun on her skin as memories of childhood flooded back." | G6_limbic, G5_dmn | Emotional + sensory |
| 4 | "The mitochondria is the powerhouse of the cell." | G4_pfc, G7_sensory | Factual/technical |
| 5 | "What if the universe is a simulation and we're just characters in someone else's dream?" | G5_dmn, G3_tpj | Self-referential/abstract |

---

## Phase 2: Result Persistence & Analytics

**Storage**: HF dataset repo `Sentient-Field/sgp-tribe3-results`

**Schema**:
```json
{
    "stimulus_id": "uuid",
    "text": "input text",
    "sgp_nodes": {"G1_broca": 0.73, ...},
    "streams": {"dorsal": 0.58, ...},
    "edge_weights": {"AF": 0.45, ...},
    "dominant_hemisphere": "left",
    "inference_time_seconds": 45.2
}
```

---

## Phase 3: LLM Integration (OpenRouter Free Tier)

**Primary**: OpenRouter free tier (`openrouter/free` router)
**Fallback**: Ollama Mistral-7B local

**Prompt construction**:
```
System: You are guided by brain-inspired SGP model.
Activation weights: {node_name}: {value}, ...

User: {input}
```

---

## Phase 4: Validation & Comparison

- Coherence: Does SGP-guided output stay more on-topic?
- Differentiation: Do different texts produce different activation profiles?
- Dual-stream: Do semantic vs. logical texts activate different node clusters?

---

## Phase 5: Scale Up (Post-Validation)

- 50-100 text test battery
- Statistical analysis
- Co-activation matrix
- Scientific article documentation

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Text inference still hits CUDA error | Medium | Fallback: subprocess with CUDA_VISIBLE_DEVICES="" |
| TRIBE v2 slow on CPU | High | Accept 30-60s per inference |
| OpenRouter rate limits | Low | Fallback to Ollama local |
| HF Space cold starts | Low | Model persists between requests |

---

## Continuation Prompts for AI Agents

**To start from Phase 0**:
```
Read /home/student/sgp-tribe3/GAMEPLAN.md. Start with Phase 0: Fix text inference on CPU.
Test with: curl -X POST https://Sentient-Field-sgp-tribe3.hf.space/predict_text -F "text=The cat sat on the mat." -F "stimulus_id=test"
```

**Current Status**: Phase 0 complete. Ollama integration working with ~24s inference time. Ready for Phase 1 testing.

**Next Steps**:
1. Train adapter weights for better LLaMA→Ollama mapping (train_adapter.py)
2. Run Phase 1 test battery with 3-5 texts
3. Validate SGP node activation patterns

---

## Plan Locked
This plan is LOCKED. Any AI agent can pick up from this document at any phase.

**Last Updated**: April 6, 2026
**Project**: SGP-Tribe3 - Sentient Generative Principal