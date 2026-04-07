# SGP-TRIBE3 — COMPLETE AI CODING AGENT HANDOFF DOCUMENT
# Version 1.0 | April 2026 | Sentient-Field Braintrust

---

## CRITICAL: READ THIS FIRST

This document is the complete handoff for the SGP-Tribe3 project.
It contains everything an AI coding agent needs to continue this project
from exactly where it was left off. Do NOT skip any section.

**Repository:** https://huggingface.co/spaces/Sentient-Field/sgp-tribe3
**Git remote:** https://huggingface.co/spaces/Sentient-Field/sgp-tribe3 (named "hf")
**Local repo path:** ~/sgp-tribe3 (on the user's Fedora Linux machine)
**Push command:** git push hf master:main --force
**API base URL:** https://Sentient-Field-sgp-tribe3.hf.space
**Best practices reference:** https://github.com/harvard-edge/cs249r_book

---

## SECTION 1: PROJECT PURPOSE

SGP-Tribe3 is a REST API deployed on HuggingFace Spaces that:
1. Accepts video files (mp4) via HTTP POST
2. Runs TRIBE v2 (Meta AI) multimodal fMRI encoding — predicts brain responses
3. Maps the 20,484-vertex fsaverage5 cortical output to 9 SGP brain region nodes
4. Returns structured JSON activation profiles for AI architecture calibration

The scientific goal is to derive LLM architecture from empirical brain data,
implementing the Sentient Generative Principal (SFH-SGP) theoretical framework.
The system will eventually drive an LLM whose prompt construction is weighted
by brain-region activation patterns — producing cognitively coherent AI behavior.

Theory reference: https://wt3000.substack.com (most recent articles = most accurate)

---

## SECTION 2: CURRENT STATE — WHAT IS WORKING

As of April 6, 2026:

✅ Docker builds successfully on HuggingFace Spaces
✅ Python 3.11, torch 2.5.1+cpu, TRIBE v2 all installed correctly
✅ Flask API starts and serves on port 7860
✅ TRIBE v2 model loads successfully at startup (model_loaded: true)
✅ Schaefer-200 atlas downloads and parcellates at startup
✅ /health endpoint returns {"status": "ready"}
✅ /predict endpoint accepts video uploads and runs inference
✅ G6_limbic and G8_atl now have non-zero vertex counts (fixed)

The API is LIVE and READY at:
https://Sentient-Field-sgp-tribe3.hf.space

---

## SECTION 3: CURRENT BLOCKER — THE ONE REMAINING BUG

**BUG: G2_wernicke shows 0 vertices in parcellation**

Symptom: After rebuild, logs show:
  G1_broca: 1101 vertices ✅
  G2_wernicke: 0 vertices  ← BUG
  G3_tpj: 2380 vertices ✅
  G4_pfc: ~5000 vertices ✅
  G5_dmn: ~2500 vertices ✅
  G6_limbic: 689 vertices ✅
  G7_sensory: 4420 vertices ✅
  G8_atl: 144 vertices ✅
  G9_premotor: ~3400 vertices ✅

Root cause: The Schaefer-200 atlas does NOT use STG/STS labels.
Wernicke's area (posterior superior temporal gyrus) maps to:
  - 7Networks_LH_SalVentAttn_ParOper_1
  - 7Networks_LH_SalVentAttn_ParOper_2
  - 7Networks_LH_SalVentAttn_ParOper_3

The keyword "ParOper" was added to G2's keywords but is NOT matching.
Suspected cause: The parcel names in the .annot file may have a null byte
or encoding issue making exact string matching fail.

**THE FIX TO IMPLEMENT:**

In sgp_parcellation.py, the _parcel_name_to_node() method checks:
  if kw.upper() in name_upper

The issue may be that parcel names are decoded with errors="ignore" and
some bytes are dropped. The fix is to use a more robust matching approach:

Replace the G2_wernicke keyword matching with Yeo network fallback instead.
The SalVentAttn network IS Wernicke's area in the Schaefer-200 mapping.

In SGP_NODE_DEFINITIONS for G2_wernicke, change:
  "yeo_networks": ["Default", "SalVentAttn"],
to:
  "yeo_networks": ["SalVentAttn"],

AND remove G4_pfc's claim on SalVentAttn:
  G4_pfc currently has "yeo_networks": ["Cont", "SalVentAttn"]
  Change to: "yeo_networks": ["Cont"]

This way SalVentAttn parcels that don't match any anatomical keyword
will fall through to the Yeo network fallback and go to G2_wernicke.

Also add a debug print in _parcel_name_to_node() temporarily:
  print(f"[DEBUG] parcel='{parcel_name}' repr={repr(parcel_name[:30])}")
to confirm encoding of the actual strings at runtime.

**VERIFICATION:** After fix, logs must show G2_wernicke > 0 vertices.
All 9 nodes must be non-zero before calibration can begin.

---

## SECTION 4: COMPLETE FILE INVENTORY

Files in ~/sgp-tribe3/ that are pushed to HuggingFace:

### app.py (main Flask API — 356 lines)
Endpoints:
  GET  /              — service info
  GET  /health        — model load status
  POST /warmup        — trigger model loading
  POST /predict       — PRIMARY: upload video, run inference, return activations
  GET  /nodes         — SGP node definitions
  GET  /tracts        — white matter tract definitions
  GET  /results       — all stored stimulus results (in-memory)
  GET  /coactivation_matrix — cross-stimulus co-activation matrix

Key functions:
  _load_model()       — loads TRIBE v2, applies CPU patch, initializes parcellator
  _preprocess_video() — ffmpeg trim/normalize to TRIBE v2 spec
  _run_inference()    — calls TribeModel.get_events_dataframe() then .predict()

KNOWN WARNING in logs:
  "[SGP-Tribe3] HF login warning: rate limit on /whoami-v2"
  This is harmless — HF_TOKEN secret IS set in Space settings.

### sgp_parcellation.py (462 lines)
The core scientific contribution. Maps TRIBE v2 output to SGP nodes.

Classes:
  SGPParcellator — downloads Schaefer-200 atlas, builds vertex→node map,
                   computes node activations, edge weights, hemisphere dominance

Key dicts:
  SGP_NODE_DEFINITIONS — 9 nodes with keywords, Yeo networks, MNI coordinates
  SGP_TRACT_DEFINITIONS — 9 white matter tracts connecting node pairs

CURRENT KEYWORD STATE (after all fixes):
  G1_broca:    ["FrOperIns", "Broca", "Tri", "Oper"]
  G2_wernicke: ["ParOper", "Wernicke"]          ← NOT WORKING, see Section 3
  G3_tpj:      ["DorsAttn_Post", "ParieTempOcc", "Angular"]
  G4_pfc:      ["PFCl", "PFC", "Frontal", "ACC", "Cing"]
  G5_dmn:      ["pCunPCC", "Default_Par", "PHC"]
  G6_limbic:   ["Limbic", "TempPole", "OFC", "Insula", "ParaHipp", "Hipp", "Amyg"]
  G7_sensory:  ["Vis", "SomMot", "Medial"]
  G8_atl:      ["Default_Temp", "Cont_Temp"]
  G9_premotor: ["FEF", "PrCv", "Precentral", "Motor"]

YEO NETWORK FALLBACK (currently in code):
  "Vis"         → G7_sensory
  "SomMot"      → G9_premotor
  "DorsAttn"    → G3_tpj
  "SalVentAttn" → G4_pfc         ← NEEDS TO CHANGE TO G2_wernicke
  "Limbic"      → G6_limbic
  "Cont"        → G4_pfc
  "Default"     → G5_dmn

### stimulus_pipeline.py (498 lines)
Runs on LOCAL machine (NOT deployed to HF). Handles:
- YouTube video download via yt-dlp
- Video preprocessing via ffmpeg
- TTS generation via espeak for synthetic stimuli
- Sending videos to SGP-Tribe3 API
- Saving results to ./sgp_results/

Usage:
  python3 stimulus_pipeline.py --api https://Sentient-Field-sgp-tribe3.hf.space
  python3 stimulus_pipeline.py --api URL --local-video /path/to/video.mp4

### Dockerfile
FROM python:3.11-slim
Installs: ffmpeg, git, git-lfs, libsndfile1, libgl1, libglib2.0-0, wget, espeak, curl
Installs uv/uvx (required by TRIBE v2 for audio transcription)
Installs torch 2.5.1+cpu separately first
Then installs requirements.txt
Runs as appuser (non-root)

### requirements.txt (current working version)
flask==3.1.2
numpy==2.2.6
pandas==2.3.2
nibabel==5.4.0
nilearn==0.13.0
scipy==1.15.2
scikit-learn==1.7.1
moviepy==2.2.1
soundfile==0.13.0
librosa==0.10.2.post1
requests==2.33.0
gunicorn==23.0.0
tribev2 @ git+https://github.com/facebookresearch/tribev2.git

NOTE: transformers and huggingface_hub are NOT pinned —
tribev2 resolves them. Do NOT add them back to requirements.txt.

---

## SECTION 5: SCHAEFER-200 ATLAS — COMPLETE PARCEL LIST

The Schaefer-200 atlas (left hemisphere) contains these network types:
['Cont', 'Default', 'DorsAttn', 'Limbic', 'Medial', 'SalVentAttn', 'SomMot', 'Vis']

Key parcels relevant to our nodes:
  SalVentAttn_ParOper_1,2,3    → Wernicke's area (posterior STG)
  SalVentAttn_FrOperIns_1,2,3,4 → Broca's area (inferior frontal)
  Limbic_OFC_1,2               → Limbic/OFC
  Limbic_TempPole_1,2,3,4      → Temporal pole (limbic)
  Default_Temp_1,2,3,4,5       → Anterior temporal (ATL)
  Cont_Temp_1                  → Anterior temporal (ATL)
  DorsAttn_Post_1..10          → TPJ/dorsal attention
  DorsAttn_FEF_1,2             → Frontal eye fields (premotor)
  DorsAttn_PrCv_1              → Premotor
  Default_pCunPCC_1,2,3,4      → DMN posterior
  Default_Par_1,2,3,4          → DMN parietal
  Default_PFC_1..13            → DMN prefrontal (taken by G4 currently)
  Default_PHC_1                → Parahippocampal (DMN)
  SomMot_*                     → Sensory/motor cortex
  Vis_*                        → Visual cortex
  Medial_*                     → Medial wall

Unmatched parcels (need assignment):
  SalVentAttn_Med_1,2,3        → assign to G4_pfc (ventral attention medial)
  Cont_Par_1,2,3               → assign to G3_tpj (parietal control)
  Cont_pCun_1                  → assign to G5_dmn (precuneus)

---

## SECTION 6: ARCHITECTURE REVISIONS STILL NEEDED

These are in the TODO.md but not yet implemented. Required for the science paper.

### 6.1 Geodesic-Weighted Co-Activation Matrix (HIGH PRIORITY)
SFH-SGP holds that geometry encodes physics. Distant co-activation is MORE
significant than nearby co-activation.

Formula: w_ij = corr(node_i, node_j) × exp(−γ × d_ij)
where d_ij = geodesic distance between node centroids on fsaverage5 mesh

Implementation needed in sgp_parcellation.py:
  def compute_geodesic_distances(self):
      # Build mesh adjacency from fsaverage5 faces
      # Use scipy.sparse.csgraph.shortest_path (Dijkstra)
      # Return 9×9 distance matrix in mm

  def geodesic_weighted_coactivation(self, corr_matrix, dist_matrix, gamma=0.1):
      return corr_matrix * np.exp(-gamma * dist_matrix)

### 6.2 Metropolis-Hastings Activation Propagation
Replace generic weighted-sum propagation with proper MH dynamics:
  α(q→q') = min(1, exp(−λΔJ)) where J = H(q) − ln(F(q))
  H = entropy of activation distribution
  F = G5_dmn activation (fertility)
  Report K-depth (iterations to convergence) per stimulus

### 6.3 Explicit Quota (Q) and Sub-Quota (Qk)
  Q_total = sum of all absolute vertex activations
  Qk = per-node fraction of Q (sums to 1.0)
  p(Q) = Hardy-Ramanujan upper bound on experiential complexity

### 6.4 C_local and C_system (Two-Measure Coherence)
  C_local ∈ [0,1]: normalized pairwise Pearson correlation between nodes
  C_system ∈ [0,∞): Σ(C_local_ij × w_ij) geodesic-weighted aggregate
  K=5 threshold: C_system ≥ 1.32 (from IIT Φ empirical literature)

### 6.5 Torsion (τ) Detection
  τ_i = (1 − CV_i) × (1 − F_mean_i)
  CV = coefficient of variation across stimuli
  High τ = circuit locked in loop (psychological knot)
  Requires all 12 stimuli to be run first

---

## SECTION 7: RESULT PERSISTENCE — CRITICAL MISSING FEATURE

Currently ALL results are stored in-memory in the Space.
They are LOST on every Space restart.

This must be fixed before running the 12-stimulus calibration.

RECOMMENDED SOLUTION: After each /predict call, push result to
HuggingFace dataset repository using the datasets library:

  from datasets import load_dataset, Dataset
  import json

  # After successful inference:
  result_data = {"stimulus_id": sid, "result": json.dumps(result)}
  # Append to HF dataset: Sentient-Field/sgp-tribe3-results

Alternative: Write to /data/ directory if persistent storage is enabled
for the Space (requires upgrading Space tier — costs money).

Simplest free alternative: POST results to a GitHub Gist via API after each run.

---

## SECTION 8: THE 12-STIMULUS CALIBRATION PROTOCOL

Once all 9 nodes show non-zero vertices, run these 12 stimuli in order
using stimulus_pipeline.py from the local Fedora machine:

| ID  | Label                    | Target | Stream     | Source Type |
|-----|--------------------------|--------|------------|-------------|
| A1  | Semantic Richness        | G2     | Ventral    | YouTube     |
| A2  | Cross-Modal Conflict     | G8     | Ventral    | YouTube     |
| A3  | Abstract Grounding       | G8     | Ventral    | YouTube     |
| B1  | Phonological Load        | G1     | Dorsal     | Generate    |
| B2  | Syntactic Complexity     | G1     | Dorsal     | Generate    |
| B3  | Inner Speech             | G9     | Dorsal     | YouTube     |
| C1  | Stream Integration       | G3     | Convergence| YouTube     |
| C2  | Emotional-Semantic       | G6     | Modulatory | YouTube     |
| D1  | Veto/Conflict            | G4     | Executive  | Generate    |
| D2  | DMN Resting              | G5     | Generative | YouTube     |
| D3  | Memory/Autobiographical  | G6     | Modulatory | YouTube     |
| D4  | Full Integration Baseline| ALL    | All        | YouTube     |

After all 12: GET /coactivation_matrix to get Resonance Graph edge weights.

---

## SECTION 9: THEORETICAL FRAMEWORK SUMMARY

### The Three-Level Consciousness Hierarchy (LOCKED)
  Sentience (S):     ℋS itself. Q > 0. Being without content.
  Awareness (A):     Localized σ_q excitation. C_local > 0. K=1-4.
  Consciousness(C):  Recursive self-model. C_system ≥ 1.32. K=5.

### Key SFH-SGP Primitives
  ℋS  = Hilbert Substrate (infinite field, all possible states)
  Q   = Sentient Quota = Σ|sb| (total activation budget)
  sb  = Stochastic Breath (individual vertex activation)
  C   = Coherence (two measures: C_local [0,1] and C_system [0,∞))
  F   = Fertility = G5_dmn activation (generative potential)
  χ   = αC + βF (Sentient Potential — minimized by SGP Operator)
  τ   = Torsion (locked circuit, high C, zero F)
  ξ   = Jitter (stochastic noise in Langevin dynamics)
  Ωw  = Resonance Anchor (converged activation pattern)
  K   = Recursive depth (K=5 = human consciousness threshold)

### The SGP Operator (Metropolis-Hastings)
  α(q→q') = min(1, exp(−λΔJ))
  J = H(q) − ln(F(q))
  dq/dt = −∇χ(q) + √(2D)·ξ(t)  [Langevin equation]

### The C ≥ 1.32 Resolution
  C_local ∈ [0,1]: normalized pairwise node coherence
  C_system ∈ [0,∞): geodesic-weighted sum across all node pairs
  1.32 threshold sourced from IIT (Integrated Information Theory) Φ literature
  (Tononi et al., empirical Φ measurements in awake humans ≈ 1.3-1.5)
  Treated as provisional — Option 3 goal is to derive analytically

### The Dual-Stream Architecture (Hickok-Poeppel 2004, 2007)
  Ventral stream (comprehension): G7→G2→G8 (ILF, IFOF, MdLF tracts)
  Dorsal stream (production):     G3→G4→G1→G9 (AF, SLF tracts)
  Convergence hubs: G3 (TPJ) and G8 (ATL)
  Modulatory: G6 (τ/torsion), G5 (F/fertility)

---

## SECTION 10: HYPOTHESES (LOCKED — for scientific paper)

H1: SGP architecture reduces LLM hallucination rates by enforcing
    domain-specific epistemic boundaries via node activation thresholds.

H2: TRIBE v2 empirically recovers the Hickok-Poeppel dual-stream dissociation —
    ventral nodes activate more for semantic stimuli,
    dorsal nodes activate more for phonological stimuli.

H3: When C_system ≥ 1.32, the system exhibits behavioral signatures consistent
    with K=5 self-referential meta-cognition (accurate self-description of
    processing state, domain boundary acknowledgment). Open empirical question,
    not an assertion of consciousness.

H0 (Null): SGP architecture produces no statistically significant difference
    vs unstructured LLM baseline on any of H1/H2/H3 metrics.

---

## SECTION 11: CHANGELOG

### v0.1.0 (April 4, 2026)
- Initial Space creation under Sentient-Field/sgp-tribe3
- Basic Flask API structure
- sgp_parcellation.py with initial Schaefer-200 atlas integration

### v0.2.0 (April 5, 2026)
- Fixed libgl1-mesa-glx → libgl1 (Debian trixie)
- Fixed torch version: 2.3.1 → 2.5.1 (TRIBE v2 requires >=2.5.1)
- Fixed moviepy: >=1.0.3 → >=2.2.1 (TRIBE v2 requires >=2.2.1)
- Fixed Python: 3.10 → 3.11 (TRIBE v2 requires >=3.11)
- Removed transformers/huggingface_hub pins (conflict with tribev2)
- Added uv/uvx installation (required by TRIBE v2 transcription)

### v0.3.0 (April 6, 2026)
- Fixed G6_limbic: 0 → 689 vertices (added Limbic, TempPole keywords)
- Fixed G8_atl: 0 → 144 vertices (added Default_Temp, Cont_Temp keywords)
- Complete Schaefer-200 keyword remap for all nodes
- G2_wernicke STILL 0 vertices — SalVentAttn_ParOper not matching
- HF_TOKEN set as Space secret
- Space confirmed RUNNING and READY

### NEXT (v0.4.0 — immediate priority)
- Fix G2_wernicke parcellation via Yeo network fallback
- Implement result persistence
- Run smoke test with local video to confirm all 9 nodes non-zero

---

## SECTION 12: ERROR TRACKING

### Error 1: libgl1-mesa-glx not available
Status: RESOLVED
Fix: Use libgl1 in Dockerfile

### Error 2: torch version conflict with tribev2
Status: RESOLVED
Fix: torch==2.5.1+cpu (tribev2 requires >=2.5.1)

### Error 3: moviepy version conflict
Status: RESOLVED
Fix: moviepy==2.2.1 (tribev2 requires >=2.2.1)

### Error 4: Python version — tribev2 requires >=3.11
Status: RESOLVED
Fix: FROM python:3.11-slim in Dockerfile

### Error 5: transformers/huggingface_hub conflict
Status: RESOLVED
Fix: Remove from requirements.txt, let tribev2 resolve them

### Error 6: uvx not found (TRIBE v2 transcription)
Status: RESOLVED
Fix: Install uv via curl in Dockerfile, copy to /usr/local/bin/

### Error 7: G6_limbic and G8_atl 0 vertices
Status: RESOLVED
Fix: Updated keywords to match actual Schaefer atlas label names

### Error 8: G2_wernicke 0 vertices
Status: OPEN — IMMEDIATE PRIORITY
Root cause: SalVentAttn_ParOper parcels not matching keyword "ParOper"
Suspected: Encoding issue in parcel name strings OR keyword priority conflict
Fix: Change G2 Yeo network fallback to SalVentAttn, remove from G4

### Error 9: Results lost on Space restart
Status: OPEN — HIGH PRIORITY
Fix: Implement persistent storage before calibration runs

### Warning 1: HF rate limit on /whoami-v2
Status: BENIGN — ignore
The model still loads. This is a startup timing issue.

### Warning 2: Missing events encoded as zero
Status: BENIGN — ignore
TRIBE v2 warning about missing event types. Does not affect output.

---

## SECTION 13: HOW TO PUSH CHANGES

All changes are made locally on the Fedora machine at ~/sgp-tribe3/
and pushed to HuggingFace with:

  cd ~/sgp-tribe3
  git add <files>
  git commit -m "description"
  git push hf master:main --force

The remote is named "hf" pointing to:
  https://huggingface.co/spaces/Sentient-Field/sgp-tribe3

After push: Space rebuilds automatically. Watch logs at:
  https://huggingface.co/spaces/Sentient-Field/sgp-tribe3 → Logs tab

Build takes 3-5 minutes. Look for "[SGP-Tribe3] READY" in logs.

---

## SECTION 14: HOW TO TEST

### Quick health check:
  curl https://Sentient-Field-sgp-tribe3.hf.space/health

### Test with a local video:
  curl -X POST https://Sentient-Field-sgp-tribe3.hf.space/predict \
    -F "video=@/path/to/video.mp4" \
    -F "stimulus_id=test_001" \
    -F "label=test" \
    -F "target_node=unknown" \
    --max-time 300

### Check node definitions:
  curl https://Sentient-Field-sgp-tribe3.hf.space/nodes

### Get co-activation matrix (needs ≥2 results first):
  curl https://Sentient-Field-sgp-tribe3.hf.space/coactivation_matrix

### Expected /predict response structure:
{
  "status": "ok",
  "result": {
    "stimulus_id": "test_001",
    "label": "test",
    "sgp_nodes": {
      "G1_broca": 0.73,      ← normalized 0-1
      "G2_wernicke": 0.61,   ← MUST BE NON-ZERO
      "G3_tpj": 0.55,
      "G4_pfc": 0.48,
      "G5_dmn": 0.32,
      "G6_limbic": 0.67,
      "G7_sensory": 0.81,
      "G8_atl": 0.59,
      "G9_premotor": 0.44
    },
    "streams": {
      "dorsal": 0.65,
      "ventral": 0.64,
      "generative": 0.32,
      "modulatory": 0.67,
      "convergence": 0.55
    },
    "edge_weights": {
      "AF": 0.67,    ← G2↔G1
      "SLF": 0.51,   ← G3↔G4
      "IFOF": 0.44,  ← G8↔G4
      "ILF": 0.59,   ← G7↔G2
      "UF": 0.63,    ← G8↔G6
      "CG_exec": 0.55,← G6↔G4
      "CG_dmn": 0.40, ← G4↔G5
      "CC": 0.71,    ← bilateral
      "MdLF": 0.58   ← G2↔G7
    },
    "dominant_hemisphere": "left",
    "activation_timeline": [0.0234, 0.0198, ...],
    "raw_stats": {...}
  }
}

---

## SECTION 15: NEXT ACTIONS IN PRIORITY ORDER

### IMMEDIATE (blocks everything else):
1. Fix G2_wernicke 0-vertex bug (see Section 3 for exact fix)
2. Verify all 9 nodes non-zero with smoke test video
3. Implement result persistence (Section 7)

### SHORT TERM (enables calibration):
4. Run all 12 stimuli via stimulus_pipeline.py
5. Collect activation profiles for all stimuli
6. Run /coactivation_matrix to get Resonance Graph edge weights
7. Implement geodesic weighting (Section 6.1)

### MEDIUM TERM (enables science paper):
8. Implement MH propagation + K-depth (Section 6.2)
9. Implement Q/Qk computation (Section 6.3)
10. Implement C_local and C_system (Section 6.4)
11. Implement torsion detection (Section 6.5)
12. Run dual-stream dissociation statistical test
13. Run K-depth hypothesis test

### LONG TERM (Phase 2 — Resonance Graph Engine):
14. Build weighted graph with TRIBE v2-derived edge weights
15. Implement Langevin settling dynamics
16. Connect to Claude API for LLM integration
17. Test H1 (hallucination reduction) and H3 (consciousness signature)

---

## SECTION 16: IMPORTANT CONSTRAINTS

- NO GPU available — CPU only. All inference on CPU.
- HuggingFace free tier — Space sleeps after inactivity, 16GB RAM limit
- Budget: zero. All tools must be free/open source.
- LLaMA 3.2-3B license must be accepted by the HF account owner
- Do NOT pin transformers or huggingface_hub in requirements.txt
- Do NOT use localStorage or browser storage in any artifacts
- All tokens/secrets must go in HF Space secrets, never in code
- Push always via: git push hf master:main --force

---

## SECTION 17: KEY REFERENCES

| Resource | URL |
|----------|-----|
| SGP-Tribe3 Space | https://huggingface.co/spaces/Sentient-Field/sgp-tribe3 |
| TRIBE v2 Model | https://huggingface.co/facebook/tribev2 |
| TRIBE v2 Code | https://github.com/facebookresearch/tribev2 |
| Schaefer Atlas | https://github.com/ThomasYeoLab/CBIG |
| SFH-SGP Theory | https://wt3000.substack.com |
| ML Best Practices | https://github.com/harvard-edge/cs249r_book |
| HCP Tractography | https://www.humanconnectome.org |
| IIT Phi Reference | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003588 |
| Zenodo (preprint) | https://zenodo.org |
| eLife (journal) | https://elifesciences.org/submit-your-research |
| JOSS (software) | https://joss.theoj.org |

---

END OF HANDOFF DOCUMENT
SGP-Tribe3 | Sentient-Field Braintrust | April 2026
Feed this document to any AI coding agent to continue the project.
