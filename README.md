---
title: SGP-Tribe3
emoji: 🧠
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: cc-by-nc-4.0
---

# SGP-Tribe3

**Sentient Generative Principal — Brain Encoding Calibration System**

A fork of [TRIBE v2](https://huggingface.co/facebook/tribev2) (Meta AI) extended with:

- Full multimodal inference: video + audio + text simultaneously
- SGP 9-node parcellation using the Schaefer-200 cortical atlas
- Dual-stream (dorsal/ventral) activation separation
- White matter tract co-activation edge weight output
- Stimulus calibration matrix accumulation across sessions

## API Endpoints

### `POST /predict`
Upload a video file (mp4, max 120s). Returns SGP node activation profile.

**Request:** `multipart/form-data` with `video` field

**Response:**
```json
{
  "stimulus_id": "uuid",
  "sgp_nodes": {
    "G1_broca": 0.73,
    "G2_wernicke": 0.61,
    "G3_tpj": 0.55,
    "G4_pfc": 0.48,
    "G5_dmn": 0.32,
    "G6_limbic": 0.67,
    "G7_sensory": 0.81,
    "G8_atl": 0.59,
    "G9_premotor": 0.44
  },
  "streams": {
    "ventral_mean": 0.64,
    "dorsal_mean": 0.58
  },
  "dominant_hemisphere": "left",
  "n_timesteps": 42,
  "n_vertices": 20484
}
```

### `GET /health`
Returns model load status.

### `POST /warmup`
Triggers model loading (call once after cold start).

## Architecture

Built on the SGP-LLM framework. Nodes map to:
- **G1 Broca** — phonological production, syntactic processing
- **G2 Wernicke** — auditory comprehension, lexical-semantic decoding  
- **G3 TPJ** — stream convergence, sensorimotor interface
- **G4 PFC** — executive control, coherence/veto
- **G5 DMN** — generativity, self-referential processing
- **G6 Limbic** — emotional weighting, memory consolidation
- **G7 Sensory** — primary perceptual input encoding
- **G8 ATL** — cross-modal semantic integration hub
- **G9 Premotor** — action planning, motor speech preparation

## Citation

Built on TRIBE v2:
```
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, Stéphane and Rapin, Jérémy and Benchetrit, Yohann and others},
  year={2026}
}
```
