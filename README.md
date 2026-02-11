# Stage 1.5 — Latent Separability Audit

## Accent × Speaker Disentanglement in TTS Backbone

---

## 🎯 Purpose

This repository implements **Stage 1.5** of the Accent Control Project.

Before training any LoRA or modifying the TTS backbone, we must answer one critical question:

> Does the frozen backbone already contain separable latent representations for Accent and Speaker?

If the answer is **no**, Stage 2 will fail regardless of engineering effort.

This audit prevents wasted training cycles.

---

## 🧠 What We Are Testing

We want to model:

```
Audio = TTS(text, S, A)
```

Where:

* `S` = speaker identity
* `A` = regional accent

For this to be feasible, the backbone must:

1. Represent accent in some internal layer.
2. Represent speaker identity.
3. Not collapse both into the same inseparable subspace.

This repository tests exactly that.

---

## 🔬 Experiment Overview

We measure separability at three levels:

1. **Audio Real (Ground Truth)**
2. **SSL Upstream Models (HuBERT / WavLM via S3PRL)**
3. **Frozen TTS Backbone (Internal Layers)**

For each level, we run:

* Accent classification (speaker-disjoint split)
* Speaker classification
* Leakage tests (cross-predictability)

---

## 📁 Repository Structure

```
stage1_5/
  data/
    manifest.jsonl
    splits/
  gen/
    synthetic_audio/
  feats/
    acoustic/
    ecapa/
    ssl/
    backbone_internal/
  probes/
  analysis/
  report/
```

---

## 📦 Dataset Requirements

Minimum specification:

* 3 Brazilian macro-regions
* ≥ 8 speakers per region
* ≥ 30 shared sentences
* Speaker-disjoint split required
* Neutral speaking style

Each entry in `manifest.jsonl`:

```json
{
  "utt_id": "spk03_NE_t17",
  "path": "wav/spk03/...",
  "speaker": "spk03",
  "accent": "NE",
  "text_id": "t17"
}
```

---

## 🧪 Experiments

### 1️⃣ Real Audio Baseline

Extract:

* MFCC statistics
* F0 statistics
* Speaking rate
* ECAPA / x-vector embeddings

Train linear probes:

* Accent prediction (speaker-disjoint)
* Speaker prediction

Goal: ensure the dataset itself is separable.

---

### 2️⃣ SSL Representation Audit

Using S3PRL:

* Extract HuBERT/WavLM features
* Run probes per layer
* Produce layer × F1 heatmap

Goal: establish an upper-bound separability baseline.

---

### 3️⃣ Frozen Backbone Audit

Steps:

1. Generate synthetic audio from frozen backbone.
2. Register hooks to capture internal representations.
3. Extract pooled embeddings per layer.
4. Train probes per layer.

Goal: determine where (if anywhere) accent appears in the backbone.

---

### 4️⃣ Leakage Tests

For each candidate representation:

* Predict speaker using accent-targeted features.
* Predict accent using speaker-targeted features.

Leakage must be close to chance.

---

## 📊 Metrics

### Accent Separability

* F1-macro
* Speaker-disjoint split

### Speaker Separability

* Top-1 accuracy

### Leakage

* Must be ≤ chance + 7 percentage points

### Text Robustness

* Train on subset of texts
* Test on different texts
* F1 drop ≤ 10pp

---

## ✅ Decision Rules

### GO (Strong)

There exists a layer where:

* Accent F1 ≥ 0.55
* Leakage low (≤ chance + 7pp)
* Robust to text variation

→ Proceed to Stage 2.

---

### GO (Conditional)

* Accent F1 ≥ 0.45
* Moderate leakage

→ Proceed with adversarial regularization in Stage 2.

---

### NOGO

* Accent F1 < 0.40 everywhere
* SSL baseline also weak

→ Pivot (dataset/backbone/accent definition).

---

## 🧾 Deliverables

The audit is complete when:

* Heatmaps are generated
* Leakage is measured
* Recommended intervention layer is identified
* Formal GO/NOGO decision is documented in `/report`

---

## ⏱ Estimated Timeline

* Setup: 3–5 days
* Feature extraction: 2 days
* Probing + analysis: 3–5 days
* Reporting: 2 days

Total: ~2 weeks

---

## 🚨 Why This Matters

If accent is not statistically separable in the frozen backbone:

* LoRA will not create clean control.
* Identity preservation will collapse.
* Stage 2 will produce cosmetic variation only.

This audit ensures scientific validity before training.

---

## 🧠 Core Insight

We are not testing "does it sound different?"

We are testing:

> Does the model internally represent Accent as a controllable factor separate from Identity?

Only if the answer is yes do we move forward.

---

## 🚀 Quickstart

1. **Install deps**

   ```bash
   pip install -e .[dev]
   ```

2. **Prepare manifest** — follow `data/manifest.example.jsonl` and place audio under `data/wav/...`.

3. **Extract features** (examples):

   ```bash
   stage1_5 features acoustic data/manifest.jsonl artifacts/features/acoustic
   stage1_5 features ecapa data/manifest.jsonl artifacts/features/ecapa --device cuda:0
   stage1_5 features ssl data/manifest.jsonl artifacts/features/ssl --model wavlm_large
   stage1_5 features backbone data/manifest_syn.jsonl data/texts.json artifacts/features/backbone --checkpoint your/backbone --layers encoder_out block_08 pre_vocoder
   ```

4. **Run the audit**

   ```bash
   stage1_5 run config/stage1_5.yaml
   ```

The pipeline will produce:

- `artifacts/analysis/metrics.csv`
- Heatmaps under `artifacts/analysis/figures/`
- A markdown report at `report/stage1_5_report.md` with GO/NOGO status.

### Run on Google Colab

- Notebook: `notebooks/stage1_5_colab.ipynb`
- Upload/run in Colab → set `REPO_URL`, ensure `data/manifest.jsonl` + audio exist, execute the cells sequentially.
- Outputs land under `artifacts/` and `report/`; sync them back to Drive if needed.
- Use the dataset section inside the notebook to automatically download an archive, extract WAVs, and build `data/manifest.jsonl` via `stage1_5 dataset ...` commands (see `docs/dataset_guidelines.md`).

### Dataset Prep Tools

- Documentation: `docs/dataset_guidelines.md`
- Download archive: `stage1_5 dataset download --url <https://...zip> --output-dir data/external`
- Build manifest from CSV metadata: `stage1_5 dataset build-manifest metadata.csv --audio-root data/wav --output data/manifest.jsonl`

## 🧰 Outputs & Diagnostics

- **Linear probes** for accent (F1-macro) and speaker (accuracy) per representation
- **Leakage metrics** (accent→speaker and speaker→accent)
- **RSA / CKA** scores correlating layers with accent/speaker similarity
- **Heatmaps** summarizing separability + leakage
- **Automated report** highlighting best insertion layer for LoRA/adapters
