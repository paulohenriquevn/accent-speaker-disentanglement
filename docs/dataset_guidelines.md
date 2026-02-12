# Stage 1.5 Dataset Preparation Guide

## Goal

Assemble a speech corpus that satisfies the Stage 1.5 experiment prerequisites:

- 3 Brazilian macro-regions (e.g., **NE**, **SE**, **S**) with comparable representation.
- ≥ 8 speakers per region (balanced gender, age if possible).
- ≥ 30 shared texts (all speakers read the exact same prompts).
- Speaker-disjoint train/test splits (no leakage between splits).
- Neutral reading style (no acting, shouting, whispering).
- High-quality recordings (16 kHz mono WAV, low background noise).

## Step-by-step Workflow

1. **Source recordings**
   - Prefer controlled corpora such as Falabrasil datasets, Common Voice pt-BR, ALIP, C-ORAL-BRASIL, or internal studio recordings.
   - Ensure each speaker explicitly identifies their region (self-annotation or metadata).
   - All speakers should read the same list of prompts (30–80 phonetically rich sentences).

2. **Normalize audio files**
   - Convert to mono 16 kHz WAV (use `sox`, `ffmpeg`, or `torchaudio`).
   - Trim leading/trailing silence if needed to reduce storage.
   - Store files under `data/wav/<speaker>/<text_id>.wav` (matching manifest references).

3. **Create `manifest.jsonl`**
   - Each line describes one utterance (real or synthetic).
   - Fields:
     ```json
     {
       "utt_id": "spk01_NE_t01",
       "path": "data/wav/spk01/t01.wav",
       "speaker": "spk01",
       "accent": "NE",
       "text_id": "t01",
       "source": "real"
     }
     ```
   - Use `source` to distinguish `real`, `syn_S`, `syn_rand`, etc.
   - Keep the manifest versioned (commit to repo or store externally with checksum).

4. **Split metadata**
   - Generate `data/splits/train_speakers.txt`, `data/splits/test_speakers.txt` (optional helper).
   - Stage 1.5 pipeline will perform speaker-disjoint splitting, but explicit lists help reproducibility.

5. **Validate coverage**
   - Run `python scripts/validate_manifest.py` (optional) to check:
     - all files exist and durations > 1s
     - speakers per region ≥ 8
     - each text appears for all speakers

6. **Upload to Colab / remote environment**
   - Zip `data/` (excluding large raw assets) or mount Google Drive.
   - Ensure `data/manifest.jsonl` and `data/wav/...` paths match the manifest entries exactly.

## Recommended Datasets

| Dataset | Notes | Link |
| --- | --- | --- |
| Falabrasil Speech Datasets | Aggregation of Brazilian corpora with region metadata | https://github.com/falabrasil/speech-datasets |
| Mozilla Common Voice pt-BR | Crowdsourced, includes user-declared accent; requires curation | https://commonvoice.mozilla.org/pt/datasets |
| ALIP / C-ORAL-BRASIL | Academic corpora with regional labels (license required) | Search via USP/UNICAMP repositories |
| CORAA-MUPE-ASR | 365h of spontaneous speech interviews with birth_state metadata; Hugging Face | https://huggingface.co/datasets/nilc-nlp/CORAA-MUPE-ASR |

## CORAA-MUPE-ASR Integration

The [MuPe Life Stories Dataset](https://huggingface.co/datasets/nilc-nlp/CORAA-MUPE-ASR) provides 289 life story interviews (365 hours) with regional metadata (`birth_state`).

### Key characteristics

- **Spontaneous speech** (interviews), not controlled reading. Each segment has a unique `text_id`; text-disjoint evaluation is limited compared to read-speech corpora.
- **Speaker types**: Interviewees (`speaker_type='R'`) have `birth_state` metadata; interviewers (`P/1`, `P/2`) do not. Only interviewees are included in the manifest.
- **Region mapping**: `birth_state` (27 Brazilian states) is automatically mapped to the 5 IBGE macro-regions: `N`, `NE`, `CO`, `SE`, `S`.

### Quick start

```bash
# Minimal: download and build manifest with default filters
stage1_5 dataset build-coraa

# With filters: only NE/SE/S, high quality, 2-30s segments, max 50 per speaker
stage1_5 dataset build-coraa \
  --regions "NE,SE,S" \
  --audio-quality high \
  --min-duration 2.0 \
  --max-duration 30.0 \
  --max-samples-per-speaker 50 \
  --output data/manifest.jsonl \
  --audio-dir data/wav/coraa
```

### CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--output` | `data/manifest.jsonl` | Output manifest path |
| `--audio-dir` | `data/wav/coraa` | Directory for exported WAV files |
| `--hf-dataset` | `nilc-nlp/CORAA-MUPE-ASR` | Hugging Face dataset identifier |
| `--hf-split` | `train` | Dataset split to load |
| `--regions` | all | Comma-separated macro-region codes (e.g. `NE,SE,S`) |
| `--min-duration` | none | Minimum segment duration in seconds |
| `--max-duration` | none | Maximum segment duration in seconds |
| `--audio-quality` | none | Filter by quality (`high` or `low`) |
| `--max-samples-per-speaker` | none | Cap segments per speaker |

### Update config

After building the manifest, update `config/stage1_5.yaml` to match:

```yaml
experiment:
  accents: ["NE", "SE", "S"]  # or whichever regions you selected

paths:
  manifest: data/manifest.jsonl
```

## Folder Layout Example

```
data/
  manifest.jsonl
  wav/
    spk01/
      t01.wav
      t02.wav
    spk02/
      t01.wav
  splits/
    speakers_train.txt
    speakers_test.txt
gen/
  syn_S/
  syn_rand/
```

## Tips

- **Phonetic balance**: include sentences covering all vowels, diphthongs, consonant clusters common in Portuguese.
- **Consistency**: record in similar environments to minimize noise variance across regions.
- **Documentation**: keep a spreadsheet with speaker demographics, recording conditions, consent forms, and checksum for each WAV.
- **Privacy**: anonymize speaker IDs; avoid storing PII in filenames or manifests.
