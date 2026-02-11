# Changelog

## [Unreleased]

### Added
- Initial Stage 1.5 pipeline scaffolding, configuration template, and CLI entrypoint (#0)
- Feature extraction modules (acoustic, ECAPA, SSL, backbone hooks) with Typer CLI (#0)
- Data manifest utilities, synthetic helpers, and speaker-disjoint splitting (#0)
- Linear probe runner with leakage/RSA/CKA metrics plus GO/NOGO decision pipeline (#0)
- Automated heatmap generation and markdown report writer (#0)
- Pytest coverage for manifests, feature store, and probe runner (#0)
- Google Colab notebook to run the full Stage 1.5 pipeline remotely (#0)
- Setuptools configuration for clean editable installs (fix pip error on Colab) (#0)
- Allow torch>=2.2 / torchaudio>=2.2 with SpeechBrain compatibility shim (#0)
- Dataset preparation docs, CLI helpers, and Colab automation for downloading + manifest creation (#0)
- Add huggingface_hub shim + upgrade to >=0.34, pin sentence-transformers<5.2, and align fsspec>=2025.3 for resolver stability (#0)
- Add critical-path tests for decision logic, backbone extraction, and end-to-end pipeline (#0)

### Changed
- Replace the SSL extractor's s3prl dependency with Hugging Face Transformers to avoid SoX requirements (#0)

### Fixed
- Colab notebook now auto-generates default texts and uses consistent paths for backbone feature extraction (#0)
- Backbone adapter now supports Qwen3-TTS checkpoints via qwen-tts registration to avoid config mapping errors (#0)
- Backbone CLI now tolerates space-delimited layer lists to avoid hook resolution errors (#0)
- Qwen3-TTS layer aliases now resolve to concrete module paths for backbone hooks (#0)
- Colab notebook now includes a Qwen3-TTS audio synthesis cell to generate synthetic WAVs for the manifest (#0)
- Fix temporal pooling to reduce over time, preventing variable-length feature vectors (#0)
- Implement text-robustness metrics and enforce them in GO/GO_CONDITIONAL decisions (#0)
- Use probe target metadata and speaker-disjoint configuration when building evaluation splits (#0)
- Make backbone/Qwen3-TTS adapter honor generation config and avoid token/text input mismatches (#0)
- Add SSL/backbone CLI options for layers, dtype, pooling, and generation controls (#0)
- Harden report generation paths and include text-robustness heatmaps/metrics (#0)
- Improve RSA computation to avoid diagonal bias and mismatched similarity/distance scales (#0)
- Validate NPZ feature keys to fail fast on inconsistent feature stores (#0)
