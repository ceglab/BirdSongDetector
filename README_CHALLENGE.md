# 🐦🔊 Birdsong ID Challenge — “Beat Merlin” Edition

Welcome to the **Birdsong Identification Challenge** — a collaborative, open repository of different approaches to recognize **bird species from audio**.  
Your mission: **build a detector that can rival (or beat) Merlin** from the Cornell Lab of Ornithology in real-world conditions.

This repo hosts **multiple detectors** side-by-side, shared infrastructure for **data, training, prediction, and evaluation**, and a lightweight **leaderboard**. Bring your ideas: classic ML, modern deep learning, self-supervision, distillation — anything goes, as long as it’s reproducible.

---

## 🧭 What’s in this repo?

```
.
├── baselines/
│   ├── rf_spectral/            # RandomForest on spectral stats + MFCCs (CPU-friendly)
│   ├── cnn_mels/               # CNN on log-mel spectrograms (Keras/PyTorch)
│   ├── ast_transformer/        # Audio Spectrogram Transformer
│   ├── birdnet_embed_linear/   # BirdNET-style embeddings + linear probe
│   └── weakly_supervised_mil/  # MIL pooling for clip-level labels
├── common/
│   ├── dataio/                 # Audio I/O, resampling, robust MP3 handling, ffmpeg helpers
│   ├── features/               # MFCC, log-mel, spectrogram builders, augmentations
│   ├── eval/                   # Metrics, bootstrapping, PR curves, calibration
│   └── utils/                  # Logging, seeding, config management, progress bars
├── data/                       # Not tracked: place datasets/manifests here
├── scripts/                    # End-to-end helpers (download, split, evaluate, etc.)
├── leaderboard/                # Markdown + JSON results from standardized eval
├── environment.yml             # Conda env
├── requirements.txt            # Pip env
└── README.md                   # (this file)
```

> Each approach lives in its own folder with a **standard CLI**: `train.py`, `predict.py`, `evaluate.py`.  
> All models must accept the **repo-wide manifest format** and produce the **repo-wide prediction format** (see below).

---

## 🎯 Challenge goals

Do as well as (or better than) **Merlin** on:
1. **Accuracy** in the wild — noisy parks, overlapping species, wind, traffic.
2. **Latency** — on a laptop/CPU or a modest GPU (≤8 GB).
3. **Robustness** — variable gain, codecs, phones vs field recorders, distant calls.

We track **Top‑1/Top‑3 accuracy**, **segment mAP**, **macro‑F1**, **calibration (ECE)**, and **runtime (RTF)**.

---

## 📦 Installation

### Option A — Conda (recommended)
```bash
conda env create -f environment.yml
conda activate birdsong-challenge
```

### Option B — Pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional but recommended:
# ffmpeg improves MP3/stream robustness
# Ubuntu/Debian: sudo apt-get install -y ffmpeg
# macOS: brew install ffmpeg
# Windows: choco install ffmpeg
```

---

## 🗂️ Data & Manifests

We rely on **open datasets** (e.g., Xeno‑canto) and your field recordings.  
All training/eval code expects a **manifest CSV**:

`manifest.csv` columns:
- `filepath` — absolute or repo-relative path to audio
- `label` — species slug (e.g., `Turdus_merula`)
- `split` — one of `train|val|test` (optional for inference-only)

Example:
```csv
filepath,label,split
/data/xc/Turdus_merula/xc12345.mp3,Turdus_merula,train
/data/xc/Passer_domesticus/xc67890.mp3,Passer_domesticus,train
/data/field/park_2025_03_01.wav,?,test
```

### Get starter data
Use or adapt:
- `scripts/xc_fetch.py` — download from Xeno‑canto with quality/type filters
- `scripts/make_manifest.py` — scan folders into `manifest.csv`
- `scripts/split_by_species.py` — stratified train/val/test splits

> ⚠️ **Licensing**: Respect **Xeno‑canto** licenses (often **CC BY‑NC**). Persist license metadata and **credit recordists** in publications.

---

## 🧪 Standardized Evaluation

All approaches **must** support the repo’s evaluation script/format for fair comparison.

### Prediction format
Each model’s `predict.py` must output a CSV with **window-level** scores:

```
file,start_s,end_s,label,score
/path/audio.wav,0.0,5.0,Turdus_merula,0.91
/path/audio.wav,2.5,7.5,Passer_domesticus,0.31
...
```

- Overlapping windows allowed.
- `score` is a probability or calibrated confidence in `[0,1]`.

### Run the official evaluation
```bash
# Aggregate windows → clip-level
python common/eval/aggregate.py --preds preds.csv --method majority --out clip_preds.csv

# Compute metrics
python common/eval/metrics.py \
  --truth data/labels_test.csv \
  --preds clip_preds.csv \
  --out leaderboard/metrics.json \
  --plots leaderboard/plots
```

Metrics reported:
- **Top‑1** and **Top‑3** accuracy (clip level)
- **Macro F1**
- **Segment‑level mAP** (optional if window preds present)
- **Expected Calibration Error** (ECE)
- **Runtime**: Real‑Time Factor (RTF = audio_seconds / wall_seconds)

> Plots include PR curves, calibration curves, reliability diagrams, and confusions.

---

## 🚀 Quick Starts

### 1) Baseline: RandomForest on spectral features (CPU-only)
```bash
cd baselines/rf_spectral

# Train
python train.py \
  --manifest ../../data/manifest.csv \
  --model out/rf.joblib --sr 22050 --duration 15 --test_size 0.2

# Predict (windowed)
python predict.py \
  --model out/rf.joblib \
  --list ../../data/test_paths.txt \
  --window_seconds 5 --hop_seconds 2.5 \
  --out preds.csv

# Evaluate
python ../../common/eval/aggregate.py --preds preds.csv --out clip_preds.csv
python ../../common/eval/metrics.py --truth ../../data/labels_test.csv --preds clip_preds.csv
```

### 2) Baseline: CNN on log-mels (PyTorch)
```bash
cd baselines/cnn_mels

python train.py \
  --manifest ../../data/manifest.csv \
  --epochs 40 --batch 64 --sr 32k --mels 128 --mixup 0.2 --label_smoothing 0.1 \
  --out out/cnn_mels.pt

python predict.py \
  --checkpoint out/cnn_mels.pt \
  --list ../../data/test_paths.txt \
  --window_seconds 5 --hop_seconds 2.5 \
  --out preds.csv

python ../../common/eval/aggregate.py --preds preds.csv --out clip_preds.csv
python ../../common/eval/metrics.py --truth ../../data/labels_test.csv --preds clip_preds.csv
```

### 3) AST (Audio Spectrogram Transformer)
```bash
cd baselines/ast_transformer
python train.py --manifest ../../data/manifest.csv --epochs 30 --warmup 1 --patch 16 --sr 16k --mels 128 --out out/ast.pt
python predict.py --checkpoint out/ast.pt --list ../../data/test_paths.txt --out preds.csv
python ../../common/eval/aggregate.py --preds preds.csv --out clip_preds.csv
python ../../common/eval/metrics.py --truth ../../data/labels_test.csv --preds clip_preds.csv
```

### 4) BirdNET-style embeddings + linear probe
```bash
cd baselines/birdnet_embed_linear
python extract_embeddings.py --manifest ../../data/manifest.csv --out embeds/
python train_probe.py --embeds embeds/ --out out/linear.joblib
python predict.py --probe out/linear.joblib --list ../../data/test_paths.txt --out preds.csv
python ../../common/eval/aggregate.py --preds preds.csv --out clip_preds.csv
python ../../common/eval/metrics.py --truth ../../data/labels_test.csv --preds clip_preds.csv
```

---

## 🧯 Robustness, Tricks & Tactics

To **close the gap with Merlin**, consider:
- **Data diversity**: multiple devices, habitats, SNRs; augment with **SpecAugment**, pitch/time shift, background noise.
- **Class imbalance**: focal loss, reweighting, repeat sampling by rare species, **mixup**/**cutmix**.
- **Multi-label** windows**: use BCE loss + sigmoid; post-process with **non-maximum suppression** in time.
- **Calibration**: temperature scaling or spline calibration improves confidence quality.
- **Streaming**: causal STFT, overlap-add windows, small receptive fields; profile **RTF < 1** on CPU.

---

## 🏁 Leaderboard

Submit your metrics with a short system card:

`leaderboard/submissions/<team>_<date>.json`
```json
{
  "team": "swift_swifts",
  "model": "cnn_mels_v3",
  "commit": "abc1234",
  "dataset": "xc_india_v1",
  "metrics": {
    "top1": 0.81,
    "top3": 0.93,
    "macro_f1": 0.76,
    "segment_map": 0.58,
    "ece": 0.06,
    "rtf": 0.45
  },
  "notes": "SpecAugment + mixup; CPU-RTF measured on i5-1240P."
}
```

Then add a row to `leaderboard/README.md`:
```
| Rank | Team          | Model         | Top‑1 | Top‑3 | F1   | mAP  | ECE  | RTF  |
|-----:|---------------|---------------|------:|------:|-----:|-----:|-----:|-----:|
| 1    | swift_swifts  | cnn_mels_v3   | 0.81  | 0.93  | 0.76 | 0.58 | 0.06 | 0.45 |
```

---

## 🧩 Repo Rules

- **Reproducibility**: `requirements.txt` or `environment.yml`, fixed seeds, and a `--config` YAML or CLI flags.
- **Standard I/O**: honor the manifest and prediction formats.
- **No secret data**: all training sets must be publicly obtainable or documented.
- **Licenses**: carry audio licenses/attribution if you redistribute clips.
- **Respect wildlife**: no playback‑based data collection that might disturb birds.

---

## 📈 What “better than Merlin” means here

We’re not reverse‑engineering Merlin; we’re **benchmarking** against open data and practical constraints:
- **Top‑3 ≥ 90%** on the repo’s public test split of common species.
- **RTF ≤ 0.5** on CPU for windowed inference (5s/2.5s hop).
- **Robust to noise**: ≤5% drop when SNR decreases by 10 dB (synthetic noise protocol).
- **Low overfit**: performance holds on a **geographic shift** test set.

Hit these marks? Claim “Merlin‑class” in this repo’s context. 🏆

---

## 🤝 Contributing

- Open a PR adding a new approach under `baselines/<your_model>/`.
- Include:
  - `README.md` with training details, hyperparams, and expected metrics.
  - `train.py`, `predict.py`, `evaluate.py` matching the standard CLIs.
  - A small **smoke test**: one short clip + a tiny manifest so `pytest -q` passes.
- Follow code style (black/ruff) and add docstrings for public functions.

---

## 🧪 Smoke Tests

```bash
pytest -q
# Runs: data I/O, feature builders, a 1‑epoch overfit test for CNN,
# and format checks for predictions/eval pipelines.
```

---

## 🔐 Ethics & Attribution

- Acknowledge recordists and platforms (e.g., Xeno‑canto).
- Avoid training on recordings that violate licenses or disturb wildlife to obtain.
- Be transparent about model limitations (e.g., confusion among sibling species).

---

## 🗺️ Roadmap

- [ ] On‑device demo (TFLite / CoreML)
- [ ] Semi‑supervised learning with pseudo‑labels
- [ ] Multi‑task: **species + call type** (song/call/alarm)
- [ ] Geolocation prior (class priors by time/region)
- [ ] On‑the‑fly diarization / source separation

---

## 💬 Need help?

Open an issue, post your logs, and share:
- command line
- manifest snippet
- a few example files
- exact error messages / stack traces

Let’s build **fast, fair, and field‑ready** birdsong detectors — and give Merlin a real run for its money. 🐤⚡
