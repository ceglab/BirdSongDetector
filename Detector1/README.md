# Birdsong Detection with Xeno‑canto Data

This repository demonstrates a **complete, end‑to‑end pipeline** to download training audio from [Xeno‑canto](https://xeno-canto.org/), train a lightweight classifier, and run **sliding‑window species detection** on your own field recordings.

It ships with:

- `xc_fetch.py` — one‑shot downloader for Xeno‑canto with sensible filters, plus a ready‑to‑train **mapping file**.
- `train_birdsong_classifier.py` — trains a Random Forest on simple spectral features (ZCR, spectral stats, MFCCs).
- `predict_birdsong.py` — runs windowed inference over WAV/MP3 and writes **window‑level** and **per‑file aggregated** predictions.
- Sample artifacts: `birdsong_rf.joblib` (trained model), `predictions.csv`, `aggregate.csv`.

> The design favors **robustness and simplicity** (few dependencies, CPU‑friendly) while handling messy real‑world MP3s via an `ffmpeg` fallback.

---

## 🔎 What the pipeline does (at a glance)

1. **Collect training audio** from Xeno‑canto for a set of target species (default: 23 common Indian species, customizable).
2. **Prepare labels**: a simple 2‑column TSV mapping `<path/to/audio>  <species_name>`.
3. **Train a classifier** on fixed‑length feature vectors per file (trim/pad to a constant duration; defaults to 15s).
4. **Predict on new audio** with a sliding window (default: 5s window / 2.5s hop), producing:
   - `predictions.csv` → one row per window: file, start/end (s), species, confidence.
   - `aggregate.csv` → **majority vote** per file with tie‑break by mean confidence.

---

## 📦 Installation

**Python:** 3.9–3.12 recommended.  
**System:** `ffmpeg` highly recommended (for robust MP3 decoding).

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install Python dependencies
pip install --upgrade pip
pip install numpy pandas joblib scikit-learn librosa soundfile requests

# 3) Install ffmpeg (strongly recommended)
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y ffmpeg
# macOS (Homebrew):
brew install ffmpeg
# Windows (chocolatey):
choco install ffmpeg
```

> Why `ffmpeg`? Some MP3s trigger mpg123/audioread quirks. The training script will **transcode via ffmpeg to WAV** when needed.

---

## 🗂️ Repository layout

```
xc_fetch.py                     # Download XC audio + make mapping + metadata
train_birdsong_classifier.py    # Train RF on spectral features
predict_birdsong.py             # Sliding-window inference + aggregation
birdsong_rf.joblib              # Example trained model bundle (joblib)
predictions.csv                 # Example window-level predictions
aggregate.csv                   # Example per-file majority vote
```

---

## 🚀 Quick start (use the provided model)

You already have `birdsong_rf.joblib` in this folder. Try it on a test file:

```bash
# Single file
python predict_birdsong.py \
  --model birdsong_rf.joblib \
  --audio path/to/field_recording.wav \
  --window_seconds 5 --hop_seconds 2.5 \
  --out predictions.csv --aggregate aggregate.csv

# Or a list of files (one path per line)
python predict_birdsong.py \
  --model birdsong_rf.joblib \
  --list audio_paths.txt \
  --out predictions.csv --aggregate aggregate.csv
```

**Outputs**
- `predictions.csv` columns: `file,start_s,end_s,species,confidence`
- `aggregate.csv` columns: `file,predicted_species,windows,mean_confidence`

> The `confidence` column is the model’s predicted probability for the winning class for each window (Random Forest `predict_proba`). If a model without probabilities is used, `confidence` will be `NaN`.

---

## 📥 Step 1 — Download training data from Xeno‑canto

`xc_fetch.py` queries the Xeno‑canto v2 API with **species, country, quality (A/B/…), and type (song/call)** filters, saves MP3s, and writes:

- `train_mapping.tsv` → two columns: `<abs_path_to_audio>  <Species_With_Underscores>`
- `xc_metadata.csv` → useful metadata (id, date, location, quality, license, etc.)

### Common commands

```bash
# Fast start: India, quality A/B, songs & calls, up to 40 recordings per species
python xc_fetch.py --out xc_data --per_species 40

# India, only quality A, only "song", 25 per species
python xc_fetch.py --out data_india_A_song --per_species 25 --quality A --type song

# Worldwide (no country filter), A/B/C quality, 60 per species
python xc_fetch.py --out data_global --country "" --quality A,B,C --per_species 60

# Use your own species list (one name per line); falls back to built‑in list otherwise
python xc_fetch.py --species_file my_species.txt --out data_custom
```

The script prints the **XC query** for each species (e.g., `gen:Pavo sp:cristatus cnt:India (q:A OR q:B) (type:song OR type:call)`), downloads files to `--out/<Species_With_Underscores>/…mp3`, and writes a ready‑to‑use `train_mapping.tsv`.

---

## 🧠 Step 2 — Train the classifier

`train_birdsong_classifier.py` reads the mapping and trains a **RandomForestClassifier** on simple features:

- Zero‑crossing rate, spectral centroid, bandwidth, rolloff, RMS (each with mean & std)
- **13 MFCCs** (mean & std per coefficient)

It also includes **robust MP3 handling**:
- Tries `librosa.load()`.
- If the file is MP3 (or load fails), **falls back to `ffmpeg → WAV`** via `soundfile`.

### Training command

```bash
python train_birdsong_classifier.py \
  --mapping xc_data/train_mapping.tsv \
  --model birdsong_rf.joblib \
  --sr 22050 \
  --duration 15.0 \
  --test_size 0.2 \
  --random_state 7
```

**Notes**
- Each training file is **trimmed/padded** to `--duration` seconds to get fixed‑length features.
- If any class has `< 2` samples (or there’s only one class), the script **trains on all data without a split** and prints a note.
- A **validation report** (precision/recall/F1 per class) prints to the console when a stratified split is possible.
- The saved `joblib` bundle contains: the fitted model, label encoder, sample rate, and feature description.

---

## 🔮 Step 3 — Predict on new audio

Use `predict_birdsong.py` on WAV/MP3. The audio is resampled to the training SR (saved in the model bundle). Inference runs over **sliding windows** and writes CSVs.

### Examples

```bash
# Single file
python predict_birdsong.py \
  --model birdsong_rf.joblib \
  --audio path/to/file.wav \
  --window_seconds 5 --hop_seconds 2.5 \
  --out predictions.csv --aggregate aggregate.csv

# Multiple files from a list
python predict_birdsong.py \
  --model birdsong_rf.joblib \
  --list audio_paths.txt \
  --window_seconds 4 --hop_seconds 1 \
  --out preds.csv --aggregate  per_file.csv

# Window-only output (no aggregation)
python predict_birdsong.py \
  --model birdsong_rf.joblib \
  --audio file.wav \
  --out windows.csv
```

**Aggregation logic**
- For each file, species counts are tallied.
- Ties are broken by **mean confidence**.
- Output columns: `predicted_species, windows, mean_confidence`.

---

## 📊 Understanding the outputs

### `predictions.csv` (window level)
| column        | meaning |
|---|---|
| `file`        | Path to the audio file |
| `start_s`     | Window start time (seconds) |
| `end_s`       | Window end time (seconds) |
| `species`     | Predicted species (label) |
| `confidence`  | Probability of that species (if model supports `predict_proba`) |

### `aggregate.csv` (per file)
| column               | meaning |
|---|---|
| `file`               | Audio file path |
| `predicted_species`  | Majority‑vote species |
| `windows`            | Number of winning windows |
| `mean_confidence`    | Mean probability across winning windows |

---

## 🧪 Tips & best practices

- **Choose window/hop wisely**: 5s/2.5s works well for many songs; for brief calls, try 2s/1s or even 1s/0.5s.
- **Balance classes**: Use `--per_species` to even out the dataset; extreme imbalance can hurt performance.
- **Quality filters**: Favor `A`/`B` quality; include `C` only if you’re data‑starved.
- **Species naming**: The training label is derived from the **directory/species slug**; be consistent across training and inference.
- **Sampling rate**: Keep `--sr` consistent between training and prediction (it’s stored in the `joblib` bundle and enforced at load time).
- **Hardware**: Everything runs comfortably on CPU; Random Forest uses `n_jobs=-1` to utilize all cores.

---

## 🛠️ Troubleshooting

**“Giving up resync after 1024 bytes – stream is not nice…”**  
This is an `mpg123`/`audioread` MP3 issue. The training script already routes MP3s through `ffmpeg → WAV`. Ensure `ffmpeg` is installed (see Install), or run with `--force_ffmpeg_for_mp3` to always use the safer path.

**No predictions produced**  
Check that your input paths exist, and try a longer `--window_seconds` (very short files are processed once as a whole if `< 1 s`).

**Validation split error**  
If you have only one class or classes with 1 sample, stratified splitting is skipped by design; train proceeds on all data.

**Librosa warnings**  
Harmless deprecation warnings are suppressed; keep `librosa` updated if you prefer.

---

## 🧩 Creating your own mapping file (without Xeno‑canto)

If your audio is in a folder structure like:
```
data/
  Species_A/*.wav
  Species_B/*.wav
```

You can create `train_mapping.tsv` like:
```bash
# Bash example
find data -type f \( -name "*.wav" -o -name "*.mp3" \) \
  | awk -F/ '{sp=$(NF-1); gsub(" ","_",sp); print $0 "\t" sp }' \
  > train_mapping.tsv
```

Then train:
```bash
python train_birdsong_classifier.py --mapping train_mapping.tsv --model birdsong_rf.joblib
```

---

## 🔁 Reproducibility

- Random seeds are controlled by `--random_state`.
- The model, label encoder, feature dim, and sample rate are stored in `birdsong_rf.joblib`.

---

## 🧱 Extending the pipeline

- **Different classifiers**: swap the `RandomForestClassifier` for e.g. `XGBoost`, `LightGBM`, or a linear SVM (remember to keep `predict_proba` or adapt the output).
- **Richer features**: add deltas to MFCCs, spectral contrast, chroma, or log‑mel features.
- **Deep learning**: generate log‑mel spectrograms and train a CNN; keep the inferencer’s sliding‑window API the same.

---

## 📚 Command reference

### `xc_fetch.py`
```
--out DIR                Output root (default: xc_data)
--per_species N          Max recordings per species (default: 40)
--country NAME           e.g. "India"; empty "" for no filter
--quality LIST           Comma‑sep (e.g., A or A,B or A,B,C)
--type LIST              Comma‑sep (e.g., song,call); empty for any
--species_file PATH      Optional newline‑separated species list
--delay SEC              Pause between API pages (default: 0.5)
```

### `train_birdsong_classifier.py`
```
--mapping PATH           Two columns: <audio> <species>
--model PATH             Output model filename (default: birdsong_rf.joblib)
--sr INT                 Target sample rate (default: 22050)
--duration SEC           Seconds per file (default: 15.0)
--test_size FLOAT        Holdout fraction (default: 0.2)
--random_state INT       RNG seed (default: 7)
--force_ffmpeg_for_mp3   Always transcode MP3 via ffmpeg
```

### `predict_birdsong.py`
```
--model PATH             Joblib model bundle
--audio PATH             Single audio file
--list PATH              Text file with one audio path per line
--window_seconds FLOAT   Sliding window length (default: 5.0)
--hop_seconds FLOAT      Hop length (default: 2.5)
--out PATH               Window‑level CSV (default: predictions.csv)
--aggregate [PATH]       Also write per‑file majority vote
                         (default name if flag given with no arg: aggregate.csv)
```

---

## ✅ Worked example

Below is a minimal sequence to **reproduce** training and prediction:

```bash
# 1) Download training audio and build mapping
python xc_fetch.py --out xc_data --per_species 30

# 2) Train on fixed‑length features
python train_birdsong_classifier.py \
  --mapping xc_data/train_mapping.tsv \
  --model birdsong_rf.joblib \
  --duration 15 --sr 22050 --test_size 0.2

# 3) Run predictions on your recordings
python predict_birdsong.py \
  --model birdsong_rf.joblib \
  --list my_recordings.txt \
  --window_seconds 5 --hop_seconds 2.5 \
  --out predictions.csv --aggregate
```

You should now see two CSVs: `predictions.csv` (per‑window) and `aggregate.csv` (per‑file). Explore them to find time ranges where each species is active.

---

*Happy birding!* 🐦🎧
