#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict bird species in audio using a trained model.

Two modes:
  1) Single audio file: --audio path.wav
  2) Multiple files listed in a text file: --list paths.txt   (one file path per line)

Sliding-window inference:
  - Window size (--window_seconds), hop size (--hop_seconds)
  - For each window, extract the same features used in training and predict
  - Outputs a CSV with [file, start_s, end_s, species, confidence]

Usage:
  python predict_birdsong.py \
    --model birdsong_rf.joblib \
    --audio field_recording.wav \
    --window_seconds 5 --hop_seconds 2.5 \
    --out windows.csv --aggregate

  # Or, with a list:
  python predict_birdsong.py \
    --model birdsong_rf.joblib \
    --list audio_list.txt \
    --out windows.csv --aggregate summary.csv
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore", category=UserWarning)

def extract_features(y, sr):
    import numpy as np
    import librosa
    # same as training
    if y.size == 0:
        return np.zeros(2 + 2 + 2 + 2 + 2 + (13 * 2), dtype=np.float32)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    feats = []
    def agg_stats(mat):
        return [float(np.mean(mat)), float(np.std(mat))]

    feats += agg_stats(zcr)
    feats += agg_stats(centroid)
    feats += agg_stats(bandwidth)
    feats += agg_stats(rolloff)
    feats += agg_stats(rms)

    for i in range(mfcc.shape[0]):
        feats.append(float(np.mean(mfcc[i])))
        feats.append(float(np.std(mfcc[i])))

    return np.array(feats, dtype=np.float32)

def predict_on_file(path, bundle, window_s, hop_s, min_len_s=1.0):
    sr_target = bundle["sr"]
    model = bundle["model"]
    le = bundle["label_encoder"]

    # Load at training SR
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    duration = len(y) / sr if sr > 0 else 0.0

    rows = []
    if duration < min_len_s:
        # short file—predict once on whole thing
        feats = extract_features(y, sr)
        probs = getattr(model, "predict_proba", None)
        if probs is not None:
            p = model.predict_proba([feats])[0]
            idx = int(np.argmax(p))
            conf = float(p[idx])
            species = le.inverse_transform([idx])[0]
        else:
            idx = int(model.predict([feats])[0])
            conf = float("nan")
            species = le.inverse_transform([idx])[0]
        rows.append((path, 0.0, duration, species, conf))
        return rows

    win = int(window_s * sr)
    hop = int(hop_s * sr)
    start = 0
    while start < len(y):
        end = min(start + win, len(y))
        yseg = y[start:end]
        if (end - start) / sr < min_len_s:
            break
        feats = extract_features(yseg, sr)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba([feats])[0]
            idx = int(np.argmax(p))
            conf = float(p[idx])
            species = le.inverse_transform([idx])[0]
        else:
            idx = int(model.predict([feats])[0])
            conf = float("nan")
            species = le.inverse_transform([idx])[0]

        rows.append((path, start / sr, end / sr, species, conf))
        if end == len(y):
            break
        start += hop

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to joblib model saved by the training script")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio", help="Single audio file to analyze")
    g.add_argument("--list", help="Text file with a list of audio file paths (one per line)")
    ap.add_argument("--window_seconds", type=float, default=5.0, help="Sliding window length in seconds")
    ap.add_argument("--hop_seconds", type=float, default=2.5, help="Hop length in seconds")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV for window-level predictions")
    ap.add_argument("--aggregate", nargs="?", const="aggregate.csv", default=None,
                    help="Also write per-file majority vote to this CSV (default name if flag is given with no arg)")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    file_list = []

    if args.audio:
        file_list = [args.audio]
    else:
        # read list file
        with open(args.list, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    file_list.append(line)

    all_rows = []
    for path in file_list:
        if not os.path.exists(path):
            print(f"Warning: not found -> {path}", file=sys.stderr)
            continue
        try:
            rows = predict_on_file(path, bundle, args.window_seconds, args.hop_seconds)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)

    if not all_rows:
        print("No predictions produced. Check inputs.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(all_rows, columns=["file", "start_s", "end_s", "species", "confidence"])
    df.to_csv(args.out, index=False)
    print(f"Wrote window-level predictions to: {args.out}")

    if args.aggregate:
        # majority vote per file (break ties by mean confidence)
        def agg_fun(sub):
            top = (
                sub.groupby("species")
                   .agg(n=("species", "size"), mean_conf=("confidence", "mean"))
                   .sort_values(["n", "mean_conf"], ascending=[False, False])
                   .iloc[0]
            )
            return pd.Series({"predicted_species": top.name, "windows": int(top["n"]), "mean_confidence": float(top["mean_conf"])})
        summary = df.groupby("file").apply(agg_fun).reset_index()
        summary.to_csv(args.aggregate, index=False)
        print(f"Wrote per-file aggregate predictions to: {args.aggregate}")

if __name__ == "__main__":
    main()
