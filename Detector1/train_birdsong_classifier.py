#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a bird song classifier from a 2-column mapping file:
<path/to/audio> <species_name>

This version is robust to quirky MP3s:
- Tries librosa load first.
- For MP3s (or when load fails), falls back to ffmpeg -> temp WAV -> load.

It also avoids stratified split errors when some classes have <2 samples:
- If any class count < 2 or only one class present, trains on ALL data, no split.
"""

import argparse
import os
import sys
import warnings
import tempfile
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- IO helpers ---------------------------

def read_mapping(mapping_path: str) -> pd.DataFrame:
    for sep in [None, "\t", ",", " ", r"\s+"]:
        try:
            df = pd.read_csv(mapping_path, sep=sep, header=None, engine="python")
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]
                df.columns = ["audio", "species"]
                return df
        except Exception:
            continue
    raise ValueError("Could not parse mapping file. Expect 2 columns: <audio_path> <species_name>.")

def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def ffmpeg_to_wav(src_path: str, target_sr: int) -> str:
    """
    Transcode src audio to a temporary mono WAV at target_sr using ffmpeg.
    Returns path to temp wav (caller should delete).
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", str(target_sr),
        "-vn",
        tmp_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or (not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0):
        # clean up if failed
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg failed on {src_path}")
    return tmp_path

def safe_load_audio(path: str, sr: int, duration: float | None):
    """
    Try librosa load first. If file is MP3 (or librosa fails), try ffmpeg->wav fallback.
    Pads/truncates to 'duration' seconds if provided (>0).
    """
    path = str(path)
    y = None
    try:
        # librosa.load uses audioread; may fail on some mp3s
        y, _sr = librosa.load(path, sr=sr, mono=True)
    except Exception:
        # force ffmpeg fallback
        pass

    # Prefer ffmpeg for mp3 to avoid mpg123 quirks even if librosa worked?
    need_ffmpeg = (y is None) or path.lower().endswith(".mp3")

    tmp_wav = None
    if need_ffmpeg:
        if not have_ffmpeg():
            if y is None:
                raise RuntimeError("MP3 decode failed and ffmpeg not found. Install ffmpeg to handle tricky MP3s.")
        else:
            tmp_wav = ffmpeg_to_wav(path, sr)
            y, _sr = sf.read(tmp_wav, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y[:, 0]

    # duration handling
    if duration is not None and duration > 0:
        target_len = int(sr * duration)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, max(0, target_len - len(y))), mode="constant")

    # cleanup
    if tmp_wav and os.path.exists(tmp_wav):
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

    return y, sr

# --------------------------- Features ---------------------------

def extract_features(y, sr):
    """Fixed-length vector: basic spectral stats + MFCC means/stds."""
    if y is None or y.size == 0:
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

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True, help="2-column file: <audio_path> <species_name>")
    ap.add_argument("--model", default="birdsong_rf.joblib", help="Output model filename")
    ap.add_argument("--sr", type=int, default=22050, help="Target sample rate for loading audio")
    ap.add_argument("--duration", type=float, default=15.0, help="Seconds to use per training file (trim/pad)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction for sanity metrics")
    ap.add_argument("--random_state", type=int, default=7, help="Random seed")
    ap.add_argument("--force_ffmpeg_for_mp3", action="store_true",
                    help="If set, always decode .mp3 via ffmpeg even if librosa succeeds.")
    args = ap.parse_args()

    df = read_mapping(args.mapping)

    X, y_labels, bad = [], [], []
    for _, row in df.iterrows():
        path = str(row["audio"])
        label = str(row["species"])
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Not found: {path}")
            # route mp3s through ffmpeg proactively if requested
            use_ffmpeg = args.force_ffmpeg_for_mp3 and path.lower().endswith(".mp3")
            if use_ffmpeg:
                y, sr = safe_load_audio(path, sr=args.sr, duration=args.duration)  # safe_load already uses ffmpeg for mp3
            else:
                y, sr = safe_load_audio(path, sr=args.sr, duration=args.duration)
            feats = extract_features(y, sr)
            X.append(feats)
            y_labels.append(label)
        except Exception as e:
            bad.append((path, str(e)))
            continue

    if not X:
        print("No features extracted. Check your mapping and audio files.", file=sys.stderr)
        if bad:
            print("\nExamples of failures:", file=sys.stderr)
            for p, err in bad[:10]:
                print(f"  {p} -> {err}", file=sys.stderr)
        sys.exit(1)

    X = np.vstack(X)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Determine if we can safely do a stratified split
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    unique_classes = counts.shape[0]

    do_split = True
    if unique_classes < 2 or min_count < 2:
        do_split = False
        print("\n[NOTE] Not doing a validation split because either:")
        if unique_classes < 2:
            print("  - Only one class present in the data.")
        if min_count < 2:
            print(f"  - At least one class has only {min_count} sample(s). Need =2 for stratified split.")
        print("Training will use ALL available data, and no validation report will be shown.")

    if do_split:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
    else:
        Xtr, ytr = X, y
        Xte = yte = None

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=args.random_state,
        class_weight="balanced_subsample",
    )
    clf.fit(Xtr, ytr)

    if do_split:
        ypred = clf.predict(Xte)
        print("\n=== Validation Report (holdout) ===")
        print(classification_report(yte, ypred, target_names=le.classes_))
    else:
        print("\n[INFO] Trained on all data (no validation split).")

    # Save model bundle
    bundle = {
        "model": clf,
        "label_encoder": le,
        "sr": args.sr,
        "train_feature_desc": "ZCR, centroid, bandwidth, rolloff, RMS (mean/std) + MFCC(13) mean/std",
        "duration": args.duration,
        "feature_dim": X.shape[1],
    }
    joblib.dump(bundle, args.model)
    print(f"\nSaved model to: {args.model}")

    if bad:
        print("\nSome files failed to process (not fatal). First few:")
        for p, err in bad[:10]:
            print(f"  {p} -> {err}")

if __name__ == "__main__":
    main()
