#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Xeno-canto one-shot downloader for a fixed species list.

- Queries the Xeno-canto v2 API with robust filters (species, country, quality, vocalization type).
- Downloads audio into per-species directories with safe filenames.
- Writes:
    1) train_mapping.tsv  ->  "<path>\t<Species_With_Underscores>"
    2) xc_metadata.csv    ->  metadata for all downloads

USAGE EXAMPLES
--------------
# Fast start (India, quality A/B, songs & calls, 40 per species):
python xc_fetch.py --out data --per_species 40

# Only India, quality A, "song" recordings, at most 25 per species:
python xc_fetch.py --out data_india_A_song --per_species 25 --quality A --type song

# From anywhere (no country filter), A/B/C quality, 60 per species:
python xc_fetch.py --out data_global --country "" --quality A,B,C --per_species 60

# If you want to load species from a text file (one name per line), use --species_file
# (the script will still fall back to the internal 23-species list if not provided):
python xc_fetch.py --species_file my_species.txt --out data_custom
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

API_BASE = "https://xeno-canto.org/api/2/recordings"

# Your species list with precise scientific names for robust queries
COMMON_TO_SCI = {
    "Indian peafowl": ("Pavo", "cristatus"),
    "Grey francolin": ("Ortygornis", "pondicerianus"),
    "Spotted dove": ("Spilopelia", "chinensis"),
    "Eastern barn owl": ("Tyto", "javanica"),     # India has javanica in many lists; adjust if you prefer T. alba s.l.
    "Plum headed parakeet": ("Psittacula", "cyanocephala"),
    "Asian green bee eater": ("Merops", "orientalis"),  # Green Bee-eater
    "Coppersmith barbet": ("Psilopogon", "haemacephalus"),
    "Indian robin": ("Copsychus", "fulicatus"),
    "Oriental magpie Robin": ("Copsychus", "saularis"),
    "Taiga flycatcher": ("Ficedula", "albicilla"),
    "Red-breasted Flycatcher": ("Ficedula", "parva"),
    "Red-vented Bulbul": ("Pycnonotus", "cafer"),
    "Large Gray Babbler": ("Argya", "malcolmi"),
    "Gray-breasted prinia": ("Prinia", "hodgsonii"),
    "Jungle Prinia": ("Prinia", "sylvatica"),
    "Ashy Prinia": ("Prinia", "socialis"),
    "Hume's Warbler": ("Phylloscopus", "humei"),
    "Common Chiffchaff": ("Phylloscopus", "collybita"),
    "Green Warbler": ("Phylloscopus", "nitidus"),
    "Greenish Warbler": ("Phylloscopus", "trochiloides"),
    "Purple Sunbird": ("Cinnyris", "asiaticus"),
    "Western yellow wagtail": ("Motacilla", "flava"),
    "Red avadavat": ("Amandava", "amandava"),
}

DEFAULT_SPECIES = list(COMMON_TO_SCI.keys())

def slugify(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s

def build_query(gen: str, sp: str, country: str, qualities: List[str], vtypes: List[str]) -> str:
    """
    Build an XC query string. Examples:
      'gen:Pavo sp:cristatus cnt:India q:A type:song'
    """
    parts = [f"gen:{gen}", f"sp:{sp}"]
    if country:
        parts.append(f"cnt:{country}")
    if qualities:
        # Multiple qualities -> OR them with parentheses
        if len(qualities) == 1:
            parts.append(f"q:{qualities[0]}")
        else:
            parts.append("(" + " OR ".join([f"q:{q}" for q in qualities]) + ")")
    if vtypes:
        if len(vtypes) == 1:
            parts.append(f"type:{vtypes[0]}")
        else:
            parts.append("(" + " OR ".join([f"type:{t}" for t in vtypes]) + ")")
    return " ".join(parts)

def fetch_all_pages(query: str, delay: float = 0.5, max_pages: int = 50) -> List[dict]:
    """Fetch all pages for a given query; return a list of recording dicts."""
    page = 1
    all_recs = []
    while page <= max_pages:
        params = {"query": query, "page": page}
        r = requests.get(API_BASE, params=params, timeout=30)
        if r.status_code != 200:
            print(f"[WARN] API status {r.status_code} on page {page} for query: {query}", file=sys.stderr)
            break
        data = r.json()
        recs = data.get("recordings", [])
        all_recs.extend(recs)
        num_pages = int(data.get("numPages", 1) or 1)
        if page >= num_pages:
            break
        page += 1
        time.sleep(delay)
    return all_recs

def pick_subset(recs: List[dict], limit: int) -> List[dict]:
    """
    Choose up to 'limit' recordings preferring:
      1) foreground 'song'/'call' via 'type' field
      2) higher 'q' alphabetical (A best)
      3) more recent (by 'date' string if present)
    """
    def score(r):
        q = r.get("q", "Z")  # A best
        t = r.get("type", "").lower()
        fg = ("song" in t) or ("call" in t)
        # invert sorting keys to prefer True, A, newer date
        return (
            0 if fg else 1,
            ord(q[0]) if q else ord("Z"),
            r.get("date", "")
        )
    sorted_recs = sorted(recs, key=score)
    return sorted_recs[:limit]

def safe_download(url: str, out_path: Path, max_retries: int = 3) -> bool:
    for i in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    time.sleep(0.75)
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 64):
                        if chunk:
                            f.write(chunk)
            return True
        except requests.RequestException:
            time.sleep(1.0 + i * 0.5)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="xc_data", help="Output root directory")
    ap.add_argument("--per_species", type=int, default=40, help="Max recordings to fetch per species")
    ap.add_argument("--country", default="India", help='Country filter for cnt:, empty string "" = no filter')
    ap.add_argument("--quality", default="A,B", help="Comma-separated quality list (e.g., A or A,B or A,B,C)")
    ap.add_argument("--type", default="song,call", help="Comma-separated types (e.g., song,call; keep empty for any)")
    ap.add_argument("--species_file", default="", help="Optional text file with species common names (one per line)")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay between API pages (seconds)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    qualities = [q.strip().upper() for q in args.quality.split(",") if q.strip()]
    vtypes = [t.strip().lower() for t in args.type.split(",") if t.strip()]

    # Load species list
    if args.species_file and Path(args.species_file).exists():
        with open(args.species_file, "r", encoding="utf-8") as fh:
            species_list = [line.strip() for line in fh if line.strip()]
    else:
        species_list = DEFAULT_SPECIES

    mapping_rows: List[Tuple[str, str]] = []
    meta_rows: List[Dict] = []

    for common in species_list:
        common_norm = common.strip()
        if common_norm not in COMMON_TO_SCI:
            print(f"[WARN] Species not in built-in map: {common_norm}. Skipping.", file=sys.stderr)
            continue

        gen, sp = COMMON_TO_SCI[common_norm]
        species_slug = slugify(common_norm)
        species_dir = out_root / species_slug

        query = build_query(gen, sp, args.country, qualities, vtypes)
        print(f"\n=== {common_norm} ({gen} {sp}) ===")
        print(f"Query: {query}")

        recs = fetch_all_pages(query, delay=args.delay)
        if not recs:
            print("[INFO] No recordings found with current filters.")
            continue

        chosen = pick_subset(recs, args.per_species)
        print(f"[INFO] Found {len(recs)} recordings, choosing {len(chosen)}.")

        for r in chosen:
            xc_id = r.get("id") or r.get("nr") or ""
            q = r.get("q", "")
            typ = r.get("type", "")
            file_url = r.get("file")  # direct download
            if not file_url:
                # Compose from id if necessary
                if xc_id:
                    file_url = f"https://xeno-canto.org/{xc_id}/download"
                else:
                    continue

            # Make a sane filename
            date = r.get("date", "")
            rec_by = r.get("rec", "") or r.get("rec_by", "")
            cnt = r.get("cnt", "")
            loc = r.get("loc", "")
            ext = ".mp3"  # XC downloads are mp3 by default
            base = f"XC{xc_id}_{q}_{slugify(typ)}_{date}_{slugify(cnt)}_{slugify(loc)}"
            fname = (species_dir / (base + ext))

            ok = safe_download(file_url, fname)
            if not ok:
                print(f"[WARN] Failed to download: {file_url}", file=sys.stderr)
                continue

            # Add mapping row
            mapping_rows.append((str(fname.resolve()), species_slug))

            # Metadata
            meta = {
                "xc_id": xc_id,
                "file_path": str(fname.resolve()),
                "common_name": common_norm,
                "genus": gen,
                "species": sp,
                "quality": q,
                "type": typ,
                "country": r.get("cnt", ""),
                "location": r.get("loc", ""),
                "latitude": r.get("lat", ""),
                "longitude": r.get("lng", ""),
                "date": date,
                "time": r.get("time", ""),
                "length": r.get("length", ""),
                "elevation": r.get("elev", ""),
                "license": r.get("lic", ""),
                "recordist": rec_by,
                "remarks": r.get("rmk", ""),
                "file_url": file_url,
                "url": r.get("url", ""),
            }
            meta_rows.append(meta)

        # be nice to the API between species
        time.sleep(1.0)

    # Write mapping TSV
    mapping_path = out_root / "train_mapping.tsv"
    with open(mapping_path, "w", encoding="utf-8") as fh:
        for audio, sp_slug in mapping_rows:
            fh.write(f"{audio}\t{sp_slug}\n")
    print(f"\n[OK] Wrote mapping file: {mapping_path} ({len(mapping_rows)} rows)")

    # Write metadata CSV
    meta_path = out_root / "xc_metadata.csv"
    if meta_rows:
        keys = list(meta_rows[0].keys())
        with open(meta_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for row in meta_rows:
                w.writerow(row)
        print(f"[OK] Wrote metadata CSV: {meta_path} ({len(meta_rows)} rows)")
    else:
        print("[INFO] No metadata rows to write.")

    print("\nAll done. You can now train with:")
    print(f"  python train_birdsong_classifier.py --mapping {mapping_path} --model birdsong_rf.joblib")
    print("and predict with your field recordings using predict_birdsong.py")

if __name__ == "__main__":
    main()
