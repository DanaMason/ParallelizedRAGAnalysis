"""
Download SIFT1M and GIST1M from ANN-Benchmarks (ann-benchmarks.com) and convert them to flat binary files the C programs can read directly.
Each file starts with a 32-byte little-endian header, followed by the data in row-major order:

    char    magic[4]    "VBIN"
    uint32  version     1
    uint32  dtype       1 = float32, 2 = int32
    uint32  elem_size   4
    uint64  rows
    uint64  cols

Files written to <out-dir>/<dataset>/:
    base.fbin     float32  base vectors
    query.fbin    float32  query vectors
    gt_ids.ibin   int32    100 exact nearest neighbors per query (0-based)
    gt_dist.fbin  float32  their Euclidean distances (not squared)
    SHA256SUMS, manifest.json

Usage: python tools/data_preparation.py [--datasets sift1m gist1m] [--raw-dir RAW] [--out-dir OUT] [--force]
"""

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
import urllib.error
import urllib.request
import h5py
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "http://ann-benchmarks.com"

# Datasets are in the order of name, source file, shape, query count.

DATASETS = {
    "sift1m": ("sift-128-euclidean.hdf5", (1_000_000, 128), 10_000),
    "gist1m": ("gist-960-euclidean.hdf5", (1_000_000, 960), 1_000),
}

# Output files, HDF5 dataset, and dtype code (1=float32, 2=int32)

OUTPUTS = [
    ("base.fbin", "train", 1),
    ("query.fbin", "test", 1),
    ("gt_ids.ibin", "neighbors", 2),
    ("gt_dist.fbin", "distances", 1),
]

HEADER = struct.Struct("<4sIIIQQ")
DTYPES = {1: np.dtype("<f4"), 2: np.dtype("<i4")}
CHUNK_ROWS = 50_000
SEED = 718

# print a message with a timestamp

def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)

# Compute the SHA256 (digital fingerprint) of a file in chunks to avoid loading all into memory at the same time.

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)

    return h.hexdigest()

# Download the file at url to dest, resuming from dest.part if an earlier attempt stops.

def download(url, dest):
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "data_preparation.py (CSC 718 coursework)"}

    if offset:
        headers["Range"] = f"bytes={offset}-"  # Resume from the last byte we have

    req = urllib.request.Request(url, headers=headers)

    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code != 416:                   
            raise
        part.unlink()  # 416: the server can't resume from this offset, so start over
        return download(url, dest)

    with resp:
        if resp.status != 206:      # 206 means the server accepted the resume. Any other status sends the whole file
            offset = 0

        length = resp.headers.get("Content-Length")
        total = offset + int(length) if length else None
        log(f"Downloading {url}" + (f" ({total / 1e6:,.0f} MB)" if total else ""))
        done, last = offset, time.monotonic()

        # Write the file in chunks, logging progress every 10 seconds.
        with open(part, "ab" if offset else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)

                if time.monotonic() - last > 10:
                    log(f"  {done / 1e6:,.0f} MB")
                    last = time.monotonic()

    if total is not None and done != total:
        sys.exit(f"Download stopped at {done} of {total} bytes; run again to resume.")

    part.replace(dest)

# Write output datasets to the flat binary format for C files to work with.

def write_vbin(dset, path, code):
    rows, cols = dset.shape

    with open(path, "wb") as f:
        f.write(HEADER.pack(b"VBIN", 1, code, 4, rows, cols))

        for start in range(0, rows, CHUNK_ROWS):
            block = np.ascontiguousarray(dset[start:start + CHUNK_ROWS], dtype=DTYPES[code])

            if code == 1 and not np.isfinite(block).all():
                raise ValueError(f"{path.name}: non-finite values near row {start}")
            
            f.write(block.tobytes())

# Conversely, read the flat binary dataset format, returning a memory-mapped array.

def read_vbin(path):
    with open(path, "rb") as f:
        magic, version, code, size, rows, cols = HEADER.unpack(f.read(HEADER.size))

    if magic != b"VBIN" or version != 1 or size != 4:
        raise ValueError(f"{path}: bad header")
    
    expected = HEADER.size + rows * cols * size
    
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {path.stat().st_size}")
    
    return np.memmap(path, DTYPES[code], "r", offset=HEADER.size, shape=(rows, cols))

# Verify the written files against HDF5 sources, recompute first 20 queries top-10 distances to verify, and return the worst relative error

def verify(f, out, n_queries=20):
    rng = np.random.default_rng(SEED)
    arrays = {}

    for file, key, _ in OUTPUTS:
        data = arrays[key] = read_vbin(out / file)
        for i in [0, len(data) - 1, *rng.integers(0, len(data), 100)]:
            if not np.array_equal(data[i], f[key][i].astype(data.dtype)):
                raise ValueError(f"{file}: row {i} does not match '{key}'")

    base, query, ids, dist = (arrays[k] for k in ("train", "test", "neighbors", "distances"))

    if ids.min() < 0 or ids.max() >= len(base):
        raise ValueError("ground-truth ids are out of range")
    
    worst = 0.0

    for q in range(min(n_queries, len(query))):
        d = np.linalg.norm(base[ids[q, :10]].astype(np.float64) - query[q], axis=1)
        ref = dist[q, :10]

        if np.any(np.diff(dist[q]) < 0) or not np.allclose(d, ref, rtol=1e-3, atol=1e-4):
            raise ValueError(f"ground truth does not match the data for query {q}")
        
        worst = max(worst, float(np.max(np.abs(d - ref) / np.maximum(ref, 1e-12))))

    return worst

# Prepare a dataset: download, convert, verify, and write manifest.

def prepare(name, raw_dir, out_dir, force):
    source_file, base_shape, n_queries = DATASETS[name]
    src, out = raw_dir / source_file, out_dir / name
    url = f"{BASE_URL}/{source_file}"

    if not src.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        download(url, src)

    if (out / "manifest.json").exists() and not force:
        log(f"{name}: already converted in {out}")
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)             # only for finished runs

    with h5py.File(src, "r") as f:
        metric = f.attrs.get("distance", "euclidean")
        metric = metric.decode() if isinstance(metric, bytes) else str(metric)
        shapes = {key: f[key].shape for _, key, _ in OUTPUTS}
        gt_shape = shapes["neighbors"]

        right_shape = (shapes["train"] == base_shape and shapes["test"] == (n_queries, base_shape[1]) 
                       and gt_shape[0] == n_queries and gt_shape[1] >= 10 and shapes["distances"] == gt_shape)
        
        if metric != "euclidean" or not right_shape:
            raise ValueError(f"{source_file} is not the expected dataset: metric={metric}, shapes={shapes}")

        for file, key, code in OUTPUTS:
            log(f"{name}: writing {file} {shapes[key]}")
            write_vbin(f[key], out / file, code)

        worst = verify(f, out)

    log(f"{name}: checks passed (max ground-truth distance error {worst:.1e})")

    sums = {file: sha256(out / file) for file, _, _ in OUTPUTS}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {file}\n" for file, h in sums.items()))

    manifest = {
        "dataset": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"url": url, "sha256": sha256(src)},
        "files": {file: {"source": key, "dtype": DTYPES[code].name,
                         "shape": list(shapes[key]), "sha256": sums[file]}
                  for file, key, code in OUTPUTS},
        "ground_truth_max_rel_error": worst,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "h5py": h5py.__version__, "platform": platform.platform()},
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"{name}: done -> {out}")

# Parse args and prep the data.

def main():
    p = argparse.ArgumentParser(description="Download and convert ANN-Benchmarks datasets.")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--raw-dir", type=Path, default=Path("raw"), help="where the .hdf5 files are kept")
    p.add_argument("--out-dir", type=Path, default=Path("data"), help="where the binary files go")
    p.add_argument("--force", action="store_true", help="convert again even if already done")
    args = p.parse_args()
    
    for name in args.datasets:
        prepare(name, args.raw_dir, args.out_dir, args.force)

    log("Done. After copying a dataset folder, check it with: sha256sum -c SHA256SUMS")


if __name__ == "__main__":
    main()