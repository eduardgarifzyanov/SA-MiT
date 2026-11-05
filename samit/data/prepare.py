"""Prepare source-grouped segmentation splits and reuse an external SSL corpus.

Splits are frozen before tiling. Run: python -m samit.data.prepare --config ...
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import fnmatch
import json
import os
from functools import lru_cache
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import yaml

from .tiler import IMAGE_EXT, atomic_index, load_grayscale, load_mask
from .integrity import file_sha256

SPLITS = ("train", "val", "test")
TILE_NAME = re.compile(r"^(.*)_(y\d+_x\d+\.[^.]+)$")
COORD_NAME = re.compile(r"^y\d+_x\d+\.[^.]+$")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_index(path) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_dataset_readme(output, report):
    text = """# SA-MiT prepared dataset

seg/train, seg/val, seg/test each contain:
  images/<dataset>/<source_id>/y......_x.......png
  masks/<dataset>/<source_id>/y......_x.......png

indices/seg_{train,val,test}.tsv: image and mask paths relative to this root.
indices/ssl_train.tsv: absolute paths to reused external SSL tiles (no copy).
manifests/plan.json: frozen raw source assignments, groups, image/mask hashes.
manifests/tiles.jsonl: provenance and coordinates of every segmentation tile.
manifests/ssl_sources.jsonl: source-level inventory of the SSL training corpus.
manifests/audit.json and READY.json: completed preparation and index hashes.

Only the listed indices define the training/evaluation data. Do not glob the
legacy SSL directory: it may also contain files outside the SSL training set.
Raw inputs and the old dataset remain unchanged; external SSL files must stay
at their recorded paths. Restart preparation with the SAME prepare.yaml to
resume. Use a new output directory for a different split or tile parameters.

Source/group and decoded duplicate checks do not prove geographic independence
without scene metadata. See audit.json for the scope of SSL duplicate checks.
Old checkpoints and metrics must not be reused as independent-test results.

## Summary

"""
    for split, counts in report["seg"].items():
        text += f"{split}: {counts['sources']} sources, {counts['tiles']} tiles\n"
    text += f"SSL train: {report['ssl']['tiles']} external tiles\n"
    text += "Train-only normalization: " + json.dumps(report["train_normalization"]) + "\n"
    with atomic_index(Path(output) / "README.md") as f:
        f.write(text)


def digest_array(array):
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def safe_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    if not name or name in (".", ".."):
        raise ValueError(f"Invalid name: {value}")
    return name


def discover_sources(cfg):
    """Read and validate raw pairs; image fingerprints also group duplicates."""
    sources = []
    seen_paths = set()
    names = set()
    for ds in cfg["labeled"]:
        name = safe_name(ds["name"])
        if name in names:
            raise ValueError(f"Duplicate dataset name: {name}")
        names.add(name)
        image_root = Path(ds["images"]).resolve(strict=True)
        mask_root = Path(ds["masks"]).resolve(strict=True)
        fmt = ds["mask_format"]
        if fmt not in ("common", "ds1", "rgb"):
            raise ValueError("mask_format must be explicit: common, ds1 or rgb")
        group_overrides = ds.get("groups", {})
        found = sorted(p for p in image_root.glob(ds.get("pattern", "**/*"))
                       if p.is_file() and p.suffix.lower() in IMAGE_EXT
                       and not any(fnmatch.fnmatch(p.relative_to(image_root).as_posix(), pat)
                                   for pat in ds.get("exclude_patterns", [])))
        if not found:
            raise ValueError(f"No images matched dataset {name}")
        for number, p in enumerate(found, 1):
            p = p.resolve()
            if p in seen_paths:
                raise ValueError(f"Image matched more than once: {p}")
            seen_paths.add(p)
            relative = p.relative_to(image_root)
            candidates = [mask_root / relative.with_suffix(ext) for ext in sorted(IMAGE_EXT)]
            candidates = [q for q in candidates if q.is_file()]
            if len(candidates) != 1:
                raise ValueError(f"Expected one matching mask for {p}, found {len(candidates)}")
            mask_path = candidates[0].resolve()
            image = load_grayscale(p)
            mask = load_mask(mask_path, fmt)
            if image.ndim != 2 or mask.shape != image.shape:
                raise ValueError(f"Image/mask shape mismatch: {p}")
            image_hash = digest_array(image)
            source_id = safe_name(p.stem) + "_" + hashlib.sha256(str(relative).encode()).hexdigest()[:10]
            group = group_overrides.get(relative.as_posix())
            if group is None:
                regex = ds.get("group_regex")
                if regex:
                    match = re.match(regex, relative.as_posix())
                    if not match:
                        raise ValueError(f"Grouping regex did not match {relative}")
                    group = match.group(1)
                else:
                    group = relative.as_posix()
            sources.append({
                "dataset": name, "source_id": source_id, "image": str(p),
                "mask": str(mask_path), "relative_source": relative.as_posix(),
                "mask_format": fmt, "group": f"{name}:{group}",
                "image_hash": image_hash, "mask_hash": digest_array(mask),
                "shape": list(image.shape),
                "image_stat": [p.stat().st_size, p.stat().st_mtime_ns],
                "mask_stat": [mask_path.stat().st_size, mask_path.stat().st_mtime_ns],
            })
            if number % 250 == 0:
                print(f"validating {name}: {number}/{len(found)} sources", flush=True)
        print(f"validated {name}: {len(found)} source pairs", flush=True)
    return sources


def assign_splits(sources, ratios=(.7, .15, .15), seed=42):
    """Union explicit groups and identical images, then split whole components."""
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Three positive split ratios must sum to 1")
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for s in sources:
        union("group:" + s["group"], "hash:" + s["image_hash"])
    buckets = defaultdict(list)
    for s in sources:
        buckets[find("group:" + s["group"])].append(s)
    # Stratify by dataset when possible. Cross-dataset duplicates stay together.
    strata = defaultdict(list)
    for key, members in buckets.items():
        stratum = "+".join(sorted({s["dataset"] for s in members}))
        strata[stratum].append(key)
    for stratum, keys in sorted(strata.items()):
        keys.sort()
        random.Random(f"{seed}:{stratum}").shuffle(keys)
        if len(keys) < 3:
            raise ValueError(f"Need at least 3 independent groups for {stratum}; got {len(keys)}")
        n_val = max(1, round(len(keys) * ratios[1]))
        n_test = max(1, round(len(keys) * ratios[2]))
        if n_val + n_test >= len(keys):
            raise ValueError(f"Too few groups for requested ratios in {stratum}")
        for i, key in enumerate(keys):
            split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            for s in buckets[key]:
                s["split"] = split
                s["component"] = key
    validate_source_splits(sources)
    return sources


def validate_source_splits(sources):
    for field in ("group", "image_hash", "component"):
        seen = {}
        for source in sources:
            value = source[field]
            if value in seen and seen[value] != source["split"]:
                raise ValueError(f"Split leakage for {field}: {value}")
            seen[value] = source["split"]


def positions(length, tile, stride):
    length = max(tile, length)
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


def _cached_tiles_valid(tiles, output, tile_size):
    for item in tiles:
        for kind in ("image", "mask"):
            path = output / item[kind]
            if not path.is_file():
                return False
            checksum = file_sha256(path)
            recorded = item.get(kind + "_sha256")
            if recorded is not None:
                if recorded != checksum:
                    return False
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.mode != "L" or image.size != (tile_size, tile_size):
                        return False
                    if kind == "mask" and not np.isin(np.asarray(image), [0, 1, 2, 3, 255]).all():
                        return False
            except (OSError, ValueError):
                return False
            item[kind + "_sha256"] = checksum
    return True


def _tile_source(task):
    source, output, tile, stride, skip_empty = task
    output = Path(output)
    split, dataset, sid = source["split"], source["dataset"], source["source_id"]
    image_dir = Path("seg") / split / "images" / dataset / sid
    mask_dir = Path("seg") / split / "masks" / dataset / sid
    done = output / "manifests" / "completed" / dataset / f"{sid}.json"
    identity = {"source": source, "tile": tile, "stride": stride, "skip_empty": skip_empty}
    for field in ("image", "mask"):
        stat = Path(source[field]).stat()
        if [stat.st_size, stat.st_mtime_ns] != source[field + "_stat"]:
            raise ValueError(f"Raw source changed after split planning: {sid}")
    if done.exists():
        cached = json.loads(done.read_text())
        if cached["identity"] != identity:
            raise ValueError(f"Resume parameters differ: {sid}")
        if _cached_tiles_valid(cached["tiles"], output, tile):
            # Upgrade older manifests after decoding their existing files.
            write_json(done, cached)
            return cached
    image = load_grayscale(Path(source["image"]))
    mask = load_mask(Path(source["mask"]), source["mask_format"])
    if digest_array(image) != source["image_hash"] or digest_array(mask) != source["mask_hash"]:
        raise ValueError(f"Raw source changed after split planning: {sid}")
    h, w = image.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw)), mode="edge")
        mask = np.pad(mask, ((0, ph), (0, pw)), constant_values=255)
    (output / image_dir).mkdir(parents=True, exist_ok=True)
    (output / mask_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    class_pixels = np.zeros(4, dtype=np.int64)
    intensity_sum = intensity_sum_sq = 0.0
    intensity_pixels = 0
    skipped = 0
    for y in positions(h, tile, stride):
        for x in positions(w, tile, stride):
            im = image[y:y+tile, x:x+tile]
            ma = mask[y:y+tile, x:x+tile]
            if (skip_empty > 0 and float((im == 0).mean()) >= skip_empty) or np.all(ma == 255):
                skipped += 1
                continue
            name = f"y{y:06d}_x{x:06d}.png"
            ip, mp = image_dir / name, mask_dir / name
            Image.fromarray(im).save(output / ip)
            Image.fromarray(ma.astype(np.uint8)).save(output / mp)
            class_pixels += np.bincount(ma[ma != 255].ravel(), minlength=4)
            values = im.astype(np.float64) / 255.0
            intensity_sum += float(values.sum())
            intensity_sum_sq += float(np.square(values).sum())
            intensity_pixels += values.size
            rows.append({"image": ip.as_posix(), "mask": mp.as_posix(), "y": y, "x": x,
                         "image_sha256": file_sha256(output / ip),
                         "mask_sha256": file_sha256(output / mp)})
    result = {"identity": identity, "tiles": rows, "skipped": skipped,
              "class_pixels": class_pixels.tolist(),
              "intensity_sum": intensity_sum, "intensity_sum_sq": intensity_sum_sq,
              "intensity_pixels": intensity_pixels}
    write_json(done, result)
    return result


def reuse_ssl(cfg, sources, output):
    """Stream the old index; source-level exclusions happen before indexing tiles."""
    ssl = cfg.get("ssl")
    index = output / "indices" / "ssl_train.tsv"
    if not ssl:
        with atomic_index(index):
            pass
        return {"tiles": 0, "sources": 0, "excluded_tiles": 0}
    legacy_root = Path(ssl["root"]).resolve(strict=True)
    raw_root = Path(ssl["raw_images"]).resolve(strict=True)
    raw_sources = {p.stem: p for p in raw_root.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXT}
    if len(raw_sources) != sum(1 for p in raw_root.iterdir()
                              if p.is_file() and p.suffix.lower() in IMAGE_EXT):
        raise ValueError("Duplicate SSL source stems")
    heldout = {Path(s["image"]).stem for s in sources if s["split"] != "train"}
    # Also check every raw SSL image with a matching byte size. This bounds the
    # work for this multi-TB corpus while catching ordinary renamed copies.
    heldout_sizes = {Path(s["image"]).stat().st_size for s in sources if s["split"] != "train"}
    heldout_hashes = {s["image_hash"] for s in sources if s["split"] != "train"}
    excluded_sources = set(ssl.get("exclude_sources", [])) | heldout
    checked_candidates = 0
    for stem, p in raw_sources.items():
        if p.stat().st_size in heldout_sizes:
            checked_candidates += 1
            if digest_array(load_grayscale(p)) in heldout_hashes:
                excluded_sources.add(stem)
    counts = Counter()
    missing_sources = Counter()
    heldout_counts = Counter()
    missing_tiles = Counter()
    image_root = legacy_root / "images"
    flat_files = {p.name for p in os.scandir(image_root) if p.is_file()}

    @lru_cache(maxsize=8)
    def source_files(stem):
        directory = image_root / stem
        if not directory.is_dir():
            return set()
        with os.scandir(directory) as entries:
            return {entry.name for entry in entries if entry.is_file()}
    excluded = 0
    with atomic_index(index) as dst:
        with Path(ssl["index"]).open(encoding="utf-8") as old:
            for n, line in enumerate(old, 1):
                entry = line.strip()
                if not entry:
                    continue
                if "\t" in entry:
                    raise ValueError(f"SSL index must have one column, line {n}")
                rel = Path(entry)
                if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "images":
                    raise ValueError(f"Unsafe legacy SSL path: {entry}")
                if len(rel.parts) == 2:
                    match = TILE_NAME.match(rel.name)
                    if not match:
                        raise ValueError(f"Cannot identify SSL source: {entry}")
                    stem, coordinates = match.groups()
                elif len(rel.parts) == 3 and COORD_NAME.match(rel.name):
                    stem, coordinates = rel.parts[1], rel.name
                else:
                    raise ValueError(f"Cannot identify SSL source: {entry}")
                if stem not in raw_sources:
                    missing_sources[stem] += 1
                    excluded += 1
                    continue
                if stem in excluded_sources:
                    heldout_counts[stem] += 1
                    excluded += 1
                    continue
                # Older indices may predate a flat -> source-directory move.
                # Resolve actual files, validating complete directory listings.
                if coordinates in source_files(stem):
                    actual = image_root / stem / coordinates
                elif f"{stem}_{coordinates}" in flat_files:
                    actual = image_root / f"{stem}_{coordinates}"
                else:
                    missing_tiles[stem] += 1
                    excluded += 1
                    continue
                dst.write(str(actual) + "\n")
                counts[stem] += 1
                if n % 1000000 == 0:
                    print(f"SSL: indexed {n:,} entries", flush=True)
    if not counts:
        raise ValueError("No SSL training tiles remain")
    with atomic_index(output / "manifests" / "ssl_sources.jsonl") as f:
        for stem, count in sorted(counts.items()):
            f.write(json.dumps({"source": stem, "raw_image": str(raw_sources[stem]),
                                "split": "train", "tiles": count}) + "\n")
    with atomic_index(output / "manifests" / "ssl_exclusions.jsonl") as f:
        for reason, values in (("missing_raw_source", missing_sources),
                               ("missing_tile_file", missing_tiles),
                               ("heldout_or_explicit_exclusion", heldout_counts)):
            for stem, count in sorted(values.items()):
                f.write(json.dumps({"source": stem, "tiles": count, "reason": reason}) + "\n")
    return {"tiles": sum(counts.values()), "sources": len(counts),
            "excluded_tiles": excluded, "checked_hash_candidates": checked_candidates,
            "missing_raw_sources": len(missing_sources),
            "missing_raw_source_tiles": sum(missing_sources.values()),
            "missing_tile_files": sum(missing_tiles.values()),
            "all_tile_paths_verified_by_directory_listing": True,
            "excluded_source_ids": sorted(set(raw_sources) & excluded_sources),
            "duplicate_check": "source names and decoded hashes for matching raw byte sizes",
            "all_ssl_content_hashed": False, "geographic_overlap_verified": False}


def prepare(cfg, plan_only=False):
    output = Path(cfg["output"]).resolve()
    tile, stride = int(cfg.get("tile", 512)), int(cfg.get("stride", 384))
    if not 0 < stride <= tile:
        raise ValueError("Require 0 < stride <= tile")
    if not 0 <= cfg.get("skip_empty", .98) <= 1:
        raise ValueError("skip_empty must be between 0 and 1")
    if int(cfg.get("workers", 4)) < 1:
        raise ValueError("workers must be positive")
    for ds in cfg["labeled"]:
        for key in ("images", "masks"):
            raw = Path(ds[key]).resolve()
            if output == raw or raw in output.parents or output in raw.parents:
                raise ValueError("Output and raw input directories must be separate")
    if output.exists() and any(output.iterdir()) and not (output / "manifests/plan.json").is_file():
        # An interrupted first scan may leave only our empty scaffold.
        scaffold = {"manifests", "indices", ".prepare.lock"}
        has_data = any(p.is_file() and p != output / ".prepare.lock"
                       for p in output.rglob("*"))
        if has_data or any(p.name not in scaffold for p in output.iterdir()):
            raise ValueError("Refusing nonempty output without this pipeline's plan.json")
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    (output / "indices").mkdir(exist_ok=True)
    # The lock survives neither crashes nor a completed process; the kernel owns it.
    import fcntl
    with (output / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = output / "manifests/plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan["config"] != cfg:
                raise ValueError("Config differs from frozen plan; use a new output directory")
            sources = plan["sources"]
            validate_source_splits(sources)
        else:
            sources = assign_splits(discover_sources(cfg), cfg.get("ratios", [.7, .15, .15]), cfg.get("seed", 42))
            plan = {"schema_version": 1, "config": cfg, "sources": sources,
                    "geographic_overlap_verified": False}
            write_json(plan_path, plan)
        split_counts = Counter(s["split"] for s in sources)
        print("Frozen source splits:", dict(split_counts), flush=True)
        if plan_only:
            return plan
        # A failed rerun must never leave a stale readiness marker behind.
        (output / "READY.json").unlink(missing_ok=True)
        report = {"seg": {}, "source_and_group_overlap": 0,
                  "decoded_image_overlap": 0, "geographic_overlap_verified": False}
        tasks = [(s, str(output), tile, stride, cfg.get("skip_empty", .98)) for s in sources]
        results = []
        with ProcessPoolExecutor(max_workers=int(cfg.get("workers", 4))) as pool:
            for i, result in enumerate(pool.map(_tile_source, tasks), 1):
                results.append(result)
                if i % 100 == 0 or i == len(tasks):
                    print(f"SEG: {i}/{len(tasks)} sources complete", flush=True)
        with atomic_index(output / "manifests/tiles.jsonl") as metadata:
            for split in SPLITS:
                pixels = np.zeros(4, dtype=np.int64)
                count = 0
                with atomic_index(output / "indices" / f"seg_{split}.tsv") as index:
                    for source, result in zip(sources, results):
                        if source["split"] != split:
                            continue
                        pixels += result["class_pixels"]
                        for row in result["tiles"]:
                            index.write(f"{row['image']}\t{row['mask']}\n")
                            metadata.write(json.dumps(dict(row, source_id=source["source_id"],
                                dataset=source["dataset"], split=split, group=source["group"])) + "\n")
                            count += 1
                if not count:
                    raise ValueError(f"No usable segmentation tiles for {split}")
                report["seg"][split] = {"sources": split_counts[split], "tiles": count,
                                       "class_pixels": pixels.tolist()}
        train_results = [r for s, r in zip(sources, results) if s["split"] == "train"]
        n_pixels = sum(r["intensity_pixels"] for r in train_results)
        mean = sum(r["intensity_sum"] for r in train_results) / n_pixels
        variance = sum(r["intensity_sum_sq"] for r in train_results) / n_pixels - mean ** 2
        report["train_normalization"] = {"mean": mean, "std": max(variance, 1e-12) ** .5,
                                           "computed_from": "seg_train_only"}
        report["ssl"] = reuse_ssl(cfg, sources, output)
        report["index_sha256"] = {
            p.name: file_sha256(p)
            for p in sorted((output / "indices").glob("*.tsv"))
        }
        write_json(output / "manifests/audit.json", report)
        write_dataset_readme(output, report)
        # Marker is written only after both segmentation and SSL indices succeed.
        write_json(output / "READY.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prepare(cfg, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
