import json
import random
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
import yaml

from samit.data.tiler import load_mask

repo = Path("/hdd_pool/egarifyzanov/SA-MiT")
cfg = yaml.safe_load((repo / "configs/default.yaml").read_text())
root = Path(cfg["common"]["data_root"])
out = repo / "results/v2/data_review"
out.mkdir(parents=True, exist_ok=True)
audit = json.loads((root / "manifests/audit.json").read_text())
plan = json.loads((root / "manifests/plan.json").read_text())
palette = np.array(cfg["common"]["palette"], dtype=np.uint8)
report = {
    "scope": (
        "Exact class totals from preparation audit; "
        "24 decoded tiles per split, balanced by dataset; "
        "6 full raw mask checks per dataset. "
        "Not an exhaustive semantic annotation review."
    ),
    "splits": {},
    "errors": [],
    "raw_checks": [],
}
rng = random.Random(42)
for split in ["train", "val", "test"]:
    buckets = defaultdict(list)
    rows = (root / f"indices/seg_{split}.tsv").read_text().splitlines()
    for row in rows:
        image, mask = row.split("\t")
        buckets[Path(image).parts[3]].append((image, mask))
    selected = []
    for dataset, pairs in sorted(buckets.items()):
        by_source = defaultdict(list)
        for pair in pairs:
            by_source[Path(pair[0]).parent.name].append(pair)
        sources = rng.sample(sorted(by_source), min(12, len(by_source)))
        selected.extend(rng.choice(by_source[s]) for s in sources)
    records = []
    for page_start in range(0, len(selected), 6):
        page = selected[page_start : page_start + 6]
        sheet = Image.new("RGB", (3 * 256, len(page) * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (image_path, mask_path) in enumerate(page):
            with Image.open(root / image_path) as im:
                image = np.array(im)
            with Image.open(root / mask_path) as im:
                mask = np.array(im)
            invalid = np.setdiff1d(np.unique(mask), [0, 1, 2, 3, 255]).tolist()
            if image.shape != (512, 512) or mask.shape != image.shape or invalid:
                report["errors"].append({"image": image_path, "invalid": invalid})
            color = np.full((*mask.shape, 3), 255, dtype=np.uint8)
            for cls in range(4):
                color[mask == cls] = palette[cls]
            rgb = np.repeat(image[:, :, None], 3, axis=2)
            overlay = rgb.copy()
            valid = mask != 255
            overlay[valid] = (0.55 * rgb[valid] + 0.45 * color[valid]).astype(np.uint8)
            for col, arr in enumerate([rgb, color, overlay]):
                sheet.paste(
                    Image.fromarray(arr).resize((256, 256), Image.Resampling.NEAREST),
                    (col * 256, k * 280 + 24),
                )
            label = (
                f"{page_start+k}: {Path(image_path).parent.parent.name}/"
                f"{Path(image_path).parent.name}/{Path(image_path).name}"
            )
            draw.text((4, k * 280 + 5), label, fill="black")
            counts = np.bincount(mask.ravel(), minlength=256)
            records.append(
                {
                    "image": image_path,
                    "class_pixels": counts[:4].tolist(),
                    "ignore_pixels": int(counts[255]),
                }
            )
            time.sleep(0.03)
        sheet.save(out / f"{split}-{page_start//6+1}.jpg", quality=90)
    counts = audit["seg"][split]["class_pixels"]
    total = sum(counts)
    report["splits"][split] = {
        "tiles": len(rows),
        "class_pixels": counts,
        "valid_pixel_percent": [round(100 * x / total, 3) for x in counts],
        "ignore_percent": round(100 * (1 - total / (len(rows) * 512 * 512)), 3),
        "sampled_tiles": records,
    }

# Check actual masks against the mapping and tile crop recorded in the frozen plan.

for dataset in ["openearthmap_sar", "fusar_map"]:
    candidates = [s for s in plan["sources"] if s["dataset"] == dataset]
    for s in rng.sample(candidates, min(6, len(candidates))):
        raw = load_mask(Path(s["mask"]), s["mask_format"])
        tile_dir = root / f"seg/{s['split']}/masks/{dataset}/{s['source_id']}"
        paths = sorted(tile_dir.glob("*.png"))
        if not paths:
            continue
        p = paths[0]
        y, x = [int(t[1:]) for t in p.stem.split("_")]
        crop = raw[y : y + 512, x : x + 512]
        expected = np.full((512, 512), 255, dtype=np.uint8)
        expected[: crop.shape[0], : crop.shape[1]] = crop
        with Image.open(p) as im:
            actual = np.array(im)
        ok = np.array_equal(actual, expected)
        report["raw_checks"].append(
            {
                "source": s["source_id"],
                "dataset": dataset,
                "tile": str(p.relative_to(root)),
                "matches": ok,
            }
        )
        if not ok:
            report["errors"].append({"crop_mismatch": str(p)})
        time.sleep(0.05)

(out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(
    json.dumps(
        {
            k: {kk: vv for kk, vv in v.items() if kk != "sampled_tiles"}
            for k, v in report["splits"].items()
        },
        indent=2,
    )
)
print("Errors:", report["errors"], "Raw checks:", len(report["raw_checks"]))
print(out)
