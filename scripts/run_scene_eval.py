"""Compare tile and source-image segmentation metrics on prepared val/test data.

Example from the project root:
    PYTHONPATH=. python scripts/run_scene_eval.py --config configs/default.yaml \
        --checkpoint checkpoints/v2/ablation/ssl/best.pt \
        --output results/v2/scene_eval/val_sample.json

The default samples two source images per dataset. Use --limit-per-dataset 0
for the full split. Pixels outside all prepared tiles are reported as uncovered.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import numpy as np
from PIL import Image
import torch
import yaml

from samit.data.integrity import validate_prepared_indices
from samit.data.prepare import digest_array, write_json
from samit.data.tiler import load_grayscale, load_mask
from samit.inference.metrics import metrics_from_confusion
from samit.inference.predictor import Predictor
from samit.inference.scene import scene_confusions
from samit.inference.report import export_comparison
from samit.models.samit import SAMiT
from samit.training.checkpoint import Checkpointer


def load_inventory(root, split, index_path):
    plan = json.loads((root / "manifests/plan.json").read_text(encoding="utf-8"))
    sources = {
        (source["dataset"], source["source_id"]): source
        for source in plan["sources"]
        if source["split"] == split
    }
    records = index_path.read_text(encoding="utf-8").splitlines()
    remaining = set(records)
    if len(remaining) != len(records):
        raise ValueError(f"Duplicate rows in {index_path}")
    tiles = defaultdict(list)
    with (root / "manifests/tiles.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            tile = json.loads(line)
            if tile["split"] != split:
                continue
            key = (tile["dataset"], tile["source_id"])
            row = f"{tile['image']}\t{tile['mask']}"
            if key not in sources or row not in remaining:
                raise ValueError(f"Tile manifest disagrees with split/index: {row}")
            remaining.remove(row)
            tiles[key].append(tile)
    if remaining:
        raise ValueError(f"Missing tile metadata for {len(remaining)} index rows")
    if not sources:
        raise ValueError(f"No source images for split {split}")
    return sources, tiles, plan["config"]["tile"]


def tile_predictions(predictor, root, tiles, tile_size):
    for tile in tiles:
        with Image.open(root / tile["image"]) as image:
            pixels = np.asarray(image.convert("L"))
        if pixels.shape != (tile_size, tile_size):
            raise ValueError(f"Unexpected tile dimensions: {tile['image']}")
        _, probs = predictor.predict(pixels, return_probs=True)
        yield tile["y"], tile["x"], probs


def summarize(results):
    tile_cms, scene_cms, counts = zip(*results)
    total = {key: sum(count[key] for count in counts) for key in counts[0]}
    total["uncovered_labeled_pixels"] = (
        total["labeled_pixels"] - total["covered_labeled_pixels"]
    )
    return {
        "sources": len(results),
        **total,
        "tile": metrics_from_confusion(np.sum(tile_cms, axis=0)),
        "scene": metrics_from_confusion(np.sum(scene_cms, axis=0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit-per-dataset", type=int, default=2)
    parser.add_argument(
        "--examples-per-dataset",
        type=int,
        default=0,
        help="Save this many source-image comparison panels per dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.limit_per_dataset, args.examples_per_dataset) < 0 or args.threads < 1:
        parser.error("Use a nonnegative source limit and at least one CPU thread")
    torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    common = cfg["common"]
    root = Path(common["data_root"])
    index_path = Path(cfg["stages"]["seg"][f"{args.split}_index"])
    ready = validate_prepared_indices(root, [index_path])
    sources, tiles, tile_size = load_inventory(root, args.split, index_path)
    if tile_size != common["image_size"]:
        raise ValueError("Prepared tile size does not match the model configuration")
    num_classes, ignore_index = common["num_classes"], common["ignore_index"]
    model = SAMiT(
        in_channels=common["in_channels"],
        num_classes=num_classes,
        hog_dim=common["hog_dim"],
    )
    epoch = Checkpointer(str(args.output.parent)).load(args.checkpoint, model)
    predictor = Predictor(
        model,
        device=args.device,
        mean=common["mean"],
        std=common["std"],
        use_amp=common.get("use_amp", True),
    )
    rng = random.Random(args.seed)
    by_dataset, per_source = {}, []
    all_results = []
    examples_dir = args.output.parent / f"{args.output.stem}_examples"
    palette = common.get("palette")
    for dataset in sorted({key[0] for key in sources}):
        keys = sorted(key for key in sources if key[0] == dataset)
        if 0 < args.limit_per_dataset < len(keys):
            keys = sorted(rng.sample(keys, args.limit_per_dataset))
        results = []
        for position, key in enumerate(keys):
            source = sources[key]
            target = load_mask(Path(source["mask"]), source["mask_format"])
            if list(target.shape) != source["shape"] or (
                digest_array(target) != source["mask_hash"]
            ):
                raise ValueError(f"Source mask differs from preparation: {key}")
            include_example = position < args.examples_per_dataset
            evaluated = scene_confusions(
                target,
                tile_predictions(predictor, root, tiles[key], tile_size),
                tile_size,
                num_classes,
                ignore_index,
                include_prediction=include_example,
            )
            example = {}
            if include_example:
                result, prediction, coverage = evaluated
                image = load_grayscale(Path(source["image"]))
                if (
                    image.shape != target.shape
                    or digest_array(image) != source["image_hash"]
                ):
                    raise ValueError(f"Source image differs from preparation: {key}")
                examples_dir.mkdir(parents=True, exist_ok=True)
                preview = examples_dir / f"{dataset}__{key[1]}_comparison.png"
                with TemporaryDirectory(dir=examples_dir) as temporary:
                    staging = Path(temporary) / preview.name
                    export_comparison(
                        image.astype(np.float32) / 255,
                        target,
                        prediction,
                        palette,
                        ignore_index,
                        staging,
                        coverage=coverage,
                    )
                    staging.replace(preview)
                example["comparison"] = preview.relative_to(
                    args.output.parent
                ).as_posix()
            else:
                result = evaluated
            results.append(result)
            per_source.append(
                {
                    "dataset": dataset,
                    "source_id": key[1],
                    **summarize([result]),
                    **example,
                }
            )
            print(f"{dataset}/{key[1]}: {result[2]['tiles']} tiles", flush=True)
        by_dataset[dataset] = summarize(results)
        all_results.extend(results)
    report = {
        "split": args.split,
        "seed": args.seed,
        "limit_per_dataset": args.limit_per_dataset,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": epoch,
        "index_sha256": ready["index_sha256"][index_path.name],
        "class_names": common.get("class_names"),
        "overall": summarize(all_results),
        "by_dataset": by_dataset,
        "per_source": per_source,
    }
    write_json(args.output, report)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
