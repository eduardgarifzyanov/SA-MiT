"""Group splitting, provenance, resumability and external SSL regressions."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from samit.data.dataset import SARDataset
from samit.data.integrity import validate_prepared_indices
from samit.data.prepare import (
    assign_splits,
    discover_sources,
    prepare,
    validate_source_splits,
)


def make_config(tmp_path, size=12):
    images, masks = tmp_path / "raw/images", tmp_path / "raw/masks"
    images.mkdir(parents=True)
    masks.mkdir()
    rng = np.random.default_rng(15)
    for i in range(6):
        Image.fromarray(rng.integers(0, 256, (size, size), dtype=np.uint8)).save(
            images / f"scene{i}.png"
        )
        label = rng.integers(0, 4, (size, size), dtype=np.uint8)
        label[0, 0] = 255
        Image.fromarray(label).save(masks / f"scene{i}.png")
    return dict(
        output=str(tmp_path / "prepared"),
        tile=8,
        stride=6,
        workers=1,
        skip_empty=0,
        seed=42,
        ratios=[0.7, 0.15, 0.15],
        labeled=[
            dict(
                name="synthetic",
                images=str(images),
                masks=str(masks),
                mask_format="common",
                pattern="*.png",
            )
        ],
    )


def test_sources_are_split_before_tiling_and_resume(tmp_path):
    cfg = make_config(tmp_path)
    report = prepare(cfg)
    root = Path(cfg["output"])
    assert report["source_and_group_overlap"] == 0
    assert sum(s["sources"] for s in report["seg"].values()) == 6
    assert sum(s["tiles"] for s in report["seg"].values()) == 24
    manifest = [
        json.loads(line)
        for line in (root / "manifests/tiles.jsonl").read_text().splitlines()
    ]
    assignments = {}
    for row in manifest:
        assert f"seg/{row['split']}/images/" in row["image"]
        assert (
            row["source_id"] not in assignments
            or assignments[row["source_id"]] == row["split"]
        )
        assignments[row["source_id"]] = row["split"]
        assert Image.open(root / row["image"]).size == (8, 8)
        assert Image.open(root / row["mask"]).size == (8, 8)
    before = {p.name: p.read_bytes() for p in (root / "indices").glob("*.tsv")}
    prepare(cfg)
    assert before == {p.name: p.read_bytes() for p in (root / "indices").glob("*.tsv")}
    validate_prepared_indices(root, [root / "indices/seg_train.tsv"])
    train_pixels = np.concatenate(
        [
            np.array(Image.open(root / row["image"])).ravel().astype(np.float64) / 255
            for row in manifest
            if row["split"] == "train"
        ]
    )
    assert abs(report["train_normalization"]["mean"] - train_pixels.mean()) < 1e-12
    assert abs(report["train_normalization"]["std"] - train_pixels.std()) < 1e-12
    dataset = SARDataset(str(root), str(root / "indices/seg_train.tsv"))
    assert dataset[0]["image"].shape == (1, 8, 8)
    with (root / "indices/seg_train.tsv").open("a") as f:
        f.write("bad\tbad\n")
    with pytest.raises(ValueError, match="audited split"):
        validate_prepared_indices(root, [root / "indices/seg_train.tsv"])


def test_duplicates_and_explicit_groups_cannot_cross_splits(tmp_path):
    cfg = make_config(tmp_path)
    ds = cfg["labeled"][0]
    shutil.copyfile(
        Path(ds["images"]) / "scene0.png", Path(ds["images"]) / "duplicate.png"
    )
    shutil.copyfile(
        Path(ds["masks"]) / "scene0.png", Path(ds["masks"]) / "duplicate.png"
    )
    ds["groups"] = {"scene1.png": "same-territory", "scene2.png": "same-territory"}
    sources = discover_sources(cfg)
    a = assign_splits(copy.deepcopy(sources))
    b = assign_splits(copy.deepcopy(sources))
    assert a == b
    by_name = {Path(s["image"]).name: s for s in a}
    assert by_name["scene0.png"]["split"] == by_name["duplicate.png"]["split"]
    assert by_name["scene1.png"]["split"] == by_name["scene2.png"]["split"]
    bad = copy.deepcopy(a)
    bad[-1]["split"] = next(
        x for x in ("train", "val", "test") if x != bad[-1]["split"]
    )
    # Deliberately assign one duplicate hash to another split.
    duplicate = next(s for s in bad if Path(s["image"]).name == "duplicate.png")
    duplicate["split"] = next(
        x for x in ("train", "val", "test") if x != by_name["scene0.png"]["split"]
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_source_splits(bad)


def test_plan_is_frozen_and_raw_changes_rejected(tmp_path):
    cfg = make_config(tmp_path)
    plan = prepare(cfg, plan_only=True)
    assert not (Path(cfg["output"]) / "READY.json").exists()
    changed = copy.deepcopy(cfg)
    changed["seed"] = 21
    with pytest.raises(ValueError, match="frozen plan"):
        prepare(changed)
    p = Path(plan["sources"][0]["image"])
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Raw source changed"):
        prepare(cfg)


def test_failed_retry_invalidates_ready(tmp_path):
    cfg = make_config(tmp_path)
    prepare(cfg)
    root = Path(cfg["output"])
    raw = Path(cfg["labeled"][0]["images"]) / "scene0.png"
    raw.write_bytes(raw.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Raw source changed"):
        prepare(cfg)
    assert not (root / "READY.json").exists()


def test_small_images_are_padded_and_masks_keep_ignore(tmp_path):
    cfg = make_config(tmp_path, size=5)
    report = prepare(cfg)
    assert sum(s["tiles"] for s in report["seg"].values()) == 6
    root = Path(cfg["output"])
    first = json.loads((root / "manifests/tiles.jsonl").read_text().splitlines()[0])
    label = np.array(Image.open(root / first["mask"]))
    assert label.shape == (8, 8) and np.all(label[5:, :] == 255)


def test_same_stems_in_different_subdirectories_do_not_collide(tmp_path):
    cfg = make_config(tmp_path)
    ds = cfg["labeled"][0]
    for folder in ("a", "b"):
        (Path(ds["images"]) / folder).mkdir()
        (Path(ds["masks"]) / folder).mkdir()
        shutil.copyfile(
            Path(ds["images"]) / "scene0.png", Path(ds["images"]) / folder / "same.png"
        )
        shutil.copyfile(
            Path(ds["masks"]) / "scene0.png", Path(ds["masks"]) / folder / "same.png"
        )
    ds["pattern"] = "**/*.png"
    plan = prepare(cfg, plan_only=True)
    ids = [s["source_id"] for s in plan["sources"]]
    assert len(ids) == len(set(ids)) == 8


def test_missing_mask_and_ambiguous_formats_fail(tmp_path):
    cfg = make_config(tmp_path)
    (Path(cfg["labeled"][0]["masks"]) / "scene0.png").unlink()
    with pytest.raises(ValueError, match="matching mask"):
        prepare(cfg)


def test_ssl_reuse_excludes_heldout_and_renamed_copy(tmp_path):
    cfg = make_config(tmp_path)
    sources = assign_splits(discover_sources(cfg))
    heldout = next(s for s in sources if s["split"] == "test")
    raw = tmp_path / "ssl_raw"
    raw.mkdir()
    legacy = tmp_path / "legacy"
    (legacy / "images").mkdir(parents=True)
    heldout_stem = Path(heldout["image"]).stem
    for stem in (heldout_stem, "renamed_copy"):
        shutil.copyfile(heldout["image"], raw / f"{stem}.png")
    Image.fromarray(np.full((12, 12), 123, dtype=np.uint8)).save(
        raw / "independent.png"
    )
    (legacy / "images/independent").mkdir()
    index = legacy / "ssl.tsv"
    lines = []
    for p in raw.iterdir():
        name = p.stem + "_y00000_x00000.png"
        destination = (
            legacy / "images/independent/y00000_x00000.png"
            if p.stem == "independent"
            else legacy / "images" / name
        )
        shutil.copyfile(p, destination)
        lines.append(f"images/{name}\n")
    lines.append("images/unknown_source_y00000_x00000.png\n")
    lines.append("images/independent_y99999_x99999.png\n")
    index.write_text("".join(lines))
    cfg["ssl"] = dict(root=str(legacy), index=str(index), raw_images=str(raw))
    report = prepare(cfg)
    assert report["ssl"]["tiles"] == 1
    assert report["ssl"]["excluded_tiles"] == 4
    assert report["ssl"]["missing_raw_source_tiles"] == 1
    assert report["ssl"]["missing_tile_files"] == 1
    target = Path(cfg["output"]) / "indices/ssl_train.tsv"
    assert "independent" in target.read_text()
    assert "renamed_copy" not in target.read_text()
    assert "unknown_source" not in target.read_text()
    assert Path(target.read_text().strip()).is_file()
    assert Path(target.read_text().strip()).parent.name == "independent"


@pytest.mark.parametrize("ratios", [[0.8, 0.2, 0.1], [1, 0, 0], [-1, 1, 1]])
def test_invalid_ratios_rejected(ratios):
    with pytest.raises(ValueError):
        assign_splits([], ratios)


def test_output_cannot_overlap_inputs(tmp_path):
    cfg = make_config(tmp_path)
    cfg["output"] = cfg["labeled"][0]["images"]
    with pytest.raises(ValueError, match="separate"):
        prepare(cfg)


def test_training_guard_requires_ready_marker(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        validate_prepared_indices(tmp_path, [])


def test_interrupted_initial_scan_can_restart(tmp_path):
    cfg = make_config(tmp_path)
    root = Path(cfg["output"])
    (root / "manifests").mkdir(parents=True)
    (root / "indices").mkdir()
    (root / ".prepare.lock").touch()
    prepare(cfg, plan_only=True)
    assert (root / "manifests/plan.json").is_file()


def test_logging_creates_nested_directory(tmp_path):
    from samit.utils.logging import get_logger

    name = "samit.prepare-test"
    logger = get_logger(name, log_file=str(tmp_path / "logs/v2/train.log"))
    try:
        logger.info("prepared")
        assert "prepared" in (tmp_path / "logs/v2/train.log").read_text()
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
