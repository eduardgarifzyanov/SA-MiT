"""Exercise CLI orchestration on a tiny model and temporary synthetic dataset."""

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from samit.models.samit import SAMiT


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny(**kwargs):
    return SAMiT(**kwargs, embed_dims=(8, 16, 32, 64), depths=(1, 1, 1, 1),
                 num_heads=(1, 2, 4, 8))


def test_cpu_training_resume_and_evaluation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(1)
        lines = []
        for i in range(2):
            Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).save(root / f"x{i}.png")
            mask = rng.integers(0, 4, (64, 64), dtype=np.uint8)
            mask[0, 0] = 255
            Image.fromarray(mask).save(root / f"y{i}.png")
            lines.append(f"x{i}.png\ty{i}.png\n")
        (root / "seg.tsv").write_text("".join(lines))
        (root / "ssl.tsv").write_text("x0.png\nx1.png\n")
        common = dict(data_root=str(root), image_size=64, in_channels=1, num_classes=4,
                      hog_dim=27, hog_cell_size=16, hog_num_bins=9, ignore_index=255,
                      mean=.42, std=.21, num_workers=0, use_amp=False, keep_last=2)
        stages = {}
        for stage in ("ssl", "seg"):
            stages[stage] = dict(
                train_index=str(root / f"{stage}.tsv"),
                val_index=str(root / "seg.tsv"), test_index=str(root / "seg.tsv"),
                batch_size=2, grad_accum=4, epochs=1, lr=1e-3, min_lr=1e-5,
                weight_decay=.01, warmup_steps=0, block_size=16, mask_ratio=.6,
                checkpoint_dir=str(root / stage), plot_dir=str(root / f"{stage}_plots"),
            )
        cfg_path = root / "cfg.yaml"
        cfg = dict(common=common, stages=stages)
        cfg_path.write_text(yaml.safe_dump(cfg))
        pretrain = _script("run_pretrain")
        args = SimpleNamespace(config=str(cfg_path), resume=None)
        with patch.object(pretrain, "parse_args", return_value=args), patch.object(pretrain, "SAMiT", side_effect=_tiny):
            pretrain.main()
            ssl_state = torch.load(root / "ssl/last.pt", weights_only=True)
            assert ssl_state["epoch"] == 0 and ssl_state["scheduler"]["step_count"] == 1
            assert np.isfinite(ssl_state["best_value"])
            stages["ssl"]["epochs"] = 2
            cfg_path.write_text(yaml.safe_dump(cfg))
            args.resume = str(root / "ssl/last.pt")
            pretrain.main()
            state = torch.load(root / "ssl/last.pt", weights_only=True)
            assert state["epoch"] == 1 and state["scheduler"]["step_count"] == 2
            history = json.loads((root / 'ssl_plots/history.json').read_text())
            assert history['epochs'] == [0, 1] and len(history['train_loss']) == 2
        finetune = _script("run_finetune")
        args = SimpleNamespace(config=str(cfg_path), resume=None,
                               pretrained=str(root / "ssl/best.pt"))
        with patch.object(finetune, "parse_args", return_value=args), patch.object(finetune, "SAMiT", side_effect=_tiny):
            finetune.main()
            stages["seg"]["epochs"] = 2
            cfg_path.write_text(yaml.safe_dump(cfg))
            args.resume = str(root / "seg/last.pt")
            finetune.main()
            state = torch.load(root / "seg/last.pt", weights_only=True)
            assert state["epoch"] == 1 and state["scheduler"]["step_count"] == 2
            history = json.loads((root / 'seg_plots/history.json').read_text())
            assert history['epochs'] == [0, 1] and len(history['train_loss']) == 2
        from samit.inference import eval_runner
        eval_cfg = SimpleNamespace(
            num_classes=4, image_size=64, checkpoint_dir=str(root / "seg"),
            checkpoint_path=str(root / "seg/best.pt"), mean=.42, std=.21,
            data_root=str(root), test_index=str(root / "seg.tsv"), use_tiled=False,
            output_dir=str(root / "eval"), sample_indices=[0], ignore_index=255,
            palette=[[46, 117, 182], [255, 192, 0], [192, 0, 0], [84, 130, 53]],
        )
        with patch.object(eval_runner, "SAMiT", side_effect=_tiny):
            result = eval_runner.run_evaluation(eval_cfg)
        assert sum(map(sum, result["confusion_matrix"])) == 2 * (64 * 64 - 1)
        assert (root / "eval/sample_0000_indices.png").is_file()
        pred = np.array(Image.open(root / 'eval/sample_0000_indices.png'))
        color = np.array(Image.open(root / 'eval/sample_0000_color.png'))
        assert np.array_equal(color, np.asarray(eval_cfg.palette, dtype=np.uint8)[pred])


def test_run_full_uses_configured_checkpoint():
    module = _script("run_full")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "custom/best.pt"
        checkpoint.parent.mkdir(); checkpoint.touch()
        cfg_path = root / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({"stages": {"ssl": {"checkpoint_dir": str(checkpoint.parent)}}}))
        with patch.object(module, "parse_args", return_value=SimpleNamespace(config=str(cfg_path))), patch.object(module, "run") as run:
            module.main()
        assert run.call_args_list[1].args[0][-2:] == ["--pretrained", str(checkpoint)]
