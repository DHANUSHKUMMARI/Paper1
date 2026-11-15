"""Train EviGround (Milestone 3 plan, step 3).

    python scripts/train.py --config base --profile unified --seed 42            # resume auto
    python scripts/train.py --config base --seed 42 --fresh                       # start over
    python scripts/train.py --config base --dry-run                               # build only
    python scripts/train.py --config base --overfit                               # CP2 overfit
    python scripts/train.py --config base model.semantic.revision=<hash>          # overrides
    python scripts/train.py --config base --stage-local                           # local copy

Run folder: ``<Paper 1>/runs/<experiment>_<profile>_seed<seed>/``. Shards: ``<Paper 1>/cache/
shards`` (``--shards`` to override). ``--fresh`` moves an existing run folder aside (never
deletes it). ``--kill-at-step N`` ends the process abruptly right after optimizer step N (the
CP2 kill-and-resume check).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eviground.config import load_config  # noqa: E402
from eviground.env import get_paths  # noqa: E402


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="base")
    ap.add_argument("--profile", choices=("unified", "detection"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--resume", default="auto", help="auto, none, or a checkpoint path")
    ap.add_argument("--fresh", action="store_true", help="move an existing run folder aside")
    ap.add_argument("--dry-run", action="store_true", help="build data and model, then stop")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--overfit", action="store_true", help="CP2 64-image overfit test")
    ap.add_argument("--kill-at-step", type=int)
    ap.add_argument("--shards", type=Path)
    ap.add_argument(
        "--stage-local",
        action="store_true",
        help="copy the shards to <local data>/shards first and read them there",
    )
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    cfg = load_config(args.config, args.overrides)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    profile = args.profile or cfg.get("train", {}).get("profile", "unified")
    paths = get_paths()
    shards = args.shards or paths.shards
    if args.stage_local:
        from eviground.data.pack import stage_shards

        local = paths.local_data / "shards"
        print(json.dumps(stage_shards(shards, local, progress=print), indent=1))
        shards = local
    name = f"{cfg.get('experiment', 'eviground')}_{profile}_seed{seed}"
    if args.overfit:
        name += "_overfit"
    run_dir = args.run_dir or paths.runs / name
    if args.fresh and run_dir.exists():
        aside = run_dir.with_name(f"{run_dir.name}.old-{time.strftime('%Y%m%d-%H%M%S')}")
        run_dir.rename(aside)
        print(f"moved the previous run folder to {aside}")

    from eviground.train.trainer import KilledForTest, Trainer

    trainer = Trainer(
        cfg,
        shards,
        run_dir,
        seed=seed,
        profile=profile,
        resume=None if args.resume == "none" or args.overfit else args.resume,
        kill_at_step=args.kill_at_step,
    )
    info = {
        "run_dir": str(run_dir),
        "train_items": len(trainer.train_ds),
        "val_items": len(trainer.val_ds),
        "left_out_never_valid": len(trainer.train_ds.dropped),
        "steps_per_epoch": trainer.steps_per_epoch,
        "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        **trainer.summary(),
    }
    print(json.dumps(info, indent=1, default=str))
    if args.dry_run:
        return
    if args.overfit:
        result = trainer.overfit()
        (run_dir / "overfit.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))
        return
    try:
        out = trainer.fit(max_steps=args.max_steps)
    except KilledForTest as e:
        print(f"{e}; exiting without cleanup", flush=True)
        os._exit(137)  # like a killed process: no cleanup, no further writes
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
