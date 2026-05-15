#!/usr/bin/env python3
"""Pre-download an instruct chat model for :class:`TransformersLocalBackend`.

Uses the Hugging Face Hub cache by default (same layout ``transformers`` expects), so you can keep
using the Hub repo id (e.g. ``Qwen/Qwen2.5-1.5B-Instruct``) in scenarios and notebook 08.

Optional ``--local-dir`` copies/snapshots into a folder under this repo; then set each agent's
``model_id`` to that **absolute or relative path** in YAML or in the UI.

Examples::

    python scripts/download_local_instruct_model.py
    python scripts/download_local_instruct_model.py --local-dir models/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--repo-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hub model id (default: catalog default instruct model).",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="If set, download snapshot into this directory (otherwise use default HF cache).",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hub token (defaults to HF_TOKEN env; optional for this public model).",
    )
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    kwargs: dict = {"repo_id": args.repo_id, "token": args.token}
    if args.local_dir is not None:
        args.local_dir = args.local_dir.resolve()
        args.local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(args.local_dir)

    path = snapshot_download(**kwargs)
    print("Download complete.")
    print("  Path:", path)
    if args.local_dir is not None:
        print("  Use this path as agent model_id for local Transformers runs.")
    else:
        print("  Use repo id", repr(args.repo_id), "with TransformersLocalBackend (Hub cache).")


if __name__ == "__main__":
    main()
