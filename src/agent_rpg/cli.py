from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend
from agent_rpg.backends.openrouter import OpenRouterBackend
from agent_rpg.engine import SimulationEngine
from agent_rpg.loader import load_scenario


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(prog="agent-rpg", description="Run LLM RPG scenario simulations")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a scenario YAML")
    run_p.add_argument("scenario", type=Path, help="Path to scenario YAML")
    run_p.add_argument(
        "--output",
        type=Path,
        default=Path("runs"),
        help="Base output directory (run id subdirectory is created)",
    )
    run_p.add_argument("--run-id", type=str, default=None, help="Optional fixed run id")
    run_p.add_argument(
        "--sqlite",
        type=Path,
        default=None,
        help="Optional explicit path for SQLite mirror of events",
    )
    run_p.add_argument(
        "--save-sqlite",
        action="store_true",
        help="Write SQLite mirror to <output>/<run-id>/events.sqlite",
    )
    run_p.add_argument(
        "--backend",
        choices=("hf", "local", "openrouter"),
        default="hf",
        help="LLM backend: hf=Inference API (HF_TOKEN), local=transformers, openrouter=OPENROUTER_API_KEY",
    )
    run_p.add_argument("--seed", type=int, default=None)

    args = p.parse_args()
    if args.cmd == "run":
        scenario = load_scenario(args.scenario)
        rid = args.run_id or str(uuid.uuid4())

        local_backend = None
        openrouter_backend = None
        if args.backend == "local":
            from agent_rpg.backends.transformers_local import TransformersLocalBackend

            tb = TransformersLocalBackend()
            default_backend = tb
            local_backend = tb
        elif args.backend == "openrouter":
            ob = OpenRouterBackend()
            default_backend = ob
            openrouter_backend = ob
        else:
            default_backend = HuggingFaceInferenceBackend(token=os.environ.get("HF_TOKEN"))

        sqlite_path: Path | None = args.sqlite
        if args.save_sqlite and sqlite_path is None:
            sqlite_path = args.output / rid / "events.sqlite"

        engine = SimulationEngine(scenario)
        out = engine.run(
            default_backend,
            local_backend=local_backend,
            openrouter_backend=openrouter_backend,
            run_id=rid,
            output_dir=args.output,
            sqlite_path=sqlite_path,
            seed=args.seed,
        )
        print(out)


if __name__ == "__main__":
    main()
