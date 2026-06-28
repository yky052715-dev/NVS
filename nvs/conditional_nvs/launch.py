"""CLI override shim used by experiment scripts.

The stable implementation lives in :mod:`nvs.conditional_nvs.cli`; this module
adds explicit sensitivity/ablation overrides without modifying saved configs.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from . import cli
from .memory import MEMORY_PROTOCOLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["mvtec", "robustad"], default="mvtec")
    parser.add_argument("--data-root")
    parser.add_argument("--perturbed-root")
    parser.add_argument("--manifest")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-protocol", choices=sorted(MEMORY_PROTOCOLS))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--prototypes", type=int)
    parser.add_argument(
        "--prototype-selection",
        choices=["proto_by_mstar", "proto_by_topk_vote_k5"],
    )
    args = parser.parse_args()
    config = cli._load_config(args.config)
    if args.perturbed_root:
        config.setdefault("data", {})["perturbed_root"] = args.perturbed_root
    if args.memory_protocol:
        config["memory"]["protocol"] = args.memory_protocol
    if args.rank is not None:
        config["subspace"]["rank"] = args.rank
    if args.prototypes is not None:
        config["subspace"]["prototypes"] = args.prototypes
    if args.prototype_selection:
        config["subspace"]["prototype_selection"] = args.prototype_selection
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.dataset == "robustad":
        cli._run_robustad(args, config)
    else:
        cli._run_mvtec(args, config)


if __name__ == "__main__":
    main()
