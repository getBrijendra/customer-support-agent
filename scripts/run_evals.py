"""Run routing evals for the seed companies' active versions and print the headline metrics.

Usage: python scripts/run_evals.py [--mode retrieval|full] [--company bookshop]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.errors import DomainError
from app.core.logging import configure_logging
from app.resources import build_resources

DEFAULT_COMPANIES = ["bookshop", "vpn-saas"]
KEYS = ("cases", "top1_accuracy", "recall_at_k", "set_f1", "no_match_recall", "stale")


async def main(mode: str, companies: list[str]) -> int:
    configure_logging("WARNING")
    resources = build_resources(get_settings())
    evals = resources.eval_service()
    code = 0
    try:
        for slug in companies:
            try:
                run, _, number = await evals.run_for_slug(slug, None, mode)  # type: ignore[arg-type]
            except DomainError as exc:
                print(f"[{slug}] {exc.code}: {exc.message}")
                code = 1
                continue
            summary = ", ".join(f"{k}={run.metrics.get(k)}" for k in KEYS)
            print(f"[{slug}] v{number} {mode}: {summary}")
            for c in run.metrics.get("top_confusions", []):
                print(f"    confused {c['expected']} -> {c['predicted']} x{c['count']}")
    finally:
        await resources.aclose()
    return code


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    p.add_argument("--company", action="append")
    a = p.parse_args()
    raise SystemExit(asyncio.run(main(a.mode, a.company or DEFAULT_COMPANIES)))
