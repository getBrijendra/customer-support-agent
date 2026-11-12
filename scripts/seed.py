"""Create the two demo companies and upload their guideline files through the service layer.

Idempotent: existing companies are reused; an upload identical to the active version is skipped.
Requires GOOGLE_API_KEY (real embeddings). Usage: python scripts/seed.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import CompanyAlreadyExists, DuplicateGuidelines  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.resources import build_resources  # noqa: E402

SEED_DIR = ROOT / "data" / "seed_guidelines"
COMPANIES = [
    ("bookshop", "Bookshop", "Online bookshop", "bookshop.txt"),
    ("vpn-saas", "VPN SaaS", "Consumer VPN subscription service", "vpn_saas.txt"),
]


async def main() -> int:
    settings = get_settings()
    configure_logging("WARNING")
    missing = [f for *_, f in COMPANIES if not (SEED_DIR / f).is_file()]
    if missing:
        print(f"Missing seed guideline files in {SEED_DIR}: {', '.join(missing)}")
        print("Place the provided files there and re-run `make seed`.")
        return 1

    resources = build_resources(settings)
    companies = resources.company_service()
    ingestion = resources.ingestion_service()
    exit_code = 0
    try:
        for slug, name, description, filename in COMPANIES:
            try:
                await companies.create(slug=slug, name=name, description=description)
                print(f"[{slug}] company created")
            except CompanyAlreadyExists:
                print(f"[{slug}] company already exists")

            content = (SEED_DIR / filename).read_bytes()
            try:
                created = await ingestion.create_version(
                    slug=slug,
                    filename=filename,
                    content=content,
                    notes="seed",
                    uploaded_by="seed-script",
                )
            except DuplicateGuidelines as exc:
                print(f"[{slug}] skipped upload: {exc.message}")
                continue
            version = await ingestion.run_ingestion(created.version.id, activate=True)
            if version is None or version.status != "ready":
                err = version.error_message if version else "version missing"
                print(f"[{slug}] v{created.version.version_number} FAILED: {err}")
                exit_code = 1
            else:
                print(
                    f"[{slug}] v{version.version_number} ready "
                    f"({version.chunk_count} chunks, model {version.embedding_model})"
                )
    finally:
        await resources.aclose()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
