"""Live smoke test for the VLM-driven figure / table narration pipeline.

Runs the real ``VLMClient`` against a previously parsed document so we can
verify end-to-end that Gemma-4 (or whichever model is configured in
``VLM_*``) returns:

- HTML for table-like images, and
- two natural-language paragraphs for figure-like images.

Usage:
    uv run python scripts/figure_narration_smoke.py \\
        --doc-dir data/parsed/2410.05779v3/hybrid_auto \\
        --max-blocks 3

This script does not touch the database or Vespa; it only calls the VLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

# Allow `uv run python scripts/...` from repo root without installing the
# package; this mirrors what other scripts in the repo do.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.parsing.figure_narrator import narrate_blocks
from app.parsing.mapping import load_middle_json, map_middle_to_parsed_blocks
from app.parsing.models import BlockType
from app.parsing.vlm_client import VLMClient


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doc-dir",
        type=Path,
        default=ROOT / "data" / "parsed" / "2410.05779v3" / "hybrid_auto",
        help="MinerU post-processed output directory (contains <name>_middle.json + images/).",
    )
    parser.add_argument("--max-blocks", type=int, default=3, help="Narrate at most N image/table blocks.")
    parser.add_argument("--max-concurrency", type=int, default=1)
    args = parser.parse_args()

    doc_dir: Path = args.doc_dir.resolve()
    if not doc_dir.is_dir():
        print(f"ERROR: doc-dir not found: {doc_dir}", file=sys.stderr)
        return 1

    middle_jsons = list(doc_dir.glob("*_middle.json"))
    if not middle_jsons:
        print(f"ERROR: no *_middle.json under {doc_dir}", file=sys.stderr)
        return 1
    middle_json_path = middle_jsons[0]

    images_dir = doc_dir / "images"
    if not images_dir.is_dir():
        print(f"ERROR: images/ not found under {doc_dir}", file=sys.stderr)
        return 1

    chat_id = uuid.uuid4()
    document_id = uuid.uuid4()

    middle = load_middle_json(middle_json_path)
    blocks = map_middle_to_parsed_blocks(middle, chat_id=chat_id, document_id=document_id)
    visual = [b for b in blocks if b.block_type in (BlockType.image, BlockType.table)]
    visual = visual[: args.max_blocks]
    print(f"Selected {len(visual)} visual blocks from {middle_json_path.name}")

    # Truncate blocks list so narrate_blocks only walks the subset we care
    # about, but keep all blocks for same-page text context.
    visual_ids = {b.block_id for b in visual}
    blocks_for_run = [b for b in blocks if b.block_type not in (BlockType.image, BlockType.table) or b.block_id in visual_ids]

    client = VLMClient()
    print(f"VLM endpoint: {client.settings.api_url} (model={client.settings.model})")

    narrations = await narrate_blocks(
        blocks_for_run,
        images_dir=images_dir,
        client=client,
        max_concurrency=args.max_concurrency,
    )
    print(f"Got {len(narrations)} narrations\n")

    for i, n in enumerate(narrations, start=1):
        print(f"--- [{i}] {n.block_type.value} p{n.page_number}  {n.image_path} ---")
        if n.error:
            print(f"ERROR: {n.error}")
            continue
        snippet = n.narrative_text.strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + " …"
        print(snippet)
        print()

    print(json.dumps(
        {
            "total": len(narrations),
            "ok": sum(1 for n in narrations if not n.error and n.narrative_text.strip()),
            "errors": [n.error for n in narrations if n.error],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
