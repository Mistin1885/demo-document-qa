"""Phase 1 MinerU PoC driver + post-processor.

Workflow per PDF:

    1. Invoke ``mineru -b hybrid-http-client -u $MINERU_SERVER_URL``.
    2. Post-process the per-doc output dir:
       a. Rename ``images/<sha256>.jpg`` → ``<doc>_p<page>_<short_hash>.jpg``
          and update every reference (markdown + middle.json).
       b. Rebuild ``<doc>.md`` from the ordered ``content_list.json`` blocks,
          wrapping each PDF page in ``<Page N>...</Page N>`` markers.
       c. Delete every output file except ``<doc>.md``, ``<doc>_middle.json``,
          and the ``images/`` directory.
    3. Run a gate check (markdown + middle.json only) and emit a JSON summary.

Usage:
    uv run python scripts/mineru_poc.py data/2410.05779v3.pdf data/parsed
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MINERU_URL = os.environ.get("MINERU_SERVER_URL", "http://localhost:8001")

PAGE_MARK_OPEN = "<Page {n}>"
PAGE_MARK_CLOSE = "</Page {n}>"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# 1) Run MinerU
# ---------------------------------------------------------------------------

def run_mineru(pdf: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "mineru",
        "-p", str(pdf),
        "-o", str(out_dir),
        "-b", "hybrid-http-client",
        "-u", MINERU_URL,
    ]
    print(f"[poc] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    parsed = out_dir / pdf.stem / "hybrid_auto"
    if not parsed.is_dir():
        raise SystemExit(f"expected output dir {parsed} missing")
    return parsed


# ---------------------------------------------------------------------------
# 2) Post-process: rename images, rebuild md with page markers, prune extras
# ---------------------------------------------------------------------------

def _short_hash(filename: str, n: int = 8) -> str:
    return Path(filename).stem[:n]


def _walk_image_paths(obj: Any, page_idx: int | None, callback) -> None:
    if isinstance(obj, dict):
        if "page_idx" in obj and isinstance(obj["page_idx"], int):
            page_idx = obj["page_idx"]
        ip = obj.get("image_path")
        if isinstance(ip, str) and ip.lower().endswith(IMAGE_EXTS):
            callback(ip, page_idx, obj)
        for v in obj.values():
            _walk_image_paths(v, page_idx, callback)
    elif isinstance(obj, list):
        for x in obj:
            _walk_image_paths(x, page_idx, callback)


def build_rename_map(middle: dict, doc_stem: str) -> dict[str, str]:
    """basename -> new basename, keyed off middle.json (covers ALL image refs)."""
    rename: dict[str, str] = {}

    def _record(image_path: str, page_idx: int | None, _block: dict) -> None:
        if image_path in rename:
            return
        page_num = (page_idx or 0) + 1
        ext = Path(image_path).suffix or ".jpg"
        new_name = f"{doc_stem}_p{page_num}_{_short_hash(image_path)}{ext}"
        rename[image_path] = new_name

    for page in middle.get("pdf_info", []):
        _walk_image_paths(page, page.get("page_idx", 0), _record)
    return rename


def apply_rename_to_disk(parsed: Path, rename: dict[str, str]) -> None:
    images = parsed / "images"
    if not images.is_dir():
        return
    for old, new in rename.items():
        src = images / old
        dst = images / new
        if src.exists() and src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))


def rewrite_middle_image_paths(middle: dict, rename: dict[str, str]) -> None:
    def _replace(obj: Any) -> None:
        if isinstance(obj, dict):
            ip = obj.get("image_path")
            if isinstance(ip, str) and ip in rename:
                obj["image_path"] = rename[ip]
            for v in obj.values():
                _replace(v)
        elif isinstance(obj, list):
            for x in obj:
                _replace(x)

    _replace(middle)


def render_md_with_page_markers(content_list: list[dict], rename: dict[str, str]) -> str:
    """Rebuild markdown from content_list grouped by page with <Page N>...</Page N>."""
    pages: dict[int, list[dict]] = {}
    for blk in content_list:
        pages.setdefault(blk.get("page_idx", 0), []).append(blk)

    out: list[str] = []
    for page_idx in sorted(pages):
        page_num = page_idx + 1
        out.append(PAGE_MARK_OPEN.format(n=page_num))
        out.append("")
        for blk in pages[page_idx]:
            out.extend(_render_block(blk, rename))
        out.append(PAGE_MARK_CLOSE.format(n=page_num))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _render_block(blk: dict, rename: dict[str, str]) -> list[str]:
    t = blk.get("type")
    if t == "text":
        text = blk.get("text", "")
        lvl = blk.get("text_level")
        prefix = ""
        if lvl == 1:
            prefix = "# "
        elif lvl == 2:
            prefix = "## "
        return [f"{prefix}{text}", ""]
    if t == "equation":
        return [blk.get("text", ""), ""]
    if t == "ref_text":
        return [blk.get("text", ""), ""]
    if t == "image":
        basename = Path(blk.get("img_path", "")).name
        new_basename = rename.get(basename, basename)
        lines = [f"![](images/{new_basename})  "]
        captions = blk.get("image_caption") or []
        for cap in captions:
            lines.append(cap)
        for fn in blk.get("image_footnote") or []:
            lines.append(fn)
        lines.append("")
        return lines
    if t == "table":
        basename = Path(blk.get("img_path", "")).name
        new_basename = rename.get(basename, basename)
        lines: list[str] = []
        for cap in blk.get("table_caption") or []:
            lines.append(cap)
            lines.append("")
        body = blk.get("table_body", "")
        if body:
            lines.append(body)
            lines.append("")
        for fn in blk.get("table_footnote") or []:
            lines.append(fn)
            lines.append("")
        if not body:
            # fall back to image link if we have no HTML body
            lines.append(f"![](images/{new_basename})  ")
            lines.append("")
        return lines
    if t in ("page_footnote", "aside_text", "page_number"):
        return []  # discarded
    # Unknown type — preserve any text we can find.
    text = blk.get("text", "")
    return ([text, ""] if text else [])


def cleanup_extras(parsed: Path, doc_stem: str) -> None:
    keep = {f"{doc_stem}.md", f"{doc_stem}_middle.json", "images"}
    for entry in parsed.iterdir():
        if entry.name in keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def postprocess(parsed: Path, doc_stem: str) -> dict[str, Any]:
    content_list_path = parsed / f"{doc_stem}_content_list.json"
    middle_path = parsed / f"{doc_stem}_middle.json"
    md_path = parsed / f"{doc_stem}.md"

    if not content_list_path.exists() or not middle_path.exists():
        raise SystemExit(
            f"post-processor needs both {content_list_path.name} and {middle_path.name}; "
            "did MinerU finish?"
        )

    content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
    middle = json.loads(middle_path.read_text(encoding="utf-8"))

    rename = build_rename_map(middle, doc_stem)
    apply_rename_to_disk(parsed, rename)
    rewrite_middle_image_paths(middle, rename)

    new_md = render_md_with_page_markers(content_list, rename)
    md_path.write_text(new_md, encoding="utf-8")
    middle_path.write_text(
        json.dumps(middle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cleanup_extras(parsed, doc_stem)
    return {"renamed_images": len(rename)}


# ---------------------------------------------------------------------------
# 3) Gate check (reads md + middle.json only)
# ---------------------------------------------------------------------------

def check(parsed: Path) -> dict:
    stem = parsed.parent.name
    md = (parsed / f"{stem}.md").read_text(encoding="utf-8")
    middle = json.loads((parsed / f"{stem}_middle.json").read_text())

    block_types: Counter[str] = Counter()
    title_levels: Counter[int] = Counter()
    display_eq = 0
    ref_count = 0
    discarded_types: Counter[str] = Counter()

    for page in middle.get("pdf_info", []):
        for b in page.get("preproc_blocks", []):
            t = b.get("type")
            block_types[t] += 1
            if t == "title":
                lvl = b.get("level")
                if isinstance(lvl, int):
                    title_levels[lvl] += 1
            elif t == "interline_equation":
                display_eq += 1
            elif t == "ref_text":
                ref_count += 1
        for b in page.get("discarded_blocks", []):
            discarded_types[b.get("type", "unknown")] += 1

    inline_math = md.count("$") - md.count("$$") * 2
    page_markers_open = len(re.findall(r"<Page \d+>", md))
    page_markers_close = len(re.findall(r"</Page \d+>", md))

    summary = {
        "doc": stem,
        "pages": len(middle.get("pdf_info", [])),
        "md_length": len(md),
        "preproc_block_count": sum(block_types.values()),
        "block_type_counts": dict(block_types),
        "title_level_counts": dict(title_levels),
        "display_equation_count": display_eq,
        "ref_text_count": ref_count,
        "inline_math_marker_estimate": inline_math,
        "discarded_type_counts": dict(discarded_types),
        "page_markers_open": page_markers_open,
        "page_markers_close": page_markers_close,
        "_backend": middle.get("_backend"),
        "_version_name": middle.get("_version_name"),
    }

    images_dir = parsed / "images"
    if images_dir.is_dir():
        files = sorted(p.name for p in images_dir.iterdir() if p.is_file())
        summary["image_file_count"] = len(files)
        summary["image_file_sample"] = files[:3]

    gate_pass = (
        summary["md_length"] > 1000
        and summary["preproc_block_count"] > 0
        and title_levels.get(1, 0) >= 1
        and title_levels.get(2, 0) >= 3
        and (display_eq > 0 or inline_math > 4)
        and ref_count >= 5
        and page_markers_open == summary["pages"]
        and page_markers_close == summary["pages"]
    )
    summary["gate_pass"] = gate_pass
    return summary


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    pdf = Path(argv[1]).resolve()
    out_dir = Path(argv[2] if len(argv) > 2 else "data/parsed").resolve()
    parsed = run_mineru(pdf, out_dir)
    post = postprocess(parsed, pdf.stem)
    summary = check(parsed) | {"postprocess": post}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
