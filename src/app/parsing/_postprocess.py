"""Pure-function post-processing helpers for MinerU hybrid output.

These functions are shared between:
- ``scripts/mineru_poc.py`` (Phase 1 PoC CLI driver)
- ``app.parsing.mineru_client`` (Phase 4.1 async client)

No I/O framework dependencies; only stdlib + pathlib.

Post-processing workflow
------------------------
1. ``build_rename_map``  — derive deterministic image rename mapping from
   ``<doc>_middle.json`` (covers ALL ``image_path`` references).
2. ``apply_rename_to_disk`` — rename files in ``images/`` sub-directory.
3. ``rewrite_middle_image_paths`` — patch all ``image_path`` values in the
   in-memory middle dict.
4. ``render_md_with_page_markers`` — rebuild ``<doc>.md`` from
   ``<doc>_content_list.json`` blocks, wrapping each page in
   ``<Page N>…</Page N>`` markers.
5. ``cleanup_extras`` — delete everything except
   ``<doc>.md``, ``<doc>_middle.json``, and ``images/``.
6. ``postprocess`` — orchestrate steps 1-5, return ``{"renamed_images": N}``.
7. ``check`` — gate check against the final ``<doc>.md`` +
   ``<doc>_middle.json``; returns a summary dict with ``gate_pass: bool``.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

PAGE_MARK_OPEN = "<Page {n}>"
PAGE_MARK_CLOSE = "</Page {n}>"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Image rename helpers
# ---------------------------------------------------------------------------


def _short_hash(filename: str, n: int = 8) -> str:
    """Return first ``n`` characters of the filename stem as a short hash."""
    return Path(filename).stem[:n]


def _walk_image_paths(obj: Any, page_idx: int | None, callback: Any) -> None:
    """Recursively walk a parsed JSON structure and call *callback* for each image_path."""
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


def build_rename_map(middle: dict[str, Any], doc_stem: str) -> dict[str, str]:
    """Return ``{old_basename: new_basename}`` for all image files referenced in *middle*."""
    rename: dict[str, str] = {}

    def _record(image_path: str, page_idx: int | None, _block: dict[str, Any]) -> None:
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
    """Move image files from old basenames to new basenames inside ``parsed/images/``."""
    images = parsed / "images"
    if not images.is_dir():
        return
    for old, new in rename.items():
        src = images / old
        dst = images / new
        if src.exists() and src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))


def rewrite_middle_image_paths(middle: dict[str, Any], rename: dict[str, str]) -> None:
    """Patch all ``image_path`` values in the middle dict in-place."""

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


# ---------------------------------------------------------------------------
# Markdown rebuild
# ---------------------------------------------------------------------------


def _render_block(blk: dict[str, Any], rename: dict[str, str]) -> list[str]:
    """Convert a single content_list block to markdown lines."""
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
        lines: list[str] = [f"![](images/{new_basename})  "]
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
        tlines: list[str] = []
        for cap in blk.get("table_caption") or []:
            tlines.append(cap)
            tlines.append("")
        body = blk.get("table_body", "")
        if body:
            tlines.append(body)
            tlines.append("")
        for fn in blk.get("table_footnote") or []:
            tlines.append(fn)
            tlines.append("")
        if not body:
            tlines.append(f"![](images/{new_basename})  ")
            tlines.append("")
        return tlines
    if t in ("page_footnote", "aside_text", "page_number"):
        return []
    text = blk.get("text", "")
    return [text, ""] if text else []


def render_md_with_page_markers(content_list: list[dict[str, Any]], rename: dict[str, str]) -> str:
    """Rebuild markdown from content_list grouped by page with ``<Page N>…</Page N>`` markers."""
    pages: dict[int, list[dict[str, Any]]] = {}
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


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_extras(parsed: Path, doc_stem: str) -> None:
    """Delete every file/dir in *parsed* except ``<doc_stem>.md``,
    ``<doc_stem>_middle.json``, and ``images/``."""
    keep = {f"{doc_stem}.md", f"{doc_stem}_middle.json", "images"}
    for entry in parsed.iterdir():
        if entry.name in keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def postprocess(parsed: Path, doc_stem: str) -> dict[str, Any]:
    """Run the full post-processing pipeline on a MinerU output directory.

    Reads ``<doc_stem>_content_list.json`` and ``<doc_stem>_middle.json``,
    renames images, rebuilds the markdown with page markers, rewrites
    middle.json, and prunes extra files.

    Returns ``{"renamed_images": N}``.

    Raises ``ValueError`` if required input files are missing.
    """
    content_list_path = parsed / f"{doc_stem}_content_list.json"
    middle_path = parsed / f"{doc_stem}_middle.json"
    md_path = parsed / f"{doc_stem}.md"

    if not content_list_path.exists() or not middle_path.exists():
        raise ValueError(
            f"post-processor needs both {content_list_path.name} and {middle_path.name}; "
            "did MinerU finish?"
        )

    content_list: list[dict[str, Any]] = json.loads(content_list_path.read_text(encoding="utf-8"))
    middle: dict[str, Any] = json.loads(middle_path.read_text(encoding="utf-8"))

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
# Gate check
# ---------------------------------------------------------------------------


def check(parsed: Path) -> dict[str, Any]:
    """Read ``<stem>.md`` and ``<stem>_middle.json`` from *parsed* and run gate checks.

    The stem is derived from ``parsed.parent.name`` (the PDF basename).

    Returns a summary dict including ``gate_pass: bool``.  A gate fails when
    page marker counts do not match the number of pages in middle.json, etc.
    """
    stem = parsed.parent.name
    md = (parsed / f"{stem}.md").read_text(encoding="utf-8")
    middle: dict[str, Any] = json.loads((parsed / f"{stem}_middle.json").read_text())

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

    summary: dict[str, Any] = {
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
