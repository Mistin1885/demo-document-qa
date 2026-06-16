# MinerU Hybrid Output Schema — Reference for Phase 4 Mapping

> Source data: `data/parsed/<doc>/hybrid_auto/` produced by
> `uv run python scripts/mineru_poc.py <pdf> data/parsed`
> MinerU 3.3.1, backend `hybrid`, model `opendatalab/MinerU2.5-2509-1.2B` via vLLM.

`scripts/mineru_poc.py` invokes the MinerU CLI and then post-processes the
output directory so that **only three artefacts survive**:

| Path | Role |
|------|------|
| `<name>.md` | Markdown rebuilt from `content_list.json` with `<Page N>...</Page N>` markers wrapping each PDF page. Image links use renamed basenames. |
| `<name>_middle.json` | **Primary structural source** for Phase 4. Per-page hierarchical `block → line → span` with bbox (PDF-point space). All `image_path` values rewritten to the renamed basenames. |
| `images/<doc>_p<page>_<short_hash>.<ext>` | Figure / table / formula crops. `<page>` is 1-indexed; `<short_hash>` is the first 8 chars of the original sha256-stem; `<ext>` preserves the original file extension. |

Intermediate files MinerU emits (`<name>_content_list.json`,
`<name>_content_list_v2.json`, `<name>_model.json`, `<name>_layout.pdf`,
`<name>_origin.pdf`) are **deleted** by the post-processor. Phase 4 code must
not depend on them.

---

## 1. `<name>.md` — markdown with page markers

Page markers are literal text — `<Page 1>` and `</Page 1>` are not valid HTML
tags (the space breaks tag-name parsing), so CommonMark renderers pass them
through as plain text instead of treating them as HTML. They survive into
chunked text and act as deterministic page anchors for citations.

Skeleton:

```markdown
<Page 1>

# DOC TITLE

Authors and affiliations...

## ABSTRACT

...

## 1 INTRODUCTION

...

</Page 1>

<Page 2>

...

</Page 2>
```

Block-by-block rendering rules (applied to `content_list.json` before deletion):

| `content_list[i].type` | Rendered as |
|---|---|
| `text` with `text_level == 1` | `# {text}` |
| `text` with `text_level == 2` | `## {text}` |
| `text` (no level) | paragraph (text as-is, inline math preserved) |
| `equation` | `$$\n{latex}\n$$` (`\tag{N}` retained when present) |
| `ref_text` | paragraph |
| `image` | `![](images/{renamed})  ` followed by each `image_caption[]`/`image_footnote[]` |
| `table` | `table_caption[]` lines → `table_body` (HTML, rowspan/colspan preserved) → `table_footnote[]` |
| `page_footnote`, `aside_text`, `page_number` | dropped (already in `discarded_blocks` of `middle.json`) |

---

## 2. `<name>_middle.json` — PRIMARY structural source

```jsonc
{
  "pdf_info": [
    {
      "page_idx": 0,
      "page_size": [612, 792],          // PDF point space
      "preproc_blocks": [ /* ordered blocks (canonical reading order) */ ],
      "para_blocks":    [ /* paragraph-grouped variant */ ],
      "discarded_blocks": [ /* aside_text / page_footnote / page_number */ ]
    },
    ...
  ],
  "_backend": "hybrid",
  "_effort": "medium",
  "_ocr_enable": true,
  "_version_name": "3.3.1"
}
```

Each block in `preproc_blocks` / `para_blocks`:

| Key | Description |
|-----|-------------|
| `type` | `title`, `text`, `interline_equation`, `image`, `table`, `ref_text` |
| `bbox` | `[x0, y0, x1, y1]` in PDF point space (same scale as `page_size`) |
| `index` | per-page reading-order index |
| `level` | only on `title` blocks: `1` = doc title, `2` = section/subsection |
| `angle` | rotation angle (usually 0) |
| `lines[]` | layout lines: each has `bbox` + `spans[]` |
| `spans[].type` | `text`, `inline_equation`, `interline_equation`, `image` |
| `spans[].content` | raw text or LaTeX (for `(inline_)equation`) |
| `spans[].image_path` | renamed basename (e.g. `2410.05779v3_p3_2fa1818e.jpg`) |
| `spans[].score` | confidence — feeds `ParsedBlock.confidence` |

For `image` / `table` blocks the structure is slightly nested (`blocks` array
with `image_body` + `image_caption` siblings):

```jsonc
{
  "type": "image",
  "bbox": [...],
  "blocks": [
    {"type": "image_body",   "lines": [{"spans": [{"type":"image", "image_path":"<renamed>"}]}]},
    {"type": "image_caption","lines": [{"spans": [{"type":"text",  "content":"Figure 1: ..."}]}]}
  ]
}
```

**Discarded types** (`discarded_blocks`): `aside_text`, `page_footnote`,
`page_number` — already filtered out of `preproc_blocks`. Phase 4 ignores
these entirely.

---

## 3. Mapping to `ParsedBlock` (Phase 4)

| `ParsedBlock` field | Source |
|---------------------|--------|
| `block_id` | `sha1(doc_id + page_idx + bbox)` (deterministic) |
| `chat_id`, `document_id` | injected by ingestion service (NOT from MinerU) |
| `page_number` | `pdf_info[i].page_idx + 1` |
| `block_type` | mapping table below |
| `text` | for `text`/`title`/`ref_text` → join `lines[].spans[].content`; for `interline_equation` → span LaTeX wrapped in `$$...$$`; for `table` → HTML `table_body` (synthesized from the nested span structure or pulled from md); for `image` → caption text |
| `bbox` | block `bbox` (PDF-point space); page dims via `pdf_info[i].page_size` |
| `reading_order` | block `index` per page (canonical) |
| `confidence` | min `score` across spans (default 1.0 if absent) |
| `image_path` | for `image`/`table`, the renamed basename (already on disk) |

### Block-type mapping (Phase 4)

| MinerU `type` (preproc_blocks) | `title.level` | Our `block_type` |
|---|---|---|
| `title` | 1 | `doc_title` |
| `title` | 2 | `heading` (use `paragraph_title` heuristic to split section vs subsection if needed) |
| `text` | — | `paragraph` |
| `interline_equation` | — | `equation_display` |
| `ref_text` | — | `reference` |
| `image` | — | `figure` (caption from sibling `image_caption` block) |
| `table` | — | `table` (caption from sibling `table_caption` block; HTML body retained) |
| inline `inline_equation` spans inside `text` | — | flagged `has_inline_math=true`; LaTeX stays embedded |

---

## 4. Known limitations observed in PoC

1. **Fractions on a single line** sometimes render as nested `<sup>/<sub>`
   instead of `$\frac{num}{den}$` (e.g. `total tokens / chunk size` on p.5).
   Math correctness preserved; visually less ideal.
2. **Numbers with thousands separators inside math** may fragment (e.g.
   `1,399 × 2 × 5,000` → `"1 $, 3 9 9 \times 2 \times 5 , 0 0 0$"`).
   Mitigation: extract numeric facts via `structured_facts` (Phase 5.3) rather
   than relying on inline LaTeX.
3. **Loose spacing in inline math** (`$\varphi ( \cdot )$`) — purely cosmetic;
   KaTeX/MathJax render identically.
4. The first page of two-column papers sometimes has an `aside_text` block
   (arXiv ID strip) — already routed to `discarded_blocks`, so harmless.

> Verdict: `<name>_middle.json` is sufficient for `ParsedBlock` + hierarchy.
> The post-processed `<name>.md` is the canonical human-readable view and the
> source of truth for citation excerpts (with `<Page N>...</Page N>` anchors).
