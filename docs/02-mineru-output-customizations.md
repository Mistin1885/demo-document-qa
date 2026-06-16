# 02 · MinerU Output Customizations

`scripts/mineru_poc.py` invokes the MinerU CLI and then post-processes the
per-document output directory so that downstream code (Phase 4 ingestion,
citations, retrieval excerpts) has a small, predictable, self-consistent
artefact set.

Three deliberate deviations from MinerU's default output:

1. **Trim**: keep only `*.md`, `*_middle.json`, `images/`.
2. **Rename**: images get human-readable, page-keyed filenames.
3. **Annotate**: markdown gets `<Page N>...</Page N>` page-boundary markers.

This document explains the *why* behind each, the implementation surface, and
the guarantees Phase 4 and later phases can rely on.

> The architectural contract for these changes lives in
> [`CLAUDE.md §6.3-6.4`](../CLAUDE.md) and the field-level schema in
> [`deploy/mineru/output-schema.md`](../deploy/mineru/output-schema.md).

---

## 1. Trim — keep `md` + `middle.json` + `images/` only

### What MinerU emits by default

```
data/parsed/<doc>/hybrid_auto/
├── <doc>.md
├── <doc>_content_list.json
├── <doc>_content_list_v2.json
├── <doc>_middle.json
├── <doc>_model.json
├── <doc>_layout.pdf
├── <doc>_origin.pdf
└── images/<sha256>.jpg
```

### What we keep

```
data/parsed/<doc>/hybrid_auto/
├── <doc>.md
├── <doc>_middle.json
└── images/<doc>_p<page>_<short_hash>.<ext>
```

### Why

- `<doc>_middle.json` is a strict super-set of `<doc>_content_list.json` —
  the latter is just a flat, page-annotated projection of `preproc_blocks`.
  Anything Phase 4 needs (reading-order, bbox in PDF-point space, spans,
  confidence, title `level`, discarded blocks) is in `middle.json` already.
- `<doc>_model.json` is the raw VLM detection list with normalized 0..1
  bboxes. Useful for debugging the VLM itself, **not** for our ingestion
  pipeline.
- `<doc>_layout.pdf` / `<doc>_span.pdf` are visual debug overlays. Heavy and
  pointless to ingest.
- `<doc>_origin.pdf` is a copy of the source. We already keep the original
  PDF in `data/storage/` (Phase 3), so duplicating it bloats disk + git
  diffs for no upside.

Removing those four files cuts the per-doc footprint by roughly 4× and means
Phase 4 has exactly **one** structural source of truth to reason about
(`middle.json`) instead of three overlapping JSON variants.

### Implementation

`scripts/mineru_poc.py::cleanup_extras` runs **after** the rename + markdown
rebuild have completed (those steps still consume `content_list.json` and the
image hashes once before the file is deleted). It is intentionally
allow-listed rather than deny-listed — a future MinerU version that adds new
sidecar files will not silently sneak them into the repo.

---

## 2. Rename — `<doc>_p<page>_<short_hash>.<ext>`

### Before

```
images/2fa1818e26e1b65e5fc1e160e8846d670e662871d2818d9bb26d50aeca3cd46f.jpg
```

### After

```
images/2410.05779v3_p3_2fa1818e.jpg
```

### Naming rules

| Segment | Rule |
|---|---|
| `<doc>` | the document basename (PDF stem). Lets you tell which paper an image came from without inspecting metadata. |
| `_p<N>` | **1-indexed** PDF page number (matches the page numbers humans cite and matches the `<Page N>` markers in markdown). |
| `_<short_hash>` | first 8 chars of the original sha256-derived stem. Keeps the name globally collision-resistant while staying short. |
| `.<ext>` | original extension (`.jpg` for figures / table crops in current MinerU; `.png` is also handled defensively). |

### Why

- Default sha256 filenames make every image opaque — to know what
  `4dce6f89...jpg` is, you have to grep middle.json. A glance at the new name
  tells you the source paper and the page.
- The renamed basenames double as **stable citation handles** in Phase 7:
  when the agent emits a `Citation` for "Figure 1 on page 3 of
  2410.05779v3", the corresponding image asset path encodes exactly that.
- The `<short_hash>` segment keeps content-addressability — if MinerU
  re-parses the same paper the image bytes hash to the same crop and the
  filename stays stable.

### Consistency guarantee

`build_rename_map` walks **`middle.json` only** so it picks up *every*
`image_path` MinerU emits, including the formula-rendering crops that
`content_list.json` did not reference. As a result:

- Every file in `images/` is referenced by the post-processed `middle.json`.
- Every `image_path` value in `middle.json` corresponds to an actual file on
  disk.
- Every `![](images/...)` link in the rebuilt markdown points at one of those
  files.

If any of those three invariants ever breaks, that is a regression — the
gate check in `scripts/mineru_poc.py::check` is the first place to look.

### Edge cases handled

- **Figure shared across pages**: each unique `image_path` is renamed exactly
  once, keyed off the *first* `page_idx` we see it referenced from. No
  duplicates produced; subsequent references still resolve.
- **Unreferenced-by-content_list images**: e.g. MinerU's per-equation rendered
  PNGs only appear in `middle.json`'s `interline_equation` spans. They are
  picked up correctly because we walk `middle.json`, not `content_list.json`.

---

## 3. Annotate — `<Page N>...</Page N>` markers in markdown

### What the markdown now looks like

```markdown
<Page 1>

# LIGHTRAG: SIMPLE AND FAST RETRIEVAL-AUGMENTED GENERATION

Zirui Guo<sup>1,2</sup>, Lianghao Xia<sup>2</sup>, ...

## ABSTRACT

...

</Page 1>

<Page 2>

To address these limitations, we propose ...

$$
\mathcal{M} = \left(\mathcal{G}, \mathcal{R} = (\varphi, \psi)\right), \ldots \tag{1}
$$

...

</Page 2>
```

### Why the exact `<Page N>` form

The user asked specifically for `<Page 1>...</Page 1>` style markers. Two
small but important properties:

1. **They are not valid HTML tags.** Tag names cannot contain a space (a
   space separates tag name from the first attribute), so `<Page 1>` does
   not match any HTML inline pattern in CommonMark. Renderers therefore pass
   it through as literal text instead of consuming it as a malformed tag.
   That means the markers stay visible to graders and to grep, and they
   never get folded into a heading or paragraph.
2. **They never look like `#` / `##` headings.** The `<` prefix prevents
   markdown's heading parser from treating them as section starts, so our
   heading-extraction in Phase 4 is unaffected.

### What we use them for

- **Citation page anchors (Phase 7).** When chunking the markdown for
  retrieval, each chunk can be tagged with its page span by scanning for the
  enclosing `<Page N>` / `</Page N>` pair. The chunker does not have to fall
  back to MinerU bboxes for the common case.
- **Human review.** A reviewer can scroll the post-processed markdown and
  immediately see the page-boundary alignment with the original PDF — no
  need to cross-reference `middle.json` bboxes.
- **Regression smoke check.** The gate condition in
  `scripts/mineru_poc.py::check` asserts `page_markers_open == pages` and
  `page_markers_close == pages`. If MinerU ever silently drops a page from
  reading-order, the PoC driver fails.

### How the markdown is rebuilt

`render_md_with_page_markers` groups `content_list.json` blocks by
`page_idx`, then emits each page wrapped in the markers. Per-block rendering
follows the rules already in `deploy/mineru/output-schema.md §1`:

| Block | Rule |
|---|---|
| `text` w/ `text_level == 1` | `# {text}` |
| `text` w/ `text_level == 2` | `## {text}` |
| `text` (no level) | paragraph as-is (inline math preserved) |
| `equation` | `$$\n{latex}\n$$` (`\tag{N}` retained) |
| `image` | `![](images/{renamed})  ` followed by `image_caption[]` / `image_footnote[]` |
| `table` | `table_caption[]` → `table_body` (HTML, rowspan/colspan kept) → `table_footnote[]` |
| `ref_text` | paragraph |
| `page_footnote` / `aside_text` / `page_number` | dropped (already in `middle.json:discarded_blocks`) |

This is also the **only** time `content_list.json` is touched — once the
markdown is rewritten, the post-processor deletes that file along with the
rest of the intermediates.

---

## 4. Invariants the rest of the codebase can rely on

After `scripts/mineru_poc.py` exits successfully on a PDF, the following
properties hold:

| Invariant | Where it is enforced |
|---|---|
| Output dir contains exactly `<doc>.md`, `<doc>_middle.json`, `images/` | `cleanup_extras` |
| `images/` filenames match `^<doc>_p\d+_[a-f0-9]{8}\.(jpg\|jpeg\|png)$` | `_short_hash` + `build_rename_map` |
| Every `image_path` in `middle.json` is a basename present on disk | `apply_rename_to_disk` + `rewrite_middle_image_paths` |
| Every `![](images/...)` in `<doc>.md` resolves to a file on disk | `render_md_with_page_markers` uses the same rename map |
| `<doc>.md` contains exactly one `<Page N>` and one `</Page N>` per PDF page, monotonically increasing | `render_md_with_page_markers` + the gate `check` |
| `<doc>_middle.json` carries `_backend == "hybrid"` and `_version_name == 3.3.1` (or newer) | preserved from MinerU; the gate prints the pair so regressions are visible |

Breaking any of these in Phase 4+ should be treated as a defect, not a
"refactor opportunity".

---

## 5. What we deliberately did NOT change

- **The `middle.json` schema.** We rewrote only `image_path` strings inside
  it. Block shapes, line / span structure, bboxes, scores, indices — all
  preserved verbatim. Phase 4 mapping code can be written against the
  upstream MinerU 3.3.1 schema with confidence.
- **Inline math.** Inline LaTeX (`$\varphi(\cdot)$`, `$\hat{\mathcal{D}}$`,
  etc.) is left exactly where MinerU placed it inside text blocks. Display
  equations keep their `\tag{N}` numbering. None of the customizations
  touch math content.
- **Tables.** `table_body` HTML (`<tr>`, `<td rowspan=...>`, `<td colspan=...>`,
  etc.) is emitted verbatim into the markdown — we do not re-render tables.
- **MinerU's reading-order decisions.** Two-column reading order, figure /
  caption association, references-section boundary detection — all left to
  MinerU. We trust its layout / VLM stack and inherit the schema as-is.

If you find yourself needing to bend any of those rules, update
`CLAUDE.md` first (per `§14` of that file) and record the decision in
`PROGRESS.md` before touching the post-processor.
