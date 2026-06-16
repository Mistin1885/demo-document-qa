# 01 · MinerU Setup & First Parse

This is the operator's quick-start for the document-parsing path used by the
Paper Notebook Agent. The end-to-end story is:

```
PDF ──► MinerU CLI ──► local hybrid pipeline (layout / table / MFR / OCR) ──┐
                                                                            │  (VLM inference offloaded)
                                                                            ▼
                                                               vLLM @ http://localhost:8001
                                                               serving opendatalab/MinerU2.5-2509-1.2B
                                                                            │
                                            ◄───────  middle.json + md + images
                                                                            │
                                                        ▼
                                   scripts/mineru_poc.py post-processor
                                                                            │
                                                        ▼
              data/parsed/<doc>/hybrid_auto/{<doc>.md, <doc>_middle.json, images/}
```

> Read this once before touching the parsing layer. Detailed field-level
> schema lives in [`deploy/mineru/output-schema.md`](../deploy/mineru/output-schema.md);
> what the post-processor changes is described in
> [`02-mineru-output-customizations.md`](./02-mineru-output-customizations.md).

---

## 1. Prerequisites

| | |
|---|---|
| Python | `3.12+` (pinned via `.python-version`) |
| Package manager | `uv` — **mandatory** per `CLAUDE.md §0`; no bare `pip`, no manual venv activation |
| GPU | a CUDA / MPS device is required for the local pipeline stage (layout / table / MFR / OCR) |
| VLM server | a running OpenAI-compatible endpoint serving `opendatalab/MinerU2.5-2509-1.2B`. The repo defaults to `http://localhost:8001`. |

### 1.1 vLLM endpoint — managed by you, not by this repo

The user / operator runs the model server outside this repo, e.g.

```bash
vllm serve opendatalab/MinerU2.5-2509-1.2B \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 16384 \
    --served-model-name opendatalab/MinerU2.5-2509-1.2B
```

Quick health check (does **not** require any deps from this repo):

```bash
curl -s http://localhost:8001/v1/models | python -m json.tool
# expect: data[0].id == "opendatalab/MinerU2.5-2509-1.2B"
```

If you need a different host / port, set `MINERU_SERVER_URL` in your local
`.env` (see `.env.example`). The driver picks it up automatically.

> Verified during Phase 1 PoC: the raw vLLM endpoint is **directly compatible**
> with MinerU's `hybrid-http-client -u <url>`. You do **not** need to put the
> `mineru-openai-server` wrapper in front of it.

---

## 2. Install client-side deps (one-time)

```bash
uv sync
```

`pyproject.toml` already pins `mineru[pipeline]` (which transitively brings in
local `torch`, OCR, table-structure, MFR) plus `six` (a transitive that
`mineru[pipeline]` forgets to pin — without it the hybrid client crashes at
import time).

Sanity:

```bash
uv run python -c "import torch, cv2; print(torch.__version__, cv2.__version__)"
uv run mineru --version
```

### 2.1 Known install gotcha — empty `cv2/` stub

On at least one workstation, `uv sync` left
`.venv/lib/python3.12/site-packages/cv2/` with only the `qt/` sub-directory
and **no binary `.so`** — `import cv2` then succeeds but returns an empty
namespace module, which makes MinerU explode much later with
`AttributeError: module 'cv2' has no attribute 'INTER_NEAREST'`.

Force-reinstall the OpenCV wheel via `uv`:

```bash
uv sync --reinstall-package opencv-python
```

If the symptom recurs, re-run that command (don't use bare `pip install` —
that violates `CLAUDE.md §0` and your tooling permission gate will block it).

---

## 3. Parse a PDF — the only command you should run

```bash
uv run python scripts/mineru_poc.py <path-to-pdf> data/parsed
```

The driver does three things in order:

1. Invokes `mineru -b hybrid-http-client -u $MINERU_SERVER_URL` on the input.
2. Post-processes the output dir (rename images, rebuild markdown with
   `<Page N>...</Page N>` markers, prune intermediate files).
3. Reports a JSON gate summary on stdout and exits non-zero if the markdown
   / structure looks degraded.

Example:

```bash
uv run python scripts/mineru_poc.py data/2410.05779v3.pdf data/parsed
```

Where the files land (relative to the output dir you pass):

```
data/parsed/<doc-basename>/hybrid_auto/
├── <doc-basename>.md                 # markdown with <Page N>...</Page N>
├── <doc-basename>_middle.json        # primary structural source
└── images/
    └── <doc-basename>_p<N>_<short_hash>.<ext>
```

Nothing else — intermediate files (`*_content_list.json`,
`*_content_list_v2.json`, `*_model.json`, `*_layout.pdf`, `*_origin.pdf`)
are deleted as soon as the post-processor has consumed them. See
[`02-mineru-output-customizations.md`](./02-mineru-output-customizations.md)
for why.

---

## 4. Sample run (Phase 1 PoC reference)

| | |
|---|---|
| Input | `data/2410.05779v3.pdf` — *LightRAG*, 16 pp., 1.1 MB |
| Wall-clock | ~2 min 13 s (cold cache, includes weight downloads) |
| Output size | 5.5 MB total (md ~66 KB, middle.json ~824 KB, 14 image crops) |
| Gate verdict | PASS — see `artifacts/evaluation/mineru-poc.md` (locally generated; gitignored) |

Re-running the same PDF reuses the cached MinerU model weights and finishes in
under a minute. The post-processor is deterministic — image filenames and
markdown page-marker positions are byte-identical across runs (modulo
upstream MinerU drift).

---

## 5. Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `HybridDependencyError: hybrid-http-client requires local pipeline dependencies` | `torch` missing or `cv2` is the empty stub | `uv sync --reinstall-package opencv-python`; verify `import torch, cv2` |
| `AttributeError: module 'cv2' has no attribute 'INTER_NEAREST'` | empty `cv2/` stub (see §2.1) | same fix as above |
| `ModuleNotFoundError: No module named 'six'` | `mineru[pipeline]` metadata gap | already pinned in `pyproject.toml`; run `uv sync` |
| Connection refused on `:8001` | vLLM server not running | start `vllm serve opendatalab/MinerU2.5-2509-1.2B --port 8001` |
| Different host/port | non-default vLLM deployment | set `MINERU_SERVER_URL=http://host:port` in `.env` |
| `expected output dir ... missing` | MinerU CLI itself failed silently | inspect stderr of the previous run; common cause is the vLLM endpoint rejecting the request because the served model name does not match |

---

## 6. Why the hybrid client, not the lightweight VLM client

MinerU exposes two server-side topologies:

- `vlm-http-client` — *all* layout / table / formula work happens on the VLM
  server; the client has no local torch. Smaller install, but you lose
  MinerU's local MFR (math-formula recognition) and table-structure
  post-processing.
- `hybrid-http-client` — the **client still does** layout / table / MFR / OCR
  locally and offloads only the VLM inference. This is the path we picked
  during Phase 1 because formula extraction is a hard requirement
  (see `artifacts/evaluation/mineru-poc.md` and the user's brief).

That trade-off is documented in `CLAUDE.md §6` and is **not** a knob users
should flip casually — switching backend would change the schema of
`middle.json` and invalidate the Phase 4 mapping.
