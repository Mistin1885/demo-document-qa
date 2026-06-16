# MinerU Hybrid Deployment

> Phase 1 PoC validated. MinerU 3.3.1 hybrid client talks directly to a
> raw vLLM-served MinerU2.5 model — no `mineru-openai-server` wrapper required.

## 1. Server side (already running)

The user runs vLLM with the MinerU 2.5 model:

```bash
# Example launch (do not run unless you maintain this server):
vllm serve opendatalab/MinerU2.5-2509-1.2B \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 16384 \
    --served-model-name opendatalab/MinerU2.5-2509-1.2B
```

Health check (run anywhere):
```bash
curl -s http://localhost:8001/v1/models | python -m json.tool
# Expect: id == "opendatalab/MinerU2.5-2509-1.2B"
```

## 2. Client side (this repo)

```bash
# 1) Install pipeline deps (must use uv — see CLAUDE.md §0).
uv add "mineru[pipeline]" six   # `six` is a transitive that mineru[pipeline] forgets to pin

# 2) Verify torch & cv2 import correctly (cv2 wheel must NOT be the empty-stub variant):
uv run python -c "import torch, cv2; print(torch.__version__, cv2.__version__)"

# 3) Parse a PDF — always use the driver (it runs MinerU + post-processes):
uv run python scripts/mineru_poc.py data/2410.05779v3.pdf data/parsed
```

The driver runs the MinerU CLI under the hood:

```
uv run mineru -p <pdf> -o <out> -b hybrid-http-client -u $MINERU_SERVER_URL
```

then post-processes the output directory so that only three artefacts remain
in `data/parsed/<basename>/hybrid_auto/`:

- `<name>.md` — markdown, with every PDF page wrapped in `<Page N>...</Page N>`
  markers and image links rewritten to the renamed basenames.
- `<name>_middle.json` — Phase 4 primary structural source (bbox / spans /
  hierarchy; all `image_path` values point to the renamed files).
- `images/<basename>_p<page>_<short_hash>.<ext>` — renamed figure / table /
  formula crops (`<page>` is 1-indexed, `<short_hash>` is 8 chars of the
  original sha256-derived filename stem).

Everything else MinerU normally emits (`*_content_list.json`,
`*_content_list_v2.json`, `*_model.json`, `*_layout.pdf`, `*_origin.pdf`) is
deleted by the post-processor. See `output-schema.md` for field-level details.

## 3. Configuration

`MINERU_SERVER_URL` (default `http://localhost:8001`) is the only required env
var — surfaced through `app.config` and the provider-profile settings.

## 4. Compatibility notes (verified)

- **Raw vLLM endpoint at `:8001` is directly compatible** with
  `-b hybrid-http-client -u <url>`. No `mineru-openai-server` wrapper needed.
- `hybrid-http-client` **requires** local torch + pipeline deps despite being
  named "client"; the *client* still does layout / table / OCR / MFR locally and
  only offloads VLM inference to the server.
- `vlm-http-client` is the lightweight variant (no local torch) — it relies on
  the server for everything but skips MFR / table-structure post-processing. We
  intentionally use `hybrid-http-client` for better table & formula fidelity.

## 5. Known install gotcha

`uv add "mineru[pipeline]"` may leave `cv2/` as a stub directory missing the
binary `.so` (observed once on this machine). If `import cv2` returns a
namespace package, run `uv sync --reinstall-package opencv-python` once. The
`six` module is also a transitive missing from the published metadata; we pin
it explicitly in `pyproject.toml`.
