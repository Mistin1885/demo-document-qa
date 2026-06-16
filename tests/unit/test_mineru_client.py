"""Unit tests for app.parsing.mineru_client.

All tests mock out the subprocess and HTTP calls — no real MinerU server or
CLI is invoked.  The fixture ``fake_pdf`` creates a minimal directory tree
that looks like MinerU output so that the post-processing pipeline can run
end-to-end.

Test matrix
-----------
a) happy-path         — parse_pdf returns MinerUParseResult with correct fields.
b) idempotent re-run  — second call returns cached result, subprocess not called.
c) force=True         — subprocess called even when result already cached.
d) subprocess rc!=0   — MinerUInvocationError raised with stderr.
e) missing output dir — MinerUInvocationError raised when subprocess succeeds
                        but output directory is absent.
f) gate failure       — MinerUGateFailure raised on page-marker count mismatch.
g) health_check       — True on 200 + correct model id; False on 500; False on
                        timeout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.parsing import (
    MinerUClient,
    MinerUGateFailure,
    MinerUInvocationError,
    MinerUParseResult,
)

# ---------------------------------------------------------------------------
# Helpers to build a minimal valid MinerU output tree
# ---------------------------------------------------------------------------

DOC_STEM = "fake_paper"

# A minimal content_list.json that will produce a markdown with at least 2 pages
# and enough content to pass the gate check.
MINIMAL_CONTENT_LIST: list[dict[str, Any]] = [
    # Page 1
    {"type": "text", "text": "# Fake Paper Title", "text_level": 1, "page_idx": 0},
    {
        "type": "text",
        "text": "Author One, Author Two",
        "page_idx": 0,
    },
    {
        "type": "text",
        "text": "Abstract: This is a fake paper about testing. " * 20,
        "page_idx": 0,
    },
    # Page 2
    {"type": "text", "text": "## Introduction", "text_level": 2, "page_idx": 1},
    {
        "type": "text",
        "text": "The introduction section contains a lot of text. " * 20,
        "page_idx": 1,
    },
    {"type": "text", "text": "## Methods", "text_level": 2, "page_idx": 1},
    {
        "type": "text",
        "text": "We use equation $x = y + z$ and $a = b$. " * 5,
        "page_idx": 1,
    },
    {"type": "equation", "text": "$$E = mc^2$$", "page_idx": 1},
    # Page 3 — references
    {"type": "text", "text": "## Conclusion", "text_level": 2, "page_idx": 2},
    {
        "type": "text",
        "text": "In conclusion we find remarkable results. " * 10,
        "page_idx": 2,
    },
    {"type": "ref_text", "text": "[1] Smith et al. 2020.", "page_idx": 2},
    {"type": "ref_text", "text": "[2] Jones et al. 2021.", "page_idx": 2},
    {"type": "ref_text", "text": "[3] Brown et al. 2022.", "page_idx": 2},
    {"type": "ref_text", "text": "[4] White et al. 2023.", "page_idx": 2},
    {"type": "ref_text", "text": "[5] Black et al. 2024.", "page_idx": 2},
]

# A minimal middle.json with 3 pages, gate-passing block structure.
MINIMAL_MIDDLE: dict[str, Any] = {
    "_backend": "hybrid",
    "_version_name": "MinerU-test-3.3.1",
    "pdf_info": [
        {
            "page_idx": 0,
            "page_size": [595.0, 842.0],
            "preproc_blocks": [
                {"type": "title", "level": 1, "lines": [{"spans": [{"content": "Fake Paper Title"}]}]},
                {"type": "text", "lines": [{"spans": [{"content": "Author One"}]}]},
            ],
            "para_blocks": [],
            "discarded_blocks": [],
        },
        {
            "page_idx": 1,
            "page_size": [595.0, 842.0],
            "preproc_blocks": [
                {"type": "title", "level": 2, "lines": [{"spans": [{"content": "Introduction"}]}]},
                {"type": "title", "level": 2, "lines": [{"spans": [{"content": "Methods"}]}]},
                {"type": "title", "level": 2, "lines": [{"spans": [{"content": "Related Work"}]}]},
                {"type": "interline_equation", "lines": [{"spans": [{"content": "E=mc^2"}]}]},
                {"type": "text", "lines": [{"spans": [{"content": "Some detail text"}]}]},
            ],
            "para_blocks": [],
            "discarded_blocks": [],
        },
        {
            "page_idx": 2,
            "page_size": [595.0, 842.0],
            "preproc_blocks": [
                {"type": "title", "level": 2, "lines": [{"spans": [{"content": "Conclusion"}]}]},
                {"type": "ref_text", "lines": [{"spans": [{"content": "[1] Smith 2020"}]}]},
                {"type": "ref_text", "lines": [{"spans": [{"content": "[2] Jones 2021"}]}]},
                {"type": "ref_text", "lines": [{"spans": [{"content": "[3] Brown 2022"}]}]},
                {"type": "ref_text", "lines": [{"spans": [{"content": "[4] White 2023"}]}]},
                {"type": "ref_text", "lines": [{"spans": [{"content": "[5] Black 2024"}]}]},
            ],
            "para_blocks": [],
            "discarded_blocks": [],
        },
    ],
}


def _setup_output_dir(output_dir: Path, stem: str = DOC_STEM) -> None:
    """Create a minimal valid MinerU output directory for *stem*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write content_list.json (consumed by postprocess, deleted afterwards)
    (output_dir / f"{stem}_content_list.json").write_text(
        json.dumps(MINIMAL_CONTENT_LIST, ensure_ascii=False),
        encoding="utf-8",
    )
    # Write middle.json
    (output_dir / f"{stem}_middle.json").write_text(
        json.dumps(MINIMAL_MIDDLE, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # images/ dir (empty is fine for these tests — no images in MINIMAL_CONTENT_LIST)
    (output_dir / "images").mkdir(exist_ok=True)


def _make_fake_subprocess(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock asyncio.Process whose communicate() returns immediately."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(
        return_value=(stdout.encode(), stderr.encode())
    )
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_parsed_root(tmp_path: Path) -> Path:
    """Return a temporary directory to use as parsed_root."""
    return tmp_path / "parsed"


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """Return a fake PDF path (file does not need to be a real PDF)."""
    p = tmp_path / f"{DOC_STEM}.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    return p


# ---------------------------------------------------------------------------
# a) Happy-path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_pdf_happy_path(fake_pdf: Path, tmp_parsed_root: Path) -> None:
    """parse_pdf returns a valid MinerUParseResult on success."""
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"

    fake_proc = _make_fake_subprocess(returncode=0)

    async def fake_create_subprocess(*args: Any, **kwargs: Any) -> MagicMock:
        # Simulate MinerU creating the output dir
        _setup_output_dir(output_dir)
        return fake_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess):
        client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)
        result = await client.parse_pdf(fake_pdf)

    assert isinstance(result, MinerUParseResult)
    assert result.output_dir == output_dir
    assert result.markdown_path.exists()
    assert result.middle_json_path.exists()
    assert result.pages == 3
    assert result.mineru_version == "MinerU-test-3.3.1"
    assert result.mineru_backend == "hybrid"
    assert result.gate_summary.get("gate_pass") is True
    assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# b) Idempotent re-run — subprocess not called again
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_rerun_skips_subprocess(fake_pdf: Path, tmp_parsed_root: Path) -> None:
    """When a gate-passing result already exists, subprocess is NOT called again."""
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"
    # Pre-populate the output directory WITH post-processed files
    # (postprocess has already run, so we need .md + _middle.json, no content_list)
    _setup_output_dir(output_dir)
    # Run postprocess so the directory is in its final state (md rebuilt, extras removed)
    from app.parsing._postprocess import postprocess as _postprocess

    _postprocess(output_dir, DOC_STEM)

    call_count = 0

    async def should_not_be_called(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return _make_fake_subprocess()

    with patch("asyncio.create_subprocess_exec", side_effect=should_not_be_called):
        client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)
        result = await client.parse_pdf(fake_pdf)

    assert call_count == 0, "subprocess should not be called when result is cached"
    assert result.gate_summary.get("gate_pass") is True
    assert result.duration_seconds == 0.0  # cached — no duration


# ---------------------------------------------------------------------------
# c) force=True always re-runs subprocess
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_reruns_subprocess(fake_pdf: Path, tmp_parsed_root: Path) -> None:
    """force=True triggers subprocess even when a valid cached result exists."""
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"
    # Pre-populate with valid post-processed result
    _setup_output_dir(output_dir)
    from app.parsing._postprocess import postprocess as _postprocess

    _postprocess(output_dir, DOC_STEM)

    call_count = 0

    async def counting_create(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        # Re-create the output dir because force=True deleted it
        _setup_output_dir(output_dir)
        return _make_fake_subprocess(returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=counting_create):
        client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)
        result = await client.parse_pdf(fake_pdf, force=True)

    assert call_count == 1, "subprocess must be called once with force=True"
    assert result.gate_summary.get("gate_pass") is True


# ---------------------------------------------------------------------------
# d) Subprocess returncode != 0 → MinerUInvocationError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subprocess_nonzero_raises_invocation_error(
    fake_pdf: Path, tmp_parsed_root: Path
) -> None:
    """Subprocess exit code != 0 raises MinerUInvocationError with stderr."""
    fake_proc = _make_fake_subprocess(returncode=1, stderr="some VLM server error")

    async def fail_proc(*args: Any, **kwargs: Any) -> MagicMock:
        return fake_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fail_proc):
        client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)
        with pytest.raises(MinerUInvocationError) as exc_info:
            await client.parse_pdf(fake_pdf)

    assert exc_info.value.returncode == 1
    assert "some VLM server error" in exc_info.value.stderr


# ---------------------------------------------------------------------------
# e) Subprocess succeeds but output dir not produced → MinerUInvocationError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subprocess_missing_output_dir_raises(
    fake_pdf: Path, tmp_parsed_root: Path
) -> None:
    """When subprocess exits 0 but the output directory is absent, raise."""
    fake_proc = _make_fake_subprocess(returncode=0)

    async def no_output(*args: Any, **kwargs: Any) -> MagicMock:
        # Do NOT create output_dir — simulate MinerU producing no output
        return fake_proc

    with patch("asyncio.create_subprocess_exec", side_effect=no_output):
        client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)
        with pytest.raises(MinerUInvocationError) as exc_info:
            await client.parse_pdf(fake_pdf)

    assert "missing" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# f) Gate failure — page marker count mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_failure_raises_mineru_gate_failure(
    fake_pdf: Path, tmp_parsed_root: Path
) -> None:
    """When the post-processed markdown has wrong page marker counts, raise MinerUGateFailure."""
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"

    async def create_bad_output(*args: Any, **kwargs: Any) -> MagicMock:
        # Create a middle.json claiming 3 pages but a markdown with only 1 page marker
        output_dir.mkdir(parents=True, exist_ok=True)
        # content_list.json with a single page only (page_idx=0)
        bad_content_list = [
            {"type": "text", "text": "Short content", "page_idx": 0},
        ]
        (output_dir / f"{DOC_STEM}_content_list.json").write_text(
            json.dumps(bad_content_list), encoding="utf-8"
        )
        # middle.json still claims 3 pages
        (output_dir / f"{DOC_STEM}_middle.json").write_text(
            json.dumps(MINIMAL_MIDDLE), encoding="utf-8"
        )
        (output_dir / "images").mkdir(exist_ok=True)
        return _make_fake_subprocess(returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=create_bad_output):
        client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)
        with pytest.raises(MinerUGateFailure) as exc_info:
            await client.parse_pdf(fake_pdf)

    gate = exc_info.value.gate_summary
    assert gate["gate_pass"] is False
    # The markdown only has 1 page marker open but middle claims 3 pages
    assert gate["page_markers_open"] != gate["pages"]


# ---------------------------------------------------------------------------
# g) health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_true_on_correct_model(
    tmp_parsed_root: Path,
) -> None:
    """health_check returns True when server responds 200 with the expected model id."""

    client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)

    # Manually patch httpx.AsyncClient to avoid needing the pytest fixture here
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(
        return_value={
            "data": [{"id": "opendatalab/MinerU2.5-2509-1.2B"}],
        }
    )

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_resp)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        result = await client.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_false_on_server_error(
    tmp_parsed_root: Path,
) -> None:
    """health_check returns False when server responds with a non-200 status."""
    client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json = MagicMock(return_value={})

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_resp)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        result = await client.health_check()

    assert result is False


@pytest.mark.asyncio
async def test_health_check_false_on_wrong_model_id(
    tmp_parsed_root: Path,
) -> None:
    """health_check returns False when a different model id is reported."""
    client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(
        return_value={"data": [{"id": "some-other-model/wrong-id"}]}
    )

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_resp)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        result = await client.health_check()

    assert result is False


@pytest.mark.asyncio
async def test_health_check_false_on_timeout(
    tmp_parsed_root: Path,
) -> None:
    """health_check returns False when the HTTP call raises an exception (e.g. timeout)."""
    client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(
        side_effect=httpx.TimeoutException("connect timeout")
    )
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        result = await client.health_check()

    assert result is False


# ---------------------------------------------------------------------------
# Additional: document_id is propagated into gate_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_document_id_in_gate_summary(fake_pdf: Path, tmp_parsed_root: Path) -> None:
    """When document_id is passed, it appears in gate_summary."""
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"
    fake_proc = _make_fake_subprocess(returncode=0)

    async def fake_create(*args: Any, **kwargs: Any) -> MagicMock:
        _setup_output_dir(output_dir)
        return fake_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)
        result = await client.parse_pdf(fake_pdf, document_id="doc-uuid-123")

    assert result.gate_summary.get("document_id") == "doc-uuid-123"


# ---------------------------------------------------------------------------
# Additional: MinerUClient reads server_url from settings when not provided
# ---------------------------------------------------------------------------


def test_client_uses_settings_server_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """MinerUClient without explicit server_url reads from app.config.get_settings()."""
    monkeypatch.setenv("MINERU_SERVER_URL", "http://custom-vlm:9999")
    from app.config import get_settings

    get_settings.cache_clear()

    client = MinerUClient(parsed_root=Path("data/parsed"))
    assert client._server_url == "http://custom-vlm:9999"

    # Cleanup cache to avoid leaking into other tests
    get_settings.cache_clear()
