"""Unit tests for app.parsing.mineru_client.

All tests mock out subprocess and HTTP calls — no real MinerU server invoked.

Test matrix
-----------
a) happy-path         — parse_pdf returns MinerUParseResult with correct fields.
b) idempotent re-run  — second call returns cached result, subprocess not called.
c) force=True         — subprocess called even when result already cached.
d) subprocess rc!=0   — MinerUInvocationError raised with stderr.
e) missing output dir — MinerUInvocationError raised when subprocess succeeds
                        but output directory is absent.
f) gate failure       — MinerUGateFailure raised on page-marker count mismatch.
g) health_check       — True on 200 + correct model id; False on 500 / wrong model
                        id / timeout (parametrized).
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
# Minimal fixture data
# ---------------------------------------------------------------------------

DOC_STEM = "fake_paper"

MINIMAL_CONTENT_LIST: list[dict[str, Any]] = [
    {"type": "text", "text": "# Fake Paper Title", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "Author One, Author Two", "page_idx": 0},
    {"type": "text", "text": "Abstract: This is a fake paper about testing. " * 20, "page_idx": 0},
    {"type": "text", "text": "## Introduction", "text_level": 2, "page_idx": 1},
    {"type": "text", "text": "The introduction section contains a lot of text. " * 20, "page_idx": 1},
    {"type": "text", "text": "## Methods", "text_level": 2, "page_idx": 1},
    {"type": "text", "text": "We use equation $x = y + z$ and $a = b$. " * 5, "page_idx": 1},
    {"type": "equation", "text": "$$E = mc^2$$", "page_idx": 1},
    {"type": "text", "text": "## Conclusion", "text_level": 2, "page_idx": 2},
    {"type": "text", "text": "In conclusion we find remarkable results. " * 10, "page_idx": 2},
    {"type": "ref_text", "text": "[1] Smith et al. 2020.", "page_idx": 2},
    {"type": "ref_text", "text": "[2] Jones et al. 2021.", "page_idx": 2},
    {"type": "ref_text", "text": "[3] Brown et al. 2022.", "page_idx": 2},
    {"type": "ref_text", "text": "[4] White et al. 2023.", "page_idx": 2},
    {"type": "ref_text", "text": "[5] Black et al. 2024.", "page_idx": 2},
]

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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}_content_list.json").write_text(
        json.dumps(MINIMAL_CONTENT_LIST, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / f"{stem}_middle.json").write_text(
        json.dumps(MINIMAL_MIDDLE, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "images").mkdir(exist_ok=True)


def _make_fake_subprocess(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_parsed_root(tmp_path: Path) -> Path:
    return tmp_path / "parsed"


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    p = tmp_path / f"{DOC_STEM}.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    return p


# ---------------------------------------------------------------------------
# a) Happy-path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_pdf_happy_path(fake_pdf: Path, tmp_parsed_root: Path) -> None:
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"
    fake_proc = _make_fake_subprocess(returncode=0)
    seen_args: tuple[Any, ...] | None = None

    async def fake_create(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal seen_args
        seen_args = args
        _setup_output_dir(output_dir)
        return fake_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
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
    assert seen_args is not None
    assert seen_args[seen_args.index("-o") + 1] == str(tmp_parsed_root)


# ---------------------------------------------------------------------------
# b) Idempotent re-run — subprocess not called again
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_rerun_skips_subprocess(fake_pdf: Path, tmp_parsed_root: Path) -> None:
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"
    _setup_output_dir(output_dir)
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

    assert call_count == 0
    assert result.gate_summary.get("gate_pass") is True
    assert result.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# c) force=True always re-runs subprocess
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_reruns_subprocess(fake_pdf: Path, tmp_parsed_root: Path) -> None:
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"
    _setup_output_dir(output_dir)
    from app.parsing._postprocess import postprocess as _postprocess

    _postprocess(output_dir, DOC_STEM)

    call_count = 0

    async def counting_create(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        _setup_output_dir(output_dir)
        return _make_fake_subprocess(returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=counting_create):
        client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)
        result = await client.parse_pdf(fake_pdf, force=True)

    assert call_count == 1
    assert result.gate_summary.get("gate_pass") is True


# ---------------------------------------------------------------------------
# d) Subprocess returncode != 0 → MinerUInvocationError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subprocess_nonzero_raises_invocation_error(
    fake_pdf: Path, tmp_parsed_root: Path
) -> None:
    fake_proc = _make_fake_subprocess(returncode=1, stderr="some VLM server error")

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=AsyncMock(return_value=fake_proc),
    ):
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
    fake_proc = _make_fake_subprocess(returncode=0)

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=AsyncMock(return_value=fake_proc),
    ):
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
    output_dir = tmp_parsed_root / DOC_STEM / "hybrid_auto"

    async def create_bad_output(*args: Any, **kwargs: Any) -> MagicMock:
        output_dir.mkdir(parents=True, exist_ok=True)
        bad_content_list = [{"type": "text", "text": "Short content", "page_idx": 0}]
        (output_dir / f"{DOC_STEM}_content_list.json").write_text(
            json.dumps(bad_content_list), encoding="utf-8"
        )
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
    assert gate["page_markers_open"] != gate["pages"]


# ---------------------------------------------------------------------------
# g) health_check (parametrized: correct model / server error / timeout)
# ---------------------------------------------------------------------------


def _make_mock_http_client(
    status_code: int = 200,
    json_body: dict | None = None,
    raise_exc: Exception | None = None,
) -> AsyncMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json = MagicMock(return_value=json_body or {})

    mock_client = AsyncMock()
    if raise_exc is not None:
        mock_client.get = AsyncMock(side_effect=raise_exc)
    else:
        mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("correct_model", True),
        ("server_error_500", False),
        ("timeout", False),
    ],
)
async def test_health_check(
    tmp_parsed_root: Path, scenario: str, expected: bool
) -> None:
    client = MinerUClient(server_url="http://localhost:8001", parsed_root=tmp_parsed_root)

    if scenario == "correct_model":
        mock_http = _make_mock_http_client(
            status_code=200,
            json_body={"data": [{"id": "opendatalab/MinerU2.5-2509-1.2B"}]},
        )
    elif scenario == "server_error_500":
        mock_http = _make_mock_http_client(status_code=500, json_body={})
    else:  # timeout
        mock_http = _make_mock_http_client(
            raise_exc=httpx.TimeoutException("connect timeout")
        )

    with patch("httpx.AsyncClient", return_value=mock_http):
        result = await client.health_check()

    assert result is expected
