import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from digit_asr.dataset import DIGIT_WORDS, prepare_gitcode_manifest, read_manifest

spec = importlib.util.spec_from_file_location(
    "download_gitcode_digits",
    Path(__file__).resolve().parents[1] / "scripts/download_gitcode_digits.py",
)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


@pytest.fixture
def archive_bytes():
    audio = io.BytesIO()
    with zipfile.ZipFile(audio, "w") as archive:
        for word in DIGIT_WORDS:
            for index in range(1, 26):
                archive.writestr(f"digits/{word}-{index}.wav", f"{word}-{index}")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("9baeb-main/数字0到9中文发音.zip", audio.getvalue())
        archive.writestr("9baeb-main/README.md", "Non-commercial only")
    return outer.getvalue()


def test_download_and_idempotence(tmp_path, monkeypatch, capsys, archive_bytes):
    calls = []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(archive_bytes)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", urlopen)
    output = tmp_path / "dataset"
    downloader.download(output)
    assert calls == [(downloader.ARCHIVE, 60)]
    assert len(list(output.rglob("*.wav"))) == 250
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["wav_count"] == 250
    assert provenance["usage"] == downloader.LICENSE_NOTE
    rows = read_manifest(prepare_gitcode_manifest(output))
    assert len(rows) == 250
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    downloader.download(output)
    assert len(calls) == 1
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert capsys.readouterr().out.count(downloader.LICENSE_NOTE) >= 2


def test_incomplete_cached_extraction_is_repaired(tmp_path, monkeypatch, archive_bytes):
    def no_network(*args, **kwargs):
        pytest.fail("Cached archive must be reused without network access")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", no_network)
    (tmp_path / "source.zip").write_bytes(archive_bytes)
    partial = tmp_path / "9baeb-main/audio/digits"
    partial.mkdir(parents=True)
    (partial / "zero-1.wav").write_bytes(b"zero-1")
    downloader.download(tmp_path)
    assert downloader.verify_count(tmp_path) == 250


def test_wrong_count_is_rejected_before_install(tmp_path, monkeypatch):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("9baeb-main/zero-1.wav", b"sample")
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content.getvalue())
    )
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="Expected exactly 250 WAV files; found 1"):
        downloader.download(output)
    assert not output.exists()


def test_archive_path_traversal_is_rejected(tmp_path):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("../escaped.wav", b"sample")
    with zipfile.ZipFile(content) as archive:
        with pytest.raises(ValueError, match="escapes"):
            downloader.extract(archive, tmp_path / "dataset")
    assert not (tmp_path / "escaped.wav").exists()


def test_default_output_is_repo_relative():
    assert downloader.DEFAULT_OUTPUT == Path(__file__).resolve().parents[1] / (
        "data/external/gitcode_digits"
    )
