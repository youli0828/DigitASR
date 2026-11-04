"""Download the restricted dataset without Git; keep all source files local."""

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://gitcode.com/Universal-Tool/9baeb"
ARCHIVE = "https://raw.gitcode.com/Universal-Tool/9baeb/archive/refs/heads/main.zip"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/external/gitcode_digits"
LICENSE_NOTE = "Non-commercial only; do not redistribute the audio or archives."
EXPECTED_WAV_COUNT = 250


def extract(archive, destination):
    destination = destination.resolve()
    for entry in archive.infolist():
        if not (destination / entry.filename).resolve().is_relative_to(destination):
            raise ValueError("Archive path escapes the dataset directory")
    archive.extractall(destination)


def verify_count(destination):
    count = len(list(destination.rglob("*.wav")))
    if count != EXPECTED_WAV_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_WAV_COUNT} WAV files; found {count}")
    return count


def download(destination):
    destination = Path(destination).resolve()
    print(LICENSE_NOTE)
    if len(list(destination.rglob("*.wav"))) == EXPECTED_WAV_COUNT:
        verify_count(destination)
        print(f"Verified 250 WAV files in {destination}; skipping download.")
        return
    cached = destination / "source.zip"
    if cached.is_file():
        content = cached.read_bytes()
    else:
        with urllib.request.urlopen(ARCHIVE, timeout=60) as response:
            content = response.read()
    with tempfile.TemporaryDirectory(prefix="gitcode-digits-") as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            extract(archive, staging)
        for nested in sorted((staging / "9baeb-main").glob("*.zip")):
            with zipfile.ZipFile(nested) as archive:
                extract(archive, nested.parent / "audio")
        verify_count(staging)
        shutil.copytree(staging, destination, dirs_exist_ok=True)
    count = verify_count(destination)
    (destination / "source.zip").write_bytes(content)
    provenance = {
        "source": SOURCE,
        "archive_url": ARCHIVE,
        "archive_sha256": hashlib.sha256(content).hexdigest(),
        "usage": LICENSE_NOTE,
        "license_note": "Upstream has MIT LICENSE and a README prohibiting commercial use.",
        "wav_count": count,
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
