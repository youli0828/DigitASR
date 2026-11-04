"""Download the restricted dataset without Git; keep all source files local."""

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://gitcode.com/Universal-Tool/9baeb"
ARCHIVE = "https://raw.gitcode.com/Universal-Tool/9baeb/archive/refs/heads/main.zip"


def extract(archive, destination):
    destination = destination.resolve()
    for entry in archive.infolist():
        if not (destination / entry.filename).resolve().is_relative_to(destination):
            raise ValueError("Archive path escapes the dataset directory")
    archive.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/gitcode_digits"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARCHIVE, timeout=60) as response:
        content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        extract(archive, args.output)
    (args.output / "source.zip").write_bytes(content)
    for nested in sorted((args.output / "9baeb-main").glob("*.zip")):
        with zipfile.ZipFile(nested) as archive:
            extract(archive, nested.parent / "audio")
    provenance = {
        "source": SOURCE,
        "archive_url": ARCHIVE,
        "archive_sha256": hashlib.sha256(content).hexdigest(),
        "usage": "Non-commercial only; do not redistribute the audio or archives.",
        "license_note": "Upstream has MIT LICENSE and a README prohibiting commercial use.",
        "wav_count": len(list(args.output.rglob("*.wav"))),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
