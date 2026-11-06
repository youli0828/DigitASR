"""Download only Mandarin digits 0-9 from the SpokeN-100 Zenodo ZIP (CC BY 4.0)."""

import argparse
import io
import json
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

URL = "https://zenodo.org/records/10810044/files/SpokeN-100.zip"


def mandarin_digit_member(name):
    parts = PurePosixPath(name).parts
    if (
        len(parts) == 4
        and parts[:2] == ("SpokeN-100", "mandarin_numbers")
        and parts[3] in {f"{digit}.wav" for digit in range(10)}
    ):
        return parts[2], int(PurePosixPath(name).stem)
    return None


def fetch_range(start, end):
    for attempt in range(5):
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise ValueError("Server must support HTTP range requests")
                data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("Incomplete HTTP range")
                return data
        except urllib.error.HTTPError as error:
            if error.code not in (429, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(max(10, int(error.headers.get("Retry-After", "60"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/spoken100_zh"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        "https://zenodo.org/api/records/10810044", timeout=120
    ) as response:
        record = json.load(response)
    archive = next(item for item in record["files"] if item["key"] == "SpokeN-100.zip")
    size = archive["size"]
    tail = fetch_range(size - 65536, size - 1)
    position = tail.rfind(b"PK\x05\x06")
    eocd = tail[position:]
    fields = struct.unpack("<4s4H2LH", eocd[:22])
    central_size, central_offset = fields[5:7]
    central = fetch_range(central_offset, central_offset + central_size - 1)
    # ZipFile adjusts offsets when only the central directory and EOCD are supplied.
    with zipfile.ZipFile(io.BytesIO(central + eocd)) as remote:
        selected = [i for i in remote.infolist() if mandarin_digit_member(i.filename)]
    if len(selected) != 320:
        raise ValueError(f"Expected 32 synthetic voices x 10 digits; found {len(selected)}")
    for index, info in enumerate(selected, 1):
        voice, digit = mandarin_digit_member(info.filename)
        target = args.output / voice / f"{digit}.wav"
        if target.exists() and zlib.crc32(target.read_bytes()) == info.CRC:
            continue
        offset = info.header_offset + central_offset
        # Upstream entries have no local extra fields; include headroom and verify below.
        block = fetch_range(offset, offset + 30 + len(info.filename.encode()) + 1024
                            + info.compress_size - 1)
        header = struct.unpack("<4s5H3L2H", block[:30])
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid local ZIP header")
        begin = 30 + header[-2] + header[-1]
        payload = block[begin:begin + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            payload = zlib.decompress(payload, -15)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Unsupported ZIP compression")
        if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
            raise ValueError(f"ZIP CRC/size mismatch: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        print(f"{index}/320 {voice}/{digit}.wav", flush=True)
        time.sleep(0.65)
    (args.output / "source.json").write_text(
        json.dumps({
            "source": "https://zenodo.org/records/10810044",
            "license": record["metadata"]["license"]["id"],
            "archive_checksum": archive["checksum"],
            "subset": "mandarin_numbers/<TTS voice>/0.wav through 9.wav only; no modulo mapping",
            "validation": "Each extracted member checked against ZIP CRC32 and size",
            "synthetic": True,
        }, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
