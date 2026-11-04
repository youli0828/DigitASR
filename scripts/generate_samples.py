"""Generate original Mandarin formant-synthesized speech with eSpeak NG (no recordings)."""

import argparse
import csv
import ctypes
import hashlib
import json
from pathlib import Path

import espeakng_loader
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DIGITS = "零一二三四五六七八九"
# Fixed, disjoint synthesis profiles; none is an augmented copy of another WAV.
PROFILES = [
    ("train", 130, 35),
    ("train", 150, 45),
    ("train", 170, 55),
    ("train", 190, 65),
    ("train", 140, 60),
    ("train", 160, 70),
    ("train", 180, 40),
    ("train", 200, 50),
    ("test", 145, 40),
    ("test", 165, 50),
    ("test", 185, 60),
    ("test", 155, 65),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/samples"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.csv").exists():
        parser.error("Output already contains a dataset; choose a new --output directory")
    library = ctypes.CDLL(espeakng_loader.get_library_path())
    library.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    library.espeak_Initialize.restype = ctypes.c_int
    sample_rate = library.espeak_Initialize(2, 0, str(espeakng_loader.get_data_path()).encode(), 0)
    if sample_rate <= 0:
        raise RuntimeError("eSpeak NG initialization failed")
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
    )
    chunks = []

    @callback_type
    def callback(wave, count, events):
        if count > 0:
            chunks.append(ctypes.string_at(wave, count * 2))
        return 0

    library.espeak_SetSynthCallback.argtypes = [callback_type]
    library.espeak_SetSynthCallback(callback)
    library.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
    library.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    library.espeak_Synth.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.espeak_Info.argtypes = [ctypes.c_void_p]
    library.espeak_Info.restype = ctypes.c_char_p
    version = library.espeak_Info(None).decode()
    if library.espeak_SetVoiceByName(b"cmn") != 0:
        raise RuntimeError("Mandarin voice cmn is unavailable")
    rows = []
    try:
        for profile, (split, rate, pitch) in enumerate(PROFILES):
            for parameter, value in ((1, rate), (3, pitch)):
                if library.espeak_SetParameter(parameter, value, 0) != 0:
                    raise RuntimeError("Unable to configure speech synthesis")
            for digit, text in enumerate(DIGITS):
                chunks.clear()
                encoded = text.encode("utf-8") + b"\0"
                buffer = ctypes.create_string_buffer(encoded)
                status = library.espeak_Synth(buffer, len(encoded), 0, 1, 0, 1, None, None)
                if status != 0 or not chunks:
                    raise RuntimeError(f"Synthesis failed: {text}")
                audio = np.frombuffer(b"".join(chunks), dtype="<i2").astype(np.float32) / 32768
                audio = resample_poly(audio, 16000, sample_rate)
                relative = f"{split}/{digit}_profile{profile:02d}.wav"
                target = args.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, audio, 16000, subtype="PCM_16")
                rows.append(
                    {
                        "path": relative,
                        "label": digit,
                        "split": split,
                        "group": f"espeak_profile{profile:02d}",
                        "text": text,
                        "rate": rate,
                        "pitch": pitch,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                )
    finally:
        library.espeak_Terminate()
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "engine": "eSpeak NG",
        "engine_version": version,
        "loader_version": "0.2.4",
        "voice": "cmn",
        "output_sample_rate": 16000,
        "engine_sample_rate": sample_rate,
        "source": "Original synthesis of the ten Unicode characters 零一二三四五六七八九",
        "profiles": PROFILES,
        "count": len(rows),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"count": len(rows), "engine_version": version}))


if __name__ == "__main__":
    main()
