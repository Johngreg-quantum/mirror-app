"""Duration of a CBR mp3, read from its frame headers.

Shared by scripts/generate-scene-audio.py, which reports it, and
scripts/build-scene-config.py, which writes it into scene_config.json for the
daily pool filter to read. No external binary: ffprobe is not a dependency
anywhere else in this repo.

Handles MPEG-2 as well as MPEG-1, which is not optional here: gpt-4o-mini-tts
returns 24 kHz MPEG-2 Layer III, where a frame is 576 samples rather than 1152
and the bitrate table is a different one. Read as MPEG-1 it reports exactly
half the true length -- wrong in the direction that would quietly admit an
over-long line to the daily pool.
"""
from pathlib import Path

# Layer III bitrate tables, in kbps, indexed by the header's 4 bitrate bits.
_BITRATE_V1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
_BITRATE_V2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
# Keyed by the header's 2 version bits: 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5.
_SAMPLE_RATE = {3: [44100, 48000, 32000],
                2: [22050, 24000, 16000],
                0: [11025, 12000, 8000]}


def mp3_duration_seconds(path) -> float:
    data = Path(path).read_bytes()
    i, total = 0, 0.0
    if data[:3] == b"ID3" and len(data) > 10:
        i = 10 + ((data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14
                  | (data[8] & 0x7F) << 7 | (data[9] & 0x7F))
    while i + 4 <= len(data):
        if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
            i += 1
            continue
        version = (data[i + 1] >> 3) & 0x03
        layer = (data[i + 1] >> 1) & 0x03
        rates = _SAMPLE_RATE.get(version)
        sr_index = (data[i + 2] & 0x0C) >> 2
        if layer != 1 or rates is None or sr_index > 2:   # layer 1 == Layer III
            i += 1
            continue
        table = _BITRATE_V1 if version == 3 else _BITRATE_V2
        bitrate = table[(data[i + 2] & 0xF0) >> 4] * 1000
        rate = rates[sr_index]
        if not bitrate:
            i += 1
            continue
        samples = 1152 if version == 3 else 576
        padding = (data[i + 2] & 0x02) >> 1
        length = int(samples / 8 * bitrate / rate) + padding
        total += samples / rate
        i += max(length, 1)
    return round(total, 2)
