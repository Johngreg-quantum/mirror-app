"""Generate one mp3 per scene with OpenAI gpt-4o-mini-tts.

    python scripts/generate-scene-audio.py last_train          # one scene
    python scripts/generate-scene-audio.py --all               # every scene
    python scripts/generate-scene-audio.py --all --dry-run     # cost only
    python scripts/generate-scene-audio.py --all --force       # re-generate

Each scene carries its own `voice` and its own `direction`; the direction is
passed as the model's `instructions`, which is what makes one line sound hurt
and the next urgent. Those two fields exist in docs/original-scenes.json for
this script and are stripped before scene data reaches the client.

Existing files are SKIPPED unless --force. Regenerating is not free and the
takes are not reproducible -- the same text and the same instructions give a
different read each time -- so re-running must never silently replace audio
that has already been listened to and approved.

Pricing (checked 19 Oct 2026, developers.openai.com/api/docs/pricing):
  text input   $0.60 per 1M tokens
  audio output $12.00 per 1M tokens
Audio tokens are not knowable before the call, so the estimate below prices
output from measured duration at the observed token rate, and the real figure
is printed per scene from the API's own usage numbers when it reports them.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "scene_config.json"
OUT_DIR = ROOT / "static" / "scenes" / "audio"

MODEL = "gpt-4o-mini-tts"
FORMAT = "mp3"
USD_PER_TEXT_TOKEN = 0.60 / 1_000_000
USD_PER_AUDIO_TOKEN = 12.00 / 1_000_000
# OpenAI bills TTS output in audio tokens. 1250 tokens/minute is the rate their
# own "about $0.015 per minute" guidance implies at $12/1M, and it is only used
# for the projection -- per-scene figures come from reported usage.
AUDIO_TOKENS_PER_SECOND = 1250 / 60


# Layer III bitrate tables, in kbps, indexed by the header's 4 bitrate bits.
_BITRATE_V1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
_BITRATE_V2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
# Keyed by the header's 2 version bits: 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5.
_SAMPLE_RATE = {3: [44100, 48000, 32000],
                2: [22050, 24000, 16000],
                0: [11025, 12000, 8000]}


def mp3_duration_seconds(path: Path) -> float:
    """Duration of a CBR mp3, summed from its frame headers.

    No external binary: ffprobe is not a dependency anywhere else in this repo
    and the daily pool filter in stage 2 needs this number at build time.

    Handles MPEG-2 as well as MPEG-1, which is not optional here: gpt-4o-mini-tts
    returns 24 kHz MPEG-2 Layer III, where a frame is 576 samples rather than
    1152 and the bitrate table is a different one. Reading it as MPEG-1 reports
    exactly half the real length, which is wrong in the direction that would
    quietly let an over-long line into the daily pool.
    """
    data = path.read_bytes()
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


def estimate_text_tokens(scene: dict) -> int:
    # ~4 characters per token is close enough for a figure in cents, and the
    # real number comes back in usage when the API reports it.
    chars = len(scene["line"]) + len(scene["direction"])
    return max(1, round(chars / 4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    scenes = config["scenes"]

    if args.all:
        targets = list(scenes)
    elif args.scene_ids:
        targets = args.scene_ids
        unknown = [s for s in targets if s not in scenes]
        if unknown:
            print("unknown scene id(s): %s" % ", ".join(unknown))
            return 1
    else:
        ap.error("name at least one scene id, or pass --all")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pending, skipped = [], []
    for sid in targets:
        if (OUT_DIR / f"{sid}.{FORMAT}").exists() and not args.force:
            skipped.append(sid)
        else:
            pending.append(sid)

    text_tokens = sum(estimate_text_tokens(scenes[s]) for s in pending)
    print("model %s   to generate %d   already present %d"
          % (MODEL, len(pending), len(skipped)))
    print("estimated text input: %d tokens = $%.5f"
          % (text_tokens, text_tokens * USD_PER_TEXT_TOKEN))
    if args.dry_run:
        print("(dry run -- nothing called, nothing charged)")
        return 0
    if not pending:
        print("nothing to do; pass --force to regenerate")
        return 0

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (env or .env)")
        return 1
    client = OpenAI()

    total_seconds, total_usd = 0.0, 0.0
    for sid in pending:
        scene = scenes[sid]
        dest = OUT_DIR / f"{sid}.{FORMAT}"
        with client.audio.speech.with_streaming_response.create(
            model=MODEL,
            voice=scene["voice"],
            input=scene["line"],
            instructions=scene["direction"],
            response_format=FORMAT,
        ) as response:
            response.stream_to_file(dest)

        seconds = mp3_duration_seconds(dest)
        audio_tokens = seconds * AUDIO_TOKENS_PER_SECOND
        usd = (estimate_text_tokens(scene) * USD_PER_TEXT_TOKEN
               + audio_tokens * USD_PER_AUDIO_TOKEN)
        total_seconds += seconds
        total_usd += usd
        print("  %-18s %-8s %5.2fs  %7d bytes  ~$%.5f"
              % (sid, scene["voice"], seconds, dest.stat().st_size, usd))

    print("\n%d file(s)   %.1fs of audio   ~$%.4f total"
          % (len(pending), total_seconds, total_usd))
    if len(pending) < len(scenes):
        per = total_usd / len(pending)
        print("at that rate, all %d scenes would cost about $%.2f"
              % (len(scenes), per * len(scenes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
