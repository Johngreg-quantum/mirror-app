"""Generate one portrait still per scene with the OpenAI image API.

    python scripts/generate-scene-art.py last_train            # one scene
    python scripts/generate-scene-art.py --all                 # every scene
    python scripts/generate-scene-art.py --all --dry-run       # cost only
    python scripts/generate-scene-art.py --all --force         # regenerate

Writes two cuts per scene, which is what scripts/build-scene-config.py looks
for before it will declare `ui.art`:

    static/scenes/art/<id>.webp      800x1200
    static/scenes/art/<id>-sm.webp   400x600

The raw 1024x1536 PNG is kept under static/scenes/art/_raw/ so a crop or a
re-encode never costs another generation. _raw is not served: nothing links to
it and it is excluded from the repo by .gitignore.

  STYLE CONSISTENCY COMES FROM THE PROMPT, NOT FROM A REFERENCE IMAGE

The first attempt attached the anchor still to every later call through the
edit endpoint. It held the grade beautifully and also held everything else:
`the_heist` and `lost_city` came back with the same woman, the same coat and
the same face as `last_train`, on a different set. A catalogue of 47 scenes
starring one character is not a catalogue.

The grade is specified in enough detail in the template -- deep blacks, warm
gold and amber, shallow depth of field, 35mm grain, anamorphic flare -- that
plain generation produces the same family on its own, which is what the anchor
image demonstrated before it was used as a reference. So every scene is
generated independently and gets its own cast.

  LIKENESS

The prompt ends by asking for original characters that resemble no real actor
and no existing film, because the entire point of this rewrite is that Mirror
owns what it ships. Anything that comes back looking like a specific person or
a specific film should be regenerated rather than shipped.

Pricing (checked 19 Oct 2026, developers.openai.com/api/docs/pricing):
  gpt-image-2.5-sunburst   text in $5/1M   image in $8/1M   image out $30/1M
A 1024x1536 portrait costs 408 output tokens at low, 1584 at medium and 6240 at
high. Medium is what this ships: low loses the grain and the falloff that make
the still look like a frame rather than a render.
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "scene_config.json"
ART_DIR = ROOT / "static" / "scenes" / "art"
RAW_DIR = ART_DIR / "_raw"

MODEL = "gpt-image-2.5-sunburst"
SIZE = "1024x1536"
QUALITY = "medium"
LARGE = (800, 1200)
SMALL = (400, 600)
WEBP_QUALITY = 82

USD_PER_TEXT_TOKEN = 5.00 / 1_000_000
USD_PER_IMAGE_OUT_TOKEN = 30.00 / 1_000_000
OUTPUT_TOKENS = {"low": 408, "medium": 1584, "high": 6240}

PROMPT = (
    "Original cinematic movie poster still, no text, no logos. {escena}. "
    "Moody film lighting, deep blacks, warm gold and amber highlights, shallow "
    "depth of field, 35mm film grain, anamorphic lens flare, dramatic "
    "composition with empty space at the top for a title. Photorealistic, "
    "original characters, not resembling any real actor or existing film."
)


# Scenes whose own `direction` trips the image safety filter. The override
# describes the same frame in terms of light and architecture instead of fear,
# which is what the still actually needs to show anyway.
#
# docs/original-scenes.json is the source of truth for the SCENE and is not
# edited to suit the image model; this map is the image model's problem, kept
# where the image model lives.
DESCRIPTION_OVERRIDES = {
    # "A frightened whisper in a dark house" reads as a home invasion.
    "locked_door": (
        "A person standing alone at the end of a dim hallway at night, lit by "
        "one lamp, looking back toward a closed door, suspense scene, the mood "
        "is still and uneasy"
    ),
}


def scene_description(scene: dict) -> str:
    """The {escena} slot: what the frame shows, from the scene's own fields.

    `direction` opens with the visual setup and then turns into vocal notes for
    the text-to-speech ("slow, deliberate, a short pause after 'train'"), which
    mean nothing to an image model. Only the first sentence is used, with the
    genre and emotion added so the grade and the face have something to follow.
    """
    override = DESCRIPTION_OVERRIDES.get(scene.get("_id"))
    if override:
        return override
    setup = (scene.get("direction") or "").split(".")[0].strip()
    bits = [setup] if setup else [scene.get("title", "")]
    genre = scene.get("genre")
    emotion = scene.get("emotion")
    if genre:
        bits.append("%s scene" % genre.lower())
    if emotion:
        bits.append("the mood is %s" % emotion.lower())
    return ", ".join(b for b in bits if b)


def write_cuts(png_bytes: bytes, scene_id: str) -> tuple:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIR / ("%s.png" % scene_id)
    raw.write_bytes(png_bytes)

    with Image.open(raw) as im:
        im = im.convert("RGB")
        for size, suffix in ((LARGE, ""), (SMALL, "-sm")):
            out = ART_DIR / ("%s%s.webp" % (scene_id, suffix))
            im.resize(size, Image.LANCZOS).save(
                out, "WEBP", quality=WEBP_QUALITY, method=6)
    large = ART_DIR / ("%s.webp" % scene_id)
    small = ART_DIR / ("%s-sm.webp" % scene_id)
    return large.stat().st_size, small.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quality", default=QUALITY, choices=sorted(OUTPUT_TOKENS))
    args = ap.parse_args()

    scenes = json.loads(CONFIG.read_text(encoding="utf-8"))["scenes"]
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

    ART_DIR.mkdir(parents=True, exist_ok=True)

    pending, skipped = [], []
    for sid in targets:
        done = ((ART_DIR / ("%s.webp" % sid)).exists()
                and (ART_DIR / ("%s-sm.webp" % sid)).exists())
        (skipped if done and not args.force else pending).append(sid)

    out_tokens = OUTPUT_TOKENS[args.quality]
    # Text is ~130 tokens a prompt; nothing is sent as image input.
    est = (len(pending) * out_tokens * USD_PER_IMAGE_OUT_TOKEN
           + len(pending) * 130 * USD_PER_TEXT_TOKEN)
    print("model %s   %s   quality %s" % (MODEL, SIZE, args.quality))
    print("to generate %d   already present %d   estimated $%.2f"
          % (len(pending), len(skipped), est))
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


    spent, made = 0.0, []
    for sid in pending:
        scene = scenes[sid]
        prompt = PROMPT.format(escena=scene_description(dict(scene, _id=sid)))
        try:
            resp = client.images.generate(
                model=MODEL, prompt=prompt, size=SIZE, quality=args.quality)
        except Exception as exc:                      # noqa: BLE001
            print("  %-18s FAILED  %s" % (sid, str(exc)[:110]))
            continue

        png = base64.b64decode(resp.data[0].b64_json)
        big, small = write_cuts(png, sid)

        usage = getattr(resp, "usage", None)
        if usage is not None:
            cost = ((getattr(usage, "input_tokens", 0) or 0) * USD_PER_TEXT_TOKEN
                    + (getattr(usage, "output_tokens", 0) or 0) * USD_PER_IMAGE_OUT_TOKEN)
        else:
            cost = out_tokens * USD_PER_IMAGE_OUT_TOKEN
        spent += cost
        made.append(sid)
        print("  %-18s %6dB / %5dB  ~$%.4f"
              % (sid, big, small, cost))

    print("\n%d image(s)   ~$%.3f total" % (len(made), spent))
    if made and len(made) < len(scenes):
        print("at that rate, all %d scenes would cost about $%.2f"
              % (len(scenes), spent / len(made) * len(scenes)))
    return 0 if len(made) == len(pending) else 1


if __name__ == "__main__":
    raise SystemExit(main())
