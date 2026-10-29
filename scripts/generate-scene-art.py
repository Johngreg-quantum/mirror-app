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
high. The blockbuster set ships at high, so that is the default: a single scene
regenerated later should match the other 46. In practice the API has billed
about $0.042 per high still, well under the table above.
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

MODEL = "gpt-image-2.5-sunburst"
SIZE = "1024x1536"
QUALITY = "high"
LARGE = (800, 1200)
SMALL = (400, 600)
WEBP_QUALITY = 82

USD_PER_TEXT_TOKEN = 5.00 / 1_000_000
USD_PER_IMAGE_OUT_TOKEN = 30.00 / 1_000_000
OUTPUT_TOKENS = {"low": 408, "medium": 1584, "high": 6240}

# Two looks, selected with --style.
#
# `blockbuster` (below) is what the shipped 47 are generated with today; the
# documentary set it replaced is kept, out of git, in _documentary-backup/.
#
# `poster` was the first batch. It reads as a poster:
# gold, backlit, dramatic, everyone facing camera -- and that is exactly what
# makes it read as generated rather than photographed.
#
# `documentary` is the answer to that. Available light instead of rim light,
# muted instead of graded, people turned away or small in frame instead of
# posed, and an explicit list of the things that signal "AI image" -- flare,
# glow, HDR, orange-and-teal, bokeh balls. Negatives matter more here than
# positives: the model's default for "cinematic" is the first look.
PROMPTS = {
    "poster": (
        "Original cinematic movie poster still, no text, no logos. {escena}. "
        "Moody film lighting, deep blacks, warm gold and amber highlights, shallow "
        "depth of field, 35mm film grain, anamorphic lens flare, dramatic "
        "composition with empty space at the top for a title. Photorealistic, "
        "original characters, not resembling any real actor or existing film."
    ),
    "documentary": (
        "Film still from a quiet independent film, shot on 35mm Kodak Portra 400. "
        "Natural available light only, muted desaturated colors, soft low contrast, "
        "subtle film grain. {escena}. {tension}Show the moment from a distance or from "
        "behind or the side; faces partly turned away or small in frame. Ordinary "
        "everyday clothes and real locations. Candid, unposed, slightly off-center "
        "imperfect framing with empty space at the top. Realistic, understated, "
        "documentary feel. No lens flare, no glow, no HDR, no orange-and-teal "
        "grading, no bokeh balls, no dramatic rim light, no text, no logos. Original "
        "people, not resembling any real actor or existing film."
    ),
    # `blockbuster` is the look of static/victory.mp4: theatrical key art,
    # colourful and epic. The documentary stills read as hazy and sad on the
    # cards; this one comes back bright on its own, which is why it is in
    # NO_LIFT_STYLES below.
    "blockbuster": (
        "Original cinematic movie poster key art, absolutely no text, no letters, "
        "no logos. {escena}. The main character is the hero of the frame, "
        "centered, full or three-quarter body, facing the camera or in a strong "
        "pose. Dramatic rim lighting outlining the figure, volumetric light rays "
        "and atmospheric haze, deep navy-blue and teal atmosphere with warm "
        "golden-amber highlights, rich saturated colors, high contrast, deep "
        "blacks, polished blockbuster poster look, epic scale. Large empty dark "
        "space at the top for a title. Original characters, not resembling any "
        "real actor or existing film."
    ),
}
DEFAULT_STYLE = "blockbuster"

# Styles whose stills are graded bright by the prompt itself. The exposure lift
# below exists for the muted documentary look; run on key art it would flatten
# the deep blacks the prompt asks for.
NO_LIFT_STYLES = {"blockbuster"}

# Genres where "quiet independent film" is the wrong instruction on its own: a
# man sprinting down a corridor is not a still life. This buys movement and
# hard light WITHOUT buying the look the documentary style exists to avoid --
# it adds practical sources and motion blur, and says nothing about flare,
# glow or grading, which the template still forbids further down.
TENSION_GENRES = {"Action", "Thriller", "Crime", "Suspense"}
TENSION_CLAUSE = (
    "Charged and kinetic: visible motion blur from real movement, deep shadow, "
    "and hard practical light sources already in the scene such as an emergency "
    "lamp, a torch or headlights. Still the same 35mm film stock, still "
    "handheld and unglamorous. "
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
    """The {escena} slot: what the frame literally shows.

    The scene's own `visual` field, which exists for exactly this. It used to
    be assembled from `direction` plus the genre and emotion, and `direction`
    opens with a mood rather than a frame -- "urgent and commanding, a leader
    in a crisis" let the model pick any crisis it liked, and `countdown`, a man
    running with a bag, came back as a flood evacuation. A sentence that says
    what is in shot cannot drift like that.

    Falls back to the old assembly for a scene with no `visual` yet, so this
    degrades to the previous behaviour rather than to an empty prompt.
    """
    visual = (scene.get("visual") or "").strip()
    if visual:
        return visual
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


# A still darker than this disappears under the card gradient. 80 is where a
# face in shadow is still readable at 180px wide, measured against the ones
# that already worked.
LIFT_BELOW = 70
LIFT_TARGET = 80


def _mean_luma(im: Image.Image) -> float:
    small = im.convert("L").resize((64, 96), Image.LANCZOS)
    data = list(small.getdata())
    return sum(data) / len(data)


def lift_exposure(im: Image.Image) -> tuple:
    """Brighten a too-dark still towards LIFT_TARGET without clipping.

    Gamma, not a multiply or a brightness offset: out = 255*(in/255)**g maps
    255 to 255 whatever g is, so the highlights cannot blow -- a lamp stays a
    lamp instead of becoming a white disc. It lifts the shadows and midtones,
    which is the part that was lost.

    The gamma is solved by bisection rather than derived, because the mean of a
    gamma-mapped image has no closed form. Twenty passes gets within a
    fraction of a level, and is instant on a 64x96 thumbnail.
    """
    before = _mean_luma(im)
    if before >= LIFT_BELOW:
        return im, before, before, 1.0

    probe = im.convert("L").resize((64, 96), Image.LANCZOS)
    lo, hi = 0.30, 1.0
    for _ in range(20):
        g = (lo + hi) / 2
        lut = [round(255 * ((i / 255) ** g)) for i in range(256)]
        mean = sum(probe.point(lut).getdata()) / (64 * 96)
        if mean < LIFT_TARGET:
            hi = g
        else:
            lo = g
    g = (lo + hi) / 2
    lut = [round(255 * ((i / 255) ** g)) for i in range(256)]
    lifted = im.point(lut * len(im.getbands()))
    return lifted, before, _mean_luma(lifted), g


def write_cuts(png_bytes: bytes, scene_id: str, out_dir: Path,
               lift: bool = True) -> tuple:
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / ("%s.png" % scene_id)
    # On a recut the bytes came FROM this file; rewriting it would be a no-op
    # that risks truncating the only copy of something the API charged for.
    if not (raw.exists() and raw.read_bytes() == png_bytes):
        raw.write_bytes(png_bytes)

    with Image.open(raw) as src:
        im = src.convert("RGB")
        if lift:
            im, before, after, gamma = lift_exposure(im)
        else:
            before = after = _mean_luma(im)
            gamma = 1.0
        for size, suffix in ((LARGE, ""), (SMALL, "-sm")):
            out = out_dir / ("%s%s.webp" % (scene_id, suffix))
            im.resize(size, Image.LANCZOS).save(
                out, "WEBP", quality=WEBP_QUALITY, method=6)
    large = out_dir / ("%s.webp" % scene_id)
    small = out_dir / ("%s-sm.webp" % scene_id)
    return large.stat().st_size, small.stat().st_size, before, after, gamma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quality", default=QUALITY, choices=sorted(OUTPUT_TOKENS))
    ap.add_argument("--style", default=DEFAULT_STYLE, choices=sorted(PROMPTS),
                    help="which look to generate (default: %s)" % DEFAULT_STYLE)
    ap.add_argument("--recut", action="store_true",
                    help="re-encode the webp cuts from the PNGs already in "
                         "_raw/, calling nothing and costing nothing")
    ap.add_argument("--out-dir", default=None,
                    help="write somewhere other than static/scenes/art, to try a "
                         "style without overwriting what is shipped")
    args = ap.parse_args()

    # Resolved against the repo root, so a relative --out-dir works from
    # anywhere and the line printed below is always repo-relative.
    out_dir = (ROOT / args.out_dir).resolve() if args.out_dir else ART_DIR
    prompt_template = PROMPTS[args.style]
    lift = args.style not in NO_LIFT_STYLES

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

    out_dir.mkdir(parents=True, exist_ok=True)

    pending, skipped = [], []
    for sid in targets:
        done = ((out_dir / ("%s.webp" % sid)).exists()
                and (out_dir / ("%s-sm.webp" % sid)).exists())
        (skipped if done and not args.force else pending).append(sid)

    # ── Recut: no network, no spend. The PNG in _raw/ is the thing the API
    # was paid for; the webp cuts are derived and can be rebuilt whenever the
    # derivation changes, which is what an exposure lift is.
    if args.recut:
        raw_dir = out_dir / "_raw"
        print("recutting from %s   (no API calls)" % raw_dir.name)
        done, lifted, missing = 0, 0, []
        for sid in targets:
            raw = raw_dir / ("%s.png" % sid)
            if not raw.exists():
                missing.append(sid)
                continue
            big, small, before, after, gamma = write_cuts(
                raw.read_bytes(), sid, out_dir, lift=lift)
            done += 1
            if gamma < 1:
                lifted += 1
                print("  %-18s %6dB / %5dB   lift %.0f -> %.0f  (gamma %.2f)"
                      % (sid, big, small, before, after, gamma))
        print("\nrecut %d   lifted %d   left alone %d" % (done, lifted, done - lifted))
        if missing:
            print("no raw PNG for: %s" % ", ".join(missing))
        return 0 if not missing else 1

    out_tokens = OUTPUT_TOKENS[args.quality]
    # Text is ~130 tokens a prompt; nothing is sent as image input.
    est = (len(pending) * out_tokens * USD_PER_IMAGE_OUT_TOKEN
           + len(pending) * 130 * USD_PER_TEXT_TOKEN)
    print("model %s   %s   quality %s   style %s" % (MODEL, SIZE, args.quality, args.style))
    try:
        where = out_dir.relative_to(ROOT)
    except ValueError:
        where = out_dir
    print("writing to %s" % where)
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
        tension = TENSION_CLAUSE if scene.get("genre") in TENSION_GENRES else ""
        prompt = prompt_template.format(
            escena=scene_description(dict(scene, _id=sid)),
            tension=tension,
        )
        try:
            resp = client.images.generate(
                model=MODEL, prompt=prompt, size=SIZE, quality=args.quality)
        except Exception as exc:                      # noqa: BLE001
            print("  %-18s FAILED  %s" % (sid, str(exc)[:110]))
            continue

        png = base64.b64decode(resp.data[0].b64_json)
        big, small, before, after, gamma = write_cuts(png, sid, out_dir, lift=lift)

        usage = getattr(resp, "usage", None)
        if usage is not None:
            cost = ((getattr(usage, "input_tokens", 0) or 0) * USD_PER_TEXT_TOKEN
                    + (getattr(usage, "output_tokens", 0) or 0) * USD_PER_IMAGE_OUT_TOKEN)
        else:
            cost = out_tokens * USD_PER_IMAGE_OUT_TOKEN
        spent += cost
        made.append(sid)
        lift = ("  lift %.0f->%.0f g%.2f" % (before, after, gamma)) if gamma < 1 else ""
        print("  %-18s %6dB / %5dB  ~$%.4f%s"
              % (sid, big, small, cost, lift))

    print("\n%d image(s)   ~$%.3f total" % (len(made), spent))
    if made and len(made) < len(scenes):
        print("at that rate, all %d scenes would cost about $%.2f"
              % (len(scenes), spent / len(made) * len(scenes)))
    return 0 if len(made) == len(pending) else 1


if __name__ == "__main__":
    raise SystemExit(main())
