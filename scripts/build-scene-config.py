"""Build scene_config.json from docs/original-scenes.json.

    python scripts/build-scene-config.py [--check]

docs/original-scenes.json is the source of truth: 47 scenes written for Mirror,
no film, actor or quoted dialogue anywhere in it. scene_config.json is the
runtime shape the server and both shells already read -- a `scenes` dict keyed
by id plus a `levels` list -- so this script translates rather than redesigns,
and the two files cannot drift because one is generated from the other.

`--check` regenerates in memory and exits non-zero if the committed file does
not match, which is what the regression suite calls. Without it the file is
rewritten in place.

Media is written only when the file is actually on disk, so the config never
points at something that is not there. `ui.audio` and `ui.audio_seconds` appear
once scripts/generate-scene-audio.py has produced the mp3; `ui.art` follows in
stage 3. The duration is measured here rather than guessed, because the daily
pool filter reads it to decide what is short enough to be one of three lines in
an evening.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mp3_duration import mp3_duration_seconds   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "original-scenes.json"
TARGET = ROOT / "scene_config.json"
AUDIO_DIR = ROOT / "static" / "scenes" / "audio"
AUDIO_URL = "/static/scenes/audio/%s.mp3"
ART_DIR = ROOT / "static" / "scenes" / "art"
ART_URL = "/static/scenes/art/%s.webp"
ART_SM_URL = "/static/scenes/art/%s-sm.webp"

# Unlock thresholds are a product decision, not something the scene list knows,
# and they are unchanged from the film version: level 1 is open, level 2 wants
# 60% on a level 1 scene, level 3 wants 70%.
UNLOCK_SCORE = {1: 0, 2: 60, 3: 70}
DIFFICULTY = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}

# One colour per genre instead of the previous 46 ad-hoc per-scene values. The
# card colour is UI styling, not scene content, so it survives the rewrite --
# but a palette keyed on genre is something a new scene inherits for free.
GENRE_COLOR = {
    "Drama":     "#8a5a1a",
    "Comedy":    "#d4a843",
    "Romance":   "#c75050",
    "Action":    "#cc2233",
    "Suspense":  "#2a3a5a",
    "Thriller":  "#1a2a4a",
    "Sports":    "#e07b39",
    "Crime":     "#3a2a4a",
    "Family":    "#5a7d8c",
    "Adventure": "#4a7c8c",
    "Sci-Fi":    "#2a5a8a",
}


def build() -> dict:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    raw = source["scenes"]

    scenes, levels = {}, {}
    for s in raw:
        level = int(s["level"])
        if level not in UNLOCK_SCORE:
            raise SystemExit("scene %s has unknown level %r" % (s["id"], s["level"]))
        if s["genre"] not in GENRE_COLOR:
            raise SystemExit("scene %s has unknown genre %r" % (s["id"], s["genre"]))

        entry = {
            "title":       s["title"],
            "line":        s["line"],
            "translation": s["translation"],
            "genre":       s["genre"],
            "emotion":     s["emotion"],
            "difficulty":  DIFFICULTY[level],
            # Read by the stage 2 audio generator, not by the app.
            "voice":       s["voice"],
            "direction":   s["direction"],
            "starter":     bool(s.get("starter")),
            "ui": {"card_color": GENRE_COLOR[s["genre"]]},
        }

        mp3 = AUDIO_DIR / ("%s.mp3" % s["id"])
        if mp3.exists():
            entry["ui"]["audio"] = AUDIO_URL % s["id"]
            entry["ui"]["audio_seconds"] = mp3_duration_seconds(mp3)

        # Both sizes or neither: a card that falls back to the typographic
        # poster looks deliberate, whereas one that stretches the 400px thumb
        # to 800px looks broken. The client picks a size; it should never have
        # to decide whether a declared file is real.
        art = ART_DIR / ("%s.webp" % s["id"])
        art_sm = ART_DIR / ("%s-sm.webp" % s["id"])
        if art.exists() and art_sm.exists():
            entry["ui"]["art"] = ART_URL % s["id"]
            entry["ui"]["art_sm"] = ART_SM_URL % s["id"]

        if s.get("idioms"):
            entry["idioms"] = list(s["idioms"])

        scenes[s["id"]] = entry
        levels.setdefault(level, []).append(s["id"])

    return {
        "scenes": scenes,
        "levels": [
            {"level": n, "unlock_score": UNLOCK_SCORE[n], "scenes": levels[n]}
            for n in sorted(levels)
        ],
    }


def serialise(config: dict) -> str:
    # CRLF and 2-space indent, matching the file this replaces, so the diff is
    # content and not whitespace.
    return json.dumps(config, indent=2, ensure_ascii=False).replace("\n", "\r\n") + "\r\n"


def main() -> int:
    config = build()
    text = serialise(config)

    counts = {lv["level"]: len(lv["scenes"]) for lv in config["levels"]}
    starters = [sid for sid, s in config["scenes"].items() if s["starter"]]

    if "--check" in sys.argv:
        current = TARGET.read_text(encoding="utf-8", newline="") if TARGET.exists() else ""
        if current != text:
            print("scene_config.json is STALE -- rerun scripts/build-scene-config.py")
            return 1
        print("scene_config.json matches docs/original-scenes.json")
    else:
        TARGET.write_text(text, encoding="utf-8", newline="")
        print("wrote %s" % TARGET.name)

    total = len(config["scenes"])
    with_audio = sum(1 for sc in config["scenes"].values() if sc["ui"].get("audio"))
    with_art = sum(1 for sc in config["scenes"].values() if sc["ui"].get("art"))
    print("  scenes %d   levels %s   starter pack %d: %s"
          % (total, counts, len(starters), ", ".join(starters)))
    print("  audio %d/%d   art %d/%d" % (with_audio, total, with_art, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
