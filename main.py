import os
import json
import re
import time
import uuid
import sqlite3
import difflib
import hashlib
import hmac
import logging
import tempfile
import traceback

logger = logging.getLogger(__name__)
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone, date
from email.utils import formatdate
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import markdown
import openai
import httpx
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

# Interactive API docs are disabled. There is no public API and no third-party
# developers, so /docs, /redoc and /openapi.json only published the full
# endpoint surface and request schemas to anyone who asked. Set these back to
# their defaults (or to unguessable paths) if an external integration ever
# needs them.
app = FastAPI(
    title="MIRROR — Movie Scene Language Learning",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    # HEAD: FastAPI/Starlette strips the body automatically, only headers are sent.
    # UptimeRobot's free plan sends HEAD; without this it 405'd and the monitor
    # showed "Down" for 2 months, defeating the Render keep-alive.
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

# ---------------------------------------------------------------------------
# CORS — only the production domain and local dev origins are allowed.
# The frontend is served from the same domain as the API, so these origins
# matter only for external/cross-origin callers.
# ---------------------------------------------------------------------------
_ALLOWED_ORIGINS = [
    "https://mirrorspeak.app",
    "https://www.mirrorspeak.app",
    # Render's own hostname still serves the app and is kept working
    # deliberately — old links, bookmarks and the health monitor use it.
    "https://mirror-app-z8wr.onrender.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_APP_DIR, "static")
_INDEX_HTML_PATH = os.path.join(_APP_DIR, "index.html")
_NEW_SHELL_INDEX_HTML_PATH = os.path.join(_STATIC_DIR, "new-shell", "index.html")
_SCENE_CONFIG_PATH = os.path.join(_APP_DIR, "scene_config.json")
_PRIVACY_MD_PATH = os.path.join(_APP_DIR, "docs", "privacy-policy.md")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
_RATE_WINDOW        = 3600          # sliding window — 1 hour in seconds
_IP_LIMIT           = 100           # max API requests per IP per window
_SUBMIT_LIMIT       = 10            # max recording submissions per user per window
_MAX_AUDIO_BYTES    = 10 * 1024 * 1024   # 10 MB upload cap
_ALLOWED_AUDIO_EXT  = {".webm", ".mp4", ".ogg", ".mp3", ".wav", ".m4a"}

# Sliding-window buckets keyed by IP address (in-memory; resets on restart,
# which is acceptable for a single-instance deployment).
_ip_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """Return the real client IP, honouring Render's X-Forwarded-For header."""
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def ip_rate_limit(request: Request, call_next):
    """Reject requests from IPs that exceed _IP_LIMIT API calls per hour.
    OPTIONS (CORS preflight) and the HTML root are exempt."""
    if request.method != "OPTIONS" and request.url.path.startswith("/api/"):
        ip  = _client_ip(request)
        now = time.monotonic()
        dq  = _ip_hits[ip]
        while dq and dq[0] < now - _RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _IP_LIMIT:
            return JSONResponse(
                {"detail": "Too many requests — please try again later"},
                status_code=429,
                headers={"Retry-After": "3600"},
            )
        dq.append(now)
    return await call_next(request)


def _check_submit_rate(user_id: int, conn) -> None:
    """Raise 429 when this user has already submitted _SUBMIT_LIMIT recordings
    in the past hour.  Uses the scores table so the check survives restarts."""
    cur = conn.cursor()
    if USE_PG:
        cur.execute(
            "SELECT COUNT(*) FROM scores "
            "WHERE user_id = %s AND created_at > NOW() - INTERVAL '1 hour'",
            (user_id,),
        )
    else:
        cur.execute(
            "SELECT COUNT(*) FROM scores "
            "WHERE user_id = ? AND created_at > datetime('now', '-1 hour')",
            (user_id,),
        )
    count = cur.fetchone()[0]
    if count >= _SUBMIT_LIMIT:
        raise HTTPException(
            429,
            f"Submission limit reached — max {_SUBMIT_LIMIT} recordings per hour",
        )


# ---------------------------------------------------------------------------
DB_PATH   = "mirror.db"

# JWT signing secret. Fail closed at startup: a missing or default secret means
# every token is forgeable by anyone who can read this public repo, so we refuse
# to boot rather than fall back to a known value. On Render a failed startup
# keeps the previous healthy deploy serving; locally it means JWT_SECRET must be
# set in the environment (or .env) before the app will run.
_DEFAULT_JWT_SECRET = "change-me-to-a-long-random-string-in-production"
SECRET    = os.getenv("JWT_SECRET", "")
if not SECRET or SECRET == _DEFAULT_JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET is not set (or is still the default placeholder). Set it to a "
        "strong random value, e.g. `python -c \"import secrets; print(secrets.token_hex(32))\"`. "
        "Refusing to start — otherwise every auth token would be forgeable."
    )
ALGORITHM = "HS256"
TOKEN_TTL = 30  # days
APP_BASE_URL = os.getenv("APP_BASE_URL", "").rstrip("/")

# Absolute origin for URLs that leave the app entirely — links handed to an
# external service, which cannot resolve a relative path. build_app_url() below
# stays relative when APP_BASE_URL is unset (correct for in-app links); this
# falls back to the canonical public domain instead, since "/" would be
# meaningless to a payment provider.
CANONICAL_BASE_URL = APP_BASE_URL or "https://mirrorspeak.app"

# Version stamped onto a user's recording consent, so a stored consent records
# which text was actually shown. Keep this equal to the effective date in the
# header of docs/privacy-policy.md. Bumping it does NOT re-prompt anyone on its
# own — the gate checks only whether recording_consent_at is set.
RECORDING_CONSENT_VERSION = "2026-08-23"

# Lemon Squeezy billing
LS_API_KEY        = os.getenv("LEMONSQUEEZY_API_KEY", "")
LS_SIGNING_SECRET = os.getenv("LEMONSQUEEZY_SIGNING_SECRET", "")
LS_MONTHLY_ID     = os.getenv("LEMONSQUEEZY_MONTHLY_VARIANT_ID", "")
LS_YEARLY_ID      = os.getenv("LEMONSQUEEZY_YEARLY_VARIANT_ID", "")


def build_app_url(path: str) -> str:
    """Return an app URL using APP_BASE_URL when configured, else a relative path."""
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{APP_BASE_URL}{path}" if APP_BASE_URL else path

# ---------------------------------------------------------------------------
# Database backend — PostgreSQL when DATABASE_URL is set, SQLite otherwise
#
# Render's free PostgreSQL addon supplies DATABASE_URL as "postgres://…".
# psycopg2 requires the "postgresql://" scheme, so we normalise it here.
# ---------------------------------------------------------------------------

_raw_db_url  = os.getenv("DATABASE_URL", "")
DATABASE_URL = _raw_db_url.replace("postgres://", "postgresql://", 1) if _raw_db_url else ""
USE_PG       = bool(DATABASE_URL)

if USE_PG:
    import ssl
    import pg8000.dbapi
    PH              = "%s"                      # PostgreSQL parameter placeholder
    _IntegrityError = pg8000.dbapi.IntegrityError
else:
    PH              = "?"                       # SQLite parameter placeholder
    _IntegrityError = sqlite3.IntegrityError


def _pg_params(url: str) -> dict:
    """Parse a postgres:// or postgresql:// URL into pg8000.dbapi.connect kwargs.
    pg8000 does not accept a connection string — it requires keyword arguments.
    Handles ?sslmode=require that Render adds to external connection URLs."""
    from urllib.parse import urlparse, parse_qs
    p  = urlparse(url)
    qs = parse_qs(p.query)
    params: dict = {
        "host":     p.hostname,
        "port":     p.port or 5432,
        "database": p.path.lstrip("/"),
        "user":     p.username,
        "password": p.password,
    }
    if qs.get("sslmode", [""])[0] == "require":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        params["ssl_context"] = ctx
    return params


def get_conn():
    """Return a fresh database connection for the configured backend."""
    if USE_PG:
        return pg8000.dbapi.connect(**_pg_params(DATABASE_URL))
    return sqlite3.connect(DB_PATH)


def get_openai_client() -> openai.OpenAI:
    """Create the OpenAI client on first use so a missing API key doesn't
    crash the process at import time — it only fails when a recording is
    actually submitted."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENAI_API_KEY environment variable is not set")
    return openai.OpenAI(api_key=api_key)


pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer  = HTTPBearer(auto_error=False)

def _load_scene_config() -> dict:
    with open(_SCENE_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = data.get("scenes")
    levels = data.get("levels")
    if not isinstance(scenes, dict) or not isinstance(levels, list):
        raise RuntimeError("scene_config.json must contain 'scenes' and 'levels'")
    return data


def _build_public_scene_config(config: dict) -> dict:
    public_scenes = {}
    for scene_id, scene in config["scenes"].items():
        public_scene = dict(scene)
        public_scene.pop("translation", None)
        public_scenes[scene_id] = public_scene
    public_config = dict(config)
    public_config["scenes"] = public_scenes
    return public_config


_SCENE_CONFIG = _load_scene_config()
_PUBLIC_SCENE_CONFIG = _build_public_scene_config(_SCENE_CONFIG)
SCENES = _SCENE_CONFIG["scenes"]
PUBLIC_SCENES = _PUBLIC_SCENE_CONFIG["scenes"]
LEVELS = _SCENE_CONFIG["levels"]

# ---------------------------------------------------------------------------
# Division / rank system
# ---------------------------------------------------------------------------
DIVISIONS = [
    {"name": "Bronze",   "min": 0,     "max": 499,   "color": "#cd7f32"},
    {"name": "Silver",   "min": 500,   "max": 1999,  "color": "#b8b8b8"},
    {"name": "Gold",     "min": 2000,  "max": 4999,  "color": "#c9a84c"},
    {"name": "Diamond",  "min": 5000,  "max": 9999,  "color": "#67e8f9"},
    {"name": "Director", "min": 10000, "max": None,  "color": "#c9a84c"},
]

def get_division(points: int) -> dict:
    for d in reversed(DIVISIONS):
        if points >= d["min"]:
            return d
    return DIVISIONS[0]

def get_next_division(points: int) -> Optional[dict]:
    for d in DIVISIONS:
        if d["min"] > points:
            return d
    return None  # already at max rank

# ---------------------------------------------------------------------------
# Daily challenge — deterministic scene selection from UTC date
# ---------------------------------------------------------------------------

def get_daily_scene_id() -> str:
    """Return today's challenge scene.  MD5 of the UTC date string gives a
    stable, evenly-distributed index — same result for every server instance."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    h = int(hashlib.md5(date_str.encode()).hexdigest(), 16)
    return list(SCENES.keys())[h % len(SCENES)]


def get_today_daily_scene(scenes):
    """Return one scene from the list, deterministic per UTC calendar day."""
    if not scenes:
        return None
    return scenes[date.today().toordinal() % len(scenes)]


# ---------------------------------------------------------------------------
# Missions — defaults and time helpers
# ---------------------------------------------------------------------------

DEFAULT_MISSIONS = [
    # (mission_id, goal, cadence)  — cadence: "daily" expires tonight, "weekly" expires next Sunday
    ("daily",           3, "daily"),
    ("pronunciation",   5, "daily"),
    ("genre_drama",     5, "weekly"),
    ("sprint",          1, "daily"),
    ("weekly_thriller", 3, "weekly"),
]


def _midnight_tonight_utc_str() -> str:
    """End of today UTC, i.e. tomorrow 00:00:00 UTC, formatted YYYY-MM-DD HH:MM:SS."""
    now = datetime.now(timezone.utc)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return end.strftime("%Y-%m-%d %H:%M:%S")


def _next_sunday_midnight_utc_str() -> str:
    """End of the upcoming Sunday UTC (i.e. next Monday 00:00:00 UTC). If today is
    already Sunday, returns end of next Sunday (7 days out)."""
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    days_until_sunday = (6 - now.weekday()) % 7  # Monday=0 ... Sunday=6
    if days_until_sunday == 0:
        days_until_sunday = 7
    end = today + timedelta(days=days_until_sunday + 1)  # Monday 00:00 == end of Sunday
    return end.strftime("%Y-%m-%d %H:%M:%S")


SCENE_GENRES = {
    # Drama
    "forrest_gump":          "drama",
    "shawshank":             "drama",
    "good_will_hunting":     "drama",
    "dead_poets":            "drama",
    "the_blind_side":        "drama",
    "pursuit_of_happyness":  "drama",
    "rain_man":              "drama",
    "as_good_as_it_gets":    "drama",
    "rocky":                 "drama",
    "whiplash":              "drama",
    "wall_street":           "drama",
    "social_network":        "drama",
    "jerry_maguire":         "drama",
    "cast_away":             "drama",
    "apollo_13":             "drama",
    "mystic_river":          "drama",
    "titanic":               "drama",
    "a_few_good_men":        "drama",
    "wolf_of_wall_street":   "drama",
    "interstellar":          "drama",
    "the_truman_show":       "drama",
    "braveheart":            "drama",
    "godfather":             "drama",
    "gladiator":             "drama",
    "breakfast_club":        "drama",
    # Thriller
    "seven":                 "thriller",
    "dark_knight":           "thriller",
    "fight_club":            "thriller",
    "the_matrix":            "thriller",
    "basic_instinct":        "thriller",
    "sixth_sense":           "thriller",
    "terminator":            "thriller",
    "taken":                 "thriller",
    "heat":                  "thriller",
    # Other (won't match drama/thriller missions)
    "home_alone":            "family",
    "back_to_the_future":    "scifi",
    "avengers":              "action",
    "men_in_black":          "scifi",
    "mrs_doubtfire":         "comedy",
    "fifth_element":         "scifi",
    "top_gun":               "action",
    "clueless":              "comedy",
    "the_intern":            "comedy",
    "devil_wears_prada":     "comedy",
    "ferris_bueller":        "comedy",
    "legally_blonde":        "comedy",
    "notting_hill":          "romance",
}

MISSION_XP = {
    "daily":           100,
    "pronunciation":    75,
    "genre_drama":     150,
    "sprint":           50,
    "weekly_thriller": 200,
}


def seed_user_missions(username: str, db) -> None:
    """Insert any default missions the user is missing or whose previous instance
    has expired. `db` is a database cursor (matches the pattern of other helpers)."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    db.execute(
        f"SELECT mission_id FROM user_missions "
        f"WHERE username = {PH} AND expires_at > {PH}",
        (username, now_str),
    )
    active_ids = {r[0] for r in db.fetchall()}

    for mission_id, goal, cadence in DEFAULT_MISSIONS:
        if mission_id in active_ids:
            continue
        expires = _midnight_tonight_utc_str() if cadence == "daily" else _next_sunday_midnight_utc_str()
        db.execute(
            f"INSERT INTO user_missions (username, mission_id, progress, goal, expires_at) "
            f"VALUES ({PH}, {PH}, 0, {PH}, {PH})",
            (username, mission_id, goal, expires),
        )


async def update_missions(username: str, scene_id: str, score: float,
                          duration_seconds: float, take_number: int, db):
    """Advance any active user_missions matching this submission. Returns
    [{mission_id, new_progress, completed, xp_earned}]. Also updates user_streak
    (current/longest streak, daily_xp, total_xp)."""
    now_str   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    today_str = date.today().strftime("%Y-%m-%d")

    db.execute(
        f"SELECT id, mission_id, progress, goal, completed, xp_awarded "
        f"FROM user_missions WHERE username = {PH} AND expires_at > {PH}",
        (username, now_str),
    )
    rows = db.fetchall()

    daily_today = get_daily_scene_id()
    genre       = SCENE_GENRES.get(scene_id, "")
    duration    = float(duration_seconds or 0)
    score_pct   = float(score or 0)

    matched = []
    for r in rows:
        row_id     = r[0]
        mid        = r[1]
        progress   = int(r[2] or 0)
        goal       = int(r[3])
        completed  = bool(r[4])
        xp_awarded = bool(r[5])
        if completed:
            continue

        advances = (
            (mid == "daily"           and scene_id == daily_today) or
            (mid == "pronunciation"   and score_pct >= SCORE_TIER_STRONG) or
            (mid == "genre_drama"     and genre == "drama")        or
            (mid == "sprint"          and take_number == 1 and duration > 0 and duration <= 240) or
            (mid == "weekly_thriller" and genre == "thriller")
        )
        if not advances:
            continue

        new_progress  = progress + 1
        new_completed = new_progress >= goal
        xp_earned     = 0

        if new_completed and not xp_awarded:
            xp_earned = MISSION_XP.get(mid, 100)
            db.execute(
                f"UPDATE user_missions SET progress = {PH}, completed = {PH}, xp_awarded = {PH} "
                f"WHERE id = {PH}",
                (new_progress, True, True, row_id),
            )
        else:
            db.execute(
                f"UPDATE user_missions SET progress = {PH} WHERE id = {PH}",
                (new_progress, row_id),
            )

        matched.append({
            "mission_id":   mid,
            "new_progress": new_progress,
            # The goal travels with the update so the post-take sequence can
            # render "3 / 5" without a second request. The client only holds
            # goals if the Missions tab happens to have been opened this
            # session, which is not something a result screen should depend on.
            "goal":         goal,
            "completed":    new_completed,
            "xp_earned":    xp_earned,
        })

    total_xp_earned = sum(m["xp_earned"] for m in matched)

    # Ensure user_streak row exists; load current state
    db.execute(
        f"SELECT current_streak, longest_streak, last_active_date, total_xp, daily_xp, daily_xp_date "
        f"FROM user_streak WHERE username = {PH}",
        (username,),
    )
    srow = db.fetchone()
    if srow is None:
        cur_streak, longest, last_active = 0, 0, ""
        total_xp,  daily_xp, daily_xp_date = 0, 0, ""
        db.execute(
            f"INSERT INTO user_streak (username) VALUES ({PH})",
            (username,),
        )
    else:
        cur_streak    = int(srow[0] or 0)
        longest       = int(srow[1] or 0)
        last_active   = (srow[2] or "")
        total_xp      = int(srow[3] or 0)
        daily_xp      = int(srow[4] or 0)
        daily_xp_date = (srow[5] or "")

    if last_active != today_str:
        cur_streak += 1
        last_active = today_str
        if cur_streak > longest:
            longest = cur_streak

    if daily_xp_date != today_str:
        daily_xp = 0
        daily_xp_date = today_str
    daily_xp += total_xp_earned
    total_xp += total_xp_earned

    db.execute(
        f"UPDATE user_streak SET current_streak = {PH}, longest_streak = {PH}, "
        f"last_active_date = {PH}, total_xp = {PH}, daily_xp = {PH}, daily_xp_date = {PH} "
        f"WHERE username = {PH}",
        (cur_streak, longest, last_active, total_xp, daily_xp, daily_xp_date, username),
    )

    return matched


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

# Columns added to `users` after the original CREATE TABLE. Shared by the SQLite
# and PostgreSQL migration paths so the two backends cannot drift apart.
# recording_consent_at / recording_consent_version record that the user accepted
# the first-run recording notice (privacy policy §6). A timestamp rather than a
# boolean because GDPR Art. 7(1) requires being able to *demonstrate* consent was
# given, and the version pins which text they saw.
_USERS_MIGRATION_COLUMNS = [
    ("points",                     "INTEGER DEFAULT 0"),
    ("streak",                     "INTEGER DEFAULT 0"),
    ("last_daily",                 "TEXT"),
    ("is_pro",                     "BOOLEAN DEFAULT FALSE"),
    ("avatar_scene_id",            "TEXT"),
    ("recording_consent_at",       "TEXT"),
    ("recording_consent_version",  "TEXT"),
    # Lemon Squeezy identifiers, captured from the webhook. Without the
    # subscription id there is no way back from a user to their subscription,
    # so deleting a subscriber's account would leave it billing a card for an
    # account that no longer exists. The portal URL is a pre-signed self-service
    # link, used to give a subscriber a working way out when we cannot cancel
    # for them.
    ("ls_subscription_id",         "TEXT"),
    ("ls_customer_portal_url",     "TEXT"),
]


def _verify_users_columns(cur, sqlite: bool) -> None:
    """Fail startup loudly if a migrated column is missing.

    The SQLite ALTER loop cannot distinguish "already exists" from a genuine
    failure without inspecting the error string, and a missed column does not
    surface until a request reads it and 500s with "no such column" — with
    nothing in the log pointing at the cause. Checking here turns that into a
    startup error naming the column, which on Render keeps the previous healthy
    deploy serving instead of shipping a broken one.
    """
    if sqlite:
        cur.execute("PRAGMA table_info(users)")
        present = {row[1] for row in cur.fetchall()}
    else:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'users'"
        )
        present = {row[0] for row in cur.fetchall()}

    missing = [col for col, _ in _USERS_MIGRATION_COLUMNS if col not in present]
    if missing:
        raise RuntimeError(
            "users table is missing migrated column(s): "
            + ", ".join(missing)
            + " — the ALTER did not apply. Fix the schema before serving."
        )


def init_db():
    conn = get_conn()
    cur  = conn.cursor()

    if USE_PG:
        # PostgreSQL: SERIAL primary key, %s placeholders.
        # CREATE TABLE IF NOT EXISTS is idempotent, so re-deploys are safe.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                username      TEXT    NOT NULL UNIQUE,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id            SERIAL PRIMARY KEY,
                scene_id      TEXT NOT NULL,
                movie         TEXT NOT NULL,
                quote         TEXT NOT NULL,
                transcription TEXT,
                sync_score    REAL,
                username      TEXT    DEFAULT '',
                user_id       INTEGER,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:
        # SQLite: AUTOINCREMENT primary key, ? placeholders.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id      TEXT NOT NULL,
                movie         TEXT NOT NULL,
                quote         TEXT NOT NULL,
                transcription TEXT,
                sync_score    REAL,
                username      TEXT    DEFAULT '',
                user_id       INTEGER,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Non-destructive migrations for pre-existing SQLite scores tables
        for col, dfn in [("username", "TEXT DEFAULT ''"), ("user_id", "INTEGER")]:
            try:
                cur.execute(f"ALTER TABLE scores ADD COLUMN {col} {dfn}")
            except sqlite3.OperationalError:
                pass  # column already exists
        # Non-destructive migration for users.points / streak / last_daily / is_pro / avatar_scene_id
        for col, dfn in _USERS_MIGRATION_COLUMNS:
            try:
                cur.execute(f"ALTER TABLE users ADD COLUMN {col} {dfn}")
            except sqlite3.OperationalError as exc:
                # Only "duplicate column name" means the migration already ran.
                # A blanket `pass` here used to swallow genuine failures too, so
                # a column could silently never be added and every read of it
                # then 500'd with "no such column". Anything else re-raises.
                if "duplicate column name" not in str(exc).lower():
                    raise
        _verify_users_columns(cur, sqlite=True)

    if USE_PG:
        # Non-destructive migrations for PostgreSQL
        for col, dfn in _USERS_MIGRATION_COLUMNS:
            cur.execute(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col} {dfn}")
        _verify_users_columns(cur, sqlite=False)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS challenges (
                id                  SERIAL PRIMARY KEY,
                challenge_id        TEXT NOT NULL UNIQUE,
                challenger_username TEXT NOT NULL,
                challenger_user_id  INTEGER,
                scene_id            TEXT NOT NULL,
                score_to_beat       REAL NOT NULL,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS challenges (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge_id        TEXT NOT NULL UNIQUE,
                challenger_username TEXT NOT NULL,
                challenger_user_id  INTEGER,
                scene_id            TEXT NOT NULL,
                score_to_beat       REAL NOT NULL,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    # Vocab tables (shared schema across PG and SQLite)
    if USE_PG:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS word_vocab (
                id         SERIAL PRIMARY KEY,
                scene_id   TEXT,
                word_en    TEXT,
                word_es    TEXT,
                phonetic   TEXT,
                example    TEXT,
                word_type  TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS word_mastery (
                user_id       INTEGER,
                scene_id      TEXT,
                word_en       TEXT,
                correct_count INTEGER DEFAULT 0,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, scene_id, word_en)
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS word_vocab (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id   TEXT,
                word_en    TEXT,
                word_es    TEXT,
                phonetic   TEXT,
                example    TEXT,
                word_type  TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS word_mastery (
                user_id       INTEGER,
                scene_id      TEXT,
                word_en       TEXT,
                correct_count INTEGER DEFAULT 0,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, scene_id, word_en)
            )
        """)

    # Missions + streak tables (dual PG / SQLite)
    if USE_PG:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_missions (
                id          SERIAL PRIMARY KEY,
                username    TEXT NOT NULL,
                mission_id  TEXT NOT NULL,
                progress    INTEGER DEFAULT 0,
                goal        INTEGER NOT NULL,
                completed   BOOLEAN DEFAULT FALSE,
                xp_awarded  BOOLEAN DEFAULT FALSE,
                expires_at  TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_streak (
                username         TEXT PRIMARY KEY,
                current_streak   INTEGER DEFAULT 0,
                longest_streak   INTEGER DEFAULT 0,
                last_active_date TEXT DEFAULT '',
                total_xp         INTEGER DEFAULT 0,
                daily_xp         INTEGER DEFAULT 0,
                daily_xp_date    TEXT DEFAULT ''
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_missions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT NOT NULL,
                mission_id  TEXT NOT NULL,
                progress    INTEGER DEFAULT 0,
                goal        INTEGER NOT NULL,
                completed   BOOLEAN DEFAULT 0,
                xp_awarded  BOOLEAN DEFAULT 0,
                expires_at  TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_streak (
                username         TEXT PRIMARY KEY,
                current_streak   INTEGER DEFAULT 0,
                longest_streak   INTEGER DEFAULT 0,
                last_active_date TEXT DEFAULT '',
                total_xp         INTEGER DEFAULT 0,
                daily_xp         INTEGER DEFAULT 0,
                daily_xp_date    TEXT DEFAULT ''
            )
        """)

    conn.commit()
    conn.close()


@app.on_event("startup")
async def startup():
    """Run DB initialisation when uvicorn is ready, not at import time.
    Errors here appear in the Render log with a full traceback instead of
    killing the process silently during module load."""
    init_db()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=30)
    email:    str = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=128)

class LoginRequest(BaseModel):
    email:    str = Field(..., max_length=255)
    password: str = Field(..., max_length=128)


class DeleteAccountRequest(BaseModel):
    """Both fields are required and both are checked server-side. The typed
    username guards against an accidental click; the password proves the person
    at the keyboard owns the account, since the token alone lives in
    localStorage for TOKEN_TTL days and a borrowed device would otherwise be
    enough to erase it."""
    password:         str = Field(..., max_length=128)
    confirm_username: str = Field(..., max_length=30)


def hash_pw(password: str) -> str:
    return pwd_ctx.hash(password)

def verify_pw(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def make_token(user_id: int, username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=TOKEN_TTL)
    return jwt.encode({"sub": str(user_id), "username": username, "exp": exp}, SECRET, algorithm=ALGORITHM)

def decode_token(creds: Optional[HTTPAuthorizationCredentials]) -> dict:
    if not creds:
        raise HTTPException(401, "Authentication required")
    try:
        payload = jwt.decode(creds.credentials, SECRET, algorithms=[ALGORITHM])
        return {"id": int(payload["sub"]), "username": payload["username"]}
    except (JWTError, KeyError, ValueError):
        raise HTTPException(401, "Invalid or expired token")

def require_live_user(user: dict) -> dict:
    """Reject a token whose account no longer exists.

    Tokens are stateless and stay valid for TOKEN_TTL days, so deleting a user
    row does not invalidate tokens already issued to them. Without this check a
    stale token can still reach the write endpoints and *re-create* the rows the
    deletion just removed — `/api/submit` and `/api/missions` both call
    seed_user_missions(), which INSERTs into the username-keyed user_missions /
    user_streak tables. Because usernames are freed for re-registration on
    delete, those resurrected rows would then be inherited by whoever next
    claims the name. One indexed primary-key lookup closes that off.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM users WHERE id = {PH}", (user["id"],))
        if cur.fetchone() is None:
            raise HTTPException(401, "Account no longer exists")
    finally:
        conn.close()
    return user


def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    return require_live_user(decode_token(creds))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()

def sync_score(expected: str, transcribed: str) -> float:
    ratio = difflib.SequenceMatcher(None, normalize(expected), normalize(transcribed)).ratio()
    return round(ratio * 100, 1)


# ---------------------------------------------------------------------------
# Score thresholds (0-100 scale)
#
# Single source of truth for every score comparison, so a scoring-engine
# recalibration is an edit here rather than a hunt through scattered magic
# numbers. The frontend mirror lives in static/app-config.js (SCORE).
#
# NOTE: the level-unlock scores in scene_config.json (L2 unlock_score = 60,
# L3 = 70) intentionally mirror SCORE_PASS / SCORE_PROFICIENT. A recalibration
# must revisit that file too — these constants do not feed it.
# ---------------------------------------------------------------------------
SCORE_PASS        = 60.0   # scene "completed": counts as done, unlocks avatar use
SCORE_PROFICIENT  = 70.0   # translation unlock, Level-1 quiz pass, first points tier
SCORE_TIER_STRONG = 85.0   # points tier
SCORE_TIER_ELITE  = 95.0   # points tier
SCORE_PERFECT     = 100.0  # points tier / perfect-take flag


def calc_points(score: float, is_first_attempt: bool) -> int:
    """Return points earned for a single submission."""
    pts = 0
    if is_first_attempt:
        pts += 10
    if score >= SCORE_PERFECT:
        pts += 100
    elif score >= SCORE_TIER_ELITE:
        pts += 75
    elif score >= SCORE_TIER_STRONG:
        pts += 50
    elif score >= SCORE_PROFICIENT:
        pts += 25
    return pts


# ---------------------------------------------------------------------------
# Routes — frontend
# ---------------------------------------------------------------------------

# Cache of (stat_key, etag, body, last_modified) per shell path. The HTML shells
# only change on deploy, so hashing 327 KB on every navigation would be pure
# waste — but keying on (mtime_ns, size) means an edited file is still picked up
# immediately in local dev without a restart.
_HTML_SHELL_CACHE: dict = {}


def _html_shell(path: str):
    """Read an HTML shell, returning (etag, body, last_modified).

    Re-reads and re-hashes only when the file's mtime or size changes.
    """
    st = os.stat(path)
    stat_key = (st.st_mtime_ns, st.st_size)

    cached = _HTML_SHELL_CACHE.get(path)
    if cached and cached[0] == stat_key:
        return cached[1], cached[2], cached[3]

    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    etag = '"%s"' % hashlib.sha1(body.encode("utf-8")).hexdigest()
    last_modified = formatdate(st.st_mtime, usegmt=True)
    _HTML_SHELL_CACHE[path] = (stat_key, etag, body, last_modified)
    return etag, body, last_modified


def _if_none_match_hit(request: Request, etag: str) -> bool:
    """True when the client already holds this exact entity.

    Handles a comma-separated list and the weak-validator "W/" prefix, both of
    which are legal in If-None-Match.
    """
    header = request.headers.get("if-none-match") if request else None
    if not header:
        return False
    if header.strip() == "*":
        return True
    for tag in header.split(","):
        tag = tag.strip()
        if tag.startswith("W/"):
            tag = tag[2:]
        if tag == etag:
            return True
    return False


def _read_html_file(path: str, missing_title: str, request: Request = None):
    """Serve an HTML shell with revalidation headers.

    These shells were previously returned as bare strings, so they carried no
    ETag, Last-Modified or Cache-Control at all — every navigation re-downloaded
    the full document (327 KB for index.html). They now revalidate: a repeat
    visit sends If-None-Match and gets a 304 with no body.

    Cache-Control is "no-cache" rather than a max-age: the shell must never be
    served stale from cache after a deploy, but it may be reused once the ETag
    is confirmed unchanged.
    """
    try:
        etag, body, last_modified = _html_shell(path)
    except (FileNotFoundError, OSError):
        return HTMLResponse(f"<h1>{missing_title}</h1>", status_code=404)
    return _revalidating_html(request, etag, body, last_modified)


def _revalidating_html(request: Request, etag: str, body: str, last_modified: str):
    """Return `body` with revalidation headers, or a 304 when the client's
    If-None-Match already matches. Shared by the HTML shells and the rendered
    markdown pages, which cache on the same (mtime_ns, size) key."""
    headers = {
        "ETag": etag,
        "Last-Modified": last_modified,
        "Cache-Control": "no-cache, must-revalidate",
    }
    if _if_none_match_hit(request, etag):
        return Response(status_code=304, headers=headers)
    return HTMLResponse(body, headers=headers)


@app.get("/", response_class=HTMLResponse)
async def cinematic_landing(request: Request):
    return _read_html_file(_INDEX_HTML_PATH, "index.html not found", request)


@app.get("/app", response_class=HTMLResponse)
@app.get("/app/auth", response_class=HTMLResponse)
@app.get("/app/levels", response_class=HTMLResponse)
@app.get("/app/progress", response_class=HTMLResponse)
@app.get("/app/daily", response_class=HTMLResponse)
@app.get("/app/scene/{scene_id}", response_class=HTMLResponse)
@app.get("/app/challenge/{challenge_id}", response_class=HTMLResponse)
async def new_shell_app_entry(
    request: Request,
    scene_id: Optional[str] = None,
    challenge_id: Optional[str] = None,
):
    return _read_html_file(_NEW_SHELL_INDEX_HTML_PATH, "app index.html not found", request)


@app.get("/scene/{scene_id}", response_class=HTMLResponse)
@app.get("/challenge/{challenge_id}", response_class=HTMLResponse)
async def new_shell_legacy_links(
    request: Request,
    scene_id: Optional[str] = None,
    challenge_id: Optional[str] = None,
):
    return _read_html_file(_NEW_SHELL_INDEX_HTML_PATH, "app index.html not found", request)


@app.get("/legacy", response_class=HTMLResponse)
async def legacy_app_entry(request: Request):
    return _read_html_file(_INDEX_HTML_PATH, "index.html not found", request)


@app.get("/legacy/challenge/{challenge_id}", response_class=HTMLResponse)
async def legacy_challenge_page(request: Request, challenge_id: str):
    """Serve the SPA for challenge links so the JS can read the path and render the challenge screen."""
    return _read_html_file(_INDEX_HTML_PATH, "index.html not found", request)


# ---------------------------------------------------------------------------
# Routes — rendered markdown documents
# ---------------------------------------------------------------------------
# docs/privacy-policy-notes.md holds the working notes for the policy — review
# questions, maintenance reminders — and is deliberately NOT routed. Only the
# policy itself is public; keep notes-to-self out of the served document.
#
# The privacy policy is rendered from docs/privacy-policy.md at request time
# rather than converted to a committed .html file, so the published page is
# always exactly the committed text — there is no second copy to drift. Parsing
# is cached on (mtime_ns, size) like the HTML shells above, so this costs one
# parse per deploy, and repeat visits still 304.

_MD_PAGE_CACHE: dict = {}

_MD_PAGE_STYLE = """
  :root { --bg:#080808; --gold:#c8a96e; --text:rgba(240,237,230,0.82); --muted:rgba(240,237,230,0.45); --line:rgba(200,169,110,0.18); }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif;
         font-weight:300; font-size:16px; line-height:1.75;
         -webkit-font-smoothing:antialiased; }
  .md-bar { border-bottom:1px solid var(--line); padding:18px 20px; }
  .md-bar a { font-family:'Bebas Neue',sans-serif; font-size:17px; letter-spacing:0.2em;
              color:var(--gold); text-decoration:none; }
  .md-wrap { max-width:44rem; margin:0 auto; padding:40px 20px 72px; }
  .md-wrap h1 { font-family:'Bebas Neue',sans-serif; font-weight:400; font-size:2rem;
                letter-spacing:0.06em; color:var(--gold); line-height:1.15; margin:0 0 28px; }
  .md-wrap h2 { font-family:'Bebas Neue',sans-serif; font-weight:400; font-size:1.3rem;
                letter-spacing:0.06em; color:var(--gold); margin:44px 0 12px;
                padding-top:20px; border-top:1px solid var(--line); }
  .md-wrap p, .md-wrap li { margin:0 0 14px; }
  .md-wrap ul, .md-wrap ol { padding-left:1.2em; }
  .md-wrap li { margin-bottom:8px; }
  .md-wrap a { color:var(--gold); text-decoration:underline; text-underline-offset:2px; }
  .md-wrap strong { color:rgba(240,237,230,0.96); font-weight:600; }
  .md-wrap hr { border:0; border-top:1px solid var(--line); margin:32px 0; }
  .md-wrap code { font-size:0.88em; background:rgba(200,169,110,0.09);
                  padding:1px 5px; border-radius:3px; color:var(--gold); }
  .md-wrap blockquote { margin:20px 0; padding:14px 18px; border-left:2px solid var(--gold);
                        background:rgba(200,169,110,0.05); color:var(--muted); }
  .md-wrap blockquote p:last-child { margin-bottom:0; }
  .md-table-wrap { overflow-x:auto; margin:0 0 20px; -webkit-overflow-scrolling:touch; }
  .md-wrap table { border-collapse:collapse; width:100%; min-width:34rem; font-size:0.86rem; }
  .md-wrap th, .md-wrap td { border:1px solid var(--line); padding:9px 12px;
                             text-align:left; vertical-align:top; }
  .md-wrap th { color:var(--gold); font-weight:500; background:rgba(200,169,110,0.06);
                white-space:nowrap; }
  .md-foot { border-top:1px solid var(--line); margin-top:56px; padding-top:20px;
             font-size:0.8rem; color:var(--muted); }
  .md-foot a { color:var(--gold); }
  @media (min-width:720px) { body { font-size:17px; } .md-wrap { padding:56px 32px 96px; }
                             .md-wrap h1 { font-size:2.6rem; } }
"""

_MD_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>__STYLE__</style>
</head>
<body>
<div class="md-bar"><a href="/">MIRROR</a></div>
<main class="md-wrap">
__BODY__
<div class="md-foot"><a href="/">&larr; Back to MIRROR</a></div>
</main>
</body>
</html>
"""


def _md_page(path: str, title: str):
    """Render a markdown file to a full HTML page, cached on (mtime_ns, size).

    Tables get an overflow-x wrapper so a wide processor table scrolls inside
    itself on a phone instead of forcing the whole page sideways.
    """
    st = os.stat(path)
    stat_key = (st.st_mtime_ns, st.st_size)

    cached = _MD_PAGE_CACHE.get(path)
    if cached and cached[0] == stat_key:
        return cached[1], cached[2], cached[3]

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    body = markdown.markdown(source, extensions=["tables"])
    body = body.replace("<table>", '<div class="md-table-wrap"><table>')
    body = body.replace("</table>", "</table></div>")

    page = (
        _MD_PAGE_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__STYLE__", _MD_PAGE_STYLE)
        .replace("__BODY__", body)
    )
    etag = '"%s"' % hashlib.sha1(page.encode("utf-8")).hexdigest()
    last_modified = formatdate(st.st_mtime, usegmt=True)
    _MD_PAGE_CACHE[path] = (stat_key, etag, page, last_modified)
    return etag, page, last_modified


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    """Public privacy policy. Path is relative, so it serves correctly on every
    domain pointed at this service."""
    try:
        etag, body, last_modified = _md_page(_PRIVACY_MD_PATH, "Privacy Policy — MIRROR")
    except (FileNotFoundError, OSError):
        return HTMLResponse("<h1>Privacy policy not found</h1>", status_code=404)
    return _revalidating_html(request, etag, body, last_modified)


# ---------------------------------------------------------------------------
# Routes — auth
# ---------------------------------------------------------------------------

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    username = req.username.strip()
    email    = req.email.lower().strip()

    if not re.match(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,28}[A-Za-z0-9]$|^[A-Za-z0-9]{2}$', username):
        raise HTTPException(400, "Username may only contain letters, numbers, dots, hyphens and underscores")
    if not re.match(r'^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{1,63}$', email):
        raise HTTPException(400, "Invalid email address")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    conn = get_conn()
    cur  = conn.cursor()
    try:
        if USE_PG:
            # RETURNING id is the PostgreSQL way to get the new row's id
            cur.execute(
                f"INSERT INTO users (username, email, password_hash) VALUES ({PH}, {PH}, {PH}) RETURNING id",
                (username, email, hash_pw(req.password)),
            )
            user_id = cur.fetchone()[0]
        else:
            cur.execute(
                f"INSERT INTO users (username, email, password_hash) VALUES ({PH}, {PH}, {PH})",
                (username, email, hash_pw(req.password)),
            )
            user_id = cur.lastrowid
        conn.commit()
    except _IntegrityError:
        raise HTTPException(400, "Email or username already taken")
    finally:
        conn.close()

    return {"access_token": make_token(user_id, username), "token_type": "bearer", "username": username}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT id, username, password_hash FROM users WHERE email = {PH}",
        (req.email.lower().strip(),),
    )
    row = cur.fetchone()
    conn.close()

    if not row or not verify_pw(req.password, row[2]):
        raise HTTPException(401, "Invalid email or password")

    return {"access_token": make_token(row[0], row[1]), "token_type": "bearer", "username": row[1]}


@app.get("/api/auth/me")
async def me(user: dict = Depends(current_user)):
    conn = get_conn()
    cur = conn.cursor()
    # Both shells call this at boot, so the consent flag rides the query that
    # was already happening — the gate costs no extra round-trip.
    try:
        cur.execute(
            f"SELECT is_pro, recording_consent_at FROM users WHERE id = {PH}",
            (user["id"],)
        )
        row = cur.fetchone()
        is_pro = bool(row[0]) if row and row[0] else False
        # Fail closed: if this read fails we report "not consented", which shows
        # the notice again. Showing it twice is harmless; skipping it is not.
        recording_consent = bool(row[1]) if row and row[1] else False
    except Exception:
        is_pro = False
        recording_consent = False
    finally:
        conn.close()
    return {
        "id": user["id"],
        "username": user["username"],
        "is_pro": is_pro,
        "recording_consent": recording_consent,
    }


@app.post("/api/consent/recording")
async def accept_recording_consent(user: dict = Depends(current_user)):
    """Record that the user accepted the first-run recording notice (§6).

    Idempotent: the UPDATE is a no-op once consent exists, so a double-tap or a
    retry cannot overwrite the original timestamp — the first acceptance is the
    one that has to be demonstrable.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            f"UPDATE users SET recording_consent_at = {PH}, "
            f"recording_consent_version = {PH} "
            f"WHERE id = {PH} AND recording_consent_at IS NULL",
            (now, RECORDING_CONSENT_VERSION, user["id"]),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("[consent] failed to record for user_id=%s", user["id"])
        raise HTTPException(500, "Could not save your choice. Please try again.")
    finally:
        conn.close()

    return {"recording_consent": True}


# Every table that stores something about a user, as
# (table, column, which key to bind). There are no foreign keys anywhere in this
# schema and therefore no ON DELETE CASCADE, so erasure has to name each table
# explicitly. Note the split: most tables key on user_id, but user_missions and
# user_streak key on *username only*. Since deleting a user frees their username
# for re-registration, missing those two would hand the next person to claim the
# name the deleted user's streak, XP and mission progress.
# challenges appears twice on purpose: current code fills both challenger_user_id
# and challenger_username, but older rows may carry only the name.
_USER_DATA_TABLES = [
    ("word_mastery",  "user_id",             "id"),
    ("scores",        "user_id",             "id"),
    ("challenges",    "challenger_user_id",  "id"),
    ("challenges",    "challenger_username", "username"),
    ("user_missions", "username",            "username"),
    ("user_streak",   "username",            "username"),
]


@app.delete("/api/account")
async def delete_account(req: DeleteAccountRequest, user: dict = Depends(current_user)):
    """Permanently erase the authenticated user's account and all data tied to
    it (GDPR Art. 17). Irreversible — there is no soft-delete tombstone.

    A subscriber's subscription is cancelled first, and only a confirmed cancel
    lets the deletion proceed. The ordering is the whole safety property: a
    database transaction cannot roll back an HTTP call to a third party, so
    deleting first and cancelling second would risk an erased account whose card
    keeps being charged. Cancelling first risks, at worst, a cancelled
    subscription on an account that still exists — annoying, retryable, and
    strictly the better failure.
    """
    # ── Phase 1: authenticate the request. No data is touched here. ──────────
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            f"SELECT username, password_hash, is_pro, ls_subscription_id, "
            f"ls_customer_portal_url FROM users WHERE id = {PH}",
            (user["id"],),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(401, "Account no longer exists")
        db_username, password_hash, is_pro = row[0], row[1], row[2]
        subscription_id, portal_url = row[3], row[4]
    finally:
        conn.close()

    # Password first: a wrong password should not reveal whether the typed
    # username matched.
    try:
        password_ok = verify_pw(req.password, password_hash)
    except Exception:
        password_ok = False
    if not password_ok:
        raise HTTPException(401, "Password is incorrect")

    # Re-check the typed confirmation server-side — the client check is a
    # convenience, not the gate.
    if req.confirm_username.strip() != db_username:
        raise HTTPException(400, "The username you typed does not match your account")

    # ── Phase 2: cancel the subscription BEFORE deleting anything. ───────────
    if is_pro:
        if not subscription_id:
            # Pre-dates the webhook storing the id, so we cannot cancel for
            # them. Hand over the self-service route rather than a dead end.
            raise HTTPException(409, _subscription_blocked_message(portal_url))
        try:
            await _cancel_ls_subscription(str(subscription_id))
        except SubscriptionCancelError as exc:
            logger.error(
                "[account] refusing to delete user_id=%s — cancel failed: %s",
                user["id"], exc,
            )
            raise HTTPException(
                502,
                "We could not cancel your MIRROR Pro subscription just now, so "
                "your account has not been deleted — deleting it would leave the "
                "subscription billing you. Please try again in a moment, or "
                "email contact@mirrorspeak.app and we will sort it out.",
            )
        logger.info("[account] cancelled subscription for user_id=%s", user["id"])

    # ── Phase 3: erase. Only reached once nothing can still bill them. ───────
    conn = get_conn()
    cur  = conn.cursor()
    try:
        # Bind username from the database row, not from the JWT — the token
        # carries a snapshot that may be stale.
        keys = {"id": user["id"], "username": db_username}
        deleted = {}
        for table, column, key in _USER_DATA_TABLES:
            cur.execute(f"DELETE FROM {table} WHERE {column} = {PH}", (keys[key],))
            deleted[f"{table}.{column}"] = cur.rowcount

        cur.execute(f"DELETE FROM users WHERE id = {PH}", (user["id"],))
        deleted["users"] = cur.rowcount

        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        logger.exception("[account] delete failed for user_id=%s", user["id"])
        raise HTTPException(500, "Could not delete the account. Nothing was changed.")
    finally:
        conn.close()

    # Deliberately logs the numeric id and row counts only — no username or
    # email — so the audit line does not re-retain what was just erased.
    logger.info("[account] deleted user_id=%s rows=%s", user["id"], deleted)
    return {"deleted": True}


# ─── Lemon Squeezy billing ───────────────────────────────────────────────────

@app.post("/api/billing/checkout")
async def create_checkout(request: Request, user: dict = Depends(current_user)):
    """Create a Lemon Squeezy checkout session and return the checkout URL."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    variant_id = body.get("variant_id")
    if not variant_id:
        raise HTTPException(status_code=400, detail="variant_id required")

    # Only allow our known variant IDs
    if str(variant_id) not in [LS_MONTHLY_ID, LS_YEARLY_ID]:
        raise HTTPException(status_code=400, detail="Invalid variant")

    if not LS_API_KEY:
        raise HTTPException(status_code=500, detail="Payment not configured")

    payload = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "checkout_data": {
                    "custom": {
                        "user_id": str(user["id"]),
                        "username": user["username"],
                    }
                },
                "product_options": {
                    "redirect_url": f"{CANONICAL_BASE_URL}/?checkout=success",
                    "receipt_link_url": f"{CANONICAL_BASE_URL}/?checkout=success",
                }
            },
            "relationships": {
                "store": {
                    "data": {"type": "stores", "id": "396208"}
                },
                "variant": {
                    "data": {"type": "variants", "id": str(variant_id)}
                }
            }
        }
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.lemonsqueezy.com/v1/checkouts",
            headers={
                "Authorization": f"Bearer {LS_API_KEY}",
                "Content-Type": "application/vnd.api+json",
                "Accept": "application/vnd.api+json",
            },
            json=payload,
            timeout=15,
        )

    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail="Checkout creation failed")

    data = resp.json()
    checkout_url = data["data"]["attributes"]["url"]
    return {"checkout_url": checkout_url}


def _subscription_blocked_message(portal_url: Optional[str]) -> str:
    """Message for a subscriber whose subscription we cannot cancel ourselves.

    Only reachable when is_pro is set but no subscription id was ever captured
    — i.e. a subscription that predates the webhook storing it. The portal URL
    is a pre-signed Lemon Squeezy self-service link; when we have one, give it
    rather than sending them to hunt through their email.
    """
    if portal_url:
        return (
            "You have an active MIRROR Pro subscription, and we could not cancel "
            "it automatically. Cancel it here first, then delete your account: "
            f"{portal_url} — or email contact@mirrorspeak.app and we will do it "
            "for you. Your account has not been deleted."
        )
    return (
        "You have an active MIRROR Pro subscription, and we could not cancel it "
        "automatically. Use the subscription management link in your Lemon "
        "Squeezy receipt email to cancel it first, then delete your account — "
        "or email contact@mirrorspeak.app and we will do it for you. "
        "Your account has not been deleted."
    )


class SubscriptionCancelError(Exception):
    """Raised when a Lemon Squeezy subscription could not be cancelled.

    Deliberately distinct from HTTPException: the caller decides what status to
    surface, and — critically — treats this as "delete nothing".
    """


async def _cancel_ls_subscription(subscription_id: str) -> None:
    """Cancel a Lemon Squeezy subscription, or raise SubscriptionCancelError.

    DELETE /v1/subscriptions/{id} stops future billing immediately. It does not
    end the subscription there and then: status becomes "cancelled" and the
    customer keeps access until `ends_at`, after which it becomes "expired".
    Lemon Squeezy has no API for immediate expiry, and that is fine here — the
    requirement is that nothing bills a deleted account, not that access stops
    to the second.

    Raising rather than returning a bool is intentional. Every failure path must
    abort the deletion, and an exception cannot be accidentally ignored the way
    a falsy return can.
    """
    if not LS_API_KEY:
        raise SubscriptionCancelError("Lemon Squeezy API key is not configured")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"https://api.lemonsqueezy.com/v1/subscriptions/{subscription_id}",
                headers={
                    "Authorization": f"Bearer {LS_API_KEY}",
                    "Accept": "application/vnd.api+json",
                },
                timeout=15,
            )
    except Exception as exc:
        # Network error or timeout — we do not know whether the cancel landed,
        # so we must assume it did not and leave the account alone.
        raise SubscriptionCancelError(f"could not reach Lemon Squeezy: {exc}") from exc

    if resp.status_code == 404:
        # Already gone from their side. Nothing left to bill, so this is success
        # for our purposes — refusing here would trap the user forever.
        logger.info("[billing] subscription %s already absent at Lemon Squeezy", subscription_id)
        return

    if resp.status_code not in (200, 201, 204):
        raise SubscriptionCancelError(
            f"Lemon Squeezy returned {resp.status_code}"
        )


@app.post("/api/billing/webhook")
async def lemonsqueezy_webhook(request: Request):
    """Receive Lemon Squeezy webhook events and update user pro status."""
    raw_body = await request.body()
    signature = request.headers.get("X-Signature", "")

    # Fail closed: an unset signing secret means we cannot verify anything, so
    # this endpoint must reject rather than silently trust the body. Previously
    # a missing secret skipped verification entirely — an unsigned request could
    # then flip any user's is_pro. This only disables the one endpoint (returns
    # 503); the rest of the app keeps serving, since billing is not core.
    if not LS_SIGNING_SECRET:
        logger.error(
            "SECURITY: /api/billing/webhook received a request but "
            "LEMONSQUEEZY_SIGNING_SECRET is not set — rejecting. Set it in the "
            "environment to enable signature verification."
        )
        raise HTTPException(status_code=503, detail="Webhook verification unavailable")

    # Verify HMAC-SHA256 signature
    expected = hmac.new(
        LS_SIGNING_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(raw_body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = request.headers.get("X-Event-Name", "")

    # Extract user_id from custom data
    try:
        custom = payload["meta"]["custom_data"]
        user_id = int(custom.get("user_id", 0))
    except (KeyError, TypeError, ValueError):
        # Can't identify user — return 200 so LS doesn't retry
        return {"ok": True}

    if not user_id:
        return {"ok": True}

    conn = get_conn()
    cur = conn.cursor()
    try:
        if event in ("subscription_created", "subscription_updated", "subscription_resumed"):
            # Check subscription status in payload
            try:
                status = payload["data"]["attributes"]["status"]
                is_active = status in ("active", "trialing")
            except (KeyError, TypeError):
                is_active = True  # assume active if we can't read status

            # Capture the identifiers this payload carries. Only is_pro used to
            # be stored, which left no route from a user back to their
            # subscription — so an account deletion could not cancel it.
            sub_id = str(payload.get("data", {}).get("id") or "") or None
            try:
                portal = payload["data"]["attributes"]["urls"]["customer_portal"] or None
            except (KeyError, TypeError):
                portal = None

            # COALESCE so a later payload that omits a field cannot blank one we
            # already hold — losing the id would silently restore the old
            # cannot-cancel state.
            cur.execute(
                f"UPDATE users SET is_pro = {PH}, "
                f"ls_subscription_id = COALESCE({PH}, ls_subscription_id), "
                f"ls_customer_portal_url = COALESCE({PH}, ls_customer_portal_url) "
                f"WHERE id = {PH}",
                (is_active, sub_id, portal, user_id)
            )
            conn.commit()

        elif event in ("subscription_cancelled", "subscription_expired"):
            # Clear the identifiers along with is_pro: the subscription is no
            # longer cancellable, and a stale id would make a later deletion
            # attempt call the API for nothing.
            cur.execute(
                f"UPDATE users SET is_pro = {PH}, ls_subscription_id = NULL, "
                f"ls_customer_portal_url = NULL WHERE id = {PH}",
                (False, user_id)
            )
            conn.commit()

        elif event == "subscription_payment_failed":
            # Don't immediately revoke — just log for now
            pass

    except Exception:
        conn.rollback()
    finally:
        conn.close()

    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes — scenes & scores
# ---------------------------------------------------------------------------

@app.get("/api/scenes")
async def get_scenes():
    return PUBLIC_SCENES


@app.get("/api/scene-config")
async def get_scene_config():
    return _PUBLIC_SCENE_CONFIG


@app.get("/api/progress")
async def get_progress(user: dict = Depends(current_user)):
    """Return the authenticated user's level, best scores per scene, and
    progress toward the next unlock threshold."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT scene_id, MAX(sync_score) FROM scores WHERE user_id = {PH} GROUP BY scene_id",
        (user["id"],),
    )
    best: dict[str, float] = {row[0]: float(row[1] or 0) for row in cur.fetchall()}
    cur.execute(
        f"SELECT COUNT(*) FROM scores WHERE user_id = {PH} AND transcription = '[quiz pass]'",
        (user["id"],),
    )
    quiz_row = cur.fetchone()
    quiz_passed = bool(quiz_row and quiz_row[0] > 0)
    conn.close()

    # Walk levels in order; each requires a qualifying score on the previous
    # level's scenes.  Break as soon as a threshold isn't met so levels can't
    # be skipped.
    current_level = 1
    for lvl in LEVELS[1:]:
        prev_scenes  = next(l["scenes"] for l in LEVELS if l["level"] == lvl["level"] - 1)
        best_on_prev = max((best.get(s, 0.0) for s in prev_scenes), default=0.0)
        if best_on_prev >= lvl["unlock_score"]:
            current_level = lvl["level"]
        else:
            break

    unlocked = [s for lvl in LEVELS if lvl["level"] <= current_level for s in lvl["scenes"]]

    # Progress info for the bar displayed below the level badge
    next_lvl_def = next((l for l in LEVELS if l["level"] == current_level + 1), None)
    next_level   = None
    if next_lvl_def:
        curr_scenes  = next(l["scenes"] for l in LEVELS if l["level"] == current_level)
        best_on_curr = max((best.get(s, 0.0) for s in curr_scenes), default=0.0)
        next_level   = {
            "level":          next_lvl_def["level"],
            "required_score": next_lvl_def["unlock_score"],
            "best_score":     round(best_on_curr, 1),
        }

    return {
        "level":           current_level,
        "best_scores":     best,
        "unlocked_scenes": unlocked,
        "next_level":      next_level,
        "quiz_passed":     quiz_passed,
    }


@app.get("/api/history")
async def get_history(user: dict = Depends(current_user)):
    """Return the authenticated user's score history and aggregate stats."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT id, scene_id, movie, sync_score, created_at "
        f"FROM scores WHERE user_id = {PH} ORDER BY created_at DESC LIMIT 100",
        (user["id"],),
    )
    rows = cur.fetchall()
    conn.close()

    history = [
        {
            "id":         r[0],
            "scene_id":   r[1],
            "movie":      r[2],
            "sync_score": float(r[3]) if r[3] is not None else 0.0,
            "created_at": r[4].isoformat() if hasattr(r[4], "isoformat") else r[4],
        }
        for r in rows
    ]

    scores = [h["sync_score"] for h in history]
    avg_score     = round(sum(scores) / len(scores), 1) if scores else 0
    best_score    = round(max(scores), 1)               if scores else 0
    first_score   = history[-1]["sync_score"]           if history else 0
    improvement   = round(best_score - first_score, 1)  if history else 0
    unique_scenes = len({h["scene_id"] for h in history})

    return {
        "history": history,
        "stats": {
            "avg_score":      avg_score,
            "best_score":     best_score,
            "total_attempts": len(history),
            "unique_scenes":  unique_scenes,
            "improvement":    improvement,
        },
    }


@app.get("/api/daily")
async def get_daily():
    """Return today's challenge scene (same for every user, resets at UTC midnight)."""
    sid   = get_daily_scene_id()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Seconds until next UTC midnight
    now     = datetime.now(timezone.utc)
    midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)
    secs_left = int((midnight - now).total_seconds())
    return {
        "scene_id":       sid,
        "scene":          PUBLIC_SCENES[sid],
        "date":           today,
        "bonus_multiplier": 2,
        "secs_until_reset": secs_left,
    }


@app.post("/api/submit")
async def submit_recording(
    scene_id: str = Form(...),
    audio: UploadFile = File(...),
    duration_seconds: float = Form(0.0),
    creds: HTTPAuthorizationCredentials = Depends(bearer),
):
    # raises 401 if the token is missing / invalid, or if the account was deleted
    user = require_live_user(decode_token(creds))

    if scene_id not in SCENES:
        raise HTTPException(400, "Invalid scene_id")

    # Per-user submission rate limit (uses DB so it survives restarts)
    conn_rl = get_conn()
    try:
        _check_submit_rate(user["id"], conn_rl)
    finally:
        conn_rl.close()

    scene          = SCENES[scene_id]
    expected_quote = scene["quote"]

    # Validate and read audio
    audio_bytes = await audio.read()
    logger.info(
        "[submit] user_id=%s scene_id=%r filename=%r content_type=%r size=%d",
        user["id"], scene_id, audio.filename, audio.content_type, len(audio_bytes),
    )
    if not audio_bytes:
        logger.warning("[submit] Rejected empty upload from user_id=%s", user["id"])
        raise HTTPException(400, "Empty audio file — please record again and ensure your microphone is working")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file too large — maximum 10 MB")

    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        ext = "." + audio.filename.rsplit(".", 1)[-1].lower()
        suffix = ext if ext in _ALLOWED_AUDIO_EXT else ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcript = get_openai_client().audio.transcriptions.create(model="whisper-1", file=f)
        transcription = transcript.text
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    score = sync_score(expected_quote, transcription)

    conn = get_conn()
    cur  = conn.cursor()

    # Check attempt count and previous best before inserting (single query)
    cur.execute(
        f"SELECT COUNT(*), MAX(sync_score) FROM scores WHERE user_id = {PH} AND scene_id = {PH}",
        (user["id"], scene_id),
    )
    pre = cur.fetchone()
    attempt_count    = pre[0]
    prev_best        = float(pre[1]) if pre[1] is not None else None
    is_first_attempt = attempt_count == 0
    is_new_pb        = prev_best is None or score > prev_best

    # Insert the new score
    cur.execute(
        f"INSERT INTO scores (scene_id, movie, quote, transcription, sync_score, username, user_id) "
        f"VALUES ({PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH})",
        (scene_id, scene["movie"], expected_quote, transcription, score, user["username"], user["id"]),
    )

    # Daily challenge detection
    daily_scene_id     = get_daily_scene_id()
    is_daily           = scene_id == daily_scene_id
    today_str          = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday_str      = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch user streak and last_daily before updating
    cur.execute(f"SELECT streak, last_daily FROM users WHERE id = {PH}", (user["id"],))
    u_row          = cur.fetchone()
    current_streak = int(u_row[0] or 0) if u_row else 0
    last_daily     = u_row[1] if u_row else None
    daily_already_done = (last_daily == today_str)

    # Award points (2× if it's the daily and not yet done today)
    base_pts   = calc_points(score, is_first_attempt)
    daily_bonus = 0
    if is_daily and not daily_already_done:
        daily_bonus = base_pts          # extra pts from doubling
        pts_earned  = base_pts * 2
    else:
        pts_earned  = base_pts

    if pts_earned > 0:
        cur.execute(
            f"UPDATE users SET points = points + {PH} WHERE id = {PH}",
            (pts_earned, user["id"]),
        )

    # Update streak if this completes today's daily for the first time
    new_streak = current_streak
    if is_daily and not daily_already_done:
        if last_daily == yesterday_str:
            new_streak = current_streak + 1
        else:
            new_streak = 1
        cur.execute(
            f"UPDATE users SET streak = {PH}, last_daily = {PH} WHERE id = {PH}",
            (new_streak, today_str, user["id"]),
        )

    # Fetch updated total points
    cur.execute(f"SELECT points FROM users WHERE id = {PH}", (user["id"],))
    total_points = cur.fetchone()[0] or 0

    # Check translation unlock: 3+ attempts AND best score >= SCORE_PROFICIENT
    cur.execute(
        f"SELECT COUNT(*), MAX(sync_score) FROM scores WHERE user_id = {PH} AND scene_id = {PH}",
        (user["id"], scene_id),
    )
    row = cur.fetchone()
    total_attempts = row[0]
    best_score_scene = float(row[1] or 0)
    translation_unlocked = total_attempts >= 3 and best_score_scene >= SCORE_PROFICIENT

    # Advance any active missions and roll user_streak XP/streak counters.
    # take_number is 1-based: this submission's position in the user's history
    # for this scene (1 = first attempt).
    seed_user_missions(user["username"], cur)
    take_number = int(attempt_count) + 1
    missions_updated = await update_missions(
        username=user["username"],
        scene_id=scene_id,
        score=score,
        duration_seconds=duration_seconds,
        take_number=take_number,
        db=cur,
    )
    total_xp_earned = sum(m["xp_earned"] for m in missions_updated)

    conn.commit()
    conn.close()

    # Division before and after this take. The previous one is derived from the
    # points total minus what this take awarded, which is exact — the client
    # could compare against its cached userProfile.division instead, but that
    # only works while the post-score refresh happens to run after the result
    # renders. Stating it here removes the ordering dependency.
    division      = get_division(total_points)
    prev_division = get_division(max(0, total_points - pts_earned))
    return {
        "transcription":        transcription,
        "expected":             expected_quote,
        "sync_score":           score,
        "scene":                PUBLIC_SCENES[scene_id],
        "points_earned":        pts_earned,
        "total_points":         total_points,
        "division":             division,
        "prev_division":        prev_division,
        "is_perfect":           score >= SCORE_PERFECT,
        "is_first_attempt":     is_first_attempt,
        "translation_unlocked": translation_unlocked,
        "translation":          scene.get("translation") if translation_unlocked else None,
        "is_daily":             is_daily,
        "daily_bonus":          daily_bonus,
        "daily_already_done":   daily_already_done,
        "streak":               new_streak,
        "is_new_pb":            is_new_pb,
        "prev_best":            prev_best,
        "missions_updated":     missions_updated,
        "total_xp_earned":      total_xp_earned,
    }


class ChallengeRequest(BaseModel):
    scene_id: str  = Field(..., max_length=50)
    score:    float = Field(..., ge=0, le=100)


@app.post("/api/challenge")
async def create_challenge(req: ChallengeRequest, user: dict = Depends(current_user)):
    """Create a shareable challenge link for the authenticated user's score."""
    if req.scene_id not in SCENES:
        raise HTTPException(400, "Invalid scene_id")
    cid  = uuid.uuid4().hex[:16]   # 16-char hex, URL-safe and short enough to share
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"INSERT INTO challenges (challenge_id, challenger_username, challenger_user_id, scene_id, score_to_beat) "
        f"VALUES ({PH}, {PH}, {PH}, {PH}, {PH})",
        (cid, user["username"], user["id"], req.scene_id, round(req.score, 1)),
    )
    conn.commit()
    conn.close()
    return {
        "challenge_id": cid,
        "url": build_app_url(f"/challenge/{cid}"),
    }


@app.get("/api/challenge/{challenge_id}")
async def get_challenge(challenge_id: str):
    """Return challenge metadata (public — no auth required)."""
    if not re.match(r'^[a-f0-9]{16}$', challenge_id):
        raise HTTPException(400, "Invalid challenge ID")
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT challenge_id, challenger_username, scene_id, score_to_beat, created_at "
        f"FROM challenges WHERE challenge_id = {PH}",
        (challenge_id,),
    )
    row = conn.cursor().fetchone() if False else cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Challenge not found")
    sid = row[2]
    return {
        "challenge_id":         row[0],
        "challenger_username":  row[1],
        "scene_id":             sid,
        "score_to_beat":        float(row[3]),
        "scene":                PUBLIC_SCENES.get(sid, {}),
        "created_at":           row[4].isoformat() if hasattr(row[4], "isoformat") else row[4],
    }


@app.get("/api/leaderboard")
async def get_leaderboard():
    """Top 10 per scene ordered by sync_score desc, with user points and division."""
    conn   = get_conn()
    cur    = conn.cursor()
    result = {}
    for sid in SCENES:
        cur.execute(
            f"SELECT s.id, s.scene_id, s.movie, s.quote, s.transcription, s.sync_score, "
            f"s.username, s.created_at, COALESCE(u.points, 0), COALESCE(u.streak, 0) "
            f"FROM scores s LEFT JOIN users u ON s.user_id = u.id "
            f"WHERE s.scene_id = {PH} ORDER BY s.sync_score DESC LIMIT 10",
            (sid,),
        )
        rows = cur.fetchall()
        result[sid] = []
        for r in rows:
            pts    = int(r[8]) if r[8] is not None else 0
            streak = int(r[9]) if r[9] is not None else 0
            div    = get_division(pts)
            result[sid].append({
                "id": r[0], "scene_id": r[1], "movie": r[2], "quote": r[3],
                "transcription": r[4], "sync_score": r[5], "username": r[6] or "",
                "created_at": r[7].isoformat() if hasattr(r[7], "isoformat") else r[7],
                "user_points": pts,
                "division": div,
                "streak": streak,
            })
    conn.close()
    return result


def _tier_points(score: float) -> int:
    """Score-tier portion of calc_points (omits first-attempt bonus)."""
    if score >= SCORE_PERFECT:     return 100
    if score >= SCORE_TIER_ELITE:  return 75
    if score >= SCORE_TIER_STRONG: return 50
    if score >= SCORE_PROFICIENT:  return 25
    return 0


@app.get("/api/ranks/social")
async def get_ranks_social(user: dict = Depends(current_user)):
    """Aggregated stats for the social/ranks panel: percentile, weekly XP,
    scenes/words progress, recent feed, top-3, and leaderboard."""
    conn = get_conn()
    cur  = conn.cursor()

    # Current user's points + streak — exact same query/logic as /api/profile
    cur.execute(f"SELECT points, streak, last_daily FROM users WHERE id = {PH}", (user["id"],))
    row = cur.fetchone()
    total_points   = int(row[0]) if row and row[0] else 0
    streak         = int(row[1]) if row and row[1] else 0
    last_daily     = row[2] if row else None  # noqa: F841 (kept for parity with /api/profile)

    # Percentile — share of users with strictly fewer points than me
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = int(cur.fetchone()[0] or 0)
    cur.execute(
        f"SELECT COUNT(*) FROM users WHERE COALESCE(points, 0) < {PH}",
        (total_points,),
    )
    lower_count = int(cur.fetchone()[0] or 0)
    percentile  = round((lower_count / total_users) * 100) if total_users else 50
    percentile  = max(1, min(99, percentile))

    # Weekly XP — rolling 7 days. The `scores` table has no points column, so
    # we re-derive XP from `sync_score` via the score-tier portion of
    # calc_points() (omitting the +10 first-attempt bonus). Date cutoff is
    # computed in Python so the query works on both Postgres and SQLite.
    seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        f"SELECT sync_score FROM scores WHERE user_id = {PH} AND created_at >= {PH}",
        (user["id"], seven_days_ago),
    )
    weekly_xp = sum(_tier_points(float(r[0] or 0)) for r in cur.fetchall())

    # Scenes completed — distinct scenes whose best score >= SCORE_PASS
    cur.execute(
        f"SELECT scene_id FROM scores WHERE user_id = {PH} "
        f"GROUP BY scene_id HAVING MAX(sync_score) >= {SCORE_PASS}",
        (user["id"],),
    )
    scenes_completed = len(cur.fetchall())

    # Words mastered
    cur.execute(
        f"SELECT COUNT(*) FROM word_mastery WHERE user_id = {PH} AND correct_count >= 3",
        (user["id"],),
    )
    words_mastered = int(cur.fetchone()[0] or 0)

    # Feed — 10 most recent submissions across all users
    # avatar_scene_id rides the existing LEFT JOIN — no extra join. It is NULL
    # for guest rows (no user_id) and for anyone who hasn't picked an avatar;
    # the client falls back to initials in both cases.
    cur.execute(
        "SELECT s.scene_id, s.sync_score, s.created_at, COALESCE(u.username, s.username), "
        "u.avatar_scene_id "
        "FROM scores s LEFT JOIN users u ON s.user_id = u.id "
        "ORDER BY s.created_at DESC LIMIT 10"
    )
    feed = []
    for r in cur.fetchall():
        sid     = r[0] or ""
        score   = float(r[1] or 0)
        created = r[2]
        uname   = r[3] or "guest"
        feed.append({
            "username":   uname,
            "initials":   (uname[:2].upper() if uname else "??"),
            "avatar_scene_id": r[4],
            "action":     "completed" if score >= SCORE_PASS else "practiced",
            "scene_id":   sid,
            "score":      round(score, 1),
            "points":     _tier_points(score),
            "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
        })

    # Top 10 leaderboard (and slice the first 3 for the podium). avatar_scene_id
    # is a plain column on the same 10 users rows — no join, no extra query. The
    # client resolves the poster URL from the scene config it already holds, so
    # only the id crosses the wire. NULL for anyone who hasn't picked one.
    cur.execute(
        "SELECT id, username, COALESCE(points, 0), avatar_scene_id FROM users "
        "ORDER BY COALESCE(points, 0) DESC LIMIT 10"
    )
    top_rows    = cur.fetchall()
    top3        = []
    leaderboard = []
    for i, r in enumerate(top_rows):
        uid   = int(r[0])
        uname = r[1] or ""
        pts   = int(r[2])
        div   = get_division(pts)
        entry = {
            "username":       uname,
            "initials":       (uname[:2].upper() if uname else "??"),
            "avatar_scene_id": r[3],
            "total_points":   pts,
            "division_name":  div["name"],
            "division_color": div["color"],
        }
        if i < 3:
            top3.append(entry)
        leaderboard.append({**entry, "is_me": uid == user["id"]})

    conn.close()
    print(f"[ranks/social] uid={user['id']} streak={streak} weekly_xp={weekly_xp}")
    return {
        "percentile":       percentile,
        "weekly_xp":        weekly_xp,
        "scenes_completed": scenes_completed,
        "words_mastered":   words_mastered,
        "streak":           streak,
        "feed":             feed,
        "top3":             top3,
        "leaderboard":      leaderboard,
    }


@app.get("/api/missions")
async def get_missions(user: dict = Depends(current_user)):
    """Return the user's active missions, daily quest, weekly challenge, and
    streak/XP summary. Seeds default missions on first call after expiry."""
    conn = get_conn()
    cur  = conn.cursor()

    seed_user_missions(user["username"], cur)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        f"SELECT mission_id, progress, goal, completed, xp_awarded, expires_at "
        f"FROM user_missions WHERE username = {PH} AND expires_at > {PH} "
        f"ORDER BY id ASC",
        (user["username"], now_str),
    )
    missions = [
        {
            "mission_id": r[0],
            "progress":   int(r[1] or 0),
            "goal":       int(r[2]),
            "completed":  bool(r[3]),
            "xp_awarded": bool(r[4]),
            "expires_at": r[5],
        }
        for r in cur.fetchall()
    ]
    by_id = {m["mission_id"]: m for m in missions}

    today_str = date.today().strftime("%Y-%m-%d")
    cur.execute(
        f"SELECT current_streak, longest_streak, last_active_date, total_xp, daily_xp, daily_xp_date "
        f"FROM user_streak WHERE username = {PH}",
        (user["username"],),
    )
    srow = cur.fetchone()
    if srow:
        streak = {
            "current":          int(srow[0] or 0),
            "longest":          int(srow[1] or 0),
            "last_active_date": srow[2] or "",
            "total_xp":         int(srow[3] or 0),
        }
        today_xp = int(srow[4] or 0) if (srow[5] or "") == today_str else 0
    else:
        streak   = {"current": 0, "longest": 0, "last_active_date": "", "total_xp": 0}
        today_xp = 0

    conn.commit()
    conn.close()

    return {
        "daily_quest":      by_id.get("daily"),
        "streak":           streak,
        "weekly_challenge": by_id.get("weekly_thriller"),
        "active_missions":  missions,
        "today_xp":         today_xp,
        "daily_xp_goal":    1000,
    }


@app.get("/api/profile")
async def get_profile(user: dict = Depends(current_user)):
    """Return the authenticated user's points, division, and scene stats."""
    conn = get_conn()
    cur  = conn.cursor()

    cur.execute(f"SELECT points, streak, last_daily, avatar_scene_id FROM users WHERE id = {PH}", (user["id"],))
    row = cur.fetchone()
    total_points    = int(row[0]) if row and row[0] else 0
    streak          = int(row[1]) if row and row[1] else 0
    last_daily      = row[2] if row else None
    avatar_scene_id = row[3] if row else None
    today_str       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_done_today = (last_daily == today_str)

    # Per-scene stats: attempt count + best score
    cur.execute(
        f"SELECT scene_id, COUNT(*), MAX(sync_score) "
        f"FROM scores WHERE user_id = {PH} GROUP BY scene_id",
        (user["id"],),
    )
    scene_rows = cur.fetchall()
    conn.close()

    scene_stats = {}
    translations_unlocked = []
    for r in scene_rows:
        sid, attempts, best = r[0], int(r[1]), float(r[2] or 0)
        scene_stats[sid] = {"attempts": attempts, "best_score": round(best, 1)}
        if attempts >= 3 and best >= SCORE_PROFICIENT:
            translations_unlocked.append(sid)

    division      = get_division(total_points)
    next_div      = get_next_division(total_points)
    pts_to_next   = (next_div["min"] - total_points) if next_div else 0

    return {
        "username":              user["username"],
        "total_points":          total_points,
        "division":              division,
        "next_division":         next_div,
        "points_to_next":        pts_to_next,
        "scene_stats":           scene_stats,
        "translations_unlocked": translations_unlocked,
        "streak":                streak,
        "daily_done_today":      daily_done_today,
        "daily_scene_id":        get_daily_scene_id(),
        "avatar_scene_id":       avatar_scene_id,
    }


@app.post("/api/profile/avatar")
async def set_avatar(payload: dict, user: dict = Depends(current_user)):
    """Set (or clear) the user's chosen avatar scene poster.

    The frontend unlock UI is cosmetic only — this endpoint is the real gate.
    A caller can POST any scene_id; the endpoint rejects it unless BOTH:
      1. scene_id is a known id from scene_config (path-safety — this value
         ends up in an <img src> client-side, so an arbitrary string can't
         reach the column).
      2. The user has scored >= SCORE_PASS on that scene.

    Pass null / empty scene_id to clear back to the default icon.

    NOTE: the 60-point unlock threshold is duplicated in three places:
      - static/app.js makeCard() (isDone gate on scene cards)
      - index.html renderLevelsPath() (Levels-path .done state)
      - this endpoint (avatar unlock gate)
    Any change to the threshold must update all three; there is no shared
    constant today.
    """
    scene_id = payload.get("scene_id")

    # null / empty → clear back to default icon
    if scene_id is None or scene_id == "":
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(f"UPDATE users SET avatar_scene_id = NULL WHERE id = {PH}", (user["id"],))
        conn.commit()
        conn.close()
        return {"avatar_scene_id": None}

    if not isinstance(scene_id, str):
        raise HTTPException(400, "scene_id must be a string or null")

    # Path-safety gate: must be a known scene id from scene_config
    if scene_id not in SCENES:
        raise HTTPException(400, "Unknown scene_id")

    # Unlock gate: user must have scored >= SCORE_PASS on this scene
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT COALESCE(MAX(sync_score), 0) FROM scores WHERE user_id = {PH} AND scene_id = {PH}",
        (user["id"], scene_id),
    )
    row = cur.fetchone()
    best = float(row[0] or 0) if row else 0.0
    if best < SCORE_PASS:
        conn.close()
        raise HTTPException(403, "Scene not yet unlocked")

    cur.execute(
        f"UPDATE users SET avatar_scene_id = {PH} WHERE id = {PH}",
        (scene_id, user["id"]),
    )
    conn.commit()
    conn.close()
    return {"avatar_scene_id": scene_id}


# Which level each quiz unlocks. "level1" is the Level 1 quiz, which opens
# Level 2. Adding a Level 3 quiz means one more entry here and nothing else.
QUIZ_UNLOCKS = {"level1": 2, "level2": 3}

# How far above the unlock threshold a quiz pass is recorded. Level 1's
# threshold is 60, so this reproduces the historical 75.0 exactly; keeping the
# margin rather than writing the threshold itself means the score never sits on
# the boundary, and never silently changes which _tier_points() band it falls
# into when a threshold moves.
QUIZ_PASS_MARGIN = 15.0


@app.post("/api/quiz-pass")
async def quiz_pass(payload: dict, user: dict = Depends(current_user)):
    """Register a qualifying score when the user passes a level quiz, so the
    existing score-driven unlock logic opens the next level.

    There is no separate grant mechanism: /api/progress decides the user's level
    purely from `scores`, so a quiz pass is recorded as a synthetic row marked
    '[quiz pass]' on a scene belonging to the level below the one being opened.

    The scene is taken from LEVELS, not from the scene's `difficulty` tag,
    because LEVELS is what the unlock check reads. Those two disagree: level 2
    lists 20 scenes while 26 are tagged "intermediate", the extra six belonging
    to level 3. Selecting by difficulty could therefore write the row onto a
    level-3 scene, which would never satisfy the level-3 unlock. They coincide
    for level 1 today, which is why the old difficulty-based lookup worked.
    """
    quiz  = payload.get("quiz", "")
    score = float(payload.get("score", 0))

    target_level = QUIZ_UNLOCKS.get(quiz)
    if target_level is None:
        raise HTTPException(400, "Unknown quiz")
    if score < SCORE_PROFICIENT:
        raise HTTPException(400, "Quiz score is below the pass mark")

    target_def = next((l for l in LEVELS if l["level"] == target_level), None)
    prev_def   = next((l for l in LEVELS if l["level"] == target_level - 1), None)
    if not target_def or not prev_def or not prev_def.get("scenes"):
        return {"unlocked": False}

    sid = prev_def["scenes"][0]
    scene = SCENES.get(sid)
    if not scene:
        return {"unlocked": False}

    # Derived from the threshold this pass has to clear, not a magic number: the
    # old hardcoded 75.0 cleared level 2's threshold of 60 only by being larger,
    # and would have silently stopped unlocking if a threshold were raised past
    # it. Written *above* the threshold rather than exactly at it — sitting on
    # the boundary leaves no headroom, and for level 1 it would drop the
    # long-standing 75.0 to 60.0, which crosses a _tier_points() boundary and
    # would quietly halve what a quiz pass is worth in weekly XP (50 -> 25).
    # QUIZ_PASS_MARGIN is 15 precisely so level 1 still grants 75.0.
    granted = min(100.0, float(target_def.get("unlock_score", SCORE_PROFICIENT)) + QUIZ_PASS_MARGIN)

    conn  = get_conn()
    cur   = conn.cursor()
    cur.execute(
        f"INSERT INTO scores (scene_id, movie, quote, transcription, sync_score, username, user_id) "
        f"VALUES ({PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH})",
        (sid, scene.get("movie", ""), scene.get("quote", ""), "[quiz pass]", granted,
         user["username"], user["id"]),
    )
    conn.commit()
    conn.close()
    return {"unlocked": True, "level": target_level, "score": granted}


@app.get("/api/translations")
async def get_translations(user: dict = Depends(current_user)):
    """Return {scene_id: spanish_translation} for every scene the user has
    unlocked translations on (3+ attempts AND best score >= SCORE_PROFICIENT)."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT scene_id, COUNT(*), MAX(sync_score) "
        f"FROM scores WHERE user_id = {PH} GROUP BY scene_id",
        (user["id"],),
    )
    rows = cur.fetchall()
    conn.close()

    result: dict[str, str] = {}
    for r in rows:
        sid, attempts, best = r[0], int(r[1]), float(r[2] or 0)
        if attempts >= 3 and best >= SCORE_PROFICIENT:
            translation = SCENES.get(sid, {}).get("translation")
            if translation:
                result[sid] = translation
    return result


@app.post("/api/vocab/mastery")
async def post_vocab_mastery(payload: dict, user: dict = Depends(current_user)):
    """Record a vocabulary answer. correct=True increments correct_count
    (capped at 3); correct=False is a no-op so failed attempts don't reset
    progress."""
    scene_id = payload.get("scene_id", "")
    word_en  = payload.get("word_en", "")
    correct  = bool(payload.get("correct", False))
    if not scene_id or not word_en:
        raise HTTPException(400, "scene_id and word_en are required")
    if not correct:
        return {"correct_count": 0, "noop": True}

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT correct_count FROM word_mastery "
        f"WHERE user_id = {PH} AND scene_id = {PH} AND word_en = {PH}",
        (user["id"], scene_id, word_en),
    )
    row = cur.fetchone()
    if row is None:
        new_count = 1
        cur.execute(
            f"INSERT INTO word_mastery (user_id, scene_id, word_en, correct_count) "
            f"VALUES ({PH}, {PH}, {PH}, {PH})",
            (user["id"], scene_id, word_en, new_count),
        )
    else:
        new_count = min(int(row[0]) + 1, 3)
        cur.execute(
            f"UPDATE word_mastery SET correct_count = {PH}, updated_at = CURRENT_TIMESTAMP "
            f"WHERE user_id = {PH} AND scene_id = {PH} AND word_en = {PH}",
            (new_count, user["id"], scene_id, word_en),
        )
    conn.commit()
    conn.close()
    return {"correct_count": new_count, "noop": False}


@app.get("/api/vocab/mastery")
async def get_vocab_mastery(scene_id: str, user: dict = Depends(current_user)):
    """Return {word_en: correct_count} for the authenticated user on a scene."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT word_en, correct_count FROM word_mastery "
        f"WHERE user_id = {PH} AND scene_id = {PH}",
        (user["id"], scene_id),
    )
    rows = cur.fetchall()
    conn.close()
    return {r[0]: int(r[1]) for r in rows}


@app.get("/api/vocab/{scene_id}")
async def get_vocab(scene_id: str, user: dict = Depends(current_user)):
    """Return 8 vocabulary items for a scene. Uses word_vocab as a cache;
    on miss, asks Claude for the list and persists it."""
    if scene_id not in SCENES:
        raise HTTPException(404, "Scene not found")

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        f"SELECT word_en, word_es, phonetic, example, word_type "
        f"FROM word_vocab WHERE scene_id = {PH} ORDER BY id ASC",
        (scene_id,),
    )
    rows = cur.fetchall()
    if len(rows) >= 8:
        conn.close()
        return [
            {"en": r[0], "es": r[1], "phonetic": r[2], "example": r[3], "type": r[4]}
            for r in rows[:8]
        ]

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        conn.close()
        raise HTTPException(500, "ANTHROPIC_API_KEY environment variable is not set")

    prompt = (
        f"Generate exactly 8 key English vocabulary words from the movie scene "
        f"'{scene_id}' for Spanish-speaking ESL learners. Return ONLY a JSON array: "
        f"[{{\"en\":\"word\",\"es\":\"español\",\"phonetic\":\"/fəˈnɛtɪk/\","
        f"\"example\":\"Short cinematic example sentence.\",\"type\":\"verb\"}}] "
        f"Types: verb, noun, adj, adv."
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      "claude-sonnet-4-6",
                    "max_tokens": 1000,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        body = ""
        try:
            body = e.response.text if hasattr(e, "response") and e.response is not None else ""
        except Exception:
            body = ""
        print(f"VOCAB ENDPOINT ERROR (httpx): {e} | body={body[:500]}")
        traceback.print_exc()
        conn.close()
        raise HTTPException(502, f"Anthropic API error: {e}")
    except Exception as e:
        print(f"VOCAB ENDPOINT ERROR: {e}")
        traceback.print_exc()
        conn.close()
        raise

    data = resp.json()
    try:
        text = data["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        conn.close()
        raise HTTPException(502, "Unexpected Anthropic response shape")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        words = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            conn.close()
            raise HTTPException(500, "Failed to parse vocab JSON from model response")
        try:
            words = json.loads(m.group(0))
        except json.JSONDecodeError:
            conn.close()
            raise HTTPException(500, "Failed to parse vocab JSON from model response")

    if not isinstance(words, list) or not words:
        conn.close()
        raise HTTPException(500, "Model returned no vocab items")

    for w in words:
        if not isinstance(w, dict):
            continue
        cur.execute(
            f"INSERT INTO word_vocab (scene_id, word_en, word_es, phonetic, example, word_type) "
            f"VALUES ({PH}, {PH}, {PH}, {PH}, {PH}, {PH})",
            (
                scene_id,
                w.get("en", ""),
                w.get("es", ""),
                w.get("phonetic", ""),
                w.get("example", ""),
                w.get("type", ""),
            ),
        )
    conn.commit()
    conn.close()
    return words
