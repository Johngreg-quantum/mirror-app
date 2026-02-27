import os
import re
import sqlite3
import difflib
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MIRROR — Movie Scene Language Learning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH   = "mirror.db"
SECRET    = os.getenv("JWT_SECRET", "change-me-to-a-long-random-string-in-production")
ALGORITHM = "HS256"
TOKEN_TTL = 30  # days

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

SCENES = {
    "fight_club": {
        "movie": "Fight Club",
        "quote": "You know what a duvet is?",
        "year": 1999, "difficulty": "Intermediate", "actor": "Brad Pitt",
    },
    "back_to_the_future": {
        "movie": "Back to the Future",
        "quote": "Where we're going, we don't need roads.",
        "year": 1985, "difficulty": "Advanced", "actor": "Christopher Lloyd",
    },
    "forrest_gump": {
        "movie": "Forrest Gump",
        "quote": "You never know what you're gonna get.",
        "year": 1994, "difficulty": "Advanced", "actor": "Tom Hanks",
    },
    "the_matrix": {
        "movie": "The Matrix",
        "quote": "I know kung fu.",
        "year": 1999, "difficulty": "Beginner", "actor": "Keanu Reeves",
    },
    "seven": {
        "movie": "Se7en",
        "quote": "What's in the box?",
        "year": 1995, "difficulty": "Beginner", "actor": "Brad Pitt",
    },
}

# Level unlock rules — must be kept in sync with LEVEL_MAP in index.html.
# Each level lists which scene IDs belong to it and what minimum sync_score
# (%) a user needs on any scene from the *previous* level to unlock it.
LEVELS = [
    {"level": 1, "scenes": ["the_matrix", "seven"],                "unlock_score": 0},
    {"level": 2, "scenes": ["fight_club"],                         "unlock_score": 60},
    {"level": 3, "scenes": ["back_to_the_future", "forrest_gump"], "unlock_score": 70},
]


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

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
    username: str
    email:    str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str


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

def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    return decode_token(creds)


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
# Routes — frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Routes — auth
# ---------------------------------------------------------------------------

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    username = req.username.strip()
    email    = req.email.lower().strip()

    if len(username) < 2:
        raise HTTPException(400, "Username must be at least 2 characters")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if "@" not in email:
        raise HTTPException(400, "Invalid email address")

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
    return user


# ---------------------------------------------------------------------------
# Routes — scenes & scores
# ---------------------------------------------------------------------------

@app.get("/api/scenes")
async def get_scenes():
    return SCENES


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
    }


@app.post("/api/submit")
async def submit_recording(
    scene_id: str = Form(...),
    audio: UploadFile = File(...),
    creds: HTTPAuthorizationCredentials = Depends(bearer),
):
    user = decode_token(creds)  # raises 401 if missing / invalid

    if scene_id not in SCENES:
        raise HTTPException(400, "Invalid scene_id")

    scene          = SCENES[scene_id]
    expected_quote = scene["quote"]

    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
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
    cur.execute(
        f"INSERT INTO scores (scene_id, movie, quote, transcription, sync_score, username, user_id) "
        f"VALUES ({PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH})",
        (scene_id, scene["movie"], expected_quote, transcription, score, user["username"], user["id"]),
    )
    conn.commit()
    conn.close()

    return {"transcription": transcription, "expected": expected_quote, "sync_score": score, "scene": scene}


@app.get("/api/leaderboard")
async def get_leaderboard():
    """Top 10 per scene ordered by sync_score desc."""
    conn   = get_conn()
    cur    = conn.cursor()
    result = {}
    for sid in SCENES:
        cur.execute(
            f"SELECT id, scene_id, movie, quote, transcription, sync_score, username, created_at "
            f"FROM scores WHERE scene_id = {PH} ORDER BY sync_score DESC LIMIT 10",
            (sid,),
        )
        rows = cur.fetchall()
        result[sid] = [
            {
                "id": r[0], "scene_id": r[1], "movie": r[2], "quote": r[3],
                "transcription": r[4], "sync_score": r[5], "username": r[6] or "",
                # PostgreSQL returns datetime objects; SQLite returns strings.
                # Both serialise correctly via FastAPI's JSON encoder.
                "created_at": r[7].isoformat() if hasattr(r[7], "isoformat") else r[7],
            }
            for r in rows
        ]
    conn.close()
    return result
