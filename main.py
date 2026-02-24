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

client    = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DB_PATH   = "mirror.db"
SECRET    = os.getenv("JWT_SECRET", "change-me-to-a-long-random-string-in-production")
ALGORITHM = "HS256"
TOKEN_TTL = 30  # days

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


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
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

    # Non-destructive migrations for pre-existing scores table
    for col, dfn in [("username", "TEXT DEFAULT ''"), ("user_id", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE scores ADD COLUMN {col} {dfn}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.commit()
    conn.close()


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

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, hash_pw(req.password)),
        )
        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email or username already taken")
    finally:
        conn.close()

    return {"access_token": make_token(user_id, username), "token_type": "bearer", "username": username}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE email = ?",
        (req.email.lower().strip(),),
    ).fetchone()
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
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        transcription = transcript.text
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    score = sync_score(expected_quote, transcription)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO scores (scene_id, movie, quote, transcription, sync_score, username, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (scene_id, scene["movie"], expected_quote, transcription, score, user["username"], user["id"]),
    )
    conn.commit()
    conn.close()

    return {"transcription": transcription, "expected": expected_quote, "sync_score": score, "scene": scene}


@app.get("/api/leaderboard")
async def get_leaderboard():
    """Top 10 per scene ordered by sync_score desc."""
    conn   = sqlite3.connect(DB_PATH)
    result = {}
    for sid in SCENES:
        rows = conn.execute(
            "SELECT id, scene_id, movie, quote, transcription, sync_score, username, created_at "
            "FROM scores WHERE scene_id = ? ORDER BY sync_score DESC LIMIT 10",
            (sid,),
        ).fetchall()
        result[sid] = [
            {"id": r[0], "scene_id": r[1], "movie": r[2], "quote": r[3],
             "transcription": r[4], "sync_score": r[5], "username": r[6] or "", "created_at": r[7]}
            for r in rows
        ]
    conn.close()
    return result
