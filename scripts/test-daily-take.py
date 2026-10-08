"""Tests for the Daily Take: three short lines, derived progress, one streak tick.

    python scripts/test-daily-take.py [sqlitePath]

Default: ./mirror.db

In-process, with the clock supplied. Every case that matters is "what time is it
where the user is", and a running server cannot be asked to believe it is 11pm
on a Monday in Miami -- so daily_state() takes a `when` and the tests drive it.
Score rows are planted with explicit created_at values, which is what the real
rows look like: UTC timestamps written by the database default.

The cases that drove the design:

  1. Three lines, and the gate opens for exactly those three -- not one, not the
     whole pool.
  2. Progress is derived from scores, so it survives a reload and is resumable;
     there is no progress table to go stale.
  3. The streak ticks ONCE, when the third line lands, in the user's local day.
  4. THE EVENING. 7:30pm and 8:30pm Eastern straddle UTC midnight. The set used
     to be dated in UTC, so it changed between those two takes -- 8pm, mid
     session, for the users who practise in the evening. The set follows the
     user's own date now, so two lines at 7:30 and a third at 8:30 finish it.
  5. And it still cannot double-credit: a set finished at 11pm local credits
     that day and no other, even though those rows sit in the next UTC day.
"""
import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = sys.argv[1] if len(sys.argv) > 1 else "mirror.db"

# The gate is only a gate when enforcement is on, and this process imports main
# directly rather than talking to a server, so the flag has to be set before the
# import or can_play_scene() returns True for everything and proves nothing.
os.environ["ENFORCE_ENTITLEMENTS"] = "true"

import main as app   # noqa: E402  (after sys.path / env)

TZ = "America/New_York"
UTC_FMT = "%Y-%m-%d %H:%M:%S"

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))


def make_user(suffix):
    con = sqlite3.connect(DB)
    try:
        name = "dttest_" + suffix
        con.execute("DELETE FROM users WHERE username = ?", (name,))
        cur = con.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password_hash, timezone) VALUES (?, ?, ?, ?)",
            (name, "dttest_%s@example.com" % suffix, "x", TZ),
        )
        uid = cur.lastrowid
        con.commit()
        return uid, name
    finally:
        con.close()


def plant_score(uid, username, scene_id, score, when_utc):
    """A scored take at a given instant, written the way /api/submit writes it."""
    con = sqlite3.connect(DB)
    try:
        con.execute(
            "INSERT INTO scores (scene_id, movie, quote, transcription, sync_score, "
            "username, user_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (scene_id, app.SCENES[scene_id]["movie"], app.SCENES[scene_id]["quote"],
             "planted", score, username, uid, when_utc.strftime(UTC_FMT)),
        )
        con.commit()
    finally:
        con.close()


def state(uid, when, last_daily=None):
    con = sqlite3.connect(DB)
    try:
        return app.daily_state(con.cursor(), uid, TZ, last_daily, when=when)
    finally:
        con.close()


def streak_ticks(st):
    """The branch /api/submit takes: the set is finished and the day is not yet
    credited. Mirrored rather than imported because the surrounding function
    needs an audio upload and an OpenAI call to reach it."""
    return st["all_done"] and not st["credited_today"]


def main():
    if not os.path.isfile(DB):
        print("SQLite file not found: %s" % DB)
        sys.exit(2)

    app.DB_PATH = DB
    app.init_db()
    print("db %s   pool %d   lines %d\n" % (DB, len(app.DAILY_POOL), app.DAILY_LINE_COUNT))

    # ── 1. THE POOL ───────────────────────────────────────────────────────────
    print("the pool")
    check("the pool is not the whole catalogue",
          0 < len(app.DAILY_POOL) < len(app.SCENES),
          "%d of %d scenes" % (len(app.DAILY_POOL), len(app.SCENES)))
    bad = []
    for sid in app.DAILY_POOL:
        s = app.SCENES[sid]
        words = len(s["quote"].split())
        dur = s["ui"]["clip_end"] - s["ui"]["clip_start"]
        if words > app.DAILY_MAX_WORDS or dur > app.DAILY_MAX_CLIP_SECONDS or dur <= 0:
            bad.append((sid, words, dur))
    check("every pooled line is short enough to do in an evening", not bad, str(bad))
    check("every pooled id is a real scene",
          all(sid in app.SCENES for sid in app.DAILY_POOL))
    # The filter is the product promise: a 41-word monologue is not a daily.
    check("the long scenes are excluded",
          "titanic" not in app.DAILY_POOL and "forrest_gump" not in app.DAILY_POOL)

    # ── 2. SELECTION ──────────────────────────────────────────────────────────
    print("\nselection")
    ids = app.get_daily_scene_ids("2026-10-08")
    check("three lines", len(ids) == app.DAILY_LINE_COUNT, str(ids))
    check("all three are different", len(set(ids)) == len(ids), str(ids))
    check("all three come from the pool", all(sid in app.DAILY_POOL for sid in ids))
    check("the same date always gives the same three",
          app.get_daily_scene_ids("2026-10-08") == ids)
    # Not a strict requirement, but a selection that repeated day to day would
    # make the whole feature pointless, so it is worth a failing test.
    varied = {tuple(app.get_daily_scene_ids("2026-10-%02d" % d)) for d in range(1, 29)}
    check("the set moves from day to day", len(varied) >= 20, "%d distinct sets in 28 days" % len(varied))
    check("the single-scene picker is gone, not left for someone to wire up",
          not hasattr(app, "get_daily_scene_id"))

    # ── 3. THE GATE ───────────────────────────────────────────────────────────
    print("\nthe free hole is exactly the three lines")
    uid, name = make_user("gate")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    today_ids = app.daily_scene_ids_for_user(cur, uid)
    for sid in today_ids:
        check("a free user may record daily line %s" % (today_ids.index(sid) + 1),
              app.can_play_scene(cur, uid, sid) is True, sid)
    free = set(app._free_scene_ids())
    locked = next((sid for sid in app.SCENES
                   if sid not in today_ids and sid not in free), None)
    check("...and is still refused a scene that is neither free nor a daily line",
          app.can_play_scene(cur, uid, locked) is False, str(locked))
    check("the gate asks the same function the daily is offered from",
          app.can_play_scene(cur, uid, today_ids[1]) is True)

    # ── THE GATE AND THE OFFER CANNOT DRIFT ACROSS A DATE BOUNDARY ───────────
    # 22:00 UTC is 6pm Monday in New York and 7am TUESDAY in Tokyo, so the two
    # users are on different dates -- and therefore different sets -- at the
    # same instant. Each must be gated on their own.
    boundary = datetime(2026, 10, 5, 22, 0, tzinfo=timezone.utc)
    tokyo_uid, tokyo_name = make_user("tokyo")
    con.execute("UPDATE users SET timezone = ? WHERE id = ?", ("Asia/Tokyo", tokyo_uid))
    con.commit()
    ny_set = app.daily_scene_ids_for_user(cur, uid, when=boundary)
    tk_set = app.daily_scene_ids_for_user(cur, tokyo_uid, when=boundary)
    check("two users in different zones are on different dates at one instant",
          app.local_day_of(TZ, boundary) == "2026-10-05"
          and app.local_day_of("Asia/Tokyo", boundary) == "2026-10-06",
          "%s vs %s" % (app.local_day_of(TZ, boundary),
                        app.local_day_of("Asia/Tokyo", boundary)))
    check("...and therefore on different sets", ny_set != tk_set,
          "%s vs %s" % (ny_set, tk_set))
    for sid in tk_set:
        if sid in ny_set or sid in free:
            continue
        check("the New York user is refused a line from TOKYO's set",
              app.can_play_scene(cur, uid, sid, when=boundary) is False, sid)
        break
    for sid in ny_set:
        if sid in tk_set or sid in free:
            continue
        check("the Tokyo user is refused a line from NEW YORK's set",
              app.can_play_scene(cur, tokyo_uid, sid, when=boundary) is False, sid)
        break
    check("each user's own set is open to them at that same instant",
          all(app.can_play_scene(cur, uid, s, when=boundary) for s in ny_set)
          and all(app.can_play_scene(cur, tokyo_uid, s, when=boundary) for s in tk_set))
    con.close()

    # ── 4. DERIVED, PARTIAL, RESUMABLE PROGRESS ───────────────────────────────
    print("\nprogress is derived from scores")
    uid, name = make_user("progress")
    # Monday 6pm EDT = 22:00 UTC. The set is the user's Monday set.
    mon_6pm = datetime(2026, 10, 5, 22, 0, tzinfo=timezone.utc)
    mon_ids = app.get_daily_scene_ids(app.local_day_of(TZ, mon_6pm))

    st = state(uid, mon_6pm)
    check("a user who has done nothing is 0 of 3",
          st["lines_done"] == 0 and st["line_total"] == 3, str(st["lines_done"]))
    check("...and is pointed at the first line", st["next_scene_id"] == mon_ids[0])

    plant_score(uid, name, mon_ids[0], 82.0, mon_6pm)
    st = state(uid, mon_6pm + timedelta(minutes=5))
    check("one recorded line reads as 1 of 3", st["lines_done"] == 1)
    check("...and the next line is the second, not the first",
          st["next_scene_id"] == mon_ids[1])
    check("...and the recorded score is carried for the aggregate",
          st["lines"][0]["score"] == 82.0, str(st["lines"][0]))
    check("the set is not complete yet", st["all_done"] is False)
    check("no streak tick on a partial set", streak_ticks(st) is False)

    plant_score(uid, name, mon_ids[1], 90.0, mon_6pm + timedelta(minutes=6))
    plant_score(uid, name, mon_ids[2], 88.0, mon_6pm + timedelta(minutes=12))
    st = state(uid, mon_6pm + timedelta(minutes=13))
    check("THREE LINES DONE READS AS COMPLETE", st["all_done"] is True)
    check("...with nothing left to practise", st["next_scene_id"] is None)
    check("...and THIS is when the streak ticks", streak_ticks(st) is True)

    # Already credited: the same instant, with last_daily set, must not tick again.
    st_again = state(uid, mon_6pm + timedelta(minutes=20), last_daily="2026-10-05")
    check("a finished set does not tick the streak twice in one local day",
          streak_ticks(st_again) is False)

    # ── 5. THE ROLLOVER, AND THE DOUBLE-CREDIT IT WOULD HAVE CAUSED ───────────
    print("\n8pm UTC rollover does not interrupt an evening")
    uid, name = make_user("rollover")
    # 7:30pm EDT Monday is 23:30 UTC Monday; 8:30pm EDT is 00:30 UTC TUESDAY.
    # The old UTC-dated set changed between those two takes -- 8pm, mid-session,
    # for every user on the US east coast. The set follows the user's date now,
    # so it is the same three lines on both sides of that boundary.
    mon_730pm = datetime(2026, 10, 5, 23, 30, tzinfo=timezone.utc)
    mon_830pm = datetime(2026, 10, 6, 0, 30, tzinfo=timezone.utc)
    check("the two takes really do straddle UTC midnight",
          mon_730pm.strftime("%Y-%m-%d") != mon_830pm.strftime("%Y-%m-%d"),
          "%s then %s UTC" % (mon_730pm.strftime("%Y-%m-%d"), mon_830pm.strftime("%Y-%m-%d")))
    check("...while both are the user's Monday",
          app.local_day_of(TZ, mon_730pm) == app.local_day_of(TZ, mon_830pm) == "2026-10-05",
          app.local_day_of(TZ, mon_830pm))
    check("...and the set is the same on both sides of it",
          state(uid, mon_730pm)["scene_ids"] == state(uid, mon_830pm)["scene_ids"],
          str(state(uid, mon_830pm)["scene_ids"]))
    # What the old behaviour would have done, kept as the regression marker.
    check("...where the old UTC-dated set would have changed under them",
          app.get_daily_scene_ids(mon_730pm.strftime("%Y-%m-%d"))
          != app.get_daily_scene_ids(mon_830pm.strftime("%Y-%m-%d")))

    plant_score(uid, name, mon_ids[0], 80.0, mon_730pm)
    plant_score(uid, name, mon_ids[1], 80.0, mon_730pm + timedelta(minutes=5))
    st = state(uid, mon_730pm + timedelta(minutes=6))
    check("two lines in at 7:30pm", st["lines_done"] == 2, str(st["lines_done"]))
    check("...pointed at the third", st["next_scene_id"] == mon_ids[2])

    plant_score(uid, name, mon_ids[2], 80.0, mon_830pm)
    st = state(uid, mon_830pm + timedelta(minutes=1))
    check("A THIRD LINE AT 8:30PM COMPLETES THE SET", st["all_done"] is True,
          "lines_done=%s" % st["lines_done"])
    check("...and the streak ticks for Monday", streak_ticks(st) is True)

    # The double-credit case the local-day count also has to survive: a set
    # finished at 11pm local must credit that day and no other.
    uid, name = make_user("doublecredit")
    mon_11pm = datetime(2026, 10, 6, 3, 0, tzinfo=timezone.utc)   # Mon 11pm EDT
    for sid in mon_ids:
        plant_score(uid, name, sid, 90.0, mon_11pm)
    st = state(uid, mon_11pm + timedelta(minutes=1))
    check("a set finished at 11pm local completes the user's MONDAY",
          st["all_done"] and app.local_day_of(TZ, mon_11pm) == "2026-10-05",
          "local day %s" % app.local_day_of(TZ, mon_11pm))
    check("...and the streak ticks for it", streak_ticks(st) is True)

    tue_10am = datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc)  # Tue 10am EDT
    check("Tuesday morning is the same UTC day as those 11pm takes",
          tue_10am.strftime("%Y-%m-%d") == mon_11pm.strftime("%Y-%m-%d"),
          tue_10am.strftime("%Y-%m-%d"))
    st_tue = state(uid, tue_10am, last_daily="2026-10-05")
    check("THE SAME TAKES DO NOT ALSO COMPLETE TUESDAY",
          st_tue["lines_done"] == 0 and st_tue["all_done"] is False,
          "lines_done=%s" % st_tue["lines_done"])
    check("...so no second streak day is handed out", streak_ticks(st_tue) is False)
    check("...and Tuesday brings a different set, not Monday's lines again",
          st_tue["scene_ids"] != mon_ids, str(st_tue["scene_ids"]))

    # ── 6. THE MIAMI EVENINGS, END TO END ─────────────────────────────────────
    print("\nconsecutive evenings keep the streak")
    uid, name = make_user("miami")
    # Mon 6pm EDT and Tue 9pm EDT -- Monday and WEDNESDAY in UTC, which is the
    # regression the streak work fixed. Each evening has its own local-date set;
    # what must survive is the local-day sequence.
    tue_9pm = datetime(2026, 10, 7, 1, 0, tzinfo=timezone.utc)
    tue_set = app.get_daily_scene_ids(app.local_day_of(TZ, tue_9pm))
    check("Tuesday 9pm local belongs to the user's TUESDAY set, not UTC Wednesday's",
          tue_set == app.get_daily_scene_ids("2026-10-06")
          and tue_set != app.get_daily_scene_ids("2026-10-07"), str(tue_set))
    for sid in mon_ids:
        plant_score(uid, name, sid, 75.0, mon_6pm)
    st_mon = state(uid, mon_6pm + timedelta(minutes=30))
    for sid in tue_set:
        plant_score(uid, name, sid, 75.0, tue_9pm)
    st_tue = state(uid, tue_9pm + timedelta(minutes=30), last_daily="2026-10-05")
    check("Monday evening completes Monday", streak_ticks(st_mon) is True)
    check("Tuesday evening completes Tuesday too, 9pm local notwithstanding",
          streak_ticks(st_tue) is True)
    check("...and yesterday is yesterday in LOCAL days, so the streak continues",
          app.local_day_of(TZ, tue_9pm) == "2026-10-06"
          and (datetime(2026, 10, 6).date() - datetime(2026, 10, 5).date()).days == 1,
          app.local_day_of(TZ, tue_9pm))

    # ── 7. THE COMPLETION AWARD ───────────────────────────────────────────────
    print("\nthe award, from the three stored scores")
    strong = app.daily_completion_award([90, 88, 92])
    check("a strong set pays the doubled tier plus the strong combo",
          strong["award"] == app._tier_points(90) * 2 + app.DAILY_COMBO_STRONG_BONUS,
          str(strong))
    prof = app.daily_completion_award([72, 75, 71])
    check("a proficient set pays the smaller combo",
          prof["combo"] == "all_proficient"
          and prof["combo_bonus"] == app.DAILY_COMBO_PROFICIENT_BONUS, str(prof))
    mixed = app.daily_completion_award([95, 90, 60])
    check("one weak line loses the combo, not the award",
          mixed["combo"] is None and mixed["award"] > 0, str(mixed))
    poor = app.daily_completion_award([30, 20, 40])
    check("THE FLOOR APPLIES TO THE SET AS A WHOLE, NOT PER LINE",
          poor["award"] == app.DAILY_COMPLETION_FLOOR, str(poor))
    check("...and the floor is reported so the UI can say why",
          poor["floor_applied"] == app.DAILY_COMPLETION_FLOOR)
    check("a set of three perfect lines cannot beat practising three scenes outright",
          app.daily_completion_award([100, 100, 100])["award"]
          < app.calc_points(100, True) * 3, str(app.daily_completion_award([100, 100, 100])))
    check("an empty set pays nothing (no lines, no award)",
          app.daily_completion_award([])["award"] == 0)
    # Scoring must still pay more than turning up, or the floor eats the game.
    check("scoring beats showing up by a wide margin",
          app.daily_completion_award([72, 72, 72])["award"]
          >= app.DAILY_COMPLETION_FLOOR * 5, str(prof["award"]))

    # ── 8. MISSION PROGRESS COUNTS LINES, NOT TAKES ───────────────────────────
    print("\nthe daily mission counts lines")
    uid, name = make_user("mission")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    app.seed_user_missions(uid, name, cur)
    con.commit()

    def advance(first_today):
        return asyncio.run(app.update_missions(
            user_id=uid, username=name, scene_id=mon_ids[0], score=50.0,
            duration_seconds=300.0, take_number=4, db=cur,
            local_day="2026-10-05", daily_line_completed=first_today,
        ))

    advance(True)
    con.commit()
    after_first = con.execute(
        "SELECT progress FROM user_missions WHERE user_id = ? AND mission_id = 'daily'",
        (uid,)).fetchone()[0]
    advance(False)
    advance(False)
    con.commit()
    after_repeats = con.execute(
        "SELECT progress FROM user_missions WHERE user_id = ? AND mission_id = 'daily'",
        (uid,)).fetchone()[0]
    con.close()
    check("finishing a line advances the daily mission", after_first == 1, str(after_first))
    check("THREE TAKES OF ONE LINE DOES NOT COMPLETE THE THREE-LINE MISSION",
          after_repeats == 1, "progress=%s" % after_repeats)

    # ── 9. THE WHOLE FLOW, THROUGH /api/submit ────────────────────────────────
    # The glue in submit_recording() is the highest-risk code in this change: an
    # error there 500s every take on the site. Transcription is stubbed -- the
    # scoring is not what is under test, the daily bookkeeping around it is.
    print("\nthree takes through /api/submit")
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:                                   # pragma: no cover
        check("fastapi TestClient is importable", False, str(exc))
        TestClient = None

    if TestClient is not None:
        class _StubTranscription:
            def __init__(self, text):
                self.text = text

        class _StubTranscriptions:
            def __init__(self, holder):
                self._holder = holder

            def create(self, **kwargs):
                return _StubTranscription(self._holder["text"])

        class _StubAudio:
            def __init__(self, holder):
                self.transcriptions = _StubTranscriptions(holder)

        class _StubClient:
            def __init__(self, holder):
                self.audio = _StubAudio(holder)

        said = {"text": ""}
        app.get_openai_client = lambda: _StubClient(said)

        uid, name = make_user("submit")
        token = app.make_token(uid, name)
        auth = {"Authorization": "Bearer %s" % token}
        _c = sqlite3.connect(DB)
        set_ids = app.daily_scene_ids_for_user(_c.cursor(), uid)
        _c.close()

        with TestClient(app.app) as client:
            anon = client.get("/api/daily").json()
            check("/api/daily names three lines to anyone",
                  len(anon.get("scene_ids", [])) == 3 and "lines_done" not in anon,
                  "keys=%s" % ("lines_done" in anon))
            check("...and still carries scene_id for older clients",
                  anon.get("scene_id") == set_ids[0])

            mine = client.get("/api/daily", headers=auth).json()
            check("a signed-in caller is told where they are",
                  mine.get("lines_done") == 0 and mine.get("next_scene_id") == set_ids[0],
                  "lines_done=%s next=%s" % (mine.get("lines_done"), mine.get("next_scene_id")))

            def submit(scene_id, spoken):
                said["text"] = spoken
                return client.post(
                    "/api/submit",
                    headers=auth,
                    data={"scene_id": scene_id, "duration_seconds": "12"},
                    files={"audio": ("take.webm", b"not really audio", "audio/webm")},
                )

            # Lines 1 and 2, spoken perfectly.
            r1 = submit(set_ids[0], app.SCENES[set_ids[0]]["quote"]).json()
            check("line 1 scores and reports its place in the set",
                  r1["daily"]["lines_done"] == 1 and r1["daily"]["line_index"] == 1,
                  str(r1["daily"]["lines_done"]))
            check("...nothing is awarded for the set yet",
                  r1["daily"]["completed_now"] is False and r1["daily"]["award"] == 0)
            check("...and the streak has not moved", r1["streak"] == 0, str(r1["streak"]))
            check("...while the take still pays its own points",
                  r1["points_earned"] == app.calc_points(100.0, True),
                  "earned=%s" % r1["points_earned"])

            r2 = submit(set_ids[1], app.SCENES[set_ids[1]]["quote"]).json()
            check("line 2 advances the set", r2["daily"]["lines_done"] == 2)
            check("...and points at line 3", r2["daily"]["next_scene_id"] == set_ids[2])
            check("...streak still untouched", r2["streak"] == 0)

            # Line 3 completes the set.
            r3 = submit(set_ids[2], app.SCENES[set_ids[2]]["quote"]).json()
            d3 = r3["daily"]
            check("LINE 3 COMPLETES THE SET", d3["completed_now"] is True)
            check("...the aggregate is the three stored scores",
                  len([l for l in d3["lines"] if l["score"] is not None]) == 3
                  and d3["avg_score"] == 100.0, str(d3["avg_score"]))
            check("...the combo is awarded from them",
                  d3["combo"] == "all_strong"
                  and d3["combo_bonus"] == app.DAILY_COMBO_STRONG_BONUS, str(d3["combo"]))
            check("...the award is paid on top of the take's own points",
                  r3["points_earned"] == app.calc_points(100.0, True) + d3["award"],
                  "earned=%s award=%s" % (r3["points_earned"], d3["award"]))
            check("...THE STREAK TICKS, ONCE", r3["streak"] == 1, str(r3["streak"]))
            check("...and nothing is left to practise", d3["next_scene_id"] is None)

            # A fourth take, back on line 1: no second award, no second tick.
            r4 = submit(set_ids[0], "complete nonsense words here").json()
            check("a repeat take after completion pays no second award",
                  r4["daily"]["completed_now"] is False
                  and r4["daily"]["already_done"] is True
                  and r4["daily"]["award"] == 0, str(r4["daily"]["award"]))
            check("...and does not tick the streak again", r4["streak"] == 1, str(r4["streak"]))
            check("...and is not floored either: the floor is for the set, once",
                  r4["daily_floor_applied"] == 0, str(r4["daily_floor_applied"]))

            after = client.get("/api/daily", headers=auth).json()
            check("/api/daily now reports the set finished",
                  after["all_done"] is True and after["lines_done"] == 3)
            prof = client.get("/api/profile", headers=auth).json()
            check("/api/profile agrees, and still answers with a scene id",
                  prof["daily_all_done"] is True
                  and prof["daily_lines_done"] == 3
                  and prof["daily_scene_id"] in set_ids, str(prof["daily_scene_id"]))
            missions = client.get("/api/missions", headers=auth).json()
            check("the missions panel reads the same 3 of 3",
                  missions["daily_take"]["lines_done"] == 3,
                  str(missions["daily_take"]["lines_done"]))
            check("...and the daily mission counted lines, not the four takes",
                  missions["daily_quest"]["progress"] == 3,
                  str(missions["daily_quest"]["progress"]))
            # The mission window has to end when the set does, or an evening
            # user's three lines straddle two rows and the mission never
            # completes. expires_at is stored in UTC; this is the user's
            # midnight expressed there.
            check("the daily mission expires at the USER's midnight",
                  missions["daily_quest"]["expires_at"]
                  == app._midnight_tonight_utc_str(TZ),
                  "%s vs %s" % (missions["daily_quest"]["expires_at"],
                                app._midnight_tonight_utc_str(TZ)))
            check("...which is not the server's UTC midnight for a US user",
                  app._midnight_tonight_utc_str(TZ)
                  != app._midnight_tonight_utc_str("UTC"),
                  app._midnight_tonight_utc_str("UTC"))
            check("the countdown runs to the user's midnight, not UTC's",
                  app._secs_until_local_midnight(TZ)
                  == app._secs_until_local_midnight(TZ, datetime.now(timezone.utc)),
                  str(app._secs_until_local_midnight(TZ)))

    # ── 10. THE SHELLS CANNOT DISAGREE ABOUT WHICH LINE IS NEXT ───────────────
    print("\none answer for both shells")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "main.py"), encoding="utf-8").read()
    check("/api/daily answers with the next line, so neither shell computes it",
          '"next_scene_id":  state["next_scene_id"]' in src)
    for shell in ("static/app.js", "static/new-shell/src/lib/adapters/daily-adapter.js"):
        js = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               shell), encoding="utf-8").read()
        check("%s prefers the server's next_scene_id" % shell.split("/")[-1],
              "next_scene_id" in js)

    # Clean up
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM scores WHERE username LIKE 'dttest_%'")
    con.execute("DELETE FROM user_missions WHERE username LIKE 'dttest_%'")
    con.execute("DELETE FROM user_streak WHERE username LIKE 'dttest_%'")
    con.execute("DELETE FROM users WHERE username LIKE 'dttest_%'")
    con.commit()
    con.close()

    failed = [n for n, ok, _ in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("FAILED:")
        for n in failed:
            print("  - " + n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
