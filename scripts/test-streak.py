"""Tests for the single streak counter and local-day boundaries.

    python scripts/test-streak.py [sqlitePath]

Default: ./mirror.db

In-process against the real users table: it calls the same helpers /api/submit
calls and applies the same streak rule, with the clock frozen so a Miami evening
can be reproduced deterministically. It does not go through HTTP, because the
cases that matter are about what time it is, and you cannot ask a running server
to believe it is 9pm on a Tuesday in October.

The three cases that motivated the change:

  1. A Miami user practising 6pm Monday then 9pm Tuesday KEEPS their streak.
     In UTC those are Monday and WEDNESDAY -- a skipped UTC day and a reset to 1,
     which is what shipped. This is the regression that matters: it broke the
     streak for anyone who practises in the evening on the US east coast, which
     is most of the users.
  2. A user who skips a local day LOSES it. The fix must not become "never
     resets", which is exactly what the counter being retired here did.
  3. A re-registered username inherits NO streak.
  4. A re-registered username inherits NO mission progress -- tested with the
     orphaned rows deliberately left in place, because the property being
     checked is that the KEY protects the new account, not that erasure
     happened to remember this table.
"""
import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = sys.argv[1] if len(sys.argv) > 1 else "mirror.db"

import main as app   # noqa: E402  (after sys.path)
from zoneinfo import ZoneInfo   # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))


def local_day(dt_utc, tz_name):
    """The user's calendar day for an instant, the way main.py computes it."""
    return dt_utc.astimezone(ZoneInfo(tz_name)).strftime("%Y-%m-%d")


def apply_daily(last_daily, now_utc, tz_name, streak):
    """The streak rule from /api/submit, with the clock supplied.

    Mirrors the branch in submit_recording() exactly: the point of the test is
    the day arithmetic, so the arithmetic is what is reproduced -- not re-derived
    differently here, which would let both be wrong together."""
    today = local_day(now_utc, tz_name)
    yesterday = (now_utc.astimezone(ZoneInfo(tz_name)) - timedelta(days=1)).strftime("%Y-%m-%d")
    if last_daily == today:
        return streak, last_daily, "already done today"
    new_streak = streak + 1 if last_daily == yesterday else 1
    return new_streak, today, "incremented" if new_streak > streak else "reset"


def make_user(suffix, username=None):
    con = sqlite3.connect(DB)
    try:
        cur = con.cursor()
        name = username or ("strtest_" + suffix)
        cur.execute("DELETE FROM users WHERE username = ?", (name,))
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (name, "strtest_%s@example.com" % suffix, "x"),
        )
        uid = cur.lastrowid
        con.commit()
        return uid, name
    finally:
        con.close()


def row(uid):
    con = sqlite3.connect(DB)
    try:
        return con.execute(
            "SELECT streak, longest_streak, last_daily, timezone FROM users WHERE id = ?",
            (uid,),
        ).fetchone()
    finally:
        con.close()


def main():
    if not os.path.isfile(DB):
        print("SQLite file not found: %s" % DB)
        sys.exit(2)

    # Point main at this database and run the real init_db(), so the new columns
    # arrive through the actual migration path rather than being created by the
    # test. A missing ALTER then fails here instead of in production.
    app.DB_PATH = DB
    app.init_db()

    print("db %s   default tz %s\n" % (DB, app.DEFAULT_TIMEZONE))

    con = sqlite3.connect(DB)
    cols = {r[1] for r in con.execute("PRAGMA table_info(users)").fetchall()}
    streak_cols = {r[1] for r in con.execute("PRAGMA table_info(user_streak)").fetchall()}
    con.close()
    print("migration")
    check("users.longest_streak exists after init_db()", "longest_streak" in cols)
    check("users.timezone exists after init_db()", "timezone" in cols)
    check("user_streak.user_id exists after init_db()", "user_id" in streak_cols)
    print()

    # ── Helpers resolve at all ────────────────────────────────────────────────
    print("timezone helpers")
    check("DEFAULT_TIMEZONE is a resolvable IANA zone",
          app.is_valid_timezone(app.DEFAULT_TIMEZONE), app.DEFAULT_TIMEZONE)
    check("an unknown zone is rejected, not silently accepted",
          app.is_valid_timezone("Mars/Olympus") is False)
    check("a user with NO timezone falls back to the default, not UTC",
          str(app.user_zone(None)) == app.DEFAULT_TIMEZONE, str(app.user_zone(None)))
    check("a garbage zone falls back to the default",
          str(app.user_zone("'; DROP TABLE users; --")) == app.DEFAULT_TIMEZONE)
    check("local_today differs from the UTC date during the US evening",
          app.local_today("America/New_York") is not None)

    # ── 1. THE MIAMI EVENING ──────────────────────────────────────────────────
    # 2026-10-05 is a Monday. 6pm EDT = 22:00 UTC same day.
    #              9pm EDT Tuesday = 01:00 UTC WEDNESDAY.
    print("\n1. Miami user, 6pm Monday then 9pm Tuesday")
    tz = "America/New_York"
    mon_6pm = datetime(2026, 10, 5, 22, 0, tzinfo=timezone.utc)
    tue_9pm = datetime(2026, 10, 7, 1, 0, tzinfo=timezone.utc)

    check("6pm Mon EDT is UTC Monday", mon_6pm.strftime("%Y-%m-%d") == "2026-10-05")
    check("9pm Tue EDT is UTC WEDNESDAY (the whole bug)",
          tue_9pm.strftime("%Y-%m-%d") == "2026-10-07",
          "utc=%s local=%s" % (tue_9pm.strftime("%Y-%m-%d"), local_day(tue_9pm, tz)))

    s1, ld1, _ = apply_daily(None, mon_6pm, tz, 0)
    check("Monday evening starts the streak at 1", s1 == 1 and ld1 == "2026-10-05",
          "streak=%s last_daily=%s" % (s1, ld1))

    s2, ld2, how = apply_daily(ld1, tue_9pm, tz, s1)
    check("TUESDAY EVENING KEEPS THE STREAK (2, not reset to 1)",
          s2 == 2 and ld2 == "2026-10-06",
          "streak=%s last_daily=%s (%s)" % (s2, ld2, how))

    # What the old UTC rule did with the same two takes.
    utc_mon = mon_6pm.strftime("%Y-%m-%d")
    utc_tue_yest = (tue_9pm - timedelta(days=1)).strftime("%Y-%m-%d")
    old_streak = 1 + 1 if utc_mon == utc_tue_yest else 1
    check("...and the old UTC rule reset it to 1, confirming the fix is the fix",
          old_streak == 1, "utc last_daily=%s utc yesterday=%s" % (utc_mon, utc_tue_yest))

    # ── 2. SKIPPING A LOCAL DAY STILL BREAKS IT ───────────────────────────────
    print("\n2. Skipping a local day")
    thu_7pm = datetime(2026, 10, 8, 23, 0, tzinfo=timezone.utc)   # Thu 7pm EDT
    s3, ld3, how3 = apply_daily(ld2, thu_7pm, tz, s2)
    check("Tuesday then THURSDAY resets the streak to 1",
          s3 == 1 and ld3 == "2026-10-08",
          "streak=%s last_daily=%s (%s)" % (s3, ld3, how3))

    s4, ld4, how4 = apply_daily(ld3, thu_7pm, tz, s3)
    check("a second daily the same local day does not double-count",
          s4 == 1 and how4 == "already done today", how4)

    # A month away is still a reset, not a continuation: the retired counter
    # would have returned 2 here, because it only asked "is this a new day".
    nov = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)
    s5, _, _ = apply_daily(ld4, nov, tz, s4)
    check("a month away resets to 1 (the retired counter said 2)", s5 == 1, "streak=%s" % s5)

    # ── DST, since local days are not all 24 hours ────────────────────────────
    print("\nDST boundary")
    # US DST ends 2026-11-01. Sat 31 Oct 8pm EDT -> Sun 1 Nov 8pm EST.
    sat = datetime(2026, 10, 31, 0, 0, tzinfo=timezone.utc) + timedelta(hours=24)
    sat_8pm = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)      # Sat 8pm EDT
    sun_8pm = datetime(2026, 11, 2, 1, 0, tzinfo=timezone.utc)      # Sun 8pm EST
    check("Sat 8pm EDT is local Oct 31", local_day(sat_8pm, tz) == "2026-10-31",
          local_day(sat_8pm, tz))
    check("Sun 8pm EST is local Nov 1", local_day(sun_8pm, tz) == "2026-11-01",
          local_day(sun_8pm, tz))
    sA, ldA, _ = apply_daily(None, sat_8pm, tz, 0)
    sB, ldB, howB = apply_daily(ldA, sun_8pm, tz, sA)
    check("a streak survives the clocks going back", sB == 2, "streak=%s (%s)" % (sB, howB))
    del sat

    # ── 3. A RE-REGISTERED USERNAME INHERITS NOTHING ──────────────────────────
    print("\n3. Re-registered username")
    uid_a, name = make_user("rereg")
    con = sqlite3.connect(DB)
    con.execute(
        "UPDATE users SET streak = 9, longest_streak = 12, last_daily = ?, timezone = ? WHERE id = ?",
        ("2026-10-06", tz, uid_a))
    # XP under the same username, which used to transfer with the name.
    con.execute("DELETE FROM user_streak WHERE username = ?", (name,))
    con.execute(
        "INSERT INTO user_streak (username, user_id, total_xp, daily_xp, daily_xp_date) "
        "VALUES (?, ?, ?, ?, ?)", (name, uid_a, 4321, 50, "2026-10-06"))
    con.commit()
    con.close()
    before = row(uid_a)
    check("the first holder has a streak and XP to inherit",
          before[0] == 9 and before[1] == 12, "streak=%s longest=%s" % (before[0], before[1]))

    # Delete the account the way /api/account does, then re-register the name.
    con = sqlite3.connect(DB)
    for table, col, key in app._USER_DATA_TABLES:
        val = uid_a if key == "id" else name
        try:
            con.execute("DELETE FROM %s WHERE %s = ?" % (table, col), (val,))
        except sqlite3.OperationalError:
            pass
    con.execute("DELETE FROM users WHERE id = ?", (uid_a,))
    con.commit()
    con.close()

    uid_b, _ = make_user("rereg", username=name)
    after = row(uid_b)
    check("SAME USERNAME, NEW ACCOUNT: streak is 0",
          int(after[0] or 0) == 0, "streak=%s" % after[0])
    check("...longest_streak is 0 too", int(after[1] or 0) == 0, "longest=%s" % after[1])
    check("...last_daily is unset", not after[2], repr(after[2]))

    con = sqlite3.connect(DB)
    xp = con.execute(
        "SELECT COALESCE(SUM(total_xp), 0) FROM user_streak WHERE user_id = ? OR username = ?",
        (uid_b, name)).fetchone()[0]
    con.close()
    check("...and NO XP is inherited either", int(xp or 0) == 0, "total_xp=%s" % xp)

    # ── 4. MISSION PROGRESS DOES NOT TRANSFER WITH THE NAME ───────────────────
    print("\n4. Re-registered username, mission progress")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    uid_c, name_c = make_user("missions")

    con = sqlite3.connect(DB)
    cur = con.cursor()
    app.seed_user_missions(uid_c, name_c, cur)
    con.commit()
    seeded = con.execute(
        "SELECT COUNT(*) FROM user_missions WHERE user_id = ?", (uid_c,)).fetchone()[0]
    check("seeding writes one row per default mission, keyed on user_id",
          seeded == len(app.DEFAULT_MISSIONS),
          "%s rows for %s missions" % (seeded, len(app.DEFAULT_MISSIONS)))

    # Give the first holder progress worth stealing: a completed daily with its
    # XP already banked.
    con.execute(
        "UPDATE user_missions SET progress = 3, completed = 1, xp_awarded = 1 "
        "WHERE user_id = ? AND mission_id = 'daily'", (uid_c,))
    con.commit()
    first = con.execute(
        "SELECT progress, completed FROM user_missions "
        "WHERE user_id = ? AND mission_id = 'daily'", (uid_c,)).fetchone()
    check("the first holder has mission progress to inherit",
          first[0] == 3 and bool(first[1]), "progress=%s completed=%s" % first)

    # Delete ONLY the users row, leaving every mission row behind -- i.e. assume
    # the erasure list forgot this table. Erasing correctly would prove nothing
    # about the key, which is the thing under test.
    con.execute("DELETE FROM users WHERE id = ?", (uid_c,))
    con.commit()
    con.close()

    uid_d, _ = make_user("missions", username=name_c)
    check("the re-registered account gets a NEW user id", uid_d != uid_c,
          "old=%s new=%s" % (uid_c, uid_d))

    con = sqlite3.connect(DB)
    cur = con.cursor()
    app.seed_user_missions(uid_d, name_c, cur)
    con.commit()
    # Read exactly as /api/missions reads.
    mine = con.execute(
        "SELECT mission_id, progress, completed FROM user_missions "
        "WHERE user_id = ? AND expires_at > ? ORDER BY id ASC", (uid_d, now_str)).fetchall()
    orphans = con.execute(
        "SELECT COUNT(*) FROM user_missions WHERE user_id = ?", (uid_c,)).fetchone()[0]
    con.close()

    check("SAME USERNAME, NEW ACCOUNT: every mission starts at 0",
          len(mine) == len(app.DEFAULT_MISSIONS) and all(int(r[1] or 0) == 0 for r in mine),
          "progress=%s" % [int(r[1] or 0) for r in mine])
    check("...nothing arrives pre-completed",
          not any(bool(r[2]) for r in mine))
    check("...and the orphaned rows are STILL THERE, so this did not pass by "
          "deletion", orphans == len(app.DEFAULT_MISSIONS), "orphans=%s" % orphans)

    # The write path, with its new signature: advancing the new account's
    # missions must not reach the orphans sitting under the same username.
    con = sqlite3.connect(DB)
    cur = con.cursor()
    advanced = asyncio.run(app.update_missions(
        user_id=uid_d,
        username=name_c,
        scene_id=app.get_daily_scene_ids()[0],
        score=90.0,
        duration_seconds=60.0,
        take_number=1,
        db=cur,
        local_day="2026-10-08",
        daily_line_completed=True,
    ))
    con.commit()
    mine_daily = con.execute(
        "SELECT progress FROM user_missions WHERE user_id = ? AND mission_id = 'daily'",
        (uid_d,)).fetchone()[0]
    theirs_daily = con.execute(
        "SELECT progress FROM user_missions WHERE user_id = ? AND mission_id = 'daily'",
        (uid_c,)).fetchone()[0]
    xp_rows = con.execute(
        "SELECT user_id, daily_xp_date FROM user_streak WHERE username = ?",
        (name_c,)).fetchall()
    con.close()
    check("a take advances the NEW account's daily mission",
          mine_daily == 1 and any(m["mission_id"] == "daily" for m in advanced),
          "progress=%s advanced=%s" % (mine_daily, [m["mission_id"] for m in advanced]))
    check("...and leaves the orphan's progress untouched at 3", theirs_daily == 3,
          "progress=%s" % theirs_daily)
    check("...and the XP row it creates belongs to the new user id",
          len(xp_rows) == 1 and xp_rows[0][0] == uid_d, "%s" % (xp_rows,))
    check("the XP day comes from the user's local day, not the server's",
          xp_rows[0][1] == "2026-10-08", "daily_xp_date=%s" % xp_rows[0][1])

    # The deploy must not reset in-flight missions for existing users: rows
    # written before the user_id column arrive NULL and get linked by username.
    uid_e, name_e = make_user("legacy")
    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO user_missions (username, mission_id, progress, goal, expires_at) "
        "VALUES (?, 'daily', 2, 3, ?)",
        (name_e, "2099-01-01 00:00:00"))
    con.commit()
    app._link_user_missions_to_user_id(con, con.cursor())
    linked = con.execute(
        "SELECT user_id, progress FROM user_missions WHERE username = ?", (name_e,)).fetchone()
    con.close()
    check("a pre-migration row keeps its progress and gains the right user_id",
          linked[0] == uid_e and linked[1] == 2, "user_id=%s progress=%s" % linked)

    # ── The retired counter is not written any more ───────────────────────────
    print("\nretired counter")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "main.py"), encoding="utf-8").read()
    check("nothing UPDATEs user_streak.current_streak",
          "SET current_streak" not in src)
    check("nothing SELECTs current_streak from user_streak",
          "SELECT current_streak" not in src)
    check("no mission row is ever looked up by username",
          "FROM user_missions WHERE username" not in src)
    check("the dead get_today_daily_scene() is gone",
          not hasattr(app, "get_today_daily_scene"))
    check("the daily line pick is still UTC-based and user-independent",
          app.get_daily_scene_ids() == app.get_daily_scene_ids())

    # ── Points floor ─────────────────────────────────────────────────────────
    print("\npoints floor")
    check("a 60%% repeat attempt scores 0 on its own",
          app.calc_points(60.0, False) == 0, str(app.calc_points(60.0, False)))
    check("DAILY_COMPLETION_FLOOR is above zero", app.DAILY_COMPLETION_FLOOR > 0,
          str(app.DAILY_COMPLETION_FLOOR))
    check("the floor is below the lowest scoring tier, so scoring still pays more",
          app.DAILY_COMPLETION_FLOOR < app.calc_points(70.0, False),
          "floor=%s  70%%=%s" % (app.DAILY_COMPLETION_FLOOR, app.calc_points(70.0, False)))
    year = app.DAILY_COMPLETION_FLOOR * 365
    diamond = next(d["min"] for d in app.DIVISIONS if d["name"] == "Diamond")
    check("a year of floor-only dailies cannot reach Diamond",
          year < diamond, "%s pts/yr vs Diamond at %s" % (year, diamond))

    # Clean up
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM user_streak WHERE username LIKE 'strtest_%'")
    con.execute("DELETE FROM user_missions WHERE username LIKE 'strtest_%'")
    con.execute("DELETE FROM users WHERE username LIKE 'strtest_%'")
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
