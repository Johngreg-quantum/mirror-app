"""Tests the one-time entitlement backfill.

    python scripts/test-backfill.py [sqlitePath]

Runs the real init_db() against a throwaway SQLite file, so it exercises the
actual startup path rather than a reimplementation of it.

The property that matters most is negative: a user who registers AFTER the
backfill must NOT be grandfathered. If the backfill ever ran per-boot, every new
signup would silently receive all of Level 1 and the free tier that decision (c)
exists to create would quietly cease to exist. Nothing else in the system would
report that as broken.
"""
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "backfill-test.db"))

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))


def ents(con, uid):
    return sorted(
        (k, r, st) for k, r, st in con.execute(
            "SELECT kind, ref, source_type FROM entitlements "
            "WHERE user_id = ? AND revoked_at IS NULL", (uid,)
        )
    )


def main():
    if os.path.exists(DB):
        os.remove(DB)
    os.environ["DATABASE_URL"] = ""          # force SQLite
    os.environ.setdefault("JWT_SECRET", "x" * 40)
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import main as app

    # Point main.py at a throwaway file instead of the real mirror.db. Swapping
    # the module global is enough because get_conn() reads it per call -- and it
    # means this test cannot touch, lock, or delete a database anything else is
    # using, which an earlier version of it managed to do.
    app.DB_PATH = DB
    real_db = DB

    try:
        # ── Boot 1: create the schema. No users yet. ─────────────────────────
        app.init_db()
        con = sqlite3.connect(real_db)
        cur = con.cursor()

        # Simulate a database that already had users before this code shipped:
        # clear the marker the empty first run wrote, then seed.
        cur.execute("DELETE FROM app_meta WHERE key = 'entitlements_backfill_at'")
        cur.execute("DELETE FROM entitlements")

        def mkuser(name, pro=0, sub=None):
            cur.execute(
                "INSERT INTO users (username, email, password_hash, is_pro, ls_subscription_id) "
                "VALUES (?, ?, 'x', ?, ?)", (name, name + "@t.test", pro, sub))
            return cur.lastrowid

        beginner = mkuser("bf_beginner")
        advanced = mkuser("bf_advanced")
        prouser = mkuser("bf_pro", pro=1)
        prowithsub = mkuser("bf_prosub", pro=1, sub="SUB-BF")

        # A perfect score on every scene, so compute_user_level walks to the top.
        for lvl in app.LEVELS:
            for sid in lvl["scenes"]:
                cur.execute(
                    "INSERT INTO scores (scene_id, movie, quote, sync_score, user_id) "
                    "VALUES (?, 'm', 'q', 100, ?)", (sid, advanced))
        con.commit()
        top_level = max(int(l["level"]) for l in app.LEVELS)

        # ── Boot 2: the backfill should run now. ─────────────────────────────
        app.init_db()

        con2 = sqlite3.connect(real_db)
        print("backfill")
        check("user with no scores gets level:1 only",
              ents(con2, beginner) == [("level", "1", "manual")],
              str(ents(con2, beginner)))
        check("user at the top level gets every level up to theirs",
              ents(con2, advanced) ==
              sorted([("level", str(n), "manual") for n in range(1, top_level + 1)]),
              str(ents(con2, advanced)))
        check("is_pro without a subscription id is sourced 'manual', not 'subscription'",
              ("pro", "", "manual") in ents(con2, prouser),
              str(ents(con2, prouser)))
        check("is_pro with a subscription id is sourced 'subscription'",
              ("pro", "", "subscription") in ents(con2, prowithsub),
              str(ents(con2, prowithsub)))
        marker = con2.execute(
            "SELECT value FROM app_meta WHERE key = 'entitlements_backfill_at'").fetchone()
        check("marker recorded", marker is not None and bool(marker[0]), str(marker))

        total_before = con2.execute("SELECT COUNT(*) FROM entitlements").fetchone()[0]
        con2.close()

        # ── Boot 3: must be a no-op. ─────────────────────────────────────────
        app.init_db()
        con3 = sqlite3.connect(real_db)
        print("\nidempotence")
        total_after = con3.execute("SELECT COUNT(*) FROM entitlements").fetchone()[0]
        check("re-running adds no rows",
              total_after == total_before, "%s -> %s" % (total_before, total_after))

        # ── The negative property. ───────────────────────────────────────────
        print("\nfree tier survives")
        cur3 = con3.cursor()
        cur3.execute("INSERT INTO users (username, email, password_hash) "
                     "VALUES ('bf_newbie', 'bf_newbie@t.test', 'x')")
        newbie = cur3.lastrowid
        con3.commit()
        con3.close()

        app.init_db()   # a later boot, with a user who registered after the cutover

        con4 = sqlite3.connect(real_db)
        check("A USER WHO REGISTERS AFTER THE BACKFILL IS NOT GRANDFATHERED",
              ents(con4, newbie) == [], str(ents(con4, newbie)))
        cur4 = con4.cursor()
        free = app._free_scene_ids()
        paid = [s for s in app.LEVELS[0]["scenes"] if s not in free]
        check("...so they see the free tier and nothing more",
              app.can_access_scene(cur4, newbie, free[0]) is True
              and app.can_access_scene(cur4, newbie, paid[0]) is False)
        check("...while a grandfathered user still reaches all of Level 1",
              app.can_access_scene(cur4, beginner, paid[0]) is True)
        con4.close()

    finally:
        # Best effort: on Windows the file stays locked briefly after close, and
        # a leftover throwaway file is not worth failing a passing run over.
        try:
            if os.path.exists(real_db):
                os.remove(real_db)
        except OSError as exc:
            print("(note: could not remove %s: %s)" % (real_db, exc))

    failed = [n for n, ok in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(failed), len(results)))
    for n in failed:
        print("  - " + n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
