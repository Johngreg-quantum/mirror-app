"""Delete rows belonging to scene ids that no longer exist.

    python scripts/purge-removed-scenes.py             # show the SQL, change nothing
    python scripts/purge-removed-scenes.py --apply     # run it

The scene catalogue was replaced wholesale: every film scene id is gone and 47
new ones took their place. Rows keyed on the old ids are now unreachable -- no
screen can render them, because no scene resolves -- but they are still there,
and `word_vocab` in particular will keep serving a cached answer for a scene id
that stopped existing.

Reads the same database the app does, through main.py, so it follows
DATABASE_URL to Postgres in production and falls back to SQLite locally without
a second connection story.

Prints the exact statements and the row counts first, every time. --apply is a
separate flag rather than a prompt because this also has to run on Render,
where nobody is at a terminal to answer one.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("JWT_SECRET", "purge-script-placeholder-0123456789abcdef")

import main as app   # noqa: E402

# table -> the column holding a scene id. Ordered so that anything a foreign
# key might hang off goes last; today nothing does, and the order is kept
# meaningful so adding a table later does not have to rediscover that.
#
# `scores` is the one with consequences. A row there is a take somebody
# recorded, and deleting it moves their level, their best scores, their
# division and their streak. It is in the list because a score against a scene
# that no longer exists is worse than no score: compute_user_level() still
# counts it toward unlocking, and /api/history renders a row whose scene cannot
# be opened. The catalogue was replaced wholesale, so every one of those rows
# refers to a film scene that is gone.
SCENE_TABLES = [
    ("word_vocab", "scene_id"),
    ("scores", "scene_id"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually run the deletes (default: show them only)")
    args = ap.parse_args()

    live = sorted(app.SCENES)
    print("database: %s" % ("PostgreSQL" if app.USE_PG else "SQLite"))
    print("live scene ids: %d" % len(live))

    conn = app.get_conn()
    cur = conn.cursor()
    ph = app.PH
    placeholders = ", ".join([ph] * len(live))

    total = 0
    plans = []
    for table, column in SCENE_TABLES:
        try:
            cur.execute(
                "SELECT COUNT(*) FROM %s WHERE %s NOT IN (%s)"
                % (table, column, placeholders), live)
            stale = cur.fetchone()[0] or 0
        except Exception as exc:                      # noqa: BLE001
            # A table that does not exist yet on this deployment is not an
            # error: init_db creates them lazily and a fresh database has none.
            print("  %-14s skipped (%s)" % (table, str(exc).splitlines()[0][:60]))
            continue

        cur.execute("SELECT COUNT(*) FROM %s" % table)
        every = cur.fetchone()[0] or 0
        print("  %-14s %d stale of %d rows" % (table, stale, every))
        total += stale
        if stale:
            plans.append((table, column))

    print("\nSQL:")
    for table, column in plans:
        print("  DELETE FROM %s WHERE %s NOT IN (<the %d live scene ids>);"
              % (table, column, len(live)))
    if not plans:
        print("  (nothing to delete)")

    # Deleting from `scores` changes what users see, so say so in numbers
    # rather than leaving it to the table name.
    if any(t == "scores" for t, _ in plans):
        try:
            cur.execute(
                "SELECT COUNT(DISTINCT user_id) FROM scores "
                "WHERE user_id IS NOT NULL AND scene_id NOT IN (%s)" % placeholders,
                live)
            users = cur.fetchone()[0] or 0
            print("\n  that removes takes belonging to %d user(s); their level, "
                  "best scores,\n  points and streak are recomputed from what is left."
                  % users)
        except Exception:                             # noqa: BLE001
            pass

    if not args.apply:
        print("\nDry run. Nothing was changed. Pass --apply to run it.")
        conn.close()
        return 0

    deleted = 0
    for table, column in plans:
        cur.execute(
            "DELETE FROM %s WHERE %s NOT IN (%s)"
            % (table, column, placeholders), live)
        deleted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    # An avatar pointing at a deleted scene renders a broken image on the
    # profile. NULL is already the "no avatar chosen" state, so this puts those
    # users back where a new account starts rather than inventing a default.
    cleared = 0
    try:
        cur.execute(
            "UPDATE users SET avatar_scene_id = NULL "
            "WHERE avatar_scene_id IS NOT NULL AND avatar_scene_id NOT IN (%s)"
            % placeholders, live)
        cleared = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    except Exception as exc:                          # noqa: BLE001
        print("  avatar_scene_id not cleared (%s)" % str(exc).splitlines()[0][:60])

    # points, streak and last_daily are stored totals, not derived, so deleting
    # the takes behind them leaves a total earned against scenes that no longer
    # exist. Only users with NO takes left are reset: a user who still has some
    # history keeps it, and recomputing a partial total from scores would
    # silently rewrite points that missions and the daily also contributed to.
    reset = 0
    try:
        cur.execute(
            "UPDATE users SET points = 0, streak = 0, longest_streak = 0, "
            "last_daily = NULL "
            "WHERE (points > 0 OR streak > 0) "
            "AND id NOT IN (SELECT user_id FROM scores WHERE user_id IS NOT NULL)")
        reset = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    except Exception as exc:                          # noqa: BLE001
        print("  totals not reset (%s)" % str(exc).splitlines()[0][:60])

    conn.commit()
    conn.close()
    print("\ndeleted %d row(s)" % deleted)
    print("cleared %d dead avatar(s)" % cleared)
    print("reset totals for %d user(s) left with no takes" % reset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
