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
SCENE_TABLES = [
    ("word_vocab", "scene_id"),
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
    conn.commit()
    conn.close()
    print("\ndeleted %d row(s)" % deleted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
