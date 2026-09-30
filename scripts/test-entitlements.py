"""End-to-end tests for entitlement grants and revocations.

    python scripts/test-entitlements.py [baseUrl] [sqlitePath]

Defaults: http://127.0.0.1:8077  and  ./mirror.db

Posts real HMAC-signed payloads at /api/billing/webhook and asserts on the
resulting rows, so this exercises signature verification, the handler, and the
SQL together rather than testing helpers in isolation.

Requires LEMONSQUEEZY_SIGNING_SECRET to match the server's, and the target
server to be running against the given SQLite file. Refuses to run against
PostgreSQL -- it writes and deletes test rows.

The refund cases were written before the refund implementation, per
docs/entitlements-spec.md §7, because revoking by user_id instead of by source
is the plausible wrong implementation and nothing else would catch it.
"""
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077").rstrip("/")
DB = sys.argv[2] if len(sys.argv) > 2 else "mirror.db"

LEVEL1_VARIANT = os.getenv("LEMONSQUEEZY_LEVEL1_VARIANT_ID", "2109931")
UNKNOWN_VARIANT = "999999999"

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))


def secret():
    s = os.getenv("LEMONSQUEEZY_SIGNING_SECRET", "").strip()
    if s:
        return s
    env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.isfile(env):
        with open(env, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("LEMONSQUEEZY_SIGNING_SECRET="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


SECRET = secret()


def post_event(event, payload, sign=True):
    body = json.dumps(payload).encode("utf-8")
    sig = hmac.new(SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    if not sign:
        sig = "0" * 64
    req = urllib.request.Request(
        BASE + "/api/billing/webhook",
        data=body,
        headers={"Content-Type": "application/json",
                 "X-Event-Name": event,
                 "X-Signature": sig},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _send(req):
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def post_json(path, payload, token=""):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return _send(req)


def get_json(path, token=""):
    req = urllib.request.Request(BASE + path)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return _send(req)


def post_multipart(path, token, scene_id, challenge_id=""):
    """A minimal /api/submit body. The audio is one junk byte on purpose: the
    gate is checked before the upload is read, so a 403 needs no real recording
    and no Whisper call. A scene that passes the gate fails further down, which
    is what distinguishes "refused" from "allowed" here."""
    b = "----mirrortest"
    parts = [("scene_id", scene_id), ("duration_seconds", "1")]
    if challenge_id:
        parts.append(("challenge_id", challenge_id))
    body = b""
    for k, v in parts:
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                 % (b, k, v)).encode()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"audio\"; "
             "filename=\"r.webm\"\r\nContent-Type: audio/webm\r\n\r\n" % b).encode()
    body += b"\x00\r\n"
    body += ("--%s--\r\n" % b).encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + b,
                 "Authorization": "Bearer " + token})
    return _send(req)


def rows(user_id):
    con = sqlite3.connect(DB)
    try:
        return con.execute(
            "SELECT kind, ref, source_type, source_id, expires_at, revoked_at, revoke_reason "
            "FROM entitlements WHERE user_id = ? ORDER BY kind, ref", (user_id,)
        ).fetchall()
    finally:
        con.close()


def make_user(suffix):
    """Insert a bare user row directly; registration would also seed missions."""
    con = sqlite3.connect(DB)
    try:
        cur = con.cursor()
        cur.execute("DELETE FROM users WHERE username = ?", ("enttest_" + suffix,))
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("enttest_" + suffix, "enttest_%s@example.com" % suffix, "x"),
        )
        uid = cur.lastrowid
        cur.execute("DELETE FROM entitlements WHERE user_id = ?", (uid,))
        con.commit()
        return uid
    finally:
        con.close()


def order_payload(user_id, order_id, variant_id):
    return {
        "meta": {"custom_data": {"user_id": str(user_id)}},
        "data": {"id": str(order_id), "type": "orders",
                 "attributes": {"first_order_item": {"variant_id": int(variant_id)}}},
    }


def sub_payload(user_id, sub_id, status="active", ends_at=None):
    attrs = {"status": status, "urls": {"customer_portal": "https://example.test/portal"}}
    if ends_at:
        attrs["ends_at"] = ends_at
    return {"meta": {"custom_data": {"user_id": str(user_id)}},
            "data": {"id": str(sub_id), "type": "subscriptions", "attributes": attrs}}


def main():
    if not SECRET:
        print("LEMONSQUEEZY_SIGNING_SECRET is not set (env or .env). Cannot sign.")
        sys.exit(2)
    if not os.path.isfile(DB):
        print("SQLite file not found: %s" % DB)
        sys.exit(2)

    print("target %s   db %s   level1 variant %s\n" % (BASE, DB, LEVEL1_VARIANT))

    # ── Signature ────────────────────────────────────────────────────────────
    print("signature")
    code, _ = post_event("order_created", order_payload(1, 1, LEVEL1_VARIANT), sign=False)
    check("unsigned request is rejected", code == 401, "HTTP %s" % code)

    # ── REFUND CASES (written first) ─────────────────────────────────────────
    print("\nrefund")
    uid = make_user("refund")
    post_event("order_created", order_payload(uid, "ORD-1", LEVEL1_VARIANT))
    post_event("subscription_created", sub_payload(uid, "SUB-1"))
    before = rows(uid)
    check("level + pro both granted before refund",
          sorted((r[0], r[1]) for r in before) == [("level", "1"), ("pro", "")],
          str(sorted((r[0], r[1]) for r in before)))

    code, _ = post_event("order_refunded", {"meta": {"custom_data": {"user_id": str(uid)}},
                                            "data": {"id": "ORD-1"}})
    after = {(r[0], r[1]): r for r in rows(uid)}
    lvl = after.get(("level", "1"))
    pro = after.get(("pro", ""))
    check("refund returns 200", code == 200, "HTTP %s" % code)
    check("refunded level is revoked with reason",
          bool(lvl) and lvl[5] is not None and lvl[6] == "refund",
          "revoked_at=%r reason=%r" % (lvl[5] if lvl else None, lvl[6] if lvl else None))
    check("PRO ROW UNTOUCHED by a level refund",
          bool(pro) and pro[5] is None,
          "pro.revoked_at=%r" % (pro[5] if pro else "missing"))
    check("refunded row is kept, not deleted", lvl is not None)

    code2, _ = post_event("order_refunded", {"meta": {"custom_data": {"user_id": str(uid)}},
                                             "data": {"id": "ORD-1"}})
    check("duplicate refund is idempotent and 200", code2 == 200, "HTTP %s" % code2)

    uid2 = make_user("refund-unknown")
    code3, _ = post_event("order_refunded", {"meta": {"custom_data": {"user_id": str(uid2)}},
                                             "data": {"id": "ORD-NOPE"}})
    check("refund for an ungranted order is a 200 no-op (not 5xx)",
          code3 == 200 and rows(uid2) == [], "HTTP %s rows=%s" % (code3, len(rows(uid2))))

    # ── ORDER GRANTS ─────────────────────────────────────────────────────────
    print("\norder_created")
    uid3 = make_user("order")
    code, _ = post_event("order_created", order_payload(uid3, "ORD-2", LEVEL1_VARIANT))
    r = rows(uid3)
    check("grants level:1 from the configured variant",
          code == 200 and len(r) == 1 and r[0][0] == "level" and r[0][1] == "1",
          "HTTP %s rows=%s" % (code, r))
    check("source recorded as the order",
          bool(r) and r[0][2] == "order" and r[0][3] == "ORD-2",
          "source=%r/%r" % (r[0][2], r[0][3]) if r else "")
    check("one-time grant does not expire", bool(r) and r[0][4] is None)

    post_event("order_created", order_payload(uid3, "ORD-2", LEVEL1_VARIANT))
    check("duplicate order_created stays one row", len(rows(uid3)) == 1,
          "rows=%s" % len(rows(uid3)))

    uid4 = make_user("order-unknown")
    code, _ = post_event("order_created", order_payload(uid4, "ORD-3", UNKNOWN_VARIANT))
    check("unconfigured variant grants nothing, returns 200",
          code == 200 and rows(uid4) == [], "HTTP %s rows=%s" % (code, len(rows(uid4))))

    # ── SUBSCRIPTION LIFECYCLE ───────────────────────────────────────────────
    print("\nsubscription")
    uid5 = make_user("sub")
    post_event("subscription_created", sub_payload(uid5, "SUB-5"))
    r = {(x[0], x[1]): x for x in rows(uid5)}.get(("pro", ""))
    check("grants pro with the subscription as source",
          bool(r) and r[2] == "subscription" and r[3] == "SUB-5",
          "row=%r" % (r,))

    post_event("subscription_cancelled",
               sub_payload(uid5, "SUB-5", status="cancelled", ends_at="2099-01-01T00:00:00Z"))
    r = {(x[0], x[1]): x for x in rows(uid5)}.get(("pro", ""))
    check("cancellation UPDATES rather than deletes", r is not None)
    check("cancellation sets the paid-through date, not immediate revocation",
          bool(r) and r[4] == "2099-01-01T00:00:00Z" and r[5] is None,
          "expires_at=%r revoked_at=%r" % (r[4] if r else None, r[5] if r else None))

    # ── ACCESS CHECK, in process against the same rows ───────────────────────
    print("\naccess check")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import main as app
        con = sqlite3.connect(DB)
        cur = con.cursor()
        free = app._free_scene_ids()
        lvl1 = app.LEVELS[0]["scenes"]
        paid_lvl1 = [s for s in lvl1 if s not in free]
        lvl2 = app.LEVELS[1]["scenes"] if len(app.LEVELS) > 1 else []

        check("free tier is the first %d scenes" % app.FREE_SCENE_COUNT,
              len(free) == app.FREE_SCENE_COUNT, str(free))

        anon = make_user("anon")
        check("a user with nothing can reach a free scene",
              app.can_access_scene(cur, anon, free[0]) is True)
        check("a user with nothing cannot reach a paid Level 1 scene",
              app.can_access_scene(cur, anon, paid_lvl1[0]) is False)

        holder = make_user("holder")
        post_event("order_created", order_payload(holder, "ORD-9", LEVEL1_VARIANT))
        check("level:1 reaches a paid Level 1 scene",
              app.can_access_scene(cur, holder, paid_lvl1[0]) is True)
        if lvl2:
            check("level:1 does NOT reach Level 2",
                  app.can_access_scene(cur, holder, lvl2[0]) is False)

        post_event("order_refunded", {"meta": {"custom_data": {"user_id": str(holder)}},
                                      "data": {"id": "ORD-9"}})
        check("revocation is visible to the check immediately",
              app.can_access_scene(cur, holder, paid_lvl1[0]) is False)

        both = make_user("both")
        post_event("order_created", order_payload(both, "ORD-10", LEVEL1_VARIANT))
        post_event("subscription_created", sub_payload(both, "SUB-10"))
        post_event("order_refunded", {"meta": {"custom_data": {"user_id": str(both)}},
                                      "data": {"id": "ORD-10"}})
        check("pro still satisfies everything after a level refund",
              app.can_access_scene(cur, both, paid_lvl1[0]) is True
              and (not lvl2 or app.can_access_scene(cur, both, lvl2[0]) is True))

        # ── The two deliberate holes ─────────────────────────────────────────
        print("\nacquisition holes")
        daily = app.get_daily_scene_id()
        hole_user = make_user("holes")
        check("the daily scene is playable without owning it",
              app.can_play_scene(cur, hole_user, daily) is True, "daily=%s" % daily)
        check("the daily scene is still NOT owned (nothing was granted)",
              app.can_access_scene(cur, hole_user, daily) is (daily in free),
              "daily=%s free=%s" % (daily, daily in free))

        # A paid scene that is not today's daily, so the hole cannot be the
        # reason it passes or fails.
        target = next((s for s in paid_lvl1 if s != daily), None)
        if target:
            check("a paid scene is refused with no challenge",
                  app.can_play_scene(cur, hole_user, target) is False, target)

            con.execute(
                "INSERT OR REPLACE INTO challenges "
                "(challenge_id, challenger_username, challenger_user_id, scene_id, score_to_beat) "
                "VALUES (?, ?, ?, ?, ?)",
                ("a" * 16, "enttest_holes", hole_user, target, 50.0))
            con.commit()
            check("a challenge opens the scene it names",
                  app.can_play_scene(cur, hole_user, target, "a" * 16) is True, target)

            # The hole must be scoped to the challenge's own scene. Trusting the
            # request's scene_id alongside any valid challenge id would open all
            # 47 scenes to anyone who has ever been sent a single link.
            other = next((s for s in (lvl2 or paid_lvl1)
                          if s not in (target, daily) and s not in free), None)
            if other:
                check("A CHALLENGE DOES NOT OPEN ANY OTHER SCENE",
                      app.can_play_scene(cur, hole_user, other, "a" * 16) is False, other)
            check("a malformed challenge id opens nothing",
                  app.can_play_scene(cur, hole_user, target, "../../etc") is False)
            check("an unknown challenge id opens nothing",
                  app.can_play_scene(cur, hole_user, target, "f" * 16) is False)

        # ── accessible_scene_ids, which /api/progress intersects against ─────
        print("\naccessible_scene_ids")
        check("with nothing held it is exactly the free set",
              app.accessible_scene_ids(cur, hole_user) == free,
              str(app.accessible_scene_ids(cur, hole_user)))

        owner = make_user("owner")
        post_event("order_created", order_payload(owner, "ORD-11", LEVEL1_VARIANT))
        check("level:1 makes all of Level 1 accessible",
              app.accessible_scene_ids(cur, owner) == list(lvl1),
              "%d ids" % len(app.accessible_scene_ids(cur, owner)))

        prouser = make_user("proaccess")
        post_event("subscription_created", sub_payload(prouser, "SUB-11"))
        all_ids = [s for l in app.LEVELS for s in l["scenes"]]
        check("pro makes every scene accessible",
              app.accessible_scene_ids(cur, prouser) == all_ids,
              "%d of %d" % (len(app.accessible_scene_ids(cur, prouser)), len(all_ids)))
        con.close()
    except Exception as exc:
        check("in-process access check ran", False, repr(exc))

    # ── ENFORCEMENT OVER HTTP ────────────────────────────────────────────────
    # The point of Deploy 2. A table nothing reads is worse than useless, so
    # these go through the real endpoints with a real token rather than calling
    # the helper again.
    print("\nenforcement (HTTP)")
    try:
        import main as app2
        con = sqlite3.connect(DB)
        free2 = app2._free_scene_ids()
        lvl1_2 = app2.LEVELS[0]["scenes"]
        daily2 = app2.get_daily_scene_id()
        paid = next(s for s in lvl1_2 if s not in free2 and s != daily2)
        con.close()

        uname = "enttest_http"
        con = sqlite3.connect(DB)
        con.execute("DELETE FROM users WHERE username = ?", (uname,))
        con.commit()
        con.close()

        code, body = post_json("/api/auth/register",
                               {"username": uname,
                                "email": "enttest_http@example.com",
                                "password": "testpass123"})
        token = (json.loads(body).get("token") or json.loads(body).get("access_token")
                 or "") if code == 200 else ""
        check("test user registered", code == 200 and bool(token),
              "HTTP %s" % code)

        if token:
            # 403 lands before the audio is read, so no real recording and no
            # Whisper call is needed to prove the gate.
            code, body = post_multipart("/api/submit", token, paid)
            check("POST /api/submit REFUSES a scene the user does not own",
                  code == 403, "HTTP %s %s" % (code, body[:90]))

            code, body = post_multipart("/api/submit", token, free2[0])
            check("a free scene gets past the gate (fails later, not 403)",
                  code != 403, "HTTP %s" % code)

            code, body = post_multipart("/api/submit", token, daily2)
            check("the daily scene gets past the gate", code != 403,
                  "HTTP %s scene=%s" % (code, daily2))

            code, body = get_json("/api/vocab/" + paid, token)
            check("GET /api/vocab refuses an unowned scene", code == 403,
                  "HTTP %s" % code)

            code, body = get_json("/api/progress", token)
            prog = json.loads(body) if code == 200 else {}
            check("/api/progress unlocked_scenes is the free set, not all of Level 1",
                  prog.get("unlocked_scenes") == free2,
                  "got %s" % (prog.get("unlocked_scenes"),))
            check("/api/progress surfaces entitlements for the cosmetic lock",
                  "entitlements" in prog and "accessible_scenes" in prog,
                  str(sorted(prog.keys())))
            check("NO SCENE IS PROMISED THAT THE SERVER WOULD REFUSE",
                  set(prog.get("unlocked_scenes") or [])
                  <= set(prog.get("accessible_scenes") or []),
                  "unlocked-minus-accessible=%s" % sorted(
                      set(prog.get("unlocked_scenes") or [])
                      - set(prog.get("accessible_scenes") or [])))

            # The one that must not regress: the ~50 users who predate the gate
            # hold level:1 from the backfill, and enforcement must not take
            # Level 1 away from them. Granting it to this user reproduces their
            # exact row shape.
            con = sqlite3.connect(DB)
            uid_http = con.execute("SELECT id FROM users WHERE username = ?",
                                   (uname,)).fetchone()[0]
            con.close()
            post_event("order_created", order_payload(uid_http, "ORD-GF", LEVEL1_VARIANT))
            code, body = get_json("/api/progress", token)
            prog2 = json.loads(body) if code == 200 else {}
            check("A GRANDFATHERED LEVEL 1 HOLDER KEEPS ALL 20 SCENES",
                  prog2.get("unlocked_scenes") == list(lvl1_2),
                  "%d scenes" % len(prog2.get("unlocked_scenes") or []))
            code, _ = post_multipart("/api/submit", token, paid)
            check("and can submit a paid Level 1 scene", code != 403, "HTTP %s" % code)
    except Exception as exc:
        check("HTTP enforcement check ran", False, repr(exc))

    # ── Summary ──────────────────────────────────────────────────────────────
    failed = [n for n, ok, _ in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("FAILED:")
        for n in failed:
            print("  - " + n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
