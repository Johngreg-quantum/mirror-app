"""Dump every product and variant in the Lemon Squeezy store, from the API.

    python scripts/ls-catalog.py

Reads LEMONSQUEEZY_API_KEY from the environment, or from a .env beside this
repo if it is not set. The key is never printed.

The dashboard shows product ids but not variant ids, and /api/billing/checkout
accepts only the two ids in LEMONSQUEEZY_MONTHLY_VARIANT_ID and
LEMONSQUEEZY_YEARLY_VARIANT_ID -- so these are the numbers that have to match.

Every object Lemon Squeezy returns carries a `test_mode` boolean. That is what
decides whether the key is a test key or a live key, so it is read from the
response rather than guessed from the shape of the key.
"""
import json
import os
import sys
import urllib.error
import urllib.request

STORE_ID = "396208"          # hardcoded in main.py's checkout payload
API = "https://api.lemonsqueezy.com/v1"


def load_key():
    key = os.getenv("LEMONSQUEEZY_API_KEY", "").strip()
    if key:
        return key, "environment"
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.isfile(env_path):
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("LEMONSQUEEZY_API_KEY="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        return v, ".env"
    return "", ""


def get(path, key):
    req = urllib.request.Request(
        API + path,
        headers={
            "Authorization": "Bearer " + key,
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        print("HTTP %s from %s\n%s" % (e.code, path, body))
        if e.code in (401, 403):
            print("\n-> The key was rejected. Check it is the same value Render holds,")
            print("   and that it has not been revoked in the Lemon Squeezy dashboard.")
        sys.exit(1)


def money(cents):
    if cents is None:
        return "-"
    try:
        return "$%.2f" % (int(cents) / 100.0)
    except (TypeError, ValueError):
        return str(cents)


def main():
    key, source = load_key()
    if not key:
        print("LEMONSQUEEZY_API_KEY is not set.\n")
        print("Either export it for one command:")
        print('  LEMONSQUEEZY_API_KEY=xxx python scripts/ls-catalog.py')
        print("or add the line LEMONSQUEEZY_API_KEY=xxx to .env (which is gitignored).")
        sys.exit(2)
    print("key source: %s  (value not printed)\n" % source)

    data = get("/products?filter[store_id]=%s&include=variants" % STORE_ID, key)

    products = data.get("data", [])
    included = data.get("included", [])
    variants = [i for i in included if i.get("type") == "variants"]

    modes = {bool(o.get("attributes", {}).get("test_mode"))
             for o in products + variants if "test_mode" in o.get("attributes", {})}
    if modes == {True}:
        mode = "TEST MODE  (this key sees the test store)"
    elif modes == {False}:
        mode = "LIVE MODE  (this key sees the live store — real charges)"
    elif modes:
        mode = "MIXED: %s — inspect per-object test_mode below" % modes
    else:
        mode = "UNKNOWN — no test_mode attribute returned"
    print("=" * 74)
    print(mode)
    print("=" * 74)

    by_product = {}
    for v in variants:
        pid = str(v.get("attributes", {}).get("product_id", ""))
        by_product.setdefault(pid, []).append(v)

    if not products:
        print("\nNo products returned for store %s." % STORE_ID)
        print("If that is unexpected, the key may belong to a different store,")
        print("or the store is in the other mode (test vs live).")

    for p in products:
        pa = p.get("attributes", {})
        print("\nPRODUCT  %s   id=%s   test_mode=%s"
              % (pa.get("name", "?"), p.get("id"), pa.get("test_mode")))
        print("  status=%s  price=%s" % (pa.get("status"), money(pa.get("price"))))
        rows = by_product.get(str(p.get("id")), [])
        if not rows:
            print("  (no variants returned)")
        for v in sorted(rows, key=lambda x: int(x.get("id", 0))):
            va = v.get("attributes", {})
            interval = va.get("interval")
            cadence = ("one-time" if not va.get("is_subscription")
                       else "every %s %s" % (va.get("interval_count", 1), interval))
            print("    VARIANT id=%-10s %-26s %-9s %-16s status=%s"
                  % (v.get("id"), (va.get("name") or "")[:26],
                     money(va.get("price")), cadence, va.get("status")))

    print("\n" + "-" * 74)
    print("What main.py needs, from the ids above:")
    print("  LEMONSQUEEZY_MONTHLY_VARIANT_ID = <Mirror Pro monthly variant id>")
    print("  LEMONSQUEEZY_YEARLY_VARIANT_ID  = <Mirror Pro yearly variant id>")
    print("These must also match the ids hardcoded in static/app.js (currently")
    print("1741149 monthly, 1741098 yearly) or checkout returns 400 Invalid variant.")


if __name__ == "__main__":
    main()
