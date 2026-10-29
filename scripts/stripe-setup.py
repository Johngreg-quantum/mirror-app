"""Create Mirror's Stripe catalogue and portal configuration, idempotently.

    python scripts/stripe-setup.py           # test mode (refuses an sk_live_ key)
    python scripts/stripe-setup.py --live    # live mode, when going live

Reads STRIPE_SECRET_KEY from the environment or .env, the same way main.py
does, and never prints it. Prints the env lines to set -- ids only, none of
them secret.

Safe to re-run: prices are found by lookup_key before anything is created, and
the portal configuration by its metadata, so a second run creates nothing and
prints the same ids. Test and live are separate catalogues with separate ids,
which is why going live is this script run once more with a live key.

The catalogue:

  Mirror Starter Pack   $1.00 one-time        STRIPE_STARTER_PRICE_ID
  Mirror Pro            $8.99 / month         STRIPE_MONTHLY_PRICE_ID
                        $47.88 / year         STRIPE_YEARLY_PRICE_ID

Pro is one product with two prices rather than two products: that is the
shape the Customer Portal expects if plan switching is ever turned on, and
the plan labels on the pricing card come from main.py, not from these names.
"""
import os
import sys

import httpx
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

API = "https://api.stripe.com"
KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
LIVE = "--live" in sys.argv

CATALOGUE = (
    # env var,                  lookup_key,           product,               cents, interval
    ("STRIPE_STARTER_PRICE_ID", "mirror_starter",     "Mirror Starter Pack",   100, None),
    ("STRIPE_MONTHLY_PRICE_ID", "mirror_pro_monthly", "Mirror Pro",            899, "month"),
    ("STRIPE_YEARLY_PRICE_ID",  "mirror_pro_yearly",  "Mirror Pro",           4788, "year"),
)

BASE_URL = "https://mirrorspeak.app"


def fail(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def call(client, method, path, **kw):
    resp = client.request(method, API + path, **kw)
    if resp.status_code != 200:
        # Stripe error bodies carry no secrets; the key is never echoed.
        fail("%s %s -> HTTP %s: %s" % (method, path, resp.status_code, resp.text[:400]))
    return resp.json()


def main():
    if not KEY:
        fail("STRIPE_SECRET_KEY is not set (env or .env).")
    is_live_key = KEY.startswith(("sk_live_", "rk_live_"))
    if is_live_key and not LIVE:
        fail("this is a LIVE key. Re-run with --live if that is what you mean.")
    if LIVE and not is_live_key:
        fail("--live was passed but the key is not a live key.")

    client = httpx.Client(headers={"Authorization": "Bearer " + KEY}, timeout=20)
    out = {}
    products = {}

    for env, lookup, product_name, cents, interval in CATALOGUE:
        found = call(client, "GET", "/v1/prices",
                     params={"lookup_keys[]": lookup, "active": "true",
                             "expand[]": "data.product"})["data"]
        if found:
            price = found[0]
            if price["unit_amount"] != cents:
                fail("price %s (%s) is %s cents, expected %s. Archive it in the "
                     "dashboard and re-run; prices cannot be edited."
                     % (price["id"], lookup, price["unit_amount"], cents))
            products[product_name] = price["product"]["id"]
            print("  exists   %-24s %s" % (lookup, price["id"]))
        else:
            if product_name not in products:
                existing = call(client, "GET", "/v1/products/search",
                                params={"query": "metadata['mirror_product']:'%s'"
                                        % product_name.replace(" ", "_").lower()})["data"]
                if existing:
                    products[product_name] = existing[0]["id"]
                else:
                    products[product_name] = call(client, "POST", "/v1/products", data={
                        "name": product_name,
                        "metadata[mirror_product]": product_name.replace(" ", "_").lower(),
                    })["id"]
            data = {
                "product": products[product_name],
                "currency": "usd",
                "unit_amount": str(cents),
                "lookup_key": lookup,
                "nickname": lookup,
            }
            if interval:
                data["recurring[interval]"] = interval
            price = call(client, "POST", "/v1/prices", data=data)
            print("  created  %-24s %s" % (lookup, price["id"]))
        out[env] = price["id"]

    # Customer Portal: cancel at period end, update card, see invoices. Plan
    # switching is off -- monthly<->yearly proration is a refund-shaped credit,
    # and all sales are final.
    configs = call(client, "GET", "/v1/billing_portal/configurations",
                   params={"active": "true", "limit": 100})["data"]
    config = next((c for c in configs if (c.get("metadata") or {}).get("mirror") == "1"), None)
    if config:
        print("  exists   portal configuration     %s" % config["id"])
    else:
        config = call(client, "POST", "/v1/billing_portal/configurations", data={
            "business_profile[headline]": "Mirror — manage your subscription",
            "business_profile[privacy_policy_url]": BASE_URL + "/privacy",
            "business_profile[terms_of_service_url]": BASE_URL + "/terms",
            "default_return_url": BASE_URL + "/",
            "features[invoice_history][enabled]": "true",
            "features[payment_method_update][enabled]": "true",
            "features[customer_update][enabled]": "true",
            "features[customer_update][allowed_updates][0]": "email",
            "features[subscription_cancel][enabled]": "true",
            "features[subscription_cancel][mode]": "at_period_end",
            "features[subscription_cancel][proration_behavior]": "none",
            "features[subscription_cancel][cancellation_reason][enabled]": "true",
            "features[subscription_cancel][cancellation_reason][options][0]": "too_expensive",
            "features[subscription_cancel][cancellation_reason][options][1]": "unused",
            "features[subscription_cancel][cancellation_reason][options][2]": "missing_features",
            "features[subscription_cancel][cancellation_reason][options][3]": "other",
            "features[subscription_update][enabled]": "false",
            "metadata[mirror]": "1",
        })
        print("  created  portal configuration     %s" % config["id"])
    out["STRIPE_PORTAL_CONFIGURATION_ID"] = config["id"]

    print("\n%s mode. Set these (ids, not secrets):\n" % ("LIVE" if LIVE else "TEST"))
    for k, v in out.items():
        print("%s=%s" % (k, v))


if __name__ == "__main__":
    main()
