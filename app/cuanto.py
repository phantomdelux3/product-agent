"""Cuanto commerce layer — MCP client tools for the shopping agent.

Cuanto exposes a stateless Streamable-HTTP MCP server (JSON responses). We call
its `tools/call` JSON-RPC method directly instead of mounting the full
McpToolset because the raw payloads are very large (affiliate stand-down rules,
shipping tables, raw network metadata, ...).

Catalogue shape (read live, nothing hard-coded): Cuanto has a *product*
catalogue and a much larger *store/offer* catalogue, both organised by the same
category codes (`list_product_categories` / `list_merchant_categories`, also
available at /catalogue/products/categories and /catalogue/offers/categories).
Many categories hold few products but hundreds of stores, so a search:
  1. looks for products of the requested type, scoped to the category,
  2. browses other products in that category ("explore the category"),
  3. adds partner stores (approved, linkable) that may sell that type.

Token-saving design (shared with app/toastd_search.py):
  * The model only sees a SLIM summary of each item (short id, title, brand,
    price, a one-line description). No URLs, no images.
  * Full display cards are written to session state under `ui_*` keys. State
    deltas stream to the frontend with the event but are never sent to the
    model.
  * The model answers with a compact `picks` block of ids + reasons; the
    frontend joins those ids with the cards.

Auth: `Authorization: Partner <CUANTO_PARTNER_TOKEN>` (env / Secret Manager).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
import time
from typing import Any

import httpx
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

CUANTO_MCP_URL = os.environ.get(
    "CUANTO_MCP_URL", "https://api.getcuanto.ai/api/v1/partners/mcp/"
)
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
_ids = itertools.count(1)

# Session-state keys the frontend listens for (see frontend lib/picks.ts).
UI_CUANTO_PRODUCTS = "ui_cuanto_products"
UI_CUANTO_MERCHANTS = "ui_cuanto_merchants"

MAX_PRODUCTS_TO_MODEL = 15
MAX_CATEGORY_PRODUCTS = 8
MAX_STORES_TO_MODEL = 8
_CATEGORY_TTL_S = 3600
_taxonomy_cache: tuple[float, dict[str, dict[str, Any]]] | None = None


# ---------------------------------------------------------------------------
# MCP transport
# ---------------------------------------------------------------------------


def _token() -> str | None:
    return os.environ.get("CUANTO_PARTNER_TOKEN")


async def _call_tool(
    client: httpx.AsyncClient, name: str, arguments: dict[str, Any]
) -> list[dict[str, Any]]:
    """Invoke a Cuanto MCP tool and return its structured `result` list."""
    token = _token()
    if not token:
        raise RuntimeError("CUANTO_PARTNER_TOKEN is not configured")

    payload = {
        "jsonrpc": "2.0",
        "id": next(_ids),
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": {k: v for k, v in arguments.items() if v is not None},
        },
    }
    res = await client.post(
        CUANTO_MCP_URL,
        json=payload,
        headers={
            "Authorization": f"Partner {token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )
    res.raise_for_status()
    body = res.json()

    if "error" in body:
        raise RuntimeError(f"Cuanto MCP error: {body['error']}")
    result = body.get("result") or {}
    if result.get("isError"):
        texts = [c.get("text", "") for c in result.get("content", [])]
        raise RuntimeError(f"Cuanto tool error: {' '.join(texts)[:300]}")
    structured = result.get("structuredContent") or {}
    items = structured.get("result", structured)
    if isinstance(items, dict):
        return [items] if items else []
    return items or []


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)


# ---------------------------------------------------------------------------
# Live taxonomy
# ---------------------------------------------------------------------------


async def _taxonomy(client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    """code -> {code, name, products, stores}, from Cuanto's live category lists."""
    global _taxonomy_cache
    now = time.monotonic()
    if _taxonomy_cache and now - _taxonomy_cache[0] < _CATEGORY_TTL_S:
        return _taxonomy_cache[1]
    product_cats, store_cats = await asyncio.gather(
        _call_tool(client, "list_product_categories", {"limit": 100}),
        _call_tool(client, "list_merchant_categories", {"limit": 100}),
    )
    tax: dict[str, dict[str, Any]] = {}
    for c in product_cats:
        if c.get("code") and c.get("name"):
            tax.setdefault(c["code"], {"code": c["code"], "name": c["name"], "products": 0, "stores": 0})
            tax[c["code"]]["products"] = int(c.get("product_count") or 0)
    for c in store_cats:
        if c.get("code") and c.get("name"):
            tax.setdefault(c["code"], {"code": c["code"], "name": c["name"], "products": 0, "stores": 0})
            tax[c["code"]]["stores"] = int(c.get("merchant_count") or 0)
    _taxonomy_cache = (now, tax)
    return tax


async def cuanto_catalogue_summary() -> str:
    """One line per live category with product/store counts, for the prompt."""
    try:
        async with _client() as c:
            tax = await _taxonomy(c)
    except Exception as e:
        logger.warning("Cuanto categories unavailable: %s", e)
        return ""
    return "; ".join(
        f"{v['name']} ({v['products']} products, {v['stores']} stores)"
        for v in sorted(tax.values(), key=lambda v: v["name"])
    )


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _resolve(hint: str | None, tax: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Match a category name/code (from the live list) to a taxonomy entry."""
    if not hint or not tax:
        return None
    h = hint.strip().lower()
    for v in tax.values():
        if h in (v["code"].lower(), v["name"].lower()):
            return v
    hw = _words(hint)
    best, best_score = None, 0
    for v in tax.values():
        score = len(hw & (_words(v["name"]) | _words(v["code"])))
        if score > best_score:
            best, best_score = v, score
    return best


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


def _is_flagged_adult(item: dict[str, Any]) -> bool:
    """Respect Cuanto's own `adult` flag (data-driven, nothing hard-coded).

    The feed does not always set it, so the agent prompt also instructs the
    model to exclude adult/explicit items, and the UI shows only ids the model
    explicitly picked.
    """
    return str(item.get("adult")).lower() in ("true", "1", "yes")


def _short(text: str | None, n: int) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _pid(uuid: str) -> str:
    return "c" + uuid.replace("-", "")[:7]


def _mid(uuid: str) -> str:
    return "m" + uuid.replace("-", "")[:7]


def _product_card(p: dict[str, Any], match: str) -> dict[str, Any]:
    return {
        "id": _pid(p["product_uuid"]),
        "title": p.get("short_title") or p.get("title"),
        "brand": p.get("brand"),
        "merchant": p.get("seller_name"),
        "merchant_logo": p.get("merchant_logo"),
        "price": p.get("price"),
        "sale_price": p.get("sale_price"),
        "currency": p.get("currency") or "USD",
        "image_url": p.get("image_link"),
        "url": p.get("redirect_url"),
        "category": p.get("product_category"),
        "availability": p.get("availability"),
        # "query" = matched the product type; "category" = same-category browse.
        "match": match,
    }


def _product_slim(p: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    slim = {
        "id": card["id"],
        "title": _short(card["title"], 80),
        "brand": card["brand"] or card["merchant"],
        "store": card["merchant"],
        "price": card["price"],
        "cur": card["currency"],
        "cat": card["category"],
        "desc": _short(p.get("description"), 110),
    }
    if card["sale_price"] is not None:
        slim["sale"] = card["sale_price"]
    return {k: v for k, v in slim.items() if v not in (None, "")}


def _store_card_and_slim(
    m: dict[str, Any], category: str | None
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    url = m.get("redirect_url") or (m.get("best_offer") or {}).get("redirect_url")
    if not url or not m.get("merchant_uuid"):
        return None
    meta = (m.get("raw_details") or {}).get("metadata") or {}
    mid = _mid(m["merchant_uuid"])
    name = m.get("seller_name") or meta.get("name")
    cats = m.get("merchant_categories") or []
    card = {
        "id": mid,
        "name": name,
        "website": m.get("seller_url"),
        "logo_url": m.get("merchant_logo"),
        "sector": meta.get("primarySector") or (cats[0] if cats else None),
        "category": category or (cats[0] if cats else None),
        "url": url,
    }
    slim = {
        "id": mid,
        "name": name,
        "sector": card["sector"],
        "desc": _short(meta.get("description"), 110),
    }
    return card, {k: v for k, v in slim.items() if v}


def _dedupe_key(p: dict[str, Any]) -> str:
    # Collapse colour/size variants of the same item ("X - Red / M", "X - Blue").
    title = (p.get("short_title") or p.get("title") or "").lower()
    title = re.split(r"\s[-/|]\s|\(", title)[0].strip()
    return f"{p.get('merchant_uuid')}::{title}"


def _merge(
    batches: list[list[dict[str, Any]]], cap: int, seen: set[str], key: str
) -> list[dict[str, Any]]:
    """Round-robin merge across batches, de-duplicating by `key` and variants."""
    out: list[dict[str, Any]] = []
    variants: set[str] = set()
    for i in range(max((len(b) for b in batches), default=0)):
        for b in batches:
            if i >= len(b) or len(out) >= cap:
                continue
            item = b[i]
            uid = item.get(key)
            if not uid or uid in seen or _is_flagged_adult(item):
                continue
            if key == "product_uuid":
                if not item.get("redirect_url"):
                    continue
                v = _dedupe_key(item)
                if v in variants:
                    continue
                variants.add(v)
            seen.add(uid)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------


async def _products_for_query(
    client: httpx.AsyncClient,
    query: str,
    code: str | None,
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Products of this type, kept to the requested category when one is given.

    Relaxes filters, then the query (keeping head nouns), until something
    matches. With a category, unscoped results from other categories are
    dropped — that is what stops "tea" matching tea-coloured sunglasses.
    """
    base = {"query_text": query, "limit": 10, "category_code": code}
    attempts = [
        {**base, **filters},
        {**base, "max_price": filters.get("max_price")},
        base,
    ]
    # Shorten long queries, but never below two words: a lone generic word
    # ("set", "gift") matches unrelated items.
    words = query.split()
    for n in range(min(len(words) - 1, 3), 1, -1):
        attempts.append({**base, "query_text": " ".join(words[-n:])})

    tried: set[str] = set()
    for args in attempts:
        key = repr(sorted((k, v) for k, v in args.items() if v is not None))
        if key in tried:
            continue
        tried.add(key)
        items = await _call_tool(client, "search_products", args)
        if code:
            items = [p for p in items if p.get("product_category_code") in (code, None)]
        if items:
            return items
    return []


def _keywords(*texts: str) -> list[str]:
    """Distinct words (in order) from the given texts, for broader searches."""
    out: list[str] = []
    for t in texts:
        for w in re.findall(r"[A-Za-z0-9']+", t):
            w = w.lower().strip("'")
            if len(w) > 2 and w not in out:
                out.append(w)
    return out


async def _category_browse(
    client: httpx.AsyncClient, cat: dict[str, Any], queries: list[str], max_price: float | None
) -> list[dict[str, Any]]:
    """Other products in the category, to explore when nothing matched.

    Keyword search needs a query, so try the user's queries, then their single
    words, then the category name's words — all scoped to the category code.
    """
    out: list[dict[str, Any]] = []
    candidates = [*queries[:2], *_keywords(*queries), *_keywords(cat["name"])]
    for q in candidates[:8]:
        for price in (max_price, None):
            items = await _call_tool(
                client,
                "search_products",
                {"query_text": q, "category_code": cat["code"], "max_price": price, "limit": 10},
            )
            if items:
                out.extend(items)
                break
        if len(out) >= MAX_CATEGORY_PRODUCTS * 2:
            break
    return out


async def _stores(
    client: httpx.AsyncClient, query: str, code: str | None
) -> list[dict[str, Any]]:
    """Stores this partner can link to. Many search hits are not approved for
    the partner (no redirect_url), so over-fetch and keep the linkable ones."""
    items = await _call_tool(
        client,
        "search_merchants",
        {"query_text": query, "merchant_category_code": code, "limit": 25},
    )
    return [
        m
        for m in items
        if m.get("redirect_url") or (m.get("best_offer") or {}).get("redirect_url")
    ]


async def _find_stores(
    client: httpx.AsyncClient, queries: list[str], codes: list[str | None]
) -> list[list[dict[str, Any]]]:
    """Stores for this product type: full queries and their single words, in
    each candidate category, then unscoped if nothing was found."""
    terms = list(dict.fromkeys([*queries[:2], *_keywords(*queries)[:3]]))
    results = await asyncio.gather(
        *(_stores(client, t, code) for code in codes for t in terms),
        return_exceptions=True,
    )
    batches = [r for r in results if isinstance(r, list) and r]
    if not batches and any(codes):
        results = await asyncio.gather(
            *(_stores(client, t, None) for t in terms), return_exceptions=True
        )
        batches = [r for r in results if isinstance(r, list) and r]
    return batches


async def search_cuanto_products(
    queries: list[str],
    tool_context: ToolContext,
    category: str | None = None,
    store_categories: list[str] | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    on_sale: bool | None = None,
) -> dict[str, Any]:
    """Searches the Cuanto partner catalogue (mostly US stores, USD prices).

    Call this every time you recommend products. It returns, in one call:
      * `products` — products of the requested type (within `category`),
      * `category_products` — other products in that category to explore,
      * `stores` — partner stores that may sell this type.

    Args:
        queries: 2-4 SHORT keyword queries (1-3 words) naming the product type
            from different angles, e.g. ["teapot", "tea infuser", "tea set"].
            Matching is keyword-heavy: use concrete product nouns.
        category: The Cuanto category the product belongs to — one of the
            exact names listed in your instructions.
        store_categories: Optional extra category names where STORES selling
            this product may be filed (e.g. tea shops under a food category).
        max_price: Optional max price in USD.
        min_price: Optional min price in USD.
        on_sale: If true, only discounted products.

    Returns:
        {"status", "category", "products", "category_products", "stores"}.
        Product items: id, title, brand, store, price, cur, sale?, cat, desc.
        Store items: id, name, sector, desc. Reference items ONLY by `id`.
    """
    queries = [q.strip() for q in (queries or []) if q and q.strip()][:4]
    if not queries:
        return {"status": "error", "products": [], "message": "no queries"}
    filters = {"max_price": max_price, "min_price": min_price, "on_sale": on_sale}

    try:
        async with _client() as c:
            try:
                tax = await _taxonomy(c)
            except Exception as e:
                logger.warning("Cuanto categories unavailable: %s", e)
                tax = {}
            cat = _resolve(category, tax)
            code = cat["code"] if cat else None

            store_codes: list[str | None] = [code]
            for extra in store_categories or []:
                r = _resolve(extra, tax)
                if r and r["code"] not in store_codes:
                    store_codes.append(r["code"])

            # Products, category browse and stores all run in parallel.
            tasks: list[Any] = [_products_for_query(c, q, code, filters) for q in queries]
            tasks.append(
                _category_browse(c, cat, queries, max_price) if cat else asyncio.sleep(0, [])
            )
            tasks.append(_find_stores(c, queries, store_codes))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Cuanto query failed: %s", r)
            ok = [r if isinstance(r, list) else [] for r in results]
            seen: set[str] = set()
            direct = _merge(ok[: len(queries)], MAX_PRODUCTS_TO_MODEL, seen, "product_uuid")
            browse = _merge([ok[len(queries)]], MAX_CATEGORY_PRODUCTS, seen, "product_uuid")
            stores_raw = _merge(ok[len(queries) + 1], MAX_STORES_TO_MODEL, set(), "merchant_uuid")
    except Exception as e:  # never break the conversation on a partner outage
        logger.warning("Cuanto search failed: %s", e)
        return {"status": "error", "products": [], "message": str(e)[:200]}

    out: dict[str, Any] = {
        "status": "success",
        "category": cat["name"] if cat else None,
        "products": [],
        "category_products": [],
        "stores": [],
    }
    cards: list[dict[str, Any]] = []
    for items, match, key in ((direct, "query", "products"), (browse, "category", "category_products")):
        for p in items:
            card = _product_card(p, match)
            cards.append(card)
            out[key].append(_product_slim(p, card))
    store_cards: list[dict[str, Any]] = []
    for m in stores_raw:
        parsed = _store_card_and_slim(m, out["category"])
        if parsed:
            store_cards.append(parsed[0])
            out["stores"].append(parsed[1])

    if cards:
        tool_context.state[UI_CUANTO_PRODUCTS] = cards
    if store_cards:
        tool_context.state[UI_CUANTO_MERCHANTS] = store_cards
    return out


async def search_cuanto_merchants(
    query_text: str,
    tool_context: ToolContext,
    category: str | None = None,
) -> dict[str, Any]:
    """Searches Cuanto partner stores (merchants) and their best offer.

    Use when the user explicitly asks for stores/brands to shop at.
    (`search_cuanto_products` already returns stores alongside products.)

    Args:
        query_text: Short keyword query, e.g. "tea", "running shoes".
        category: Optional category name from the list in your instructions.

    Returns:
        {"status", "stores": [{id, name, sector, desc}]}. Reference stores
        ONLY by `id` in the `merchants` list of your picks block.
    """
    try:
        async with _client() as c:
            try:
                cat = _resolve(category, await _taxonomy(c))
            except Exception:
                cat = None
            items = await _stores(c, query_text, cat["code"] if cat else None)
    except Exception as e:
        logger.warning("Cuanto search_merchants failed: %s", e)
        return {"status": "error", "stores": [], "message": str(e)[:200]}

    cards, slims = [], []
    for m in _merge([items], MAX_STORES_TO_MODEL, set(), "merchant_uuid"):
        parsed = _store_card_and_slim(m, cat["name"] if cat else None)
        if parsed:
            cards.append(parsed[0])
            slims.append(parsed[1])
    if cards:
        tool_context.state[UI_CUANTO_MERCHANTS] = cards
    return {"status": "success", "stores": slims}
