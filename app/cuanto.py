"""Cuanto commerce layer — MCP client tools for the shopping agent.

Cuanto exposes a stateless Streamable-HTTP MCP server (JSON responses). We call
its `tools/call` JSON-RPC method directly instead of mounting the full
McpToolset because the raw payloads are very large (affiliate stand-down rules,
shipping tables, raw network metadata, ...).

Token-saving design (shared with app/toastd_search.py):
  * The model only sees a SLIM summary of each item (short id, title, brand,
    price, a one-line description). No URLs, no images.
  * The full display card (image, tracked redirect_url, prices) is written to
    session state under a `ui_*` key. State deltas stream to the frontend with
    the event but are never sent to the model, so the UI gets exact structured
    data while the model never has to read or re-type it.
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

# --- Content safety ---------------------------------------------------------
# The partner feed does not set `adult` reliably (adult stores are filed under
# "Health & Beauty"), and keyword search routes everyday queries such as "toy"
# or "massager" to them. Block known adult merchants plus explicit terms.
# Extra merchants can be blocked via CUANTO_BLOCKED_MERCHANTS (comma-separated,
# case-insensitive substrings of the merchant name or website).
_BLOCKED_MERCHANTS = {"adorime"} | {
    m.strip().lower()
    for m in os.environ.get("CUANTO_BLOCKED_MERCHANTS", "").split(",")
    if m.strip()
}
_ADULT_TERMS = re.compile(
    r"\b(dildo|vibrat(or|ing egg)|sex(ual)?\s?(toy|wellness)|sex\b|clit|g-?spot|"
    r"anal\b|butt\s?plug|masturbat|stroker|penis|erotic|bdsm|bondage|fetish|"
    r"strap-?on|nipple|orgasm|arous|lubricant|lube\b|condom|porn|"
    r"adult\s?toy|wand massager|suction (toy|stimulator)|kegel|cock\s?ring)",
    re.IGNORECASE,
)


def _is_blocked(item: dict[str, Any]) -> bool:
    """True for adult / unsafe items that must never be recommended."""
    if item.get("adult") is True or str(item.get("adult")).lower() == "true":
        return True
    merchant = " ".join(
        str(item.get(k) or "") for k in ("seller_name", "seller_url")
    ).lower()
    if any(b in merchant for b in _BLOCKED_MERCHANTS):
        return True
    text = " ".join(
        str(item.get(k) or "")
        for k in ("title", "short_title", "description", "product_type")
    )
    return bool(_ADULT_TERMS.search(text))


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


def _short(text: str | None, n: int) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _pid(uuid: str) -> str:
    return "c" + uuid.replace("-", "")[:7]


def _mid(uuid: str) -> str:
    return "m" + uuid.replace("-", "")[:7]


def _product_card(p: dict[str, Any]) -> dict[str, Any]:
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
    }


def _product_slim(p: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    slim = {
        "id": card["id"],
        "title": _short(card["title"], 80),
        "brand": card["brand"] or card["merchant"],
        "price": card["price"],
        "cur": card["currency"],
        "cat": card["category"],
        "desc": _short(p.get("description"), 110),
    }
    if card["sale_price"] is not None:
        slim["sale"] = card["sale_price"]
    return {k: v for k, v in slim.items() if v not in (None, "")}


async def _search_one(
    client: httpx.AsyncClient,
    query: str,
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """One query with progressive relaxation (filters, then shorter query)."""
    base = {"query_text": query, "limit": limit}
    attempts = [
        {**base, **filters},
        {**base, "max_price": filters.get("max_price")},
        base,
    ]
    words = query.split()
    for n in range(min(len(words) - 1, 3), 0, -1):
        attempts.append({**base, "query_text": " ".join(words[-n:])})

    seen: set[str] = set()
    for args in attempts:
        key = repr(sorted((k, v) for k, v in args.items() if v is not None))
        if key in seen:
            continue
        seen.add(key)
        items = [
            p
            for p in await _call_tool(client, "search_products", args)
            if not _is_blocked(p)
        ]
        if items:
            return items
    return []


def _dedupe_key(p: dict[str, Any]) -> str:
    # Collapse colour/size variants of the same item ("X - Red / M", "X - Blue").
    title = (p.get("short_title") or p.get("title") or "").lower()
    title = re.split(r"\s[-/|–]\s|\(", title)[0].strip()
    return f"{p.get('merchant_uuid')}::{title}"


async def search_cuanto_products(
    queries: list[str],
    tool_context: ToolContext,
    max_price: float | None = None,
    min_price: float | None = None,
    category_code: str | None = None,
    on_sale: bool | None = None,
) -> dict[str, Any]:
    """Searches the Cuanto partner catalogue (mostly US stores, USD prices).

    Call this every time you recommend products. Pass 2-4 SHORT keyword
    queries (1-3 words each) covering different angles of the request, e.g.
    ["skincare gift set", "face serum", "body lotion"]. Cuanto matching is
    keyword-heavy, so short concrete product nouns work best.

    Args:
        queries: 2-4 short keyword queries naming product types.
        max_price: Optional max price in USD.
        min_price: Optional min price in USD.
        category_code: Optional category code, e.g. "health-beauty",
            "apparel-accessories", "home-garden", "electronics",
            "food-beverages-tobacco", "sporting-goods", "arts-entertainment".
        on_sale: If true, only discounted products.

    Returns:
        {"status", "products": [{id, title, brand, price, cur, sale?, cat,
        desc}]}. Reference products ONLY by `id` in your picks block.
    """
    queries = [q.strip() for q in (queries or []) if q and q.strip()][:4]
    if not queries:
        return {"status": "error", "products": [], "message": "no queries"}
    filters = {
        "max_price": max_price,
        "min_price": min_price,
        "category_code": category_code,
        "on_sale": on_sale,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as c:
            batches = await asyncio.gather(
                *(_search_one(c, q, filters, 10) for q in queries),
                return_exceptions=True,
            )
    except Exception as e:  # never break the conversation on a partner outage
        logger.warning("Cuanto search_products failed: %s", e)
        return {"status": "error", "products": [], "message": str(e)[:200]}

    merged: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    # Round-robin across queries so every angle is represented.
    lists = [b for b in batches if isinstance(b, list)]
    for b in batches:
        if isinstance(b, Exception):
            logger.warning("Cuanto query failed: %s", b)
    query_of: dict[str, str] = {}
    for i in range(max((len(b) for b in lists), default=0)):
        for qi, b in enumerate(lists):
            if i >= len(b):
                continue
            p = b[i]
            query_of.setdefault(p.get("product_uuid") or "", queries[qi] if len(lists) == len(queries) else "")
            if not p.get("redirect_url") or not p.get("product_uuid"):
                continue
            if p["product_uuid"] in merged or _dedupe_key(p) in seen_keys:
                continue
            seen_keys.add(_dedupe_key(p))
            merged[p["product_uuid"]] = p
            if len(merged) >= MAX_PRODUCTS_TO_MODEL:
                break
        if len(merged) >= MAX_PRODUCTS_TO_MODEL:
            break

    cards, slims = [], []
    for p in merged.values():
        card = _product_card(p)
        card["query"] = query_of.get(p["product_uuid"], "")
        cards.append(card)
        slims.append(_product_slim(p, card))

    if cards:
        tool_context.state[UI_CUANTO_PRODUCTS] = cards
    return {"status": "success", "count": len(slims), "products": slims}


async def search_cuanto_merchants(
    query_text: str,
    tool_context: ToolContext,
    merchant_category_code: str | None = None,
) -> dict[str, Any]:
    """Searches Cuanto partner merchants (stores) and their best offer.

    Use when the user wants a store to shop at, or when product search
    returns fewer than 3 relevant products.

    Args:
        query_text: Short keyword query, e.g. "music gifts", "skincare".
        merchant_category_code: Optional exact merchant category code.

    Returns:
        {"status", "merchants": [{id, name, sector, desc}]}. Reference
        merchants ONLY by `id` in your picks block.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as c:
            items = await _call_tool(
                c,
                "search_merchants",
                {
                    "query_text": query_text,
                    "merchant_category_code": merchant_category_code,
                    "limit": 6,
                },
            )
    except Exception as e:
        logger.warning("Cuanto search_merchants failed: %s", e)
        return {"status": "error", "merchants": [], "message": str(e)[:200]}

    cards, slims = [], []
    for m in items:
        if _is_blocked(m):
            continue
        url = m.get("redirect_url") or (m.get("best_offer") or {}).get("redirect_url")
        if not url or not m.get("merchant_uuid"):
            continue
        meta = (m.get("raw_details") or {}).get("metadata") or {}
        mid = _mid(m["merchant_uuid"])
        name = m.get("seller_name") or meta.get("name")
        cards.append(
            {
                "id": mid,
                "name": name,
                "website": m.get("seller_url"),
                "logo_url": m.get("merchant_logo"),
                "sector": meta.get("primarySector"),
                "url": url,
            }
        )
        slims.append(
            {
                k: v
                for k, v in {
                    "id": mid,
                    "name": name,
                    "sector": meta.get("primarySector"),
                    "desc": _short(meta.get("description"), 110),
                }.items()
                if v
            }
        )
    if cards:
        tool_context.state[UI_CUANTO_MERCHANTS] = cards
    return {"status": "success", "count": len(slims), "merchants": slims}
