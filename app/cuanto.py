"""Cuanto commerce layer — MCP client tools for the shopping agent.

Cuanto exposes a stateless Streamable-HTTP MCP server (JSON responses). We call
its `tools/call` JSON-RPC method directly instead of mounting the full
McpToolset because the raw payloads are very large (affiliate stand-down rules,
shipping tables, raw network metadata, ...). Each tool here trims results down
to the fields the agent actually needs, which keeps Gemini's context small and
the recommendations accurate.

Auth: `Authorization: Partner <CUANTO_PARTNER_TOKEN>` (set via env / Secret
Manager — never hard-code it).
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CUANTO_MCP_URL = os.environ.get(
    "CUANTO_MCP_URL", "https://api.getcuanto.ai/api/v1/partners/mcp/"
)
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
_ids = itertools.count(1)


def _token() -> str | None:
    return os.environ.get("CUANTO_PARTNER_TOKEN")


async def _call_tool(name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
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
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
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


def _trim_product(p: dict[str, Any]) -> dict[str, Any]:
    desc = (p.get("description") or "").strip()
    return {
        "product_uuid": p.get("product_uuid"),
        "title": p.get("short_title") or p.get("title"),
        "brand": p.get("brand"),
        "merchant": p.get("seller_name"),
        "category": p.get("product_category"),
        "price": p.get("price"),
        "sale_price": p.get("sale_price"),
        "currency": p.get("currency"),
        "availability": p.get("availability"),
        "image_url": p.get("image_link"),
        "redirect_url": p.get("redirect_url"),
        "region": p.get("region") or p.get("product_region"),
        "description": desc[:240] + ("…" if len(desc) > 240 else ""),
        "similarity_score": p.get("similarity_score"),
    }


def _trim_merchant(m: dict[str, Any]) -> dict[str, Any]:
    meta = (m.get("raw_details") or {}).get("metadata") or {}
    desc = (meta.get("description") or "").strip()
    return {
        "merchant_uuid": m.get("merchant_uuid"),
        "name": m.get("seller_name") or meta.get("name"),
        "website": m.get("seller_url"),
        "logo_url": m.get("merchant_logo"),
        "categories": m.get("merchant_categories") or [],
        "sector": meta.get("primarySector"),
        "region": m.get("region"),
        "redirect_url": m.get("redirect_url")
        or (m.get("best_offer") or {}).get("redirect_url"),
        "description": desc[:240] + ("…" if len(desc) > 240 else ""),
    }


async def search_cuanto_products(
    query_text: str,
    max_price: float | None = None,
    min_price: float | None = None,
    category_code: str | None = None,
    brand: str | None = None,
    gender: str | None = None,
    on_sale: bool | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    """Searches the Cuanto partner catalogue for products to recommend.

    Use this alongside the Toastd product search every time you produce
    product recommendations, so the user also gets Cuanto picks.

    Args:
        query_text: SHORT keyword query of 1-3 words naming the product type,
            e.g. "skincare gift set", "leather wallet", "running shoes".
            Cuanto matching is keyword-heavy; long descriptive sentences
            return fewer results.
        max_price: Optional maximum price (in the product's own currency,
            usually USD).
        min_price: Optional minimum price.
        category_code: Optional exact category code, e.g. "health-beauty",
            "apparel-accessories", "home-garden", "electronics",
            "food-beverages-tobacco", "sporting-goods", "arts-entertainment".
        brand: Optional brand name to narrow results.
        gender: Optional gender filter, e.g. "male", "female", "unisex".
        on_sale: If true, only return discounted products.
        limit: Number of products to return (1-10, default 6).

    Returns:
        A dict with `status` and `products` (each has product_uuid, title,
        brand, merchant, price, sale_price, currency, image_url, redirect_url,
        description, similarity_score).
    """
    base = {"query_text": query_text, "limit": max(1, min(int(limit or 6), 10))}
    strict = {
        **base,
        "max_price": max_price,
        "min_price": min_price,
        "category_code": category_code,
        "brand": brand,
        "gender": gender,
        "on_sale": on_sale,
    }
    # Exact-match filters (gender, category, brand, ...) are often unset in the
    # partner feeds, so progressively relax them until something matches.
    attempts = [strict, {**base, "max_price": max_price}, base]
    # Then shorten the query, keeping the trailing (head-noun) words:
    # "natural minimalist skincare gift set" -> "skincare gift set" -> ...
    words = query_text.split()
    for n in range(min(len(words) - 1, 3), 0, -1):
        attempts.append({**base, "query_text": " ".join(words[-n:])})
    items: list[dict[str, Any]] = []
    try:
        seen: set[str] = set()
        for args in attempts:
            key = repr(sorted((k, v) for k, v in args.items() if v is not None))
            if key in seen:
                continue
            seen.add(key)
            items = await _call_tool("search_products", args)
            if items:
                break
    except Exception as e:  # never break the conversation on a partner outage
        logger.warning("Cuanto search_products failed: %s", e)
        return {"status": "error", "products": [], "message": str(e)[:200]}

    products = [_trim_product(p) for p in items if p.get("redirect_url")]
    return {"status": "success", "count": len(products), "products": products}


async def search_cuanto_merchants(
    query_text: str,
    merchant_category_code: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Searches Cuanto partner merchants (stores) and their best offer.

    Use this when the user is looking for a store/brand to shop at rather than
    a specific product, or when product search returns nothing useful.

    Args:
        query_text: Natural-language description, e.g. "music gifts store".
        merchant_category_code: Optional exact merchant category code.
        limit: Number of merchants to return (1-10, default 5).

    Returns:
        A dict with `status` and `merchants` (each has merchant_uuid, name,
        website, logo_url, sector, redirect_url, description).
    """
    try:
        items = await _call_tool(
            "search_merchants",
            {
                "query_text": query_text,
                "merchant_category_code": merchant_category_code,
                "limit": max(1, min(int(limit or 5), 10)),
            },
        )
    except Exception as e:
        logger.warning("Cuanto search_merchants failed: %s", e)
        return {"status": "error", "merchants": [], "message": str(e)[:200]}

    merchants = [_trim_merchant(m) for m in items if m.get("redirect_url")]
    return {"status": "success", "count": len(merchants), "merchants": merchants}
