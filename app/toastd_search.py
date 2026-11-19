"""Toastd catalogue search (Vertex AI Search) as a token-lean function tool.

Raw product documents are ~7 KB each (14 image URLs, variant inventory, a long
AI context paragraph...), so a single 8-result search used to push ~14k tokens
into the model. This tool:
  * returns a SLIM summary per product to the model (short id, title, brand,
    INR prices, category, gender/aesthetic, a ~120-char description), and
  * writes full display cards (image, URL, variants) to session state under
    `ui_toastd_products`, which streams to the frontend but never reaches the
    model. See app/cuanto.py for the shared design.

It also replaces the built-in VertexAiSearchTool, which cannot be combined with
other function tools on one agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from google.adk.tools import ToolContext
from google.cloud import discoveryengine_v1beta as discoveryengine

logger = logging.getLogger(__name__)

UI_TOASTD_PRODUCTS = "ui_toastd_products"
MAX_PRODUCTS_TO_MODEL = 12

DATASTORE_ID = os.environ.get(
    "VERTEX_DATASTORE_ID",
    "projects/toastd/locations/global/collections/default_collection/dataStores/product-search-store",
)

_client: discoveryengine.SearchServiceAsyncClient | None = None


def _get_client() -> discoveryengine.SearchServiceAsyncClient:
    global _client
    if _client is None:
        _client = discoveryengine.SearchServiceAsyncClient()
    return _client


def _plain(value: Any) -> Any:
    """Recursively convert proto-plus Map/Repeated composites to dict/list."""
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "__iter__"):
        return [_plain(v) for v in value]
    return str(value)


def _num(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _short(text: str | None, n: int) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _context(data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("ai_product_context")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _variants(data: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for opt in data.get("options") or []:
        vals = [v for v in (opt.get("values") or []) if v and v != "Default Title"]
        if vals:
            out.append(f"{opt.get('name')}: {', '.join(vals[:8])}")
    return out


def _to_card_and_slim(
    doc: discoveryengine.Document,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not doc.struct_data:
        return None
    data = _plain(doc.struct_data)
    ctx = _context(data)
    pid = "t" + re.sub(r"[^0-9a-z]", "", (doc.id or "").lower())[:7]
    original = _num(data.get("price_original"))
    discounted = _num(data.get("price_discounted"))
    if original is not None and discounted is not None and discounted >= original:
        discounted = None
    images = data.get("images") or []
    variants = _variants(data)

    card = {
        "id": pid,
        "title": data.get("title"),
        "brand": ctx.get("brand") or data.get("brand"),
        "store": data.get("brand"),
        "price": original,
        "sale_price": discounted,
        "currency": "INR",
        "image_url": images[0] if images else None,
        "url": data.get("brand_url") or data.get("uri") or data.get("link"),
        "variants": variants,
        "available": data.get("is_available", True),
    }
    slim = {
        "id": pid,
        "title": _short(card["title"], 80),
        "brand": card["brand"],
        "inr": discounted if discounted is not None else original,
        "mrp": original if discounted is not None else None,
        "cat": data.get("category"),
        "for": _short(ctx.get("gender"), 24) or None,
        "style": _short(ctx.get("aesthetic"), 40) or None,
        "opts": _short("; ".join(variants), 80) if variants else None,
        "desc": _short(ctx.get("context_paragraph"), 120),
        "in_stock": None if card["available"] else False,
    }
    slim = {k: v for k, v in slim.items() if v not in (None, "", [])}
    return card, slim


async def _search_one(query: str, page_size: int) -> list[discoveryengine.Document]:
    pager = await _get_client().search(
        discoveryengine.SearchRequest(
            serving_config=f"{DATASTORE_ID}/servingConfigs/default_search",
            query=query,
            page_size=page_size,
        )
    )
    docs: list[discoveryengine.Document] = []
    async for res in pager:
        docs.append(res.document)
        if len(docs) >= page_size:
            break
    return docs


async def search_toastd_products(
    queries: list[str],
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Searches the Toastd product catalogue (Indian D2C brands, INR prices).

    Pass 2-3 SHORT queries (1-3 words each) covering different angles of the
    request, e.g. ["skincare", "face serum", "self care gift"]. Short concrete
    product nouns match far better than long descriptive sentences.

    Args:
        queries: 2-3 short product queries.

    Returns:
        {"status", "products": [{id, title, brand, inr, mrp?, cat, for?,
        style?, opts?, desc}]}. Reference products ONLY by `id` in your picks
        block.
    """
    queries = [q.strip() for q in (queries or []) if q and q.strip()][:3]
    if not queries:
        return {"status": "error", "products": [], "message": "no queries"}
    batches = await asyncio.gather(
        *(_search_one(q, 6) for q in queries), return_exceptions=True
    )

    cards, slims, seen = [], [], set()
    lists = []
    for b in batches:
        if isinstance(b, Exception):
            logger.warning("Toastd search failed: %s", b)
        else:
            lists.append(b)
    if not lists:
        return {"status": "error", "products": [], "message": "search failed"}
    # Round-robin merge so each query angle is represented.
    for i in range(max(len(b) for b in lists)):
        for qi, b in enumerate(lists):
            if i >= len(b) or b[i].id in seen:
                continue
            seen.add(b[i].id)
            parsed = _to_card_and_slim(b[i])
            if parsed:
                parsed[0]["query"] = queries[qi] if len(lists) == len(queries) else ""
                cards.append(parsed[0])
                slims.append(parsed[1])
            if len(slims) >= MAX_PRODUCTS_TO_MODEL:
                break
        if len(slims) >= MAX_PRODUCTS_TO_MODEL:
            break

    if cards:
        tool_context.state[UI_TOASTD_PRODUCTS] = cards
    return {"status": "success", "count": len(slims), "products": slims}
