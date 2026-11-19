"""Toastd catalogue search (Vertex AI Search) usable alongside function tools.

Gemini's built-in VertexAiSearchTool can't be combined with other function
tools (e.g. the Cuanto tools) on one agent, so ADK swaps in
DiscoveryEngineSearchTool. Its structured-data parser calls json.dumps on raw
proto-plus containers (RepeatedComposite / MapComposite), which raises
"Object of type RepeatedComposite is not JSON serializable" for product
documents that contain lists (images, variants, ...). This subclass converts
the struct data to plain Python first.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.tools.discovery_engine_search_tool import DiscoveryEngineSearchTool
from google.cloud import discoveryengine_v1beta as discoveryengine


def _plain(value: Any) -> Any:
    """Recursively convert proto-plus Map/Repeated composites to dict/list."""
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "__iter__"):
        return [_plain(v) for v in value]
    return str(value)


class ToastdSearchTool(DiscoveryEngineSearchTool):
    def _parse_document_result(self, doc: discoveryengine.Document) -> dict[str, str]:
        if not doc.struct_data:
            return super()._parse_document_result(doc)
        data = _plain(doc.struct_data)
        title = data.pop("title", "")
        uri = data.pop("uri", data.pop("link", ""))
        return {
            "title": str(title),
            "url": str(uri),
            "content": json.dumps(data, ensure_ascii=False, default=str),
        }
