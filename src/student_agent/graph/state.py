from __future__ import annotations

from typing import Any, TypedDict

from ..mcp_gateway import EvidenceGateway
from ..trace import TraceWriter


class CaseGraphState(TypedDict, total=False):
    """State passed between bounded workflow nodes for exactly one case."""

    case: dict[str, Any]
    case_id: str
    gateway: EvidenceGateway
    trace: TraceWriter
    entities: dict[str, list[str]]
    evidence: dict[str, dict[str, Any]]
    evidence_refs: list[str]
    decision: dict[str, Any]
    output: dict[str, Any]
