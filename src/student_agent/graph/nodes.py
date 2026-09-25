from __future__ import annotations

from typing import Any

from ..policy import decide
from ..verifier import emit_verification, repair_output, verify_output
from .state import CaseGraphState


def _values(case: dict[str, Any], *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        raw = case.get(name, [])
        for item in raw if isinstance(raw, list) else [raw]:
            if isinstance(item, str) and item and item not in values:
                values.append(item)
    return values


def _append(values: list[str], value: Any) -> None:
    if isinstance(value, str) and value and value not in values:
        values.append(value)


def _add_evidence_entities(entities: dict[str, list[str]], evidence: dict[str, Any]) -> None:
    """Copy only identifiers present in the server evidence into output scope."""
    data = evidence.get("data")
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        if not isinstance(row, dict):
            continue
        _append(entities["order_ids"], row.get("order_id"))
        _append(entities["item_ids"], row.get("order_item_id"))
        _append(entities["seller_ids"], row.get("seller_id"))
        _append(entities["payment_references"], row.get("payment_id"))
        _append(entities["payment_references"], row.get("payment_reference"))
        _append(entities["shipment_ids"], row.get("shipment_id"))


async def coordinator_node(state: CaseGraphState) -> dict[str, Any]:
    case = state["case"]
    request = case.get("customer_request", {})
    if not isinstance(request, dict):
        request = {}
    entities = {
        "order_ids": _values(case, "order_id", "order_ids")
        or _values(request, "claimed_order_id", "order_id", "order_ids"),
        "item_ids": _values(case, "item_id", "item_ids", "order_item_id"),
        "seller_ids": _values(case, "seller_id", "seller_ids"),
        "payment_references": _values(
            case, "payment_reference", "payment_references", "payment_id"
        ),
        "shipment_ids": _values(case, "shipment_id", "shipment_ids"),
    }
    trace = state["trace"]
    for actor in ("order-item-agent", "payment-agent", "shipment-agent"):
        trace.emit(
            case_id=state["case_id"],
            event_type="task_assigned",
            actor="coordinator",
            target=actor,
        )
    return {"entities": entities, "evidence": {}, "evidence_refs": []}


async def _collect(
    state: CaseGraphState, *, actor: str, tools: tuple[str, ...]
) -> dict[str, Any]:
    order_ids = state["entities"]["order_ids"]
    if not order_ids:
        return {}
    evidence_by_tool = dict(state.get("evidence", {}))
    refs = list(state.get("evidence_refs", []))
    entities = {name: list(ids) for name, ids in state["entities"].items()}
    for tool in tools:
        try:
            evidence = await state["gateway"].call(
                tool, case_id=state["case_id"], actor=actor, order_id=order_ids[0]
            )
        except (RuntimeError, ValueError):
            # Gateway already performs its single, timeout-only retry.  A failed
            # specialist task is represented by absent evidence, never invented data.
            continue
        ref = evidence["evidence_ref"]
        evidence_by_tool[tool] = evidence
        _add_evidence_entities(entities, evidence)
        if ref not in refs:
            refs.append(ref)
        state["trace"].emit(
            case_id=state["case_id"],
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool, evidence_refs=[ref],
        )
        state["trace"].emit(
            case_id=state["case_id"],
            event_type="handoff",
            actor=actor,
            target="policy-agent", evidence_refs=[ref],
        )
    return {"entities": entities, "evidence": evidence_by_tool, "evidence_refs": refs}


async def order_node(state: CaseGraphState) -> dict[str, Any]:
    return await _collect(
        state, actor="order-item-agent", tools=("get_order", "get_order_items")
    )


async def payment_node(state: CaseGraphState) -> dict[str, Any]:
    return await _collect(
        state,
        actor="payment-agent",
        tools=("get_order_payments", "get_payment_timeline", "get_refund_timeline"),
    )


async def shipment_node(state: CaseGraphState) -> dict[str, Any]:
    return await _collect(state, actor="shipment-agent", tools=("get_shipment_summary",))


async def policy_node(state: CaseGraphState) -> dict[str, Any]:
    evidence = dict(state.get("evidence", {}))
    order_ids = state["entities"]["order_ids"]
    if order_ids:
        try:
            policy_evidence = await state["gateway"].call(
                "get_policy",
                case_id=state["case_id"],
                actor="policy-agent",
                policy_version=str(state["case"].get("policy_version", "")),
            )
        except (RuntimeError, ValueError):
            pass
        else:
            ref = policy_evidence["evidence_ref"]
            evidence["get_policy"] = policy_evidence
            state["trace"].emit(
                case_id=state["case_id"],
                event_type="tool_result_consumed",
                actor="policy-agent",
                tool_name="get_policy",
                evidence_refs=[ref],
            )

    decision = decide(state["case"], evidence)
    refs = list(dict.fromkeys([*state.get("evidence_refs", []), *decision["evidence_refs"]]))
    state["trace"].emit(
        case_id=state["case_id"],
        event_type="policy_decided",
        actor="policy-agent",
        decision_code=decision["assessment"]["primary_issue"],
        evidence_refs=decision["evidence_refs"],
    )
    state["trace"].emit(
        case_id=state["case_id"],
        event_type="handoff",
        actor="policy-agent",
        target="verifier-agent",
        evidence_refs=decision["evidence_refs"],
    )
    return {"evidence": evidence, "evidence_refs": refs, "decision": decision}


async def verifier_node(state: CaseGraphState) -> dict[str, Any]:
    draft = {
        "schema_version": "day09-l3a-output-v2",
        "case_id": state["case_id"],
        "affected_entities": state["entities"],
        **state["decision"],
    }
    output = repair_output(
        draft,
        case=state["case"],
        evidence=state.get("evidence", {}).values(),
    )
    report = verify_output(
        output,
        case=state["case"],
        evidence=state.get("evidence", {}).values(),
        contracts=state["trace"].contracts,
        consumed_refs=state.get("evidence_refs", []),
    )
    emit_verification(state["trace"], report, evidence_refs=output["evidence_refs"])
    return {"output": output}
