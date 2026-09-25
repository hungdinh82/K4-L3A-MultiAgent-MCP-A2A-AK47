from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.contracts import Contracts
from student_agent.trace import TraceWriter
from student_agent.verifier import (
    calibrate_confidence,
    emit_verification,
    evidence_entity_scope,
    repair_output,
    total_paid_brl,
    verify_output,
)

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "L3A_CASE_001"
ORDER_ID = "e2a03ccf5ea816036608b2d8c3ab8e60"
ITEM_ID = "item-e2a03ccf5ea8"
SELLER_ID = "seller-e2a03ccf5ea8"
POLICY_EXAMPLE_SELLER = "seller-e58fb7bfd033"

REF_ORDER = "ev_order_AAAAAAAAAAAAAAAAAAAAAA"
REF_ITEMS = "ev_items_AAAAAAAAAAAAAAAAAAAAAA"
REF_PAYMENTS = "ev_payments_AAAAAAAAAAAAAAAAAAA"
REF_SHIPMENT = "ev_shipment_AAAAAAAAAAAAAAAAAAA"
REF_POLICY = "ev_policy_AAAAAAAAAAAAAAAAAAAAA"
REF_FOREIGN = "ev_other_case_AAAAAAAAAAAAAAAAAA"
HASH = "sha256:" + "0" * 64


def envelope(ref: str, domain: str, data: Any) -> dict[str, Any]:
    return {
        "schema_version": "day09-mcp-evidence-v1",
        "evidence_ref": ref,
        "result_hash": HASH,
        "domain": domain,
        "data": data,
        "warnings": [],
    }


CASE: dict[str, Any] = {
    "case_id": CASE_ID,
    "opened_at": "2018-01-01T09:00:00-03:00",
    "customer_request": {
        "language": "vi",
        "message": "Đơn hàng có dấu hiệu bất thường sau thanh toán.",
        "claimed_order_id": ORDER_ID,
        "claims": [
            {"claim_id": "claim-001-a", "topic": "canceled_order_paid"},
            {"claim_id": "claim-001-b", "topic": "requested_full_refund"},
        ],
    },
    "policy_version": "EC_POLICY_V1",
}

# Shapes mirror real MCP responses for L3A_CASE_001.
EVIDENCE: list[dict[str, Any]] = [
    envelope(
        REF_ORDER,
        "order",
        {"order_id": ORDER_ID, "order_status": "canceled", "customer_id": "customer-row-x"},
    ),
    envelope(
        REF_ITEMS,
        "item",
        [
            {
                "order_id": ORDER_ID,
                "order_item_id": ITEM_ID,
                "seller_id": SELLER_ID,
                "price": "79.00",
                "freight_value": "10.00",
            }
        ],
    ),
    envelope(
        REF_PAYMENTS,
        "payment",
        [
            {"order_id": ORDER_ID, "payment_sequential": "1", "payment_value": "79.00"},
            {"order_id": ORDER_ID, "payment_sequential": "1", "payment_value": "18.00"},
        ],
    ),
    envelope(
        REF_SHIPMENT,
        "shipment",
        {"order_id": ORDER_ID, "order_status": "canceled", "events": []},
    ),
    envelope(
        REF_POLICY,
        "policy",
        {
            "currency": "BRL",
            "policy_version": "EC_POLICY_V1",
            "rules": {
                "canceled_order_paid": {
                    "case_status": "action_required",
                    "recommended_action": "issue_refund",
                    "refund_brl": 79.0,
                    "responsible_parties": [{"party_id": None, "party_type": "platform"}],
                },
                "late_delivery_seller": {
                    "case_status": "action_required",
                    "recommended_action": "refund_freight",
                    "refund_brl": 18.0,
                    "responsible_parties": [
                        {"party_id": POLICY_EXAMPLE_SELLER, "party_type": "seller"}
                    ],
                },
            },
        },
    ),
]


def good_output() -> dict[str, Any]:
    return {
        "schema_version": "day09-l3a-output-v2",
        "case_id": CASE_ID,
        "assessment": {
            "primary_issue": "canceled_order_paid",
            "case_status": "action_required",
            "confidence": 0.85,
        },
        "affected_entities": {
            "order_ids": [ORDER_ID],
            "item_ids": [ITEM_ID],
            "seller_ids": [SELLER_ID],
            "payment_references": [],
            "shipment_ids": [],
        },
        "claim_assessments": [
            {
                "claim_id": "claim-001-a",
                "verdict": "supported",
                "confidence": 0.85,
                "evidence_refs": [REF_ORDER, REF_PAYMENTS],
            }
        ],
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": "ORDER_CANCELED_AFTER_CAPTURE", "rank": 1}],
            "responsible_parties": [{"party_type": "platform", "party_id": None}],
        },
        "evidence_refs": [REF_ORDER, REF_ITEMS, REF_PAYMENTS, REF_POLICY],
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 79.0,
            "refund_lines": [
                {"reason_code": "CANCELED_ORDER_REFUND", "amount_brl": 79.0, "entity_id": ITEM_ID}
            ],
        },
        "resolution_actions": ["issue_refund"],
    }


@pytest.fixture(scope="module")
def contracts() -> Contracts:
    return Contracts(ROOT / "contracts" / "schemas")


def verify(output: dict[str, Any], contracts: Contracts | None = None, **kwargs: Any):
    return verify_output(output, case=CASE, evidence=EVIDENCE, contracts=contracts, **kwargs)


# --------------------------------------------------------------------------- happy path


def test_good_output_passes_all_checks(contracts: Contracts) -> None:
    report = verify(good_output(), contracts)
    assert report.ok, report.issues
    assert report.issues == []
    assert report.decision_code == "verified"


def test_evidence_scope_ignores_policy_template_ids() -> None:
    scope = evidence_entity_scope(EVIDENCE)
    assert scope["order_ids"] == {ORDER_ID}
    assert scope["item_ids"] == {ITEM_ID}
    assert scope["seller_ids"] == {SELLER_ID}
    assert POLICY_EXAMPLE_SELLER not in scope["seller_ids"]


def test_total_paid_does_not_double_count_payment_timeline() -> None:
    timeline = envelope(
        "ev_timeline_AAAAAAAAAAAAAAAAAAA",
        "payment",
        {"payments": [{"payment_value": "79.00"}, {"payment_value": "18.00"}], "events": []},
    )
    assert total_paid_brl([*EVIDENCE, timeline]) == 97.0
    assert total_paid_brl([timeline]) == 97.0
    assert total_paid_brl([EVIDENCE[0]]) is None


# --------------------------------------------------------------------------- hard gates


def test_schema_violation_is_error(contracts: Contracts) -> None:
    output = good_output()
    output["assessment"]["primary_issue"] = "made_up_issue"
    assert "schema_invalid" in verify(output, contracts).codes()


def test_case_id_mismatch_is_error() -> None:
    output = good_output()
    output["case_id"] = "L3A_CASE_002"
    report = verify(output)
    assert not report.ok
    assert "case_id_mismatch" in report.codes()


def test_unknown_evidence_ref_is_error() -> None:
    output = good_output()
    output["evidence_refs"].append(REF_FOREIGN)
    output["claim_assessments"][0]["evidence_refs"].append(REF_FOREIGN)
    report = verify(output)
    assert report.codes().count("unknown_evidence_ref") == 2


def test_concrete_issue_without_evidence_is_error() -> None:
    output = good_output()
    output["evidence_refs"] = []
    output["claim_assessments"] = []
    assert "missing_required_evidence" in verify(output).codes()


def test_insufficient_evidence_may_have_no_refs() -> None:
    output = good_output()
    output.update(
        evidence_refs=[],
        claim_assessments=[],
        resolution_actions=["collect_missing_evidence"],
        financial_resolution={"currency": "BRL", "recommended_refund_brl": 0, "refund_lines": []},
    )
    output["assessment"] = {
        "primary_issue": "insufficient_evidence",
        "case_status": "needs_investigation",
        "confidence": 0.3,
    }
    output["root_cause_analysis"]["responsible_parties"] = []
    assert verify(output).ok


def test_untraced_evidence_is_warning() -> None:
    report = verify(good_output(), consumed_refs=[REF_ORDER])
    assert report.ok
    assert "evidence_not_traced" in report.codes()


# --------------------------------------------------------------------------- entities & claims


def test_entity_not_in_evidence_is_error() -> None:
    output = good_output()
    output["affected_entities"]["seller_ids"].append(POLICY_EXAMPLE_SELLER)
    assert "entity_out_of_scope" in verify(output).codes()


def test_unverified_payment_reference_is_only_warning() -> None:
    output = good_output()
    output["affected_entities"]["payment_references"] = ["pay-guess"]
    report = verify(output)
    assert report.ok
    assert "entity_unverified" in report.codes()


def test_unknown_or_duplicate_claim_is_error() -> None:
    output = good_output()
    claim = output["claim_assessments"][0]
    output["claim_assessments"] = [claim, copy.deepcopy(claim), {**claim, "claim_id": "x"}]
    codes = verify(output).codes()
    assert "duplicate_claim_id" in codes
    assert "unknown_claim_id" in codes


def test_supported_claim_needs_evidence() -> None:
    output = good_output()
    output["claim_assessments"][0]["evidence_refs"] = []
    assert "claim_without_evidence" in verify(output).codes()


# --------------------------------------------------------------------------- money & consistency


def test_refund_lines_must_sum_to_recommended() -> None:
    output = good_output()
    output["financial_resolution"]["recommended_refund_brl"] = 89.0
    assert "refund_total_mismatch" in verify(output).codes()


def test_refund_cannot_exceed_amount_paid() -> None:
    output = good_output()
    output["financial_resolution"]["recommended_refund_brl"] = 120.0
    output["financial_resolution"]["refund_lines"][0]["amount_brl"] = 120.0
    assert "refund_exceeds_paid" in verify(output).codes()


def test_no_action_cannot_refund() -> None:
    output = good_output()
    output["assessment"]["case_status"] = "no_action"
    codes = verify(output).codes()
    assert "no_action_with_refund" in codes
    assert "no_action_with_refund_action" in codes
    assert "refund_requires_action" in codes


def test_no_action_issue_cannot_refund() -> None:
    output = good_output()
    output["assessment"]["primary_issue"] = "valid_split_payment"
    assert "no_action_issue_with_refund" in verify(output).codes()


def test_insufficient_evidence_requires_investigation_status() -> None:
    output = good_output()
    output["assessment"]["primary_issue"] = "insufficient_evidence"
    assert "insufficient_evidence_status" in verify(output).codes()


def test_duplicate_actions_ignore_case() -> None:
    output = good_output()
    output["resolution_actions"] = ["issue_refund", "Issue_Refund"]
    assert "duplicate_actions" in verify(output).codes()


def test_seller_party_must_be_affected_seller() -> None:
    output = good_output()
    output["root_cause_analysis"]["responsible_parties"] = [
        {"party_type": "seller", "party_id": POLICY_EXAMPLE_SELLER},
        {"party_type": "seller", "party_id": None},
    ]
    codes = verify(output).codes()
    assert "seller_party_not_affected" in codes
    assert "seller_party_without_id" in codes


def test_cause_ranks_must_be_contiguous() -> None:
    output = good_output()
    output["root_cause_analysis"]["ranked_causes"] = [
        {"cause_code": "A_CAUSE", "rank": 1},
        {"cause_code": "B_CAUSE", "rank": 3},
    ]
    assert "invalid_cause_ranks" in verify(output).codes()


def test_conflict_selected_source_must_be_listed() -> None:
    output = good_output()
    output["data_conflicts"] = [
        {
            "field": "order_status",
            "sources": ["get_order", "get_shipment_summary"],
            "selected_source": "customer_message",
            "resolution_code": "PREFER_ORDER_SYSTEM",
        }
    ]
    assert "conflict_source_unknown" in verify(output).codes()


def test_policy_mismatch_is_warning_not_error() -> None:
    output = good_output()
    output["resolution_actions"] = ["issue_refund_manually"]
    output["root_cause_analysis"]["responsible_parties"] = [
        {"party_type": "payment_provider", "party_id": None}
    ]
    report = verify(output)
    assert report.ok
    assert {"policy_action_missing", "policy_party_mismatch"} <= set(report.codes())


# --------------------------------------------------------------------------- calibration & repair


def test_calibrate_confidence_caps() -> None:
    assert calibrate_confidence(1.0, evidence=EVIDENCE) == 0.95
    assert calibrate_confidence(0.0, evidence=EVIDENCE) == 0.05
    assert calibrate_confidence(0.9, evidence=EVIDENCE[:1]) == 0.6
    assert calibrate_confidence(0.9, evidence=EVIDENCE, has_conflicts=True) == 0.8
    assert calibrate_confidence(0.9, evidence=EVIDENCE, has_errors=True) == 0.3


def test_repair_fixes_safe_problems_without_inventing(contracts: Contracts) -> None:
    broken = good_output()
    broken["evidence_refs"] = [REF_ORDER, REF_ORDER, REF_FOREIGN]
    broken["claim_assessments"][0]["evidence_refs"] = [REF_PAYMENTS, REF_FOREIGN]
    broken["affected_entities"]["seller_ids"].append(POLICY_EXAMPLE_SELLER)
    broken["financial_resolution"]["recommended_refund_brl"] = 50.0
    broken["resolution_actions"] = ["issue_refund", "ISSUE_REFUND"]
    broken["root_cause_analysis"]["ranked_causes"] = [
        {"cause_code": "B_CAUSE", "rank": 4},
        {"cause_code": "A_CAUSE", "rank": 2},
    ]
    broken["assessment"]["confidence"] = 1.0
    original = copy.deepcopy(broken)

    fixed = repair_output(broken, case=CASE, evidence=EVIDENCE)

    assert broken == original, "repair must not mutate its input"
    assert fixed["evidence_refs"] == [REF_ORDER, REF_PAYMENTS]
    assert fixed["claim_assessments"][0]["evidence_refs"] == [REF_PAYMENTS]
    assert fixed["affected_entities"]["seller_ids"] == [SELLER_ID]
    assert fixed["financial_resolution"]["recommended_refund_brl"] == 79.0
    assert fixed["resolution_actions"] == ["issue_refund"]
    assert [c["cause_code"] for c in fixed["root_cause_analysis"]["ranked_causes"]] == [
        "A_CAUSE",
        "B_CAUSE",
    ]
    assert [c["rank"] for c in fixed["root_cause_analysis"]["ranked_causes"]] == [1, 2]
    assert fixed["assessment"]["confidence"] == 0.95
    assert verify(fixed, contracts).ok


def test_repair_caps_refund_at_amount_paid() -> None:
    output = good_output()
    output["financial_resolution"]["recommended_refund_brl"] = 500.0
    output["financial_resolution"]["refund_lines"] = []
    fixed = repair_output(output, case=CASE, evidence=EVIDENCE)
    assert fixed["financial_resolution"]["recommended_refund_brl"] == 97.0


# --------------------------------------------------------------------------- trace


def test_emit_verification_writes_valid_trace_event(tmp_path: Path, contracts: Contracts) -> None:
    trace = TraceWriter(tmp_path / "trace.jsonl", contracts)
    output = good_output()
    output["case_id"] = "L3A_CASE_999"
    report = verify(output)
    event = emit_verification(trace, report, evidence_refs=output["evidence_refs"])

    assert event["event_type"] == "verification_completed"
    assert event["case_id"] == CASE_ID
    assert event["decision_code"] == "verification_failed"
    assert event["attributes"]["first_issue"] == "case_id_mismatch"
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    contracts.validate_trace(json.loads(lines[0]), "trace")


# --------------------------------------------------------------------------- end-to-end


class FakeGateway:
    """Offline MCP stand-in that serves the fixture evidence by tool name."""

    BY_TOOL = {
        "get_order": EVIDENCE[0],
        "get_order_items": EVIDENCE[1],
        "get_order_payments": EVIDENCE[2],
        "get_shipment_summary": EVIDENCE[3],
        "get_policy": EVIDENCE[4],
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def list_tools(self) -> list[str]:
        return sorted(self.BY_TOOL)

    async def call(self, tool_name: str, *, case_id: str, **_: Any) -> dict[str, Any]:
        self.calls.append((tool_name, case_id))
        if tool_name not in self.BY_TOOL:
            raise RuntimeError(f"MCP tool {tool_name} failed: not available offline")
        return copy.deepcopy(self.BY_TOOL[tool_name])


def test_solve_case_output_passes_verifier(tmp_path: Path, contracts: Contracts) -> None:
    from student_agent.workflow import solve_case

    gateway = FakeGateway()
    trace = TraceWriter(tmp_path / "trace.jsonl", contracts)
    try:
        output = asyncio.run(solve_case(copy.deepcopy(CASE), gateway, trace))  # type: ignore[arg-type]
    except NotImplementedError:
        pytest.skip("solve_case is not implemented on this branch yet")

    contracts.validate_output(output, "solve_case output")
    assert output["case_id"] == CASE_ID
    assert all(case_id == CASE_ID for _, case_id in gateway.calls)

    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    consumed = {
        ref
        for e in events
        if e["event_type"] == "tool_result_consumed"
        for ref in e.get("evidence_refs", [])
    }
    report = verify_output(
        output, case=CASE, evidence=EVIDENCE, contracts=contracts, consumed_refs=consumed
    )
    assert report.ok, [(i.code, i.message) for i in report.errors]
    assert "verification_completed" in {e["event_type"] for e in events}
