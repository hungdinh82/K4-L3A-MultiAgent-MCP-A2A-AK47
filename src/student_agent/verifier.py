"""Verifier agent: deterministic checks on a draft L3A output before finalize.

The verifier never calls MCP and never invents data. It receives the draft output, the
case input and the evidence envelopes collected for *this* case, then:

1. reports every violated invariant as a ``VerificationIssue``;
2. optionally applies safe, deterministic repairs (dedupe, drop unknown refs, fix refund
   totals, clamp confidence) via ``repair_output``;
3. emits one ``verification_completed`` trace event via ``emit_verification``.

Issue severity:
- ``error``: the output would hit a scorer hard gate or is internally inconsistent; do not
  finalize it unchanged.
- ``warning``: suspicious but not provably wrong (e.g. differs from the policy template).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .contracts import ContractError, Contracts
from .trace import TraceWriter

MONEY_TOLERANCE = 0.01
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.95
UNVERIFIED_CONFIDENCE_CAP = 0.3
CONFLICT_CONFIDENCE_CAP = 0.8
THIN_EVIDENCE_CONFIDENCE_CAP = 0.6

# Evidence data keys that identify case entities, per ``affected_entities`` field.
ENTITY_KEYS: dict[str, tuple[str, ...]] = {
    "order_ids": ("order_id",),
    "item_ids": ("order_item_id", "item_id"),
    "seller_ids": ("seller_id",),
    "payment_references": ("payment_reference", "payment_id", "payment_sequential"),
    "shipment_ids": ("shipment_id",),
}
# Entity fields whose ids must be proven by evidence. The MCP data does not always expose
# payment/shipment identifiers, so unknown ones there are only warnings.
STRICT_ENTITY_FIELDS = ("order_ids", "item_ids", "seller_ids")
# Policy evidence contains template/example ids that are not part of the case.
NON_ENTITY_DOMAINS = frozenset({"policy"})

REFUND_ACTION_WORDS = ("refund",)
NO_ACTION_ISSUES = frozenset({"unsupported_claim", "valid_split_payment"})


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    severity: str  # "error" | "warning"
    field: str
    message: str


@dataclass
class VerificationReport:
    case_id: str
    issues: list[VerificationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[VerificationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[VerificationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def decision_code(self) -> str:
        if self.errors:
            return "verification_failed"
        if self.warnings:
            return "verified_with_warnings"
        return "verified"

    def codes(self) -> list[str]:
        return [issue.code for issue in self.issues]

    def _add(self, code: str, severity: str, field_name: str, message: str) -> None:
        self.issues.append(VerificationIssue(code, severity, field_name, message))

    def error(self, code: str, field_name: str, message: str) -> None:
        self._add(code, "error", field_name, message)

    def warning(self, code: str, field_name: str, message: str) -> None:
        self._add(code, "warning", field_name, message)


# --------------------------------------------------------------------------- helpers


def _walk_values(value: Any, keys: frozenset[str]) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and isinstance(item, str | int) and not isinstance(item, bool):
                yield str(item)
            else:
                yield from _walk_values(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item, keys)


def evidence_refs_of(evidence: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(record["evidence_ref"]) for record in evidence if "evidence_ref" in record}


def evidence_entity_scope(evidence: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Collect entity ids that MCP evidence (excluding policy templates) actually shows."""
    records = [r for r in evidence if r.get("domain") not in NON_ENTITY_DOMAINS]
    return {
        name: set(_walk_values([r.get("data") for r in records], frozenset(keys)))
        for name, keys in ENTITY_KEYS.items()
    }


def _to_money(value: Any) -> float | None:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount if math.isfinite(amount) else None


def total_paid_brl(evidence: Iterable[Mapping[str, Any]]) -> float | None:
    """Sum payment_value from payment evidence; ``None`` when no payment rows were seen.

    ``get_order_payments`` returns a list of rows and ``get_payment_timeline`` nests the same
    rows under ``payments``; use the first shape found to avoid double counting.
    """
    list_rows: list[Mapping[str, Any]] = []
    nested_rows: list[Mapping[str, Any]] = []
    for record in evidence:
        if record.get("domain") != "payment":
            continue
        data = record.get("data")
        if isinstance(data, list):
            list_rows = list_rows or [row for row in data if isinstance(row, Mapping)]
        elif isinstance(data, Mapping) and isinstance(data.get("payments"), list):
            nested_rows = nested_rows or [r for r in data["payments"] if isinstance(r, Mapping)]
    rows = list_rows or nested_rows
    if not rows:
        return None
    amounts = [_to_money(row.get("payment_value")) for row in rows]
    return round(sum(a for a in amounts if a is not None), 2)


def policy_rules_of(evidence: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    for record in evidence:
        data = record.get("data")
        if record.get("domain") == "policy" and isinstance(data, Mapping):
            rules = data.get("rules")
            if isinstance(rules, Mapping):
                return dict(rules)
    return {}


def _get(mapping: Any, *path: str, default: Any = None) -> Any:
    for key in path:
        if not isinstance(mapping, Mapping) or key not in mapping:
            return default
        mapping = mapping[key]
    return mapping


# --------------------------------------------------------------------------- checks


def _check_schema(
    output: Mapping[str, Any], contracts: Contracts | None, r: VerificationReport
) -> None:
    if contracts is None:
        return
    try:
        contracts.validate_output(output, "draft output")
    except ContractError as exc:
        r.error("schema_invalid", "$", str(exc))


def _check_case_scope(
    output: Mapping[str, Any], case: Mapping[str, Any], r: VerificationReport
) -> None:
    if output.get("case_id") != case.get("case_id"):
        r.error(
            "case_id_mismatch",
            "case_id",
            f"output case_id {output.get('case_id')!r} != input {case.get('case_id')!r}",
        )


def _check_evidence(
    output: Mapping[str, Any],
    known_refs: set[str],
    consumed_refs: set[str] | None,
    r: VerificationReport,
) -> None:
    top_refs = list(output.get("evidence_refs") or [])
    issue = _get(output, "assessment", "primary_issue")
    if not top_refs and issue != "insufficient_evidence":
        r.error(
            "missing_required_evidence",
            "evidence_refs",
            "a concrete primary_issue must cite at least one MCP evidence_ref",
        )
    for ref in top_refs:
        if ref not in known_refs:
            r.error("unknown_evidence_ref", "evidence_refs", f"{ref} was not returned by MCP")
        elif consumed_refs is not None and ref not in consumed_refs:
            r.warning(
                "evidence_not_traced",
                "evidence_refs",
                f"{ref} has no tool_result_consumed trace event",
            )
    top_set = set(top_refs)
    for index, claim in enumerate(output.get("claim_assessments") or []):
        for ref in claim.get("evidence_refs") or []:
            path = f"claim_assessments.{index}.evidence_refs"
            if ref not in known_refs:
                r.error("unknown_evidence_ref", path, f"{ref} was not returned by MCP")
            elif ref not in top_set:
                r.warning("claim_ref_not_in_output", path, f"{ref} missing from evidence_refs")


def _check_claims(
    output: Mapping[str, Any], case: Mapping[str, Any], r: VerificationReport
) -> None:
    claims = _get(case, "customer_request", "claims", default=[]) or []
    case_claim_ids = {c.get("claim_id") for c in claims if isinstance(c, Mapping)}
    seen: set[str] = set()
    for index, claim in enumerate(output.get("claim_assessments") or []):
        claim_id = claim.get("claim_id")
        path = f"claim_assessments.{index}.claim_id"
        if case_claim_ids and claim_id not in case_claim_ids:
            r.error("unknown_claim_id", path, f"{claim_id!r} is not a claim of this case")
        if claim_id in seen:
            r.error("duplicate_claim_id", path, f"{claim_id!r} assessed more than once")
        seen.add(claim_id)
        verdict = claim.get("verdict")
        if verdict in {"supported", "partially_supported"} and not claim.get("evidence_refs"):
            r.error(
                "claim_without_evidence",
                f"claim_assessments.{index}.evidence_refs",
                f"verdict {verdict!r} needs at least one evidence_ref",
            )


def _check_entities(
    output: Mapping[str, Any],
    case: Mapping[str, Any],
    scope: Mapping[str, set[str]],
    r: VerificationReport,
) -> None:
    entities = output.get("affected_entities") or {}
    for name in ENTITY_KEYS:
        for entity_id in entities.get(name) or []:
            if entity_id in scope.get(name, set()):
                continue
            message = f"{entity_id!r} does not appear in this case's MCP evidence"
            if name in STRICT_ENTITY_FIELDS:
                r.error("entity_out_of_scope", f"affected_entities.{name}", message)
            else:
                r.warning("entity_unverified", f"affected_entities.{name}", message)
    claimed = _get(case, "customer_request", "claimed_order_id")
    order_ids = entities.get("order_ids") or []
    if claimed and order_ids and claimed not in order_ids:
        r.warning(
            "claimed_order_not_affected",
            "affected_entities.order_ids",
            f"claimed order {claimed!r} is not listed; confirm this is intentional",
        )
    issue = _get(output, "assessment", "primary_issue")
    if issue not in {"insufficient_evidence", None} and not order_ids:
        r.warning("no_affected_order", "affected_entities.order_ids", "no order id listed")


def _check_money(
    output: Mapping[str, Any],
    scope: Mapping[str, set[str]],
    paid: float | None,
    r: VerificationReport,
) -> None:
    financial = output.get("financial_resolution") or {}
    recommended = _to_money(financial.get("recommended_refund_brl"))
    lines = financial.get("refund_lines") or []
    line_total = round(sum(_to_money(line.get("amount_brl")) or 0.0 for line in lines), 2)
    if recommended is None:
        return
    if lines and abs(line_total - recommended) > MONEY_TOLERANCE:
        r.error(
            "refund_total_mismatch",
            "financial_resolution",
            f"recommended {recommended:.2f} != sum of refund_lines {line_total:.2f}",
        )
    if recommended > MONEY_TOLERANCE and not lines:
        r.warning("refund_without_lines", "financial_resolution.refund_lines", "no breakdown")
    if paid is not None and recommended > paid + MONEY_TOLERANCE:
        r.error(
            "refund_exceeds_paid",
            "financial_resolution.recommended_refund_brl",
            f"refund {recommended:.2f} exceeds total paid {paid:.2f}",
        )
    all_ids = set().union(*scope.values()) if scope else set()
    for index, line in enumerate(lines):
        entity_id = line.get("entity_id")
        if entity_id is not None and entity_id not in all_ids:
            r.warning(
                "refund_line_entity_unverified",
                f"financial_resolution.refund_lines.{index}.entity_id",
                f"{entity_id!r} is not an entity seen in evidence",
            )


def _check_consistency(output: Mapping[str, Any], r: VerificationReport) -> None:
    assessment = output.get("assessment") or {}
    issue = assessment.get("primary_issue")
    status = assessment.get("case_status")
    refund = _to_money(_get(output, "financial_resolution", "recommended_refund_brl")) or 0.0
    actions = [str(a) for a in output.get("resolution_actions") or []]

    if status == "no_action" and refund > MONEY_TOLERANCE:
        r.error("no_action_with_refund", "assessment.case_status", "no_action but refund > 0")
    if status == "no_action" and any(w in a for a in actions for w in REFUND_ACTION_WORDS):
        r.error("no_action_with_refund_action", "resolution_actions", "no_action but refunds")
    if refund > MONEY_TOLERANCE and status != "action_required":
        r.error("refund_requires_action", "assessment.case_status", "refund needs action_required")
    if issue == "insufficient_evidence" and status != "needs_investigation":
        r.error(
            "insufficient_evidence_status",
            "assessment.case_status",
            "insufficient_evidence must be needs_investigation",
        )
    if issue in NO_ACTION_ISSUES and refund > MONEY_TOLERANCE:
        r.error("no_action_issue_with_refund", "financial_resolution", f"{issue} refunds money")
    if status == "action_required" and not actions:
        r.error("action_required_without_action", "resolution_actions", "no action listed")

    normalized = [a.strip().lower() for a in actions]
    if len(set(normalized)) != len(normalized):
        r.error("duplicate_actions", "resolution_actions", "duplicate resolution actions")

    root = output.get("root_cause_analysis") or {}
    ranks = [cause.get("rank") for cause in root.get("ranked_causes") or []]
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        r.error("invalid_cause_ranks", "root_cause_analysis.ranked_causes", f"ranks {ranks}")
    seller_ids = set(_get(output, "affected_entities", "seller_ids", default=[]) or [])
    for index, party in enumerate(root.get("responsible_parties") or []):
        path = f"root_cause_analysis.responsible_parties.{index}"
        if party.get("party_type") == "seller":
            party_id = party.get("party_id")
            if not party_id:
                r.error("seller_party_without_id", path, "seller responsibility needs party_id")
            elif party_id not in seller_ids:
                r.error("seller_party_not_affected", path, f"{party_id!r} not in seller_ids")

    for index, conflict in enumerate(output.get("data_conflicts") or []):
        selected = conflict.get("selected_source")
        if selected is not None and selected not in (conflict.get("sources") or []):
            r.error(
                "conflict_source_unknown",
                f"data_conflicts.{index}.selected_source",
                f"{selected!r} is not one of the listed sources",
            )


def _check_policy(
    output: Mapping[str, Any],
    rules: Mapping[str, Mapping[str, Any]],
    r: VerificationReport,
) -> None:
    issue = _get(output, "assessment", "primary_issue")
    rule = rules.get(issue) if issue else None
    if not rule:
        return
    status = _get(output, "assessment", "case_status")
    if rule.get("case_status") and status != rule["case_status"]:
        r.warning(
            "policy_status_mismatch",
            "assessment.case_status",
            f"policy says {rule['case_status']!r} for {issue}, output has {status!r}",
        )
    action = rule.get("recommended_action")
    if action and action not in (output.get("resolution_actions") or []):
        r.warning("policy_action_missing", "resolution_actions", f"policy expects {action!r}")
    expected_types = {p.get("party_type") for p in rule.get("responsible_parties") or []}
    actual_types = {
        p.get("party_type")
        for p in _get(output, "root_cause_analysis", "responsible_parties", default=[]) or []
    }
    if expected_types and actual_types and not expected_types & actual_types:
        r.warning(
            "policy_party_mismatch",
            "root_cause_analysis.responsible_parties",
            f"policy expects {sorted(expected_types)}, output has {sorted(actual_types)}",
        )


def _check_confidence(
    output: Mapping[str, Any], evidence: list[Mapping[str, Any]], r: VerificationReport
) -> None:
    confidence = _to_money(_get(output, "assessment", "confidence"))
    if confidence is None:
        return
    if confidence > MAX_CONFIDENCE:
        r.warning("overconfident", "assessment.confidence", f"{confidence} > {MAX_CONFIDENCE}")
    domains = {rec.get("domain") for rec in evidence} - NON_ENTITY_DOMAINS
    if len(domains) < 2 and confidence > THIN_EVIDENCE_CONFIDENCE_CAP:
        r.warning(
            "confidence_thin_evidence",
            "assessment.confidence",
            f"{confidence} with evidence from {len(domains)} domain(s)",
        )
    if output.get("data_conflicts") and confidence > CONFLICT_CONFIDENCE_CAP:
        r.warning("confidence_with_conflicts", "assessment.confidence", "unresolved conflicts")


# --------------------------------------------------------------------------- public API


def verify_output(
    output: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    evidence: Iterable[Mapping[str, Any]],
    contracts: Contracts | None = None,
    consumed_refs: Iterable[str] | None = None,
    policy_rules: Mapping[str, Mapping[str, Any]] | None = None,
) -> VerificationReport:
    """Check a draft output against the case, its evidence and the public contract.

    ``evidence`` must be the MCP envelopes collected for this case only. ``consumed_refs``
    (refs already emitted in ``tool_result_consumed`` events) enables the trace-linkage
    check. ``policy_rules`` defaults to the rules found in the policy evidence.
    """
    records = list(evidence)
    report = VerificationReport(case_id=str(case.get("case_id")))
    scope = evidence_entity_scope(records)
    rules = dict(policy_rules) if policy_rules is not None else policy_rules_of(records)

    _check_schema(output, contracts, report)
    _check_case_scope(output, case, report)
    _check_evidence(
        output,
        evidence_refs_of(records),
        set(consumed_refs) if consumed_refs is not None else None,
        report,
    )
    _check_claims(output, case, report)
    _check_entities(output, case, scope, report)
    _check_money(output, scope, total_paid_brl(records), report)
    _check_consistency(output, report)
    _check_policy(output, rules, report)
    _check_confidence(output, records, report)
    return report


def calibrate_confidence(
    confidence: float,
    *,
    evidence: Iterable[Mapping[str, Any]],
    has_conflicts: bool = False,
    has_errors: bool = False,
) -> float:
    """Clamp confidence to what the collected evidence can justify."""
    domains = {rec.get("domain") for rec in evidence} - NON_ENTITY_DOMAINS
    value = min(max(float(confidence), MIN_CONFIDENCE), MAX_CONFIDENCE)
    if len(domains) < 2:
        value = min(value, THIN_EVIDENCE_CONFIDENCE_CAP)
    if has_conflicts:
        value = min(value, CONFLICT_CONFIDENCE_CAP)
    if has_errors:
        value = min(value, UNVERIFIED_CONFIDENCE_CAP)
    return round(value, 2)


def repair_output(
    output: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    evidence: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply safe deterministic fixes. Never adds evidence, entities or new conclusions.

    - drop evidence refs that MCP never returned (output and claims), dedupe the rest;
    - add claim refs that are valid but missing from top-level ``evidence_refs``;
    - drop strict entity ids that evidence does not show;
    - set ``recommended_refund_brl`` to the refund line total, capped at the amount paid;
    - dedupe resolution actions and renumber cause ranks 1..n;
    - clamp confidence to what the evidence supports.
    """
    records = list(evidence)
    known = evidence_refs_of(records)
    scope = evidence_entity_scope(records)
    fixed: dict[str, Any] = copy.deepcopy(dict(output))
    fixed["case_id"] = case.get("case_id", fixed.get("case_id"))

    refs = [ref for ref in dict.fromkeys(fixed.get("evidence_refs") or []) if ref in known]
    for claim in fixed.get("claim_assessments") or []:
        claim_refs = [
            ref for ref in dict.fromkeys(claim.get("evidence_refs") or []) if ref in known
        ]
        claim["evidence_refs"] = claim_refs
        refs.extend(ref for ref in claim_refs if ref not in refs)
    fixed["evidence_refs"] = refs

    entities = fixed.get("affected_entities") or {}
    for name in STRICT_ENTITY_FIELDS:
        if name in entities:
            entities[name] = [e for e in dict.fromkeys(entities[name]) if e in scope[name]]

    financial = fixed.get("financial_resolution")
    if isinstance(financial, dict):
        lines = financial.get("refund_lines") or []
        if lines:
            total = round(sum(_to_money(line.get("amount_brl")) or 0.0 for line in lines), 2)
            financial["recommended_refund_brl"] = total
        paid = total_paid_brl(records)
        recommended = _to_money(financial.get("recommended_refund_brl"))
        if paid is not None and recommended is not None and recommended > paid:
            financial["recommended_refund_brl"] = paid

    actions = fixed.get("resolution_actions")
    if isinstance(actions, list):
        seen: set[str] = set()
        unique: list[str] = []
        for action in actions:
            key = str(action).strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(action)
        fixed["resolution_actions"] = unique

    causes = _get(fixed, "root_cause_analysis", "ranked_causes")
    if isinstance(causes, list):
        causes.sort(key=lambda cause: cause.get("rank", 99))
        for rank, cause in enumerate(causes, 1):
            cause["rank"] = rank

    assessment = fixed.get("assessment")
    if isinstance(assessment, dict) and _to_money(assessment.get("confidence")) is not None:
        remaining = verify_output(fixed, case=case, evidence=records)
        assessment["confidence"] = calibrate_confidence(
            assessment["confidence"],
            evidence=records,
            has_conflicts=bool(fixed.get("data_conflicts")),
            has_errors=not remaining.ok,
        )
    return fixed


def emit_verification(
    trace: TraceWriter,
    report: VerificationReport,
    *,
    evidence_refs: Iterable[str] = (),
    actor: str = "verifier-agent",
    target: str | None = "coordinator",
) -> dict[str, Any]:
    """Emit the observable ``verification_completed`` event (codes only, no reasoning)."""
    return trace.emit(
        case_id=report.case_id,
        event_type="verification_completed",
        actor=actor,
        target=target,
        decision_code=report.decision_code,
        evidence_refs=list(dict.fromkeys(evidence_refs))[:20] or None,
        attributes={
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "first_issue": report.issues[0].code if report.issues else None,
        },
    )
