"""Structured role outputs: fail explicitly instead of silently inventing fields."""
from __future__ import annotations

import json

from ..schema import VERDICTS


ROLE_FIELDS = {
    "evidence": {"claim_id": str, "evidence_id": str, "assessment": str,
                 "observed_facts": list, "evidence_references": list,
                 "unresolved_assertions": list, "rationale": str},
    "aggregator": {"provisional_verdict": str, "assertion_evidence_map": list,
                   "supporting_observations": list, "contradictory_observations": list,
                   "unresolved_conflicts": list, "explanation": str},
    "counterfactual": {"target_premise": str, "observed_evidence_references": list,
                       "counterfactual_scenarios": list, "dependency_analysis": str,
                       "observed_evidence_recommendation": str, "unresolved_issues": list},
    "standard": {"supported_steps": list, "unsupported_steps": list, "contradictions": list,
                 "counterfactual_audit": list, "verdict_recommendation": str,
                 "evidence_references": list, "required_corrections": list},
    "judge": {"verdict": str, "decisive_evidence_references": list, "arbitration_rationale": str,
              "accepted_corrections": list, "unresolved_issues": list,
              "consensus_reached": bool, "termination_reason": str},
    "rewriter": {"verdict": str, "explanation": str, "evidence_references": list, "limitations": list},
    "direct": {"verdict": str, "explanation": str},
    "retrieval_planner": {"sufficient": bool, "queries": list, "unresolved_assertions": list,
                          "evidence_references": list},
    "one_shot": {"verdict": str, "explanation": str, "evidence_references": list, "limitations": list},
}
VERDICT_FIELDS = {"assessment", "provisional_verdict", "observed_evidence_recommendation",
                  "verdict_recommendation", "verdict"}
REFERENCE_FIELDS = {"evidence_references", "observed_evidence_references", "decisive_evidence_references"}


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if lines[-1].strip() == "```" else text
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Agent must return one JSON object")
    return value


def validate_role(role: str, record: dict, permitted_ids: set[str], context: dict):
    for key, expected in ROLE_FIELDS[role].items():
        if key not in record or not isinstance(record[key], expected):
            raise ValueError(f"Role {role} requires field {key} of type {expected.__name__}")
    for key in VERDICT_FIELDS & record.keys():
        if record[key] not in VERDICTS:
            raise ValueError(f"Invalid canonical verdict in {key}: {record[key]}")
    _validate_references(record, permitted_ids)
    if role == "evidence":
        if record["claim_id"] != context["claim"]["claim_id"] or record["evidence_id"] != context["assigned_evidence_id"]:
            raise ValueError("Evidence-specific agent changed its assigned identity")
        if context.get("previous_assessment") is not None:
            for key in ("previous_assessment", "revision_reason"):
                if key not in record or not isinstance(record[key], str):
                    raise ValueError(f"Later-round assessment requires {key}")
            if record["previous_assessment"] != context["previous_assessment"]:
                raise ValueError("Previous assessment must match controller record")
    if role == "counterfactual":
        identifiers = set()
        for scenario in record["counterfactual_scenarios"]:
            if not isinstance(scenario, dict):
                raise ValueError("Counterfactual scenarios must be objects")
            required = {"scenario_id": str, "hypothetical": bool, "original_premise": str,
                        "assumed_change": str, "fixed_observations": list, "effect_on_judgment": str}
            for key, kind in required.items():
                if not isinstance(scenario.get(key), kind):
                    raise ValueError(f"Counterfactual scenario requires {key}")
            if not scenario["scenario_id"].startswith("CF:") or scenario["hypothetical"] is not True:
                raise ValueError("Counterfactual IDs must start CF: and be explicitly hypothetical")
            if scenario["scenario_id"] in identifiers:
                raise ValueError("Duplicate counterfactual scenario ID")
            identifiers.add(scenario["scenario_id"])
    if role == "rewriter" and record["verdict"] != context["judgment"]["verdict"]:
        raise ValueError("Rewriter cannot change adjudicated verdict")
    if role == "judge":
        if record["consensus_reached"] != context["controller_converged"]:
            raise ValueError("Judge must apply the controller's documented stopping criterion")
        if record["termination_reason"] != context["controller_termination_reason"]:
            raise ValueError("Judge must preserve the controller termination reason")
    if role == "retrieval_planner" and any(not isinstance(q, str) or not q.strip() for q in record["queries"]):
        raise ValueError("Planner queries must be nonempty strings")
    return record


def _validate_references(value, permitted_ids):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in REFERENCE_FIELDS:
                if not isinstance(item, list) or any(not isinstance(x, str) or x not in permitted_ids for x in item):
                    raise ValueError(f"Unknown or hypothetical observed-evidence reference in {key}")
            if key == "evidence_id" and item not in permitted_ids:
                raise ValueError(f"Unknown evidence identity: {item}")
            _validate_references(item, permitted_ids)
    elif isinstance(value, list):
        for item in value:
            _validate_references(item, permitted_ids)
