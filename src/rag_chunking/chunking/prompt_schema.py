"""Strict boundary-plan schema for prompt-based chunking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


PLANNER_SCHEMA_VERSION = "prompt_boundary_plan_v1"


class PlanValidationError(ValueError):
    """A planner response is not a complete, contiguous block partition."""


@dataclass(frozen=True, slots=True)
class PlannerGroup:
    start_block_index: int
    end_block_index: int
    reason: str


@dataclass(frozen=True, slots=True)
class BoundaryPlan:
    groups: tuple[PlannerGroup, ...]


def planner_json_schema() -> dict[str, Any]:
    return {
        "name": PLANNER_SCHEMA_VERSION,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["groups"],
            "properties": {
                "groups": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["start_block_index", "end_block_index", "reason"],
                        "properties": {
                            "start_block_index": {"type": "integer"},
                            "end_block_index": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                    },
                }
            },
        },
    }


def parse_boundary_plan(raw: str | dict[str, Any], start: int, end: int) -> BoundaryPlan:
    """Parse a strict plan and prove every supplied block occurs exactly once."""

    try:
        value = json.loads(raw, object_pairs_hook=_unique_object) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, PlanValidationError) as error:
        raise PlanValidationError(f"malformed JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {"groups"}:
        raise PlanValidationError("response must be an object containing only 'groups'")
    groups_value = value["groups"]
    if not isinstance(groups_value, list) or not groups_value:
        raise PlanValidationError("groups must be a non-empty array")
    groups: list[PlannerGroup] = []
    expected = start
    for position, item in enumerate(groups_value):
        if not isinstance(item, dict) or set(item) != {
            "start_block_index", "end_block_index", "reason"
        }:
            raise PlanValidationError(f"group {position} has an invalid schema")
        group_start = item["start_block_index"]
        group_end = item["end_block_index"]
        reason = item["reason"]
        if type(group_start) is not int or type(group_end) is not int or not isinstance(reason, str):
            raise PlanValidationError(f"group {position} has invalid field types")
        if group_start != expected:
            raise PlanValidationError(
                f"group {position} must start at {expected}; received {group_start}"
            )
        if group_end < group_start or group_end > end:
            raise PlanValidationError(f"group {position} has an impossible end index {group_end}")
        groups.append(PlannerGroup(group_start, group_end, reason))
        expected = group_end + 1
    if expected != end + 1:
        raise PlanValidationError(f"plan ends at {expected - 1}; expected {end}")
    return BoundaryPlan(tuple(groups))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlanValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
