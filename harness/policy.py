from dataclasses import dataclass
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


@dataclass(frozen=True)
class PolicyRequest:
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    task_id: str


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str
    policy_name: str