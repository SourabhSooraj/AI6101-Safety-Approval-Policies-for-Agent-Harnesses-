import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class QEstimate:
    sample_count: int
    changes: int
    q: float


def estimate_q_table(
    log_path: str | Path,
    group_by: str = "tool_name",
    baseline_threshold: float = 0.5,
    min_samples: int = 1,
) -> dict[str, float]:
    """
    从 dev JSONL 日志估计 q = P(reviewer 改变自动决定)。

    日志至少需要：
    - decision: REQUIRE_APPROVAL
    - risk_score: 0 到 1
    - reviewer_verdict: ALLOW/APPROVE 或 DENY
    - tool_name: 默认分组字段
    """

    grouped = _collect_q_estimates(
        _read_jsonl(log_path),
        group_by=group_by,
        baseline_threshold=baseline_threshold,
    )
    return {
        key: estimate.q
        for key, estimate in grouped.items()
        if estimate.sample_count >= min_samples
    }


def estimate_q_details(
    log_path: str | Path,
    group_by: str = "tool_name",
    baseline_threshold: float = 0.5,
) -> dict[str, QEstimate]:
    """返回每组样本数、改变次数和 q，便于报告检查。"""

    return _collect_q_estimates(
        _read_jsonl(log_path),
        group_by=group_by,
        baseline_threshold=baseline_threshold,
    )


def _read_jsonl(log_path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(log_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on log line {line_number}"
                ) from error


def _collect_q_estimates(
    records: Iterable[dict[str, Any]],
    group_by: str,
    baseline_threshold: float,
) -> dict[str, QEstimate]:
    counts: dict[str, int] = {}
    changes: dict[str, int] = {}

    for record in records:
        if str(record.get("decision", "")).upper() != "REQUIRE_APPROVAL":
            continue

        reviewer_decision = _normalize_reviewer_verdict(
            record.get("reviewer_verdict")
        )
        risk_score = record.get("risk_score")
        if reviewer_decision is None or not isinstance(
            risk_score, (int, float)
        ):
            continue

        group_value = record.get(group_by, "unknown")
        group_key = str(group_value)
        baseline_decision = (
            "ALLOW" if float(risk_score) < baseline_threshold else "DENY"
        )

        counts[group_key] = counts.get(group_key, 0) + 1
        if reviewer_decision != baseline_decision:
            changes[group_key] = changes.get(group_key, 0) + 1

    return {
        group: QEstimate(
            sample_count=count,
            changes=changes.get(group, 0),
            q=changes.get(group, 0) / count,
        )
        for group, count in counts.items()
    }


def _normalize_reviewer_verdict(value: Any) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip().upper()
    if normalized in {"ALLOW", "APPROVE", "APPROVED"}:
        return "ALLOW"
    if normalized in {"DENY", "DENIED", "BLOCK", "BLOCKED"}:
        return "DENY"
    return None
