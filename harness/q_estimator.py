import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .budget_allocator import DEFAULT_ALLOW_THRESHOLD


@dataclass(frozen=True)
class QEstimate:
    sample_count: int
    changes: int
    q: float


def estimate_q_table(
    log_path: str | Path,
    group_by: str = "tool_name",
    baseline_threshold: float = DEFAULT_ALLOW_THRESHOLD,
    min_samples: int = 1,
) -> dict[str, float]:
    """
    从 dev JSONL 日志估计 q = P(reviewer 改变自动决定)。

    日志至少需要：
    - decision: REQUIRE_APPROVAL
    - risk_score: 0 到 1
    - reviewer_verdict: ALLOW/APPROVE 或 DENY
    - tool_name: 默认分组字段

    reviewer_success 为 false 的记录会被跳过：失败时的 DENY 是 fail-closed
    默认值，不是 reviewer 的真实判断。
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
    baseline_threshold: float = DEFAULT_ALLOW_THRESHOLD,
) -> dict[str, QEstimate]:
    """返回每组样本数、改变次数和 q，便于报告检查。"""

    return _collect_q_estimates(
        _read_jsonl(log_path),
        group_by=group_by,
        baseline_threshold=baseline_threshold,
    )


def load_q_table(path: str | Path) -> dict[str, float]:
    """读取 build_q_table.py 输出的 JSON，并校验每个 q 在 0 到 1 之间。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("q table must be a JSON object of tool_name -> q")

    q_table: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"q for {key!r} must be a number")
        if not 0 <= value <= 1:
            raise ValueError(f"q for {key!r} must be between 0 and 1")
        q_table[str(key)] = float(value)
    return q_table


def _read_jsonl(log_path: str | Path) -> Iterable[dict[str, Any]]:
    # utf-8-sig 兼容 Windows PowerShell 5.1 写入的 BOM。
    with Path(log_path).open("r", encoding="utf-8-sig") as file:
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
        # 旧日志没有 reviewer_success 字段，缺失时视为成功。
        if record.get("reviewer_success", True) is False:
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
