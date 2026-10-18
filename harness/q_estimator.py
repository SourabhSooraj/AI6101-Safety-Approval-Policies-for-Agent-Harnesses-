import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

from .budget_allocator import DEFAULT_ALLOW_THRESHOLD


LogPaths = str | Path | Sequence[str | Path]


@dataclass(frozen=True)
class QEstimate:
    sample_count: int
    changes: int
    q: float


def estimate_q_table(
    log_path: LogPaths,
    group_by: str = "tool_name",
    baseline_threshold: float = DEFAULT_ALLOW_THRESHOLD,
    min_samples: int = 1,
    allow_mixed_models: bool = False,
) -> dict[str, float]:
    """
    从 dev JSONL 日志估计 q = P(reviewer 改变自动决定)。

    日志至少需要：
    - decision: REQUIRE_APPROVAL
    - risk_score: 0 到 1
    - reviewer_verdict: ALLOW/APPROVE 或 DENY
    - tool_name: 默认分组字段

    reviewer_success 为 false 的记录会被跳过：失败时的 DENY 是 fail-closed
    默认值，不是 reviewer 的真实判断。stage 不是 p6 的记录也会被跳过。

    log_path 可以是一个或多个 JSONL 文件。q 依赖 Judge 和 reviewer 模型，
    计入的记录来自多个 (judge_model, reviewer_model) 组合时默认报错，
    避免本地开发日志混入 DeepSeek 的 q table。
    """

    grouped = _collect_q_estimates(
        _read_jsonl_files(log_path),
        group_by=group_by,
        baseline_threshold=baseline_threshold,
        allow_mixed_models=allow_mixed_models,
    )
    return {
        key: estimate.q
        for key, estimate in grouped.items()
        if estimate.sample_count >= min_samples
    }


def estimate_q_details(
    log_path: LogPaths,
    group_by: str = "tool_name",
    baseline_threshold: float = DEFAULT_ALLOW_THRESHOLD,
    allow_mixed_models: bool = False,
) -> dict[str, QEstimate]:
    """返回每组样本数、改变次数和 q，便于报告检查。"""

    return _collect_q_estimates(
        _read_jsonl_files(log_path),
        group_by=group_by,
        baseline_threshold=baseline_threshold,
        allow_mixed_models=allow_mixed_models,
    )


def models_in_logs(log_path: LogPaths) -> set[tuple[Any, Any]]:
    """返回日志中出现的 (judge_model, reviewer_model) 组合。"""

    return {_model_key(record) for record in _read_jsonl_files(log_path)}


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


def _read_jsonl_files(log_path: LogPaths) -> Iterable[dict[str, Any]]:
    paths = (
        [log_path] if isinstance(log_path, (str, Path)) else list(log_path)
    )
    for path in paths:
        yield from _read_jsonl(path)


def _model_key(record: dict[str, Any]) -> tuple[Any, Any]:
    # 旧日志没有这两个字段，记为 None。
    return record.get("judge_model"), record.get("reviewer_model")


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
                    f"Invalid JSON on line {line_number} of {log_path}"
                ) from error


def _collect_q_estimates(
    records: Iterable[dict[str, Any]],
    group_by: str,
    baseline_threshold: float,
    allow_mixed_models: bool = False,
) -> dict[str, QEstimate]:
    counts: dict[str, int] = {}
    changes: dict[str, int] = {}
    model_keys: set[tuple[Any, Any]] = set()

    for record in records:
        if str(record.get("decision", "")).upper() != "REQUIRE_APPROVAL":
            continue
        # 旧日志没有 reviewer_success 字段，缺失时视为成功。
        if record.get("reviewer_success", True) is False:
            continue
        # q 衡量 reviewer 改变 P6 自动决定的概率；baseline 或 P1-only 的
        # risk_score 是固定值，不是 Judge 分数，不能混入。
        if record.get("stage", "p6") != "p6":
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

        model_keys.add(_model_key(record))
        counts[group_key] = counts.get(group_key, 0) + 1
        if reviewer_decision != baseline_decision:
            changes[group_key] = changes.get(group_key, 0) + 1

    if len(model_keys) > 1 and not allow_mixed_models:
        found = ", ".join(
            f"judge={judge} reviewer={reviewer}"
            for judge, reviewer in sorted(model_keys, key=str)
        )
        raise ValueError(
            "Logs mix several Judge/reviewer models; q depends on both, so "
            f"build one table per pair ({found}). "
            "Pass allow_mixed_models=True to override."
        )

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
