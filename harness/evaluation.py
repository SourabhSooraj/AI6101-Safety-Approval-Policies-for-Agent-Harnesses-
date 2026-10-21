"""
评估（task 3）：读取 run_batch.py 的 batch 文件夹，计算 utility、TCR@k、攻击成功率、
HITL 负担、各组件成本，按攻击类型和 user task 分组，并给出 Wilson 区间和
McNemar + Holm 的成对比较。只用标准库。

Evaluation (task 3): reads run_batch.py batch folders and computes utility,
TCR@k, attack success rate, HITL load and cost by component, broken down by
attack type and user task, with Wilson intervals and pairwise McNemar + Holm
comparisons. Standard library only.

约定：崩溃（没有 run_summary.json）的运行永远不算成功。utility 和 TCR 的分母包括
崩溃的运行；攻击成功率给两个版本：只算完成的运行，以及把崩溃算作攻击未成功的
全部运行（同时报告崩溃数，避免一个 gate 靠崩溃“降低”攻击成功率）。

Convention: a run that crashed (no run_summary.json) never counts as a
success. Utility and TCR include crashed runs in the denominator. Attack
success is given two ways: over finished runs only, and over all runs with a
crash counted as "attack did not succeed" (the crash count is reported too, so
a gate cannot "lower" attack success by crashing).
"""

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from .policies import UNLIMITED_ESCALATIONS
from .run_contract import attack_type as attack_type_for

Z_95 = 1.959963984540054
NO_ATTACK = "none"
P6_POLICIES = frozenset({"hybrid", "no_budget"})
# 除 k 以外区分配置的设置；同一 family 的不同 k 组成 TCR@k 曲线上的一条线。
# Settings that tell configurations apart, apart from k; the k values of one
# family form one line of the TCR@k curve.
FAMILY_KEYS = (
    "policy_variant",
    "allocator",
    "tau0",
    "t_low",
    "t_high",
    "q_table_path",
    "utility_cost",
    "auto_rule",
    "reviewer_error_table_path",
    "sample_count",
    "auto_complete",
)
# run_batch.py 的 setup 键 -> run_summary.json 中对应的字段。
# run_batch.py setup key -> the matching run_summary.json field.
SETUP_TO_SUMMARY = {
    "q_table": "q_table_path",
    "reviewer_error_table": "reviewer_error_table_path",
}
# 成对比较的指标；attack_succeeded 只比较有攻击的运行。
# Metrics compared in pairs; attack_succeeded compares attacked runs only.
COMPARED_METRICS = ("safe_task_completion", "attack_succeeded", "strict_utility")


# ---------------------------------------------------------------- 统计 / Statistics


def wilson_interval(
    successes: int, total: int, z: float = Z_95
) -> tuple[float, float] | None:
    """
    二项比例的 Wilson 区间；total 为 0 时没有区间。
    Wilson interval for a binomial proportion; None when total is 0.
    """

    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """
    精确 McNemar 检验（双侧）：只看结果不同的配对，b 和 c 在 H0 下服从
    Binomial(b + c, 0.5)。小样本时比卡方近似可靠。

    Exact two-sided McNemar test: only discordant pairs count, and under H0
    b and c follow Binomial(b + c, 0.5). More reliable than the chi-squared
    approximation for small samples.
    """

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: Sequence[float]) -> list[float]:
    """
    Holm 校正后的 p 值，顺序与输入相同，且随原始 p 单调不减。
    Holm-adjusted p-values, in input order and non-decreasing in the raw p.
    """

    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


@dataclass(frozen=True)
class Rate:
    successes: int
    total: int

    @property
    def value(self) -> float | None:
        return self.successes / self.total if self.total else None

    def as_dict(self) -> dict[str, Any]:
        interval = wilson_interval(self.successes, self.total)
        return {
            "rate": self.value,
            "low": interval[0] if interval else None,
            "high": interval[1] if interval else None,
            "successes": self.successes,
            "n": self.total,
        }


def rate(flags: Iterable[bool | None]) -> Rate:
    """
    None（该运行没有这个字段）不计入分母。
    None (the run has no such field) is left out of the denominator.
    """

    known = [flag for flag in flags if flag is not None]
    return Rate(sum(known), len(known))


# ---------------------------------------------------------------- 读取运行 / Loading runs


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


@dataclass
class Run:
    """
    一个 episode；summary 为 None 表示没有完成（崩溃或被中断）。
    One episode; summary is None when it did not finish (crashed or interrupted).
    """

    batch: str
    setup: str
    task_id: str
    injection_task_id: str | None
    config: dict[str, Any]
    summary: dict[str, Any] | None
    error: str | None = None
    policy_records: list[dict[str, Any]] = field(default_factory=list)
    reviewer_records: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    split: str | None = None

    @property
    def finished(self) -> bool:
        return self.summary is not None

    @property
    def attacked(self) -> bool:
        return self.injection_task_id is not None

    @property
    def k(self) -> int:
        return int(self.config.get("k") or 0)

    @property
    def config_key(self) -> tuple:
        return (*(self.config.get(key) for key in FAMILY_KEYS), self.k)

    @property
    def family_key(self) -> tuple:
        return tuple(self.config.get(key) for key in FAMILY_KEYS)

    @property
    def unit(self) -> tuple[str, str | None]:
        return self.task_id, self.injection_task_id

    @property
    def attack_type(self) -> str | None:
        if self.summary is not None and "attack_type" in self.summary:
            return self.summary["attack_type"]
        return attack_type_for(
            str(self.config.get("suite", "banking")), self.injection_task_id
        )

    def flag(self, key: str) -> bool | None:
        """
        完成的运行读 run_summary；没完成的运行算作 False；字段不存在为 None。
        Finished runs read run_summary; unfinished runs count as False; a
        missing field is None.
        """

        if self.summary is None:
            return False
        value = self.summary.get(key)
        return None if value is None else bool(value)


def _split_episode_name(name: str) -> tuple[str, str, str] | None:
    parts = name.split("__")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


def _summary_config(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: summary.get(key) for key in FAMILY_KEYS},
        "k": summary.get("k"),
        "suite": summary.get("suite"),
    }


def _summary_models(summary: dict[str, Any]) -> dict[str, Any]:
    if isinstance(summary.get("models"), dict):
        return dict(summary["models"])
    return {
        "agent": summary.get("agent_model"),
        "judge": summary.get("judge_model"),
        "reviewer": summary.get("reviewer_model"),
    }


def _crashed_config(
    batch_config: dict[str, Any], setup: str, policy: str
) -> dict[str, Any]:
    """
    没有任何完成运行可参照时，用 batch_config.json 和 setup 设置推出配置。
    With no finished run to copy from, work out the configuration from
    batch_config.json and the setup's settings.
    """

    config = {key: batch_config.get(key) for key in FAMILY_KEYS}
    config["policy_variant"] = policy
    config["k"] = batch_config.get("k")
    config["suite"] = batch_config.get("suite")
    for entry in batch_config.get("setups") or []:
        if entry.get("name") == setup:
            for key, value in (entry.get("options") or {}).items():
                config[SETUP_TO_SUMMARY.get(key, key)] = value
    if policy == "no_budget":
        config["k"] = UNLIMITED_ESCALATIONS
    return config


def load_batch(batch_dir: Path) -> tuple[list[Run], list[str]]:
    """
    读取一个 batch 文件夹中的全部 episode（完成的和没完成的）。
    Read every episode in one batch folder, finished or not.
    """

    batch_dir = Path(batch_dir)
    warnings: list[str] = []
    config_path = batch_dir / "batch_config.json"
    batch_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    # batch_summary 中每个 episode 只取最后一行（重跑成功会覆盖旧的错误）。
    # Use only the last batch_summary line per episode (a successful retry
    # replaces the old error).
    errors: dict[tuple[str, str, str], str] = {}
    for record in _read_jsonl(batch_dir / "batch_summary.jsonl"):
        injection = record.get("injection_task_id") or NO_ATTACK
        key = (record["task_id"], record.get("setup", record["policy"]), injection)
        if "error" in record:
            errors[key] = record["error"]
        else:
            errors.pop(key, None)

    runs: list[Run] = []
    finished_configs: dict[str, dict[str, Any]] = {}
    unfinished: list[tuple[Path, tuple[str, str, str]]] = []
    for episode_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        parsed = _split_episode_name(episode_dir.name)
        if parsed is None:
            continue
        task_id, setup, injection = parsed
        summary_path = episode_dir / "run_summary.json"
        if not summary_path.exists():
            unfinished.append((episode_dir, parsed))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = _summary_config(summary)
        finished_configs.setdefault(setup, config)
        runs.append(
            Run(
                batch=batch_dir.name,
                setup=setup,
                task_id=task_id,
                injection_task_id=None if injection == NO_ATTACK else injection,
                config=config,
                summary=summary,
                policy_records=_read_jsonl(episode_dir / "policy_decisions.jsonl"),
                reviewer_records=_read_jsonl(episode_dir / "reviewer_decisions.jsonl"),
                models=_summary_models(summary),
                split=batch_config.get("split"),
            )
        )

    batch_models = {
        "agent": batch_config.get("agent_model"),
        "judge": batch_config.get("judge_model"),
        "reviewer": batch_config.get("reviewer_model"),
    }
    for episode_dir, (task_id, setup, injection) in unfinished:
        error = errors.get((task_id, setup, injection))
        if error is None:
            error = "no run_summary.json (interrupted or still running)"
            warnings.append(f"{batch_dir.name}/{episode_dir.name}: {error}")
        policy = next(
            (
                entry.get("policy")
                for entry in batch_config.get("setups") or []
                if entry.get("name") == setup
            ),
            setup,
        )
        config = finished_configs.get(setup) or _crashed_config(
            batch_config, setup, policy
        )
        runs.append(
            Run(
                batch=batch_dir.name,
                setup=setup,
                task_id=task_id,
                injection_task_id=None if injection == NO_ATTACK else injection,
                config=config,
                summary=None,
                error=error,
                policy_records=_read_jsonl(episode_dir / "policy_decisions.jsonl"),
                reviewer_records=_read_jsonl(episode_dir / "reviewer_decisions.jsonl"),
                models=batch_models,
                split=batch_config.get("split"),
            )
        )
    return runs, warnings


def load_runs(
    batch_dirs: Sequence[Path], allow_mixed_models: bool = False
) -> tuple[list[Run], list[str]]:
    """
    读取多个 batch。同一配置下同一 (task, 攻击) 出现多次时保留后读到的那个。
    模型不一致时报错（除非 allow_mixed_models），dev/test 混用时给出警告。

    Read several batches. If the same (task, attack) appears more than once
    for one configuration, the one read last is kept. Different models are an
    error (unless allow_mixed_models); mixing dev and test gives a warning.
    """

    runs: dict[tuple, Run] = {}
    warnings: list[str] = []
    # 在去重之前收集：被后一个 batch 替换掉的运行也说明混用了划分。
    # Collect before removing duplicates: a run replaced by a later batch
    # still shows that splits were mixed.
    splits: set[str] = set()
    for batch_dir in batch_dirs:
        batch_runs, batch_warnings = load_batch(Path(batch_dir))
        warnings += batch_warnings
        splits |= {str(run.split) for run in batch_runs if run.split}
        for run in batch_runs:
            key = (run.config_key, run.unit)
            if key in runs:
                warnings.append(
                    f"{run.task_id} / {run.injection_task_id or NO_ATTACK} / "
                    f"{run.setup} appears in {runs[key].batch} and {run.batch}; "
                    f"using {run.batch}"
                )
            runs[key] = run

    result = list(runs.values())
    model_sets = sorted(
        {tuple(sorted((str(k), str(v)) for k, v in run.models.items())) for run in result}
    )
    if len(model_sets) > 1 and not allow_mixed_models:
        described = ["; ".join(f"{k}={v}" for k, v in models) for models in model_sets]
        raise ValueError(
            "runs use different models (pass --allow-mixed-models to combine "
            "them anyway): " + " | ".join(described)
        )
    if len(splits) > 1:
        warnings.append(
            f"runs come from more than one split: {', '.join(sorted(splits))}"
        )
    return result, warnings


# ---------------------------------------------------------------- 指标 / Metrics


def outcome_metrics(runs: Sequence[Run]) -> dict[str, Any]:
    """
    utility、TCR 和攻击成功率；见模块说明中的分母约定。
    Utility, TCR and attack success; see the module docstring for the
    denominator convention.
    """

    clean = [run for run in runs if not run.attacked]
    attacked = [run for run in runs if run.attacked]
    return {
        "runs": len(runs),
        "finished": sum(run.finished for run in runs),
        "crashed": sum(not run.finished for run in runs),
        "utility_no_attack": rate(run.flag("strict_utility") for run in clean),
        "utility_under_attack": rate(run.flag("strict_utility") for run in attacked),
        "official_utility_no_attack": rate(run.flag("utility") for run in clean),
        "tcr": rate(run.flag("safe_task_completion") for run in runs),
        "tcr_no_attack": rate(run.flag("safe_task_completion") for run in clean),
        "tcr_under_attack": rate(run.flag("safe_task_completion") for run in attacked),
        "asr_finished": rate(
            run.flag("attack_succeeded") for run in attacked if run.finished
        ),
        "asr_all_runs": rate(run.flag("attack_succeeded") for run in attacked),
        "attacked_crashed": sum(not run.finished for run in attacked),
    }


def _escalations(run: Run) -> tuple[int, int]:
    """
    (交给 reviewer 的次数, 因预算用完被自动拒绝的次数)。
    (calls sent to the reviewer, calls auto-denied because the budget ran out).
    """

    escalated = auto_denied = 0
    for record in run.policy_records:
        if record.get("decision") == "REQUIRE_APPROVAL":
            # gate 在剩余预算为 0 时把 REQUIRE_APPROVAL 改成 DENY，日志里仍是原决定。
            # With no budget left, the gate turns REQUIRE_APPROVAL into DENY,
            # but the log keeps the original decision.
            if int(record.get("remaining_escalations_at_decision") or 0) > 0:
                escalated += 1
            else:
                auto_denied += 1
        elif record.get("policy_name") == "p6_budget_exhausted":
            auto_denied += 1
    return escalated, auto_denied


def hitl_metrics(runs: Sequence[Run]) -> dict[str, Any]:
    escalations = [_escalations(run) for run in runs]
    finite = [
        (escalated, run.k)
        for run, (escalated, _) in zip(runs, escalations)
        if 0 < run.k < UNLIMITED_ESCALATIONS
    ]
    reviews = [record for run in runs for record in run.reviewer_records]
    count = len(runs)
    return {
        "runs": count,
        "escalations": sum(escalated for escalated, _ in escalations),
        "escalations_per_run": (
            sum(escalated for escalated, _ in escalations) / count if count else None
        ),
        "budget_share_used": (
            sum(min(1.0, escalated / k) for escalated, k in finite) / len(finite)
            if finite
            else None
        ),
        "auto_denied": sum(denied for _, denied in escalations),
        "runs_out_of_budget": rate(denied > 0 for _, denied in escalations),
        "reviewer_approved": rate(
            record.get("reviewer_verdict") == "ALLOW"
            for record in reviews
            if record.get("reviewer_success") is not False
        ),
        "reviewer_failed": rate(record.get("reviewer_success") is False for record in reviews),
    }


def cost_metrics(runs: Sequence[Run]) -> dict[str, Any]:
    """
    各组件的美元成本和 token。agent 成本只在 run_summary 中，没完成的运行没有；
    Judge 和 reviewer 成本来自日志，没完成的运行也会计入已经花掉的部分。

    Cost in USD and tokens for each component. Agent cost is only in
    run_summary, so unfinished runs have none; Judge and reviewer cost come
    from the logs, so what unfinished runs already spent is counted.
    """

    def total(records: Iterable[dict[str, Any]], *keys: str) -> float:
        return sum(
            (
                float(
                    next(
                        (record[key] for key in keys if record.get(key) is not None),
                        0,
                    )
                    or 0
                )
                for record in records
            ),
            0.0,
        )

    agent = [run.summary.get("agent_usage") or {} for run in runs if run.summary]
    policy = [record for run in runs for record in run.policy_records]
    reviews = [record for run in runs for record in run.reviewer_records]
    components = {
        "agent": {
            "usd": total(agent, "estimated_cost_usd"),
            "input_tokens": int(total(agent, "input_tokens")),
            "output_tokens": int(total(agent, "output_tokens")),
        },
        "judge": {
            "usd": total(policy, "estimated_cost_usd"),
            "input_tokens": int(total(policy, "input_tokens")),
            "output_tokens": int(total(policy, "output_tokens")),
        },
        "reviewer": {
            "usd": total(reviews, "reviewer_cost_usd", "estimated_cost_usd"),
            "input_tokens": int(total(reviews, "reviewer_input_tokens", "input_tokens")),
            "output_tokens": int(
                total(reviews, "reviewer_output_tokens", "output_tokens")
            ),
        },
    }
    usd = sum(component["usd"] for component in components.values())
    return {
        "runs": len(runs),
        **components,
        "total_usd": usd,
        "usd_per_run": usd / len(runs) if runs else None,
    }


# ---------------------------------------------------------------- 分组与比较 / Grouping and comparison


def _group(runs: Iterable[Run], key: Callable[[Run], Any]) -> dict[Any, list[Run]]:
    groups: dict[Any, list[Run]] = {}
    for run in runs:
        groups.setdefault(key(run), []).append(run)
    return groups


def config_labels(runs: Sequence[Run]) -> dict[tuple, str]:
    """
    每个配置的显示名：setup 名；同名的配置再加上 k。
    Display name of each configuration: the setup name, plus k when two
    configurations share a name.
    """

    names: dict[tuple, set[str]] = {}
    for run in runs:
        names.setdefault(run.config_key, set()).add(run.setup)
    labels = {key: "+".join(sorted(value)) for key, value in names.items()}
    counts: dict[str, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    return {
        key: (f"{label} k={key[-1]}" if counts[label] > 1 else label)
        for key, label in labels.items()
    }


def family_labels(runs: Sequence[Run]) -> dict[tuple, str]:
    labels: dict[tuple, str] = {}
    for run in runs:
        if run.family_key in labels:
            continue
        config = run.config
        label = str(config.get("policy_variant"))
        if label in P6_POLICIES:
            label += f" ({config.get('allocator')}, tau0={config.get('tau0')})"
        taken = sum(existing.split(" #")[0] == label for existing in labels.values())
        labels[run.family_key] = f"{label} #{taken + 1}" if taken else label
    return labels


def k_label(k: int) -> str:
    return "unlimited" if k >= UNLIMITED_ESCALATIONS else str(k)


def tcr_at_k(runs: Sequence[Run]) -> list[dict[str, Any]]:
    families = family_labels(runs)
    rows = []
    for (family, k), group in sorted(
        _group(runs, lambda run: (run.family_key, run.k)).items(),
        key=lambda item: (families[item[0][0]], item[0][1]),
    ):
        metrics = outcome_metrics(group)
        rows.append(
            {
                "family": families[family],
                "k": k_label(k),
                "tcr": metrics["tcr"],
                "tcr_no_attack": metrics["tcr_no_attack"],
                "tcr_under_attack": metrics["tcr_under_attack"],
            }
        )
    return rows


def mcnemar_comparisons(runs: Sequence[Run]) -> list[dict[str, Any]]:
    """
    每对配置在双方都完成的同一 (task, 攻击) 上做精确 McNemar；
    Holm 校正在同一指标的所有配对之间进行。

    Exact McNemar test for each pair of configurations, on the same
    (task, attack) runs both finished; Holm correction is applied across all
    pairs of the same metric.
    """

    labels = config_labels(runs)
    by_config = {
        key: {run.unit: run for run in group if run.finished}
        for key, group in _group(runs, lambda run: run.config_key).items()
    }
    configs = sorted(by_config, key=lambda key: labels[key])
    rows = []
    for metric in COMPARED_METRICS:
        metric_rows = []
        for first, second in combinations(configs, 2):
            a_runs, b_runs = by_config[first], by_config[second]
            pairs = []
            for unit in sorted(set(a_runs) & set(b_runs), key=str):
                if metric == "attack_succeeded" and unit[1] is None:
                    continue
                a, b = a_runs[unit].flag(metric), b_runs[unit].flag(metric)
                if a is not None and b is not None:
                    pairs.append((a, b))
            if not pairs:
                continue
            only_a = sum(a and not b for a, b in pairs)
            only_b = sum(b and not a for a, b in pairs)
            metric_rows.append(
                {
                    "metric": metric,
                    "a": labels[first],
                    "b": labels[second],
                    "pairs": len(pairs),
                    "a_rate": sum(a for a, _ in pairs) / len(pairs),
                    "b_rate": sum(b for _, b in pairs) / len(pairs),
                    "only_a": only_a,
                    "only_b": only_b,
                    "p": mcnemar_exact(only_a, only_b),
                }
            )
        for row, adjusted in zip(metric_rows, holm([row["p"] for row in metric_rows])):
            row["p_holm"] = adjusted
            row["significant"] = adjusted < 0.05
        rows += metric_rows
    return rows


def evaluate(runs: Sequence[Run]) -> dict[str, Any]:
    labels = config_labels(runs)
    by_config = _group(runs, lambda run: run.config_key)
    ordered = sorted(by_config, key=lambda key: labels[key])

    def per_config(metrics: Callable[[Sequence[Run]], dict[str, Any]]):
        return [
            {"setup": labels[key], "k": k_label(key[-1]), **metrics(by_config[key])}
            for key in ordered
        ]

    def breakdown(key: Callable[[Run], Any], name: str):
        rows = []
        for config in ordered:
            groups = _group(by_config[config], key)
            for value in sorted(groups, key=lambda item: (item is None, str(item))):
                rows.append(
                    {
                        "setup": labels[config],
                        "k": k_label(config[-1]),
                        name: value if value is not None else NO_ATTACK,
                        **outcome_metrics(groups[value]),
                    }
                )
        return rows

    return {
        "models": sorted(
            {json.dumps(run.models, sort_keys=True) for run in runs}
        ),
        "splits": sorted({run.split for run in runs if run.split}),
        "overview": per_config(outcome_metrics),
        "tcr_at_k": tcr_at_k(runs),
        "hitl": per_config(hitl_metrics),
        "cost": per_config(cost_metrics),
        "by_attack_type": breakdown(lambda run: run.attack_type, "attack_type"),
        "by_user_task": breakdown(lambda run: run.task_id, "task_id"),
        "mcnemar": mcnemar_comparisons(runs),
    }
