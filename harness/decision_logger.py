import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .policy import PolicyRequest, PolicyResult


class DecisionLogger:
    """将每次 Policy 决定追加写入 JSONL，供指标、q 和成本分析使用。"""

    def __init__(
        self,
        path: str | Path,
        run_id: str | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.extra_context = dict(extra_context or {})
        self._lock = threading.Lock()

    def log(
        self,
        request: PolicyRequest,
        result: PolicyResult,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "task_id": request.task_id,
            "tool_call_id": request.tool_call_id,
            "tool_name": request.tool_name,
            "arguments": request.arguments,
            "user_request": request.user_request,
            "decision": result.decision.value,
            "policy_name": result.policy_name,
            "reason": result.reason,
            "stage": result.stage,
            "risk_score": result.risk_score,
            "impact": result.impact,
            "q": result.q,
            "value": result.value,
            "tau": result.tau,
            "remaining_escalations_at_decision": request.remaining_escalations,
            "judge_success": result.judge_success,
            "judge_disagreement": result.judge_disagreement,
            "model_name": result.model_name,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cached_input_tokens": result.cached_input_tokens,
            "estimated_cost_usd": result.estimated_cost_usd,
            **self.extra_context,
            **(extra or {}),
        }

        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
