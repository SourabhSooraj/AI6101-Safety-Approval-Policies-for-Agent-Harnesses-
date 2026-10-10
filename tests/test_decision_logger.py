import json
import tempfile
import unittest
from pathlib import Path

from harness.decision_logger import DecisionLogger
from harness.policy import Decision, PolicyRequest, PolicyResult


class DecisionLoggerTests(unittest.TestCase):
    def test_writes_jsonl_record(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "decisions.jsonl"
            logger = DecisionLogger(
                log_path,
                run_id="run-1",
                extra_context={"suite": "banking"},
            )
            request = PolicyRequest(
                tool_name="send_money",
                tool_call_id="call-1",
                arguments={"amount": 500},
                task_id="task-1",
                user_request="Pay rent.",
            )
            result = PolicyResult(
                decision=Decision.REQUIRE_APPROVAL,
                reason="Risk is high.",
                policy_name="p6_value_above_tau",
                risk_score=0.9,
                impact=4,
                q=0.5,
                value=1.8,
                tau=1.0,
                stage="p6",
                judge_success=True,
                model_name="deepseek-flash",
                input_tokens=100,
                output_tokens=20,
                estimated_cost_usd=0.001,
            )

            logger.log(request, result)

            record = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["run_id"], "run-1")
            self.assertEqual(record["suite"], "banking")
            self.assertEqual(record["task_id"], "task-1")
            self.assertEqual(record["decision"], "REQUIRE_APPROVAL")
            self.assertEqual(record["input_tokens"], 100)
            self.assertAlmostEqual(record["estimated_cost_usd"], 0.001)


if __name__ == "__main__":
    unittest.main()
