import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from harness.decision_logger import DecisionLogger
from harness.policy import Decision, PolicyRequest, PolicyResult
from scripts.run_batch import episode_cost, print_batch_table

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_BATCH = REPO_ROOT / "tests" / "fixtures" / "sample_runs" / "sample_batch"

# 运行目录 -> 这个样例要展示的情况。
FINISHED_RUNS = {
    "user_task_1__hybrid__none": "allow",
    "user_task_11__hybrid__none": "escalation",
    "user_task_1__hybrid__injection_task_0": "deny",
    "user_task_5__allow_all__injection_task_0": "successful_attack",
}
CRASHED_RUN = "user_task_2__hybrid__none"

# run_episode 写入的 run_summary.json 字段。
CURRENT_SUMMARY_KEYS = {
    "run_id", "task_id", "suite", "policy", "policy_variant", "attack_mode",
    "injection_task_id", "k", "sample_count", "allocator", "tau0",
    "utility_cost", "auto_rule", "reviewer_error_table_path", "t_low",
    "t_high", "auto_complete", "agent_thinking", "provider", "local_server",
    "agent_model", "agent_base_url", "judge_model", "judge_base_url",
    "judge_temperature", "reviewer_model", "reviewer_base_url",
    "attack_model_name", "q_table_path", "utility", "strict_utility",
    "utility_passes_without_action", "attack_succeeded", "agent_usage",
    "policy_log", "reviewer_log", "agentdojo_trace_dir",
}
# 接口约定中 a2 新增、harness 尚未写入的字段。
A2_SUMMARY_KEYS = {
    "commit", "models", "attack_type", "safe_task_completion",
    "unauthorized_calls", "data_stealing",
}
# PolicyGateToolsExecutor 在 reviewer 日志中额外写入的字段。
REVIEWER_EXTRA_KEYS = {
    "event_type", "reviewer_verdict", "reviewer_success", "reviewer_reason",
    "reviewer_error", "reviewer_model_name", "reviewer_input_tokens",
    "reviewer_output_tokens", "reviewer_cached_input_tokens",
    "reviewer_cost_usd",
}
# run_episode 写进每条决策记录的 log_context：run_summary 中除结果外的设置。
LOG_CONTEXT_KEYS = CURRENT_SUMMARY_KEYS - {
    "run_id", "task_id", "q_table_path", "utility", "strict_utility",
    "utility_passes_without_action", "attack_succeeded", "agent_usage",
    "policy_log", "reviewer_log", "agentdojo_trace_dir",
}


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def decision_logger_keys():
    """真实 DecisionLogger 写出的字段，用来发现样例与代码的格式漂移。"""

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        DecisionLogger(path, run_id="r").log(
            PolicyRequest(
                tool_name="get_balance",
                tool_call_id="call_1",
                arguments={},
                task_id="user_task_0",
            ),
            PolicyResult(
                decision=Decision.ALLOW,
                reason="",
                policy_name="p1_read_only",
            ),
        )
        return set(read_jsonl(path)[0])


def summary(run):
    path = SAMPLE_BATCH / run / "run_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def decisions(run):
    return read_jsonl(SAMPLE_BATCH / run / "policy_decisions.jsonl")


def reviews(run):
    return read_jsonl(SAMPLE_BATCH / run / "reviewer_decisions.jsonl")


class SampleRunFormatTests(unittest.TestCase):
    def test_policy_records_have_the_logger_and_context_fields(self):
        expected = decision_logger_keys() | LOG_CONTEXT_KEYS
        for run in [*FINISHED_RUNS, CRASHED_RUN]:
            for record in decisions(run):
                with self.subTest(run=run, call=record["tool_call_id"]):
                    self.assertEqual(set(record), expected)

    def test_reviewer_records_add_the_reviewer_fields(self):
        expected = decision_logger_keys() | LOG_CONTEXT_KEYS | REVIEWER_EXTRA_KEYS
        records = [record for run in FINISHED_RUNS for record in reviews(run)]
        self.assertTrue(records)
        for record in records:
            self.assertEqual(set(record), expected)

    def test_summaries_have_current_and_contract_fields(self):
        for run in FINISHED_RUNS:
            with self.subTest(run=run):
                self.assertEqual(
                    set(summary(run)),
                    CURRENT_SUMMARY_KEYS | A2_SUMMARY_KEYS,
                )

    def test_summary_log_paths_point_at_the_run_folder(self):
        for run in FINISHED_RUNS:
            data = summary(run)
            for key in ("policy_log", "reviewer_log", "agentdojo_trace_dir"):
                with self.subTest(run=run, key=key):
                    path = REPO_ROOT / data[key]
                    self.assertEqual(path.parent, SAMPLE_BATCH / run)
            self.assertTrue((REPO_ROOT / data["policy_log"]).is_file())
            self.assertTrue((REPO_ROOT / data["agentdojo_trace_dir"]).is_dir())

    def test_no_path_has_a_colon(self):
        # Windows 文件名不能含 ":"，否则无法 checkout。
        for path in SAMPLE_BATCH.rglob("*"):
            self.assertNotIn(":", path.name)

    def test_batch_summary_has_one_line_per_episode(self):
        lines = read_jsonl(SAMPLE_BATCH / "batch_summary.jsonl")
        self.assertEqual(len(lines), len(FINISHED_RUNS) + 1)
        for line in lines:
            if "error" in line:
                continue
            run = line["run_id"].split("/", 1)[1]
            self.assertEqual(
                {key: value for key, value in line.items()
                 if key != "total_cost_usd"},
                summary(run),
            )


class SampleRunScenarioTests(unittest.TestCase):
    def test_allow(self):
        run = "user_task_1__hybrid__none"
        self.assertEqual(
            [record["decision"] for record in decisions(run)], ["ALLOW"]
        )
        self.assertEqual(reviews(run), [])
        data = summary(run)
        self.assertTrue(data["utility"])
        self.assertIsNone(data["attack_succeeded"])
        self.assertIsNone(data["attack_type"])
        self.assertTrue(data["safe_task_completion"])

    def test_escalation(self):
        run = "user_task_11__hybrid__none"
        records = decisions(run)
        self.assertEqual(
            [record["decision"] for record in records],
            ["ALLOW", "REQUIRE_APPROVAL"],
        )
        self.assertEqual(records[1]["stage"], "p6")
        [review] = reviews(run)
        self.assertEqual(review["tool_call_id"], records[1]["tool_call_id"])
        self.assertEqual(review["reviewer_verdict"], "ALLOW")
        self.assertTrue(review["reviewer_success"])
        self.assertTrue(summary(run)["utility"])

    def test_deny(self):
        run = "user_task_1__hybrid__injection_task_0"
        records = decisions(run)
        self.assertEqual(records[-1]["tool_name"], "send_money")
        self.assertEqual(records[-1]["decision"], "DENY")
        self.assertEqual(records[-1]["stage"], "judge")
        data = summary(run)
        self.assertTrue(data["utility"])
        self.assertFalse(data["attack_succeeded"])
        self.assertEqual(data["attack_type"], "data_stealing")
        self.assertEqual(data["unauthorized_calls"], [])
        self.assertEqual(data["data_stealing"]["send_decision"], "DENY")

    def test_successful_attack(self):
        run = "user_task_5__allow_all__injection_task_0"
        self.assertTrue(
            all(record["decision"] == "ALLOW" for record in decisions(run))
        )
        data = summary(run)
        self.assertTrue(data["attack_succeeded"])
        self.assertFalse(data["safe_task_completion"])
        self.assertEqual(
            [call["tool_call_id"] for call in data["unauthorized_calls"]],
            ["call_2"],
        )
        self.assertEqual(data["data_stealing"]["send_decision"], "ALLOW")

    def test_crash(self):
        run_dir = SAMPLE_BATCH / CRASHED_RUN
        self.assertFalse((run_dir / "run_summary.json").exists())
        self.assertTrue(decisions(CRASHED_RUN))
        [error_line] = [
            line for line in read_jsonl(SAMPLE_BATCH / "batch_summary.jsonl")
            if "error" in line
        ]
        # run_batch.py 写失败 episode 时只有这四个字段。
        self.assertEqual(
            set(error_line),
            {"task_id", "policy", "injection_task_id", "error"},
        )
        self.assertEqual(error_line["task_id"], "user_task_2")


class SampleRunToolTests(unittest.TestCase):
    def test_batch_table_reads_the_samples(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print_batch_table(SAMPLE_BATCH / "batch_summary.jsonl")
        self.assertIn("1 episode(s) failed", out.getvalue())

    def test_episode_cost_is_zero_for_local_models(self):
        for run in FINISHED_RUNS:
            data = summary(run)
            data = {
                **data,
                "policy_log": str(REPO_ROOT / data["policy_log"]),
                "reviewer_log": str(REPO_ROOT / data["reviewer_log"]),
            }
            self.assertEqual(episode_cost(data), 0.0)


if __name__ == "__main__":
    unittest.main()
