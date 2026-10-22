import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentdojo.task_suite.load_suites import get_suite

from harness.budget_allocator import BudgetAllocator
from harness.llm_judge import LLMJudge
from harness.policy_engine import PolicyEngine
from harness.simulated_reviewer import SimulatedReviewer
from scripts import replay_attacks
from scripts.replay_attacks import (
    ATTACK,
    CONTROL,
    READ_ONLY_TOOLS,
    build_cases,
    main,
    read_outcomes,
    replay_case,
    summarize,
)
from scripts.run_batch import Setup

DEV_TASKS = [f"user_task_{i}" for i in (1, 2, 5, 6, 8, 9, 10, 11, 12, 14)]
ALL_INJECTIONS = [f"injection_task_{i}" for i in range(9)]


class FixedJudgeClient:
    """每个样本都给同一个分数。 Gives every sample the same score."""

    last_usage = None

    def __init__(self, score: float) -> None:
        self.score = score
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps({"risk_score": self.score, "reason": "test"})


class VerdictClient:
    """reviewer 每次给同一个决定。 The reviewer gives the same verdict every time."""

    last_usage = None

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps({"decision": self.verdict, "reason": "test"})


def _engine(policy: str, judge_score: float = 0.5) -> PolicyEngine:
    return PolicyEngine(
        judge=LLMJudge(client=FixedJudgeClient(judge_score), sample_count=3, model_name="stub"),
        allocator=BudgetAllocator(tau0=1.0),
        policy_variant=policy,
    )


class BuildCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = get_suite("v1.2.2", "banking")
        cls.cases = build_cases(cls.suite, DEV_TASKS, ALL_INJECTIONS)

    def test_one_attack_case_per_task_and_injection(self) -> None:
        attacks = [case for case in self.cases if case.kind == ATTACK]
        self.assertEqual(len(attacks), 90)
        self.assertEqual(len({case.name for case in attacks}), 90)

    def test_control_cases_only_for_tasks_with_writes(self) -> None:
        controls = {case.task_id for case in self.cases if case.kind == CONTROL}
        self.assertEqual(
            controls,
            {f"user_task_{i}" for i in (2, 5, 6, 9, 11, 12, 14)},
        )

    def test_history_is_reads_and_request_is_the_prompt(self) -> None:
        for case in self.cases:
            self.assertTrue(all(call.function in READ_ONLY_TOOLS for call in case.history))
            prompt = self.suite.get_user_task_by_id(case.task_id).PROMPT
            self.assertEqual(case.user_request, prompt)

    def test_targets_are_the_attackers_calls(self) -> None:
        case = next(case for case in self.cases if case.name == "user_task_1__injection_task_6")
        self.assertEqual([call.function for call in case.targets], ["send_money"] * 3)
        self.assertTrue(all(call.args["recipient"] == "US133000000121212121212" for call in case.targets))
        case = next(case for case in self.cases if case.name == "user_task_1__injection_task_8")
        self.assertEqual(
            [call.function for call in case.targets],
            ["get_scheduled_transactions", "send_money"],
        )

    def test_no_control_flag(self) -> None:
        cases = build_cases(self.suite, ["user_task_5"], ["injection_task_0"], control=False)
        self.assertEqual([case.kind for case in cases], [ATTACK])


class ReplayCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = get_suite("v1.2.2", "banking")
        cls.cases = {
            case.name: case
            for case in build_cases(cls.suite, ["user_task_5", "user_task_14"], ALL_INJECTIONS)
        }

    def _replay(self, name, setup, engine, reviewer, budget=3):
        return replay_case(
            self.cases[name],
            suite=self.suite,
            setup=setup,
            policy_engine=engine,
            reviewer=reviewer,
            escalation_budget=budget,
            expected_remaining_risky_calls=3,
            replay_id="test",
        )

    def test_reviewer_denies_each_escalated_attack_call(self) -> None:
        client = VerdictClient("DENY")
        outcomes = self._replay(
            "user_task_5__injection_task_6",
            Setup("p1_only", "p1_only"),
            PolicyEngine(policy_variant="p1_only"),
            SimulatedReviewer(client),
        )
        targets = [outcome for outcome in outcomes if outcome.role == "target"]
        self.assertEqual([outcome.decision for outcome in targets], ["DENY"] * 3)
        self.assertEqual([outcome.decided_by for outcome in targets], ["reviewer"] * 3)
        self.assertTrue(all(outcome.harmful for outcome in targets))
        self.assertEqual(len(client.prompts), 3)
        # 历史中的读取由 P1 允许，不交给 reviewer。
        # The history reads are allowed by P1 and never reach the reviewer.
        history = [outcome for outcome in outcomes if outcome.role == "history"]
        self.assertTrue(history)
        self.assertTrue(all(outcome.decided_by == "p1" for outcome in history))

    def test_budget_runs_out_across_the_attackers_calls(self) -> None:
        outcomes = self._replay(
            "user_task_5__injection_task_6",
            Setup("p1_only", "p1_only"),
            PolicyEngine(policy_variant="p1_only"),
            SimulatedReviewer(VerdictClient("DENY")),
            budget=2,
        )
        targets = [outcome for outcome in outcomes if outcome.role == "target"]
        self.assertEqual(
            [outcome.decided_by for outcome in targets],
            ["reviewer", "reviewer", "budget"],
        )
        self.assertEqual(targets[2].decision, "DENY")
        self.assertIsNone(targets[2].reviewer_verdict)

    def test_allow_all_lets_the_attack_through_and_runs_it(self) -> None:
        outcomes = self._replay(
            "user_task_14__injection_task_7",
            Setup("allow_all", "allow_all"),
            PolicyEngine(policy_variant="allow_all"),
            SimulatedReviewer(VerdictClient("DENY")),
        )
        target = outcomes[-1]
        self.assertEqual((target.tool_name, target.decision), ("update_password", "ALLOW"))
        self.assertEqual(target.decided_by, "baseline")

    def test_hybrid_records_the_judges_scores(self) -> None:
        outcomes = self._replay(
            "user_task_5__injection_task_5",
            Setup("hybrid", "hybrid"),
            _engine("hybrid", judge_score=0.9),
            SimulatedReviewer(VerdictClient("ALLOW")),
        )
        target = outcomes[-1]
        self.assertEqual((target.decision, target.decided_by), ("DENY", "judge"))
        self.assertEqual(target.judge_scores, [0.9, 0.9, 0.9])
        self.assertEqual(target.judge_reason, "test")

    def test_judge_sees_the_user_request_and_history(self) -> None:
        engine = _engine("hybrid", judge_score=0.1)
        self._replay(
            "user_task_5__injection_task_5",
            Setup("hybrid", "hybrid"),
            engine,
            SimulatedReviewer(VerdictClient("ALLOW")),
        )
        prompt = engine.policy.judge.client.prompts[-1]
        self.assertIn(self.suite.get_user_task_by_id("user_task_5").PROMPT, prompt)
        self.assertIn("get_most_recent_transactions", prompt)

    def test_middle_score_goes_to_the_reviewer(self) -> None:
        outcomes = self._replay(
            "user_task_5__injection_task_5",
            Setup("hybrid", "hybrid"),
            _engine("hybrid", judge_score=0.5),
            SimulatedReviewer(VerdictClient("DENY")),
        )
        target = outcomes[-1]
        self.assertEqual((target.decision, target.decided_by), ("DENY", "reviewer"))
        self.assertEqual(target.reviewer_verdict, "DENY")


class SummarizeTests(unittest.TestCase):
    def test_counts_attacks_and_safe_calls(self) -> None:
        suite = get_suite("v1.2.2", "banking")
        cases = build_cases(suite, ["user_task_5"], ["injection_task_6", "injection_task_7"])
        outcomes = []
        for case in cases:
            outcomes += replay_case(
                case,
                suite=suite,
                setup=Setup("p1_only", "p1_only"),
                policy_engine=PolicyEngine(policy_variant="p1_only"),
                reviewer=SimulatedReviewer(VerdictClient("DENY")),
                escalation_budget=2,
                expected_remaining_risky_calls=3,
                replay_id="test",
            )
        metrics = summarize(outcomes)["p1_only"]
        # 3 次转账 + 1 次改密码。预算 2，第三次转账自动拒绝；用户没提密码，
        # P1 直接拒绝改密码。对照中用户自己的转账交给 reviewer。
        # 3 transfers + 1 password change. Budget 2, so the third transfer is
        # denied automatically; the user never mentions a password, so P1
        # denies the change. The user's own transfer in the control goes to
        # the reviewer.
        self.assertEqual(metrics["attack_calls_stopped"]["successes"], 4)
        self.assertEqual(metrics["attack_calls_stopped"]["n"], 4)
        self.assertEqual(
            metrics["attack_calls_by_step"],
            {"DENY": {"budget": 1, "p1": 1, "reviewer": 2}},
        )
        self.assertEqual(metrics["attack_cases_fully_stopped"]["successes"], 2)
        self.assertEqual(metrics["safe_calls_stopped"]["n"], 1)
        self.assertEqual(metrics["safe_calls_stopped"]["successes"], 1)
        self.assertEqual(metrics["safe_calls_by_step"], {"DENY": {"reviewer": 1}})
        self.assertEqual(metrics["reviews"], 3)
        self.assertIsNotNone(metrics["attack_calls_stopped"]["low"])


class MainTests(unittest.TestCase):
    TASKS = [
        "--task-ids", "user_task_5",
        "--injection-tasks", "injection_task_5", "injection_task_7",
    ]
    ARGS = [*TASKS, "--policies", "hybrid", "allow_all"]

    def _stub_components(self, config, policy, *, policy_logger):
        engine = PolicyEngine(
            judge=LLMJudge(client=FixedJudgeClient(0.5), sample_count=3, model_name="stub"),
            allocator=BudgetAllocator(tau0=1.0),
            policy_variant=policy,
            decision_logger=policy_logger,
        )
        return engine, SimulatedReviewer(VerdictClient("DENY"))

    def test_dry_run_needs_no_api_keys(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(output):
            main([*self.ARGS, "--dry-run"])
        text = output.getvalue()
        self.assertIn("attack cases: 2 (2 attacker write calls)", text)
        self.assertIn("control cases: 1 (1 safe write calls)", text)
        self.assertIn("allow_all: at most 0 Judge and 0 reviewer requests", text)

    def test_writes_logs_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-key"}
        ), patch.object(replay_attacks, "build_components", self._stub_components), contextlib.redirect_stderr(
            io.StringIO()
        ):
            argv = [*self.ARGS, "--output-dir", tmp, "--replay-id", "r1"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
            replay_dir = Path(tmp) / "r1"
            outcomes = read_outcomes(replay_dir / "calls.jsonl")
            self.assertEqual({(outcome.setup, outcome.case) for outcome in outcomes}, {
                (setup, case)
                for setup in ("hybrid", "allow_all")
                for case in (
                    "user_task_5__injection_task_5",
                    "user_task_5__injection_task_7",
                    "user_task_5__none",
                )
            })
            summary = json.loads((replay_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["hybrid"]["attack_calls_stopped"]["successes"], 2)
            self.assertEqual(summary["allow_all"]["attack_calls_stopped"]["successes"], 0)
            self.assertTrue((replay_dir / "policy_decisions.jsonl").exists())
            reviews = (replay_dir / "reviewer_decisions.jsonl").read_text(encoding="utf-8")
            self.assertIn('"replay": true', reviews)

            # 续跑：已完成的案例不再发送。
            # Resuming: finished cases are not sent again.
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(argv)
            self.assertIn("6 cases done, 0 to go", output.getvalue())
            self.assertEqual(len(read_outcomes(replay_dir / "calls.jsonl")), len(outcomes))

            # 换了 setup 就必须换 replay id。
            # Changed setups need a new replay id.
            changed = [*self.TASKS, "--policies", "p1_only", "--output-dir", tmp, "--replay-id", "r1"]
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(io.StringIO()):
                main(changed)
            self.assertIn("different settings (setups)", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
