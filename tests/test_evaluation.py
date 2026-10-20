import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from harness.evaluation import (
    evaluate,
    holm,
    load_runs,
    mcnemar_exact,
    wilson_interval,
)
from harness.policies import UNLIMITED_ESCALATIONS
from scripts import evaluate as evaluate_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_BATCH = REPO_ROOT / "tests" / "fixtures" / "sample_runs" / "sample_batch"
MODELS = {"agent": "qwen3:8b", "judge": "qwen3:8b", "reviewer": "llama3.1:8b"}


class BatchBuilder:
    """在临时目录中写出 run_batch.py 格式的 batch 文件夹。"""

    def __init__(self, root: Path, name: str, split: str = "dev", models=MODELS):
        self.dir = root / name
        self.dir.mkdir(parents=True)
        self.models = dict(models)
        self.config = {
            "suite": "banking",
            "agent_model": models["agent"],
            "judge_model": models["judge"],
            "reviewer_model": models["reviewer"],
            "k": 3,
            "allocator": "value",
            "tau0": 1.0,
            "t_low": 0.3,
            "t_high": 0.7,
            "split": split,
        }
        self.write_config()
        self.lines: list[dict] = []

    def write_config(self, **extra):
        self.config.update(extra)
        (self.dir / "batch_config.json").write_text(json.dumps(self.config))

    def run(
        self,
        task_id,
        setup,
        injection=None,
        *,
        policy=None,
        k=3,
        strict_utility=True,
        safe=True,
        attack_succeeded=None,
        decisions=(),
        reviews=(),
        agent_usd=0.0,
        models=None,
    ):
        policy = policy or setup
        folder = self.dir / f"{task_id}__{setup}__{injection or 'none'}"
        folder.mkdir()
        summary = {
            "run_id": f"{self.dir.name}/{folder.name}",
            "task_id": task_id,
            "suite": "banking",
            "policy": policy,
            "policy_variant": policy,
            "injection_task_id": injection,
            "k": UNLIMITED_ESCALATIONS if policy == "no_budget" else k,
            "allocator": "value",
            "tau0": 1.0,
            "t_low": 0.3,
            "t_high": 0.7,
            "utility": strict_utility,
            "strict_utility": strict_utility,
            "attack_succeeded": attack_succeeded if injection else None,
            "safe_task_completion": safe,
            "agent_usage": {
                "estimated_cost_usd": agent_usd,
                "input_tokens": 100,
                "output_tokens": 10,
            },
            "models": models or self.models,
        }
        (folder / "run_summary.json").write_text(json.dumps(summary))
        self._logs(folder, decisions, reviews)
        self.lines.append({**summary, "setup": setup})
        self._write_lines()

    def crash(self, task_id, setup, injection=None, *, policy=None, decisions=()):
        folder = self.dir / f"{task_id}__{setup}__{injection or 'none'}"
        folder.mkdir()
        self._logs(folder, decisions, ())
        self.lines.append(
            {
                "task_id": task_id,
                "policy": policy or setup,
                "injection_task_id": injection,
                "error": "openai.APITimeoutError: Request timed out.",
                "setup": setup,
            }
        )
        self._write_lines()

    @staticmethod
    def _logs(folder, decisions, reviews):
        if decisions:
            (folder / "policy_decisions.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in decisions)
            )
        if reviews:
            (folder / "reviewer_decisions.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in reviews)
            )

    def _write_lines(self):
        (self.dir / "batch_summary.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in self.lines)
        )


def row(rows, **match):
    [found] = [
        item for item in rows if all(item.get(key) == value for key, value in match.items())
    ]
    return found


class StatisticsTests(unittest.TestCase):
    def test_wilson_matches_known_values(self):
        self.assertIsNone(wilson_interval(0, 0))
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=4)
        self.assertAlmostEqual(high, 0.7634, places=4)
        low, high = wilson_interval(1, 1)
        self.assertAlmostEqual(low, 0.2065, places=4)
        self.assertEqual(high, 1.0)
        self.assertEqual(wilson_interval(0, 4)[0], 0.0)

    def test_exact_mcnemar(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(mcnemar_exact(1, 9), 22 / 1024)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)
        self.assertEqual(mcnemar_exact(2, 7), mcnemar_exact(7, 2))

    def test_holm_adjusts_in_order_and_stays_monotone(self):
        adjusted = holm([0.01, 0.04, 0.03])
        for value, expected in zip(adjusted, [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(value, expected)
        self.assertEqual(holm([0.5, 0.9]), [1.0, 1.0])
        self.assertEqual(holm([]), [])


class SampleRunsTests(unittest.TestCase):
    def setUp(self):
        self.runs, self.warnings = load_runs([SAMPLE_BATCH])
        self.results = evaluate(self.runs)

    def test_crashed_run_is_loaded_with_its_setup(self):
        self.assertEqual(len(self.runs), 5)
        [crashed] = [run for run in self.runs if not run.finished]
        self.assertEqual(crashed.task_id, "user_task_2")
        self.assertEqual(crashed.config["policy_variant"], "hybrid")
        self.assertIn("APITimeoutError", crashed.error)
        self.assertEqual(self.warnings, [])

    def test_outcomes_count_a_crash_as_not_done(self):
        hybrid = row(self.results["overview"], setup="hybrid")
        self.assertEqual((hybrid["runs"], hybrid["crashed"]), (4, 1))
        self.assertEqual((hybrid["tcr"].successes, hybrid["tcr"].total), (3, 4))
        self.assertEqual(hybrid["utility_no_attack"].total, 3)
        self.assertEqual(hybrid["asr_finished"].successes, 0)

        allow_all = row(self.results["overview"], setup="allow_all")
        self.assertEqual(allow_all["asr_finished"].value, 1.0)
        self.assertEqual(allow_all["tcr"].value, 0.0)

    def test_hitl_counts_the_one_escalation(self):
        hybrid = row(self.results["hitl"], setup="hybrid")
        self.assertEqual(hybrid["escalations"], 1)
        self.assertEqual(hybrid["reviewer_approved"].successes, 1)
        self.assertEqual(hybrid["auto_denied"], 0)

    def test_breakdowns(self):
        data_stealing = row(
            self.results["by_attack_type"], setup="allow_all", attack_type="data_stealing"
        )
        self.assertEqual(data_stealing["asr_finished"].value, 1.0)
        task_2 = row(self.results["by_user_task"], setup="hybrid", task_id="user_task_2")
        self.assertEqual(task_2["crashed"], 1)


class SyntheticBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def batch(self, name="b1", **kwargs):
        return BatchBuilder(self.root, name, **kwargs)

    def test_tcr_at_k_has_one_line_per_family(self):
        batch = self.batch()
        for k in (0, 1, 3):
            batch.run("user_task_1", f"hybrid_k{k}", policy="hybrid", k=k, safe=k > 0)
            batch.run("user_task_1", f"allow_all_k{k}", policy="allow_all", k=k, safe=False)
        batch.run("user_task_1", "no_budget")

        rows = evaluate(load_runs([batch.dir])[0])["tcr_at_k"]

        hybrid = [item for item in rows if item["family"].startswith("hybrid")]
        self.assertEqual([item["k"] for item in hybrid], ["0", "1", "3"])
        self.assertEqual([item["tcr"].value for item in hybrid], [0.0, 1.0, 1.0])
        self.assertEqual(
            [item["k"] for item in rows if item["family"] == "allow_all"], ["0", "1", "3"]
        )
        self.assertEqual(
            row(rows, family="no_budget (value, tau0=1.0)")["k"], "unlimited"
        )

    def test_attack_success_over_finished_and_all_runs(self):
        batch = self.batch()
        batch.run("user_task_1", "hybrid", "injection_task_0", attack_succeeded=True)
        batch.run("user_task_2", "hybrid", "injection_task_0", attack_succeeded=False)
        batch.crash("user_task_3", "hybrid", "injection_task_4")

        overview = row(evaluate(load_runs([batch.dir])[0])["overview"], setup="hybrid")

        self.assertEqual(
            (overview["asr_finished"].successes, overview["asr_finished"].total), (1, 2)
        )
        self.assertEqual(
            (overview["asr_all_runs"].successes, overview["asr_all_runs"].total), (1, 3)
        )
        self.assertEqual(overview["attacked_crashed"], 1)

    def test_crashed_run_gets_its_attack_type_from_the_injection_id(self):
        batch = self.batch()
        batch.run("user_task_1", "hybrid")
        batch.crash("user_task_3", "hybrid", "injection_task_4")

        rows = evaluate(load_runs([batch.dir])[0])["by_attack_type"]

        self.assertEqual(row(rows, attack_type="direct_harm")["crashed"], 1)

    def test_crashed_setup_without_a_finished_run_uses_the_batch_config(self):
        batch = self.batch()
        batch.write_config(
            setups=[{"name": "strict", "policy": "hybrid", "options": {"k": 1}}]
        )
        batch.crash("user_task_1", "strict", policy="hybrid")

        [run] = load_runs([batch.dir])[0]

        self.assertEqual((run.config["policy_variant"], run.k), ("hybrid", 1))

    def test_hitl_load_counts_reviews_and_auto_denials(self):
        batch = self.batch()
        batch.run(
            "user_task_1",
            "hybrid",
            k=2,
            decisions=[
                {"decision": "REQUIRE_APPROVAL", "remaining_escalations_at_decision": 2},
                {"decision": "REQUIRE_APPROVAL", "remaining_escalations_at_decision": 1},
                {"decision": "REQUIRE_APPROVAL", "remaining_escalations_at_decision": 0},
                {"decision": "DENY", "policy_name": "p6_budget_exhausted"},
                {"decision": "ALLOW", "policy_name": "p1_read_only"},
            ],
            reviews=[
                {"reviewer_verdict": "ALLOW", "reviewer_success": True},
                {"reviewer_verdict": "DENY", "reviewer_success": False},
            ],
        )
        batch.run("user_task_2", "hybrid", k=2)

        hitl = row(evaluate(load_runs([batch.dir])[0])["hitl"], setup="hybrid")

        self.assertEqual(hitl["escalations"], 2)
        self.assertEqual(hitl["escalations_per_run"], 1.0)
        self.assertEqual(hitl["budget_share_used"], 0.5)
        self.assertEqual(hitl["auto_denied"], 2)
        self.assertEqual(hitl["runs_out_of_budget"].successes, 1)
        # 失败的 review 不算 reviewer 的判断。
        self.assertEqual(
            (hitl["reviewer_approved"].successes, hitl["reviewer_approved"].total), (1, 1)
        )
        self.assertEqual(hitl["reviewer_failed"].successes, 1)

    def test_cost_is_split_by_component(self):
        batch = self.batch()
        batch.run(
            "user_task_1",
            "hybrid",
            agent_usd=0.5,
            decisions=[{"decision": "ALLOW", "estimated_cost_usd": 0.25,
                        "input_tokens": 40, "output_tokens": 4}],
            reviews=[{"reviewer_verdict": "ALLOW", "reviewer_cost_usd": 0.125,
                      "estimated_cost_usd": 0.125, "reviewer_input_tokens": 30,
                      "reviewer_output_tokens": 3}],
        )
        batch.crash(
            "user_task_2", "hybrid",
            decisions=[{"decision": "ALLOW", "estimated_cost_usd": 0.25}],
        )

        cost = row(evaluate(load_runs([batch.dir])[0])["cost"], setup="hybrid")

        self.assertEqual(cost["agent"]["usd"], 0.5)
        self.assertEqual(cost["judge"]["usd"], 0.5)
        self.assertEqual(cost["reviewer"]["usd"], 0.125)
        self.assertEqual(cost["reviewer"]["input_tokens"], 30)
        self.assertEqual(cost["total_usd"], 1.125)
        self.assertEqual(cost["usd_per_run"], 1.125 / 2)

    def test_mcnemar_pairs_only_runs_both_setups_finished(self):
        batch = self.batch()
        for number in range(1, 9):
            task = f"user_task_{number}"
            batch.run(task, "hybrid", "injection_task_0", attack_succeeded=False)
            batch.run(task, "allow_all", "injection_task_0", attack_succeeded=True, safe=False)
        batch.run("user_task_1", "hybrid", safe=True)
        batch.crash("user_task_9", "hybrid", "injection_task_0")
        batch.run("user_task_9", "allow_all", "injection_task_0", attack_succeeded=True)

        rows = evaluate(load_runs([batch.dir])[0])["mcnemar"]

        asr = row(rows, metric="attack_succeeded")
        self.assertEqual((asr["a"], asr["b"], asr["pairs"]), ("allow_all", "hybrid", 8))
        self.assertEqual((asr["only_a"], asr["only_b"]), (8, 0))
        self.assertAlmostEqual(asr["p"], 2 / 256)
        self.assertTrue(asr["significant"])
        # 无攻击的 user_task_1 只有 hybrid 跑了，不能配对。
        self.assertEqual(row(rows, metric="safe_task_completion")["pairs"], 8)

    def test_holm_is_applied_across_pairs_of_one_metric(self):
        batch = self.batch()
        for number in range(1, 7):
            task = f"user_task_{number}"
            batch.run(task, "a", safe=True, policy="hybrid", k=1)
            batch.run(task, "b", safe=False, policy="hybrid", k=2)
            batch.run(task, "c", safe=number > 3, policy="hybrid", k=3)

        rows = [
            item
            for item in evaluate(load_runs([batch.dir])[0])["mcnemar"]
            if item["metric"] == "safe_task_completion"
        ]

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [item["p_holm"] for item in rows], holm([item["p"] for item in rows])
        )

    def test_mixed_models_are_refused_unless_allowed(self):
        first = self.batch("b1")
        first.run("user_task_1", "hybrid")
        second = self.batch("b2", models={**MODELS, "agent": "deepseek-flash"})
        second.run("user_task_2", "hybrid")

        with self.assertRaisesRegex(ValueError, "deepseek-flash"):
            load_runs([first.dir, second.dir])
        runs, _ = load_runs([first.dir, second.dir], allow_mixed_models=True)
        self.assertEqual(len(runs), 2)

    def test_mixed_splits_and_duplicates_are_reported(self):
        first = self.batch("b1", split="dev")
        first.run("user_task_1", "hybrid", safe=False)
        second = self.batch("b2", split="test")
        second.run("user_task_1", "hybrid", safe=True)

        runs, warnings = load_runs([first.dir, second.dir])

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].batch, "b2")
        self.assertTrue(any("more than one split" in warning for warning in warnings))
        self.assertTrue(any("appears in b1 and b2" in warning for warning in warnings))

    def test_folder_without_summary_or_error_is_unfinished(self):
        batch = self.batch()
        batch.run("user_task_1", "hybrid")
        (batch.dir / "user_task_2__hybrid__none").mkdir()

        runs, warnings = load_runs([batch.dir])

        self.assertEqual(sum(not run.finished for run in runs), 1)
        self.assertTrue(any("interrupted" in warning for warning in warnings))

    def test_retry_that_succeeded_replaces_the_error(self):
        batch = self.batch()
        batch.crash("user_task_1", "hybrid")
        (batch.dir / "user_task_1__hybrid__none").rmdir()
        batch.run("user_task_1", "hybrid")

        runs, _ = load_runs([batch.dir])

        self.assertEqual([run.finished for run in runs], [True])


class EvaluateScriptTests(unittest.TestCase):
    def test_writes_json_and_csv_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                evaluate_script.main([str(SAMPLE_BATCH), "--out", str(out)])

            for table in evaluate_script.TABLES:
                self.assertTrue((out / f"{table}.csv").exists(), table)
            # TCR@k 曲线同时保存为 CSV、PNG 和 PDF。
            self.assertTrue(
                (out / "tcr_at_k.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            )
            self.assertTrue((out / "tcr_at_k.pdf").read_bytes().startswith(b"%PDF-"))
            data = json.loads((out / "results.json").read_text())
            hybrid = row(data["overview"], setup="hybrid")
            self.assertEqual(hybrid["tcr"]["n"], 4)
            self.assertAlmostEqual(hybrid["tcr"]["rate"], 0.75)
            header = (out / "overview.csv").read_text().splitlines()[0].split(",")
            self.assertIn("tcr_low", header)
            self.assertIn("Outcomes", printed.getvalue())

    def test_plot_covers_every_k_and_the_unlimited_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = BatchBuilder(root, "b1")
            for k in (0, 1, 3):
                batch.run("user_task_1", f"hybrid_k{k}", policy="hybrid", k=k)
            batch.run("user_task_1", "no_budget")
            rows = evaluate(load_runs([batch.dir])[0])["tcr_at_k"]

            paths = evaluate_script.plot_tcr_at_k(rows, root)

            self.assertEqual(
                [path.name for path in paths], ["tcr_at_k.png", "tcr_at_k.pdf"]
            )
            for path in paths:
                self.assertGreater(path.stat().st_size, 0)

    def test_pdf_is_the_same_for_the_same_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = evaluate(load_runs([SAMPLE_BATCH])[0])["tcr_at_k"]
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            first.mkdir()
            second.mkdir()

            evaluate_script.plot_tcr_at_k(rows, first)
            evaluate_script.plot_tcr_at_k(rows, second)

            self.assertEqual(
                (first / "tcr_at_k.pdf").read_bytes(),
                (second / "tcr_at_k.pdf").read_bytes(),
            )

    def test_missing_folder_stops_with_a_message(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluate_script.main([str(SAMPLE_BATCH / "no_such_batch")])


if __name__ == "__main__":
    unittest.main()
