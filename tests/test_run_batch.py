import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_batch
from scripts.make_dev_test_split import make_split
from scripts.run_batch import (
    DEFAULT_SPLIT_FILE,
    Setup,
    build_parser,
    check_setups,
    given_episode_flags,
    load_setups,
    load_split,
    main,
    setup_flags,
)


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        main(argv)
    return out.getvalue()


class LoadSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write_split(self, **fields):
        data = {
            "suite": "banking",
            "benchmark_version": "v1.2.2",
            "dev": ["user_task_1"],
            "test": ["user_task_0"],
            **fields,
        }
        path = Path(self.tmp.name) / "split.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_default_file_matches_the_seeded_split(self):
        expected = make_split()
        for side in ("dev", "test"):
            self.assertEqual(
                load_split(
                    DEFAULT_SPLIT_FILE,
                    side,
                    suite="banking",
                    benchmark_version="v1.2.2",
                ),
                expected[side],
            )

    def test_wrong_suite_or_version_is_refused(self):
        path = self.write_split()
        with self.assertRaisesRegex(ValueError, "workspace"):
            load_split(path, "dev", suite="workspace", benchmark_version="v1.2.2")
        with self.assertRaisesRegex(ValueError, "v1.2.1"):
            load_split(path, "dev", suite="banking", benchmark_version="v1.2.1")

    def test_task_on_both_sides_is_refused(self):
        path = self.write_split(test=["user_task_0", "user_task_1"])
        with self.assertRaisesRegex(ValueError, "user_task_1"):
            load_split(path, "test", suite="banking", benchmark_version="v1.2.2")

    def test_empty_side_is_refused(self):
        path = self.write_split(test=[])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            load_split(path, "dev", suite="banking", benchmark_version="v1.2.2")


class RunBatchSplitTests(unittest.TestCase):
    def episode_tasks(self, output):
        return [
            line.split()[0]
            for line in output.splitlines()
            if line.startswith("user_task_")
        ]

    def test_dry_run_dev_lists_the_dev_tasks(self):
        output = run_main(["--split", "dev", "--dry-run"])

        self.assertEqual(self.episode_tasks(output), make_split()["dev"])
        self.assertIn("10 episodes", output)

    def test_dry_run_test_crosses_policies_and_injections(self):
        output = run_main(
            [
                "--split", "test",
                "--policies", "hybrid", "allow_all",
                "--injection-tasks", "none", "injection_task_0",
                "--dry-run",
            ]
        )

        tasks = self.episode_tasks(output)
        self.assertEqual(set(tasks), set(make_split()["test"]))
        self.assertIn(f"{6 * 2 * 2} episodes", output)

    def test_missing_split_file_stops_the_run(self):
        missing = Path(tempfile.gettempdir()) / "no_such_split_6101.json"
        with self.assertRaises(SystemExit):
            run_main(["--split", "dev", "--split-file", str(missing), "--dry-run"])


class SetupFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, setups):
        path = Path(self.tmp.name) / "setups.json"
        path.write_text(json.dumps({"setups": setups}), encoding="utf-8")
        return path

    def test_example_file_loads(self):
        setups = load_setups(
            Path(run_batch.REPO_ROOT) / "configs" / "setups" / "policy_comparison.json"
        )

        self.assertIn("hybrid_k3", [setup.name for setup in setups])

    def test_name_defaults_to_the_policy(self):
        [setup] = load_setups(self.write([{"policy": "p1_only", "k": 2}]))

        self.assertEqual(setup, Setup(name="p1_only", policy="p1_only", options={"k": 2}))

    def test_misspelt_key_is_refused(self):
        with self.assertRaisesRegex(ValueError, "tau"):
            load_setups(self.write([{"policy": "hybrid", "tau": 2}]))

    def test_model_flags_cannot_be_set_per_setup(self):
        with self.assertRaisesRegex(ValueError, "judge_model"):
            load_setups(self.write([{"policy": "hybrid", "judge_model": "x"}]))

    def test_unknown_policy_is_refused(self):
        with self.assertRaisesRegex(ValueError, "policy"):
            load_setups(self.write([{"name": "a", "policy": "hybird"}]))

    def test_duplicate_and_unsafe_names_are_refused(self):
        with self.assertRaisesRegex(ValueError, "twice"):
            load_setups(self.write([{"policy": "hybrid"}, {"policy": "hybrid"}]))
        with self.assertRaisesRegex(ValueError, "folder name"):
            load_setups(self.write([{"name": "a__b", "policy": "hybrid"}]))

    def test_bad_value_is_caught_before_running(self):
        setup = Setup(name="s", policy="hybrid", options={"allocator": "fast"})
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "'s'"):
                check_setups(["--provider", "ollama"], [setup], resolve=False)

    def test_conflicting_options_are_caught_before_running(self):
        setup = Setup(
            name="s",
            policy="hybrid",
            options={"allocator": "value", "reviewer_error_table": "x.json"},
        )
        with self.assertRaisesRegex(ValueError, "reviewer-error-table"):
            check_setups(["--provider", "ollama"], [setup], resolve=True)


class FlagForwardingTests(unittest.TestCase):
    def test_only_given_model_flags_are_forwarded(self):
        argv = [
            "--task-ids", "user_task_1",
            "--provider", "ollama",
            "--model", "llama3.1:8b",
            "--agent-base-url", "http://localhost:8000/v1",
            "--api-key-env", "AGENT_KEY",
            "--judge-model", "qwen3:8b",
            "--judge-base-url", "http://localhost:8001/v1",
            "--judge-api-key-env", "JUDGE_KEY",
            "--attack-model-name", "Llama",
            "--agent-thinking",
        ]

        flags = given_episode_flags(build_parser(), argv)

        self.assertEqual(
            flags,
            [
                "--provider", "ollama",
                "--model", "llama3.1:8b",
                "--agent-base-url", "http://localhost:8000/v1",
                "--agent-api-key-env", "AGENT_KEY",
                "--agent-thinking",
                "--judge-model", "qwen3:8b",
                "--judge-base-url", "http://localhost:8001/v1",
                "--judge-api-key-env", "JUDGE_KEY",
                "--attack-model-name", "Llama",
            ],
        )

    def test_defaults_are_left_to_the_child(self):
        parser = build_parser()

        self.assertEqual(given_episode_flags(parser, ["--task-ids", "user_task_1"]), [])
        # 解析器的默认值在之后仍然可用。
        self.assertEqual(parser.parse_args(["--task-ids", "x"]).k, 3)

    def test_setup_flags_come_after_and_override_model_flags(self):
        setup = Setup(name="s", policy="p4_only", options={"k": 1, "t_low": 0.2})

        self.assertEqual(
            setup_flags(setup), ["--policy", "p4_only", "--k", "1", "--t-low", "0.2"]
        )


class FakeChild:
    """替代 subprocess.run：按命令写 run_summary.json，或模拟一次崩溃。"""

    def __init__(self, crash_tasks=()):
        self.crash_tasks = set(crash_tasks)
        self.commands = []

    def __call__(self, command, cwd, stdout, stderr):
        self.commands.append(command)
        log_dir = Path(command[command.index("--log-dir") + 1])
        task_id = command[command.index("--task-id") + 1]
        policy = command[command.index("--policy") + 1]
        injection = (
            command[command.index("--injection-task") + 1]
            if "--injection-task" in command
            else None
        )
        if task_id in self.crash_tasks:
            stdout.write("Traceback (most recent call last):\n")
            stdout.write("openai.APITimeoutError: Request timed out.\n")
            (log_dir / "policy_decisions.jsonl").write_text("{}\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 1)
        summary = {
            "run_id": command[command.index("--run-id") + 1],
            "task_id": task_id,
            "policy": policy,
            "injection_task_id": injection,
            "utility": True,
            "strict_utility": True,
            "attack_succeeded": False if injection else None,
            "agent_usage": {"estimated_cost_usd": 0.0},
            "policy_log": str(log_dir / "policy_decisions.jsonl"),
            "reviewer_log": str(log_dir / "reviewer_decisions.jsonl"),
        }
        stdout.write(f"run_id={summary['run_id']}\n")
        (log_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)


class ProcessPerRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(run_batch, "REPO_ROOT", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.batch_dir = Path(self.tmp.name) / "logs" / "agentdojo_batch" / "b1"

    def run_batch(self, child, *extra, stderr=None):
        argv = [
            "--task-ids", "user_task_1", "user_task_2",
            "--policies", "hybrid", "allow_all",
            "--provider", "ollama",
            "--batch-id", "b1",
            *extra,
        ]
        with patch.object(run_batch.subprocess, "run", child):
            if stderr is None:
                return run_main(argv)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                main(argv)

    def summary_lines(self):
        path = self.batch_dir / "batch_summary.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_each_episode_runs_in_its_own_process_and_folder(self):
        child = FakeChild()
        self.run_batch(child, "--model", "qwen3:8b")

        self.assertEqual(len(child.commands), 4)
        log_dirs = {command[command.index("--log-dir") + 1] for command in child.commands}
        self.assertEqual(len(log_dirs), 4)
        for command in child.commands:
            self.assertIn("--model", command)
            self.assertNotIn("--k", command)
        lines = self.summary_lines()
        self.assertEqual(
            sorted(line["setup"] for line in lines),
            ["allow_all", "allow_all", "hybrid", "hybrid"],
        )
        self.assertTrue(all(line["total_cost_usd"] == 0.0 for line in lines))
        self.assertTrue((self.batch_dir / "user_task_1__hybrid__none" / "run.log").exists())

    def test_finished_runs_are_skipped_and_crashes_rerun(self):
        self.run_batch(FakeChild(crash_tasks={"user_task_2"}))
        errors = [line for line in self.summary_lines() if "error" in line]
        self.assertEqual(len(errors), 2)
        for error in errors:
            self.assertEqual(
                error["error"], "openai.APITimeoutError: Request timed out."
            )
            self.assertEqual(
                set(error),
                {"task_id", "policy", "injection_task_id", "error", "setup"},
            )

        retry = FakeChild()
        output = self.run_batch(retry)

        self.assertEqual(
            sorted(command[command.index("--task-id") + 1] for command in retry.commands),
            ["user_task_2", "user_task_2"],
        )
        self.assertEqual(output.count("already done, skipping"), 2)
        crashed_dir = self.batch_dir / "user_task_2__hybrid__none"
        self.assertFalse((crashed_dir / "policy_decisions.jsonl").exists())
        self.assertIn("run_id=", (crashed_dir / "run.log").read_text())

    def test_resuming_with_other_models_is_refused(self):
        self.run_batch(FakeChild(crash_tasks={"user_task_2"}), "--model", "qwen3:8b")

        err = io.StringIO()
        with self.assertRaises(SystemExit):
            self.run_batch(FakeChild(), "--model", "llama3.1:8b", stderr=err)
        self.assertIn("agent_model", err.getvalue())

    def test_dry_run_starts_no_process(self):
        child = FakeChild()
        output = self.run_batch(child, "--dry-run")

        self.assertEqual(child.commands, [])
        self.assertIn("--log-dir", output)
        self.assertIn("4 episodes", output)
        self.assertFalse(self.batch_dir.exists())


if __name__ == "__main__":
    unittest.main()
