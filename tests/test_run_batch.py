import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_dev_test_split import make_split
from scripts.run_batch import DEFAULT_SPLIT_FILE, load_split, main


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


if __name__ == "__main__":
    unittest.main()
