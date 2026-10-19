import json
import unittest

from agentdojo.task_suite.load_suites import get_suite

from scripts.make_dev_test_split import (
    BENCHMARK_VERSION,
    DEFAULT_OUTPUT,
    KEPT_TOGETHER,
    STRATA,
    SUITE,
    all_tasks,
    make_split,
)


class DevTestSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    def test_saved_file_matches_seeded_split(self) -> None:
        self.assertEqual(self.saved, make_split(seed=self.saved["seed"]))

    def test_sizes_and_no_overlap(self) -> None:
        dev, test = set(self.saved["dev"]), set(self.saved["test"])
        self.assertEqual(len(dev), 10)
        self.assertEqual(len(test), 6)
        self.assertFalse(dev & test)
        self.assertEqual(dev | test, set(all_tasks()))

    def test_near_duplicates_stay_together(self) -> None:
        test = set(self.saved["test"])
        for group in KEPT_TOGETHER:
            sides = {task in test for task in group}
            self.assertEqual(len(sides), 1, group)

    def test_every_stratum_on_both_sides(self) -> None:
        test = set(self.saved["test"])
        for name, stratum in self.saved["strata"].items():
            tasks = set(stratum["tasks"])
            self.assertTrue(tasks & test, f"{name} has no test task")
            self.assertTrue(tasks - test, f"{name} has no dev task")

    def test_strata_cover_agentdojo_banking_tasks(self) -> None:
        suite = get_suite(BENCHMARK_VERSION, SUITE)
        listed = [
            task
            for stratum in STRATA.values()
            for unit in stratum["units"]
            for task in unit
        ]
        self.assertEqual(len(listed), len(set(listed)))
        self.assertEqual(set(listed), set(suite.user_tasks))


if __name__ == "__main__":
    unittest.main()
