import json
import tempfile
import unittest
from pathlib import Path

from harness.budget_allocator import ReviewerErrorRates, load_reviewer_error_table
from harness.reviewer_errors import (
    LABEL_COLUMNS,
    error_table_json,
    estimate_reviewer_errors,
    read_labels,
    read_reviewer_log,
    write_label_sheet,
)


def review(call_id, verdict, tool="send_money", run_id="run-1", success=True):
    return {
        "run_id": run_id,
        "tool_call_id": call_id,
        "tool_name": tool,
        "reviewer_verdict": verdict,
        "reviewer_success": success,
    }


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)


class EstimateReviewerErrorsTests(unittest.TestCase):
    def test_counts_both_error_types_with_laplace_smoothing(self):
        reviews = [
            review("u1", "ALLOW"),
            review("u2", "DENY"),
            review("u3", "DENY"),
            review("s1", "DENY"),
            review("s2", "ALLOW"),
        ]
        labels = {
            ("run-1", "u1"): "unsafe",
            ("run-1", "u2"): "unsafe",
            ("run-1", "u3"): "unsafe",
            ("run-1", "s1"): "safe",
            ("run-1", "s2"): "safe",
        }

        estimates = estimate_reviewer_errors(reviews, labels, min_samples=1)

        send = estimates["send_money"]
        self.assertEqual(
            (send.unsafe_count, send.unsafe_approved, send.safe_count, send.safe_denied),
            (3, 1, 2, 1),
        )
        # (1+1)/(3+2) 和 (1+1)/(2+2)
        self.assertAlmostEqual(send.rates.false_approve, 0.4)
        self.assertAlmostEqual(send.rates.false_deny, 0.5)

    def test_unlabelled_reviews_are_ignored(self):
        estimates = estimate_reviewer_errors(
            [review("u1", "ALLOW"), review("x", "ALLOW")],
            {("run-1", "u1"): "unsafe"},
            min_samples=1,
        )

        self.assertEqual(estimates["send_money"].unsafe_count, 1)
        self.assertEqual(estimates["send_money"].safe_count, 0)

    def test_small_groups_fall_back_to_the_pooled_rate(self):
        reviews = [review(f"u{i}", "DENY") for i in range(6)] + [
            review("p1", "ALLOW", tool="update_password"),
        ]
        labels = {("run-1", f"u{i}"): "unsafe" for i in range(6)}
        labels[("run-1", "p1")] = "unsafe"

        estimates = estimate_reviewer_errors(reviews, labels, min_samples=5)

        pooled = estimates["default"].rates.false_approve
        # 合并：7 个不安全调用中 1 个被放行 -> (1+1)/(7+2)
        self.assertAlmostEqual(pooled, 2 / 9)
        # send_money 有 6 个样本，用自己的估计 (0+1)/(6+2)
        self.assertAlmostEqual(
            estimates["send_money"].rates.false_approve, 1 / 8
        )
        # update_password 只有 1 个样本，改用合并估计
        self.assertAlmostEqual(
            estimates["update_password"].rates.false_approve, pooled
        )

    def test_no_labels_gives_prior_mean(self):
        estimates = estimate_reviewer_errors([review("u1", "ALLOW")], {})

        self.assertEqual(
            estimates["default"].rates,
            ReviewerErrorRates(false_approve=0.5, false_deny=0.5),
        )

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            estimate_reviewer_errors([], {}, min_samples=-1)
        with self.assertRaises(ValueError):
            estimate_reviewer_errors([], {}, prior_errors=0, prior_correct=0)

    def test_table_json_round_trips_through_the_allocator_loader(self):
        estimates = estimate_reviewer_errors(
            [review("u1", "ALLOW"), review("s1", "ALLOW")],
            {("run-1", "u1"): "unsafe", ("run-1", "s1"): "safe"},
            min_samples=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errors.json"
            path.write_text(json.dumps(error_table_json(estimates)), encoding="utf-8")

            table = load_reviewer_error_table(path)

        self.assertEqual(set(table), {"default", "send_money"})
        self.assertAlmostEqual(table["send_money"].false_approve, round(2 / 3, 4))


class LabelSheetFileTests(TempDirTestCase):
    def test_sheet_round_trip_skips_blank_labels(self):
        path = self.directory / "labels.csv"
        write_label_sheet(
            [
                {"run_id": "run-1", "tool_call_id": "a", "label": "unsafe"},
                {"run_id": "run-1", "tool_call_id": "b", "label": ""},
                {"run_id": "run-1", "tool_call_id": "c", "label": "Safe"},
            ],
            path,
        )

        header = path.read_text(encoding="utf-8-sig").splitlines()[0]
        self.assertEqual(header.split(","), list(LABEL_COLUMNS))
        self.assertEqual(
            read_labels(path),
            {("run-1", "a"): "unsafe", ("run-1", "c"): "safe"},
        )

    def test_bad_labels_are_rejected(self):
        path = self.directory / "labels.csv"
        write_label_sheet(
            [{"run_id": "run-1", "tool_call_id": "a", "label": "maybe"}], path
        )
        with self.assertRaises(ValueError):
            read_labels(path)

        write_label_sheet(
            [
                {"run_id": "run-1", "tool_call_id": "a", "label": "safe"},
                {"run_id": "run-1", "tool_call_id": "a", "label": "unsafe"},
            ],
            path,
        )
        with self.assertRaises(ValueError):
            read_labels(path)

    def test_missing_columns_are_rejected(self):
        path = self.directory / "labels.csv"
        path.write_text("run_id,label\nrun-1,safe\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            read_labels(path)

    def test_reviewer_log_skips_failed_and_unparsed_reviews(self):
        path = self.directory / "reviewer_decisions.jsonl"
        lines = [
            review("a", "ALLOW"),
            review("b", "DENY", success=False),
            review("c", None),
            review("d", "DENY"),
        ]
        path.write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n\n",
            encoding="utf-8-sig",
        )

        records = read_reviewer_log(path)

        self.assertEqual([record["tool_call_id"] for record in records], ["a", "d"])


if __name__ == "__main__":
    unittest.main()
