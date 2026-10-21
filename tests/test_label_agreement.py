import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from harness.label_agreement import (
    LABELLER_COLUMNS,
    cohen_kappa,
    confusion_matrix,
    fleiss_kappa,
    item_id,
    labeller_rows,
    percent_agreement,
    read_labeller_sheet,
)
from harness.reviewer_errors import read_labels
from scripts import label_agreement as agreement_script
from scripts import make_label_sheet

ATTACKER = "US133000000121212121212"


def counts_to_labels(table):
    """{(a, b): n} -> 两个标注序列。 / {(a, b): n} -> two label sequences."""

    first, second = [], []
    for (a, b), n in table.items():
        first += [a] * n
        second += [b] * n
    return first, second


def review(call_id, run_id, tool, arguments, task, injection=None):
    return {
        "run_id": run_id,
        "tool_call_id": call_id,
        "task_id": task,
        "injection_task_id": injection,
        "tool_name": tool,
        "arguments": arguments,
        "user_request": f"request for {task}",
        "reviewer_verdict": "ALLOW",
        "reviewer_success": True,
    }


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def fill(path, labels):
    """按 item_id 写入标签，模拟标注者在 Excel 中填表。
    Write labels by item_id, as a labeller filling the copy in Excel would."""

    rows = read_csv(path)
    for row in rows:
        row["label"] = labels.get(row["item_id"], "")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class StatisticsTests(unittest.TestCase):
    def test_cohen_kappa_matches_a_worked_example(self):
        # p_o = 35/50 = 0.7, p_e = 0.5 * 0.6 + 0.5 * 0.4 = 0.5, kappa = 0.4.
        first, second = counts_to_labels(
            {("safe", "safe"): 20, ("safe", "unsafe"): 5,
             ("unsafe", "safe"): 10, ("unsafe", "unsafe"): 15}
        )

        self.assertAlmostEqual(cohen_kappa(first, second), 0.4)
        self.assertAlmostEqual(percent_agreement(first, second), 0.7)
        self.assertEqual(
            confusion_matrix(first, second),
            {"safe": {"safe": 20, "unsafe": 5}, "unsafe": {"safe": 10, "unsafe": 15}},
        )

    def test_perfect_and_chance_agreement(self):
        labels = ["safe", "unsafe", "safe", "unsafe"]
        self.assertEqual(cohen_kappa(labels, labels), 1.0)
        self.assertAlmostEqual(
            cohen_kappa(["safe", "safe", "unsafe", "unsafe"],
                        ["safe", "unsafe", "safe", "unsafe"]),
            0.0,
        )

    def test_kappa_is_undefined_when_everyone_uses_one_label(self):
        labels = ["safe"] * 5
        self.assertIsNone(cohen_kappa(labels, labels))
        self.assertEqual(percent_agreement(labels, labels), 1.0)
        self.assertIsNone(cohen_kappa([], []))

    def test_fleiss_kappa_matches_a_hand_calculation(self):
        # P-bar = (1 + 1 + 1/3 + 1/3) / 4 = 2/3, P_e = 0.5^2 + 0.5^2 = 0.5,
        # kappa = (2/3 - 0.5) / 0.5 = 1/3.
        ratings = [
            ["safe", "safe", "safe"],
            ["unsafe", "unsafe", "unsafe"],
            ["safe", "safe", "unsafe"],
            ["safe", "unsafe", "unsafe"],
        ]

        self.assertAlmostEqual(fleiss_kappa(ratings), 1 / 3)
        with self.assertRaises(ValueError):
            fleiss_kappa([["safe", "safe"], ["safe"]])


class LabellerCopyTests(unittest.TestCase):
    def setUp(self):
        self.rows = make_label_sheet.build_rows(
            [
                review(f"c{number}", f"b1/user_task_{number}__hybrid__injection_task_0",
                       "send_money", {"recipient": "X", "amount": number},
                       f"user_task_{number}", "injection_task_0")
                for number in range(1, 13)
            ]
        )
        self.rows[0]["label"] = "safe"
        self.rows[0]["label_source"] = "ground_truth"

    def test_item_id_is_stable_and_reveals_nothing(self):
        first = item_id("b1/user_task_1__hybrid__injection_task_0", "c1")

        self.assertEqual(first, item_id("b1/user_task_1__hybrid__injection_task_0", "c1"))
        self.assertNotEqual(first, item_id("b1/user_task_1__hybrid__none", "c1"))
        self.assertNotIn("injection", first)
        self.assertEqual(self.rows[0]["item_id"], first)

    def test_copy_hides_run_ids_attack_ids_and_pre_labels(self):
        copy = labeller_rows(self.rows, "alice", seed=1)

        self.assertEqual(len(copy), 11)
        for row in copy:
            self.assertEqual(tuple(row), LABELLER_COLUMNS)
            self.assertEqual(row["label"], "")
            self.assertNotIn("injection", json.dumps(row))

    def test_prelabelled_rows_can_be_included_with_the_label_hidden(self):
        copy = labeller_rows(self.rows, "alice", seed=1, include_prelabelled=True)

        self.assertEqual(len(copy), 12)
        self.assertTrue(all(row["label"] == "" for row in copy))

    def test_each_labeller_gets_their_own_reproducible_order(self):
        alice = [row["item_id"] for row in labeller_rows(self.rows, "alice", seed=1)]
        bob = [row["item_id"] for row in labeller_rows(self.rows, "bob", seed=1)]

        self.assertEqual(sorted(alice), sorted(bob))
        self.assertNotEqual(alice, bob)
        self.assertEqual(
            alice, [row["item_id"] for row in labeller_rows(self.rows, "alice", seed=1)]
        )
        self.assertEqual(
            alice,
            [row["item_id"] for row in labeller_rows(list(reversed(self.rows)), "alice", seed=1)],
        )
        self.assertNotEqual(
            alice, [row["item_id"] for row in labeller_rows(self.rows, "alice", seed=2)]
        )


class EndToEndTests(unittest.TestCase):
    """make_label_sheet.py --labellers, then label_agreement.py on the copies."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        run_dir = root / "batch" / "user_task_3__hybrid__injection_task_1"
        run_dir.mkdir(parents=True)
        run_id = f"batch/{run_dir.name}"
        records = [
            # 由 ground truth 判定（不进标注者的表）。
            # Settled by ground truth (not in the labellers' copies).
            review("c1", run_id, "send_money",
                   {"recipient": "GB29NWBK60161331926819", "amount": 4.0},
                   "user_task_3", "injection_task_1"),
            *[
                review(f"c{number}", run_id, "update_user_info",
                       {"city": f"City {number}"}, "user_task_3", "injection_task_1")
                for number in range(2, 12)
            ],
        ]
        (run_dir / "reviewer_decisions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        self.master = root / "labels" / "sheet.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            make_label_sheet.main(
                [str(root / "batch"), "--output", str(self.master), "--ground-truth",
                 "--labellers", "alice", "bob", "carol", "--seed", "7"]
            )
        self.copies = {
            name: self.master.with_name(f"sheet_{name}.csv")
            for name in ("alice", "bob", "carol")
        }
        self.items = [row["item_id"] for row in read_csv(self.master) if not row["label"]]
        self.out = root / "agreement"

    def run_agreement(self, *names):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            report = agreement_script.main(
                [str(self.master), *[str(self.copies[name]) for name in names],
                 "--out", str(self.out)]
            )
        return report, printed.getvalue()

    def test_copies_are_written_without_attack_hints(self):
        self.assertEqual(len(self.items), 10)
        for path in self.copies.values():
            text = path.read_text(encoding="utf-8-sig")
            self.assertNotIn("injection_task", text)
            self.assertNotIn("batch/", text)
            self.assertEqual(len(read_csv(path)), 10)
        orders = [[row["item_id"] for row in read_csv(path)] for path in self.copies.values()]
        self.assertEqual(len({tuple(order) for order in orders}), 3)

    def test_two_labellers(self):
        alice = {item: "safe" for item in self.items}
        bob = dict(alice)
        bob[self.items[0]] = "unsafe"
        alice[self.items[1]] = "unsafe"
        bob[self.items[1]] = "unsafe"
        fill(self.copies["alice"], alice)
        fill(self.copies["bob"], bob)

        report, printed = self.run_agreement("alice", "bob")

        [pair] = report["pairs"]
        self.assertEqual((pair["a"], pair["b"], pair["items"]), ("alice", "bob", 10))
        self.assertAlmostEqual(pair["percent_agreement"], 0.9)
        self.assertEqual(pair["confusion_matrix"]["safe"]["unsafe"], 1)
        self.assertIn("Cohen's kappa", printed)
        self.assertIsNone(report["fleiss_kappa"])

        [disagreement] = read_csv(self.out / "disagreements.csv")
        self.assertEqual(disagreement["item_id"], self.items[0])
        self.assertEqual(
            (disagreement["alice_label"], disagreement["bob_label"], disagreement["final_label"]),
            ("safe", "unsafe", ""),
        )
        merged = read_csv(self.out / "merged_labels.csv")
        sources = {row["item_id"]: row["label_source"] for row in merged}
        self.assertEqual(sources[self.items[1]], "labellers_agree")
        self.assertEqual(sources[self.items[0]], "")
        self.assertIn("ground_truth", sources.values())
        # merged_labels.csv 能直接给 build_reviewer_error_table.py 用。
        # merged_labels.csv can go straight into build_reviewer_error_table.py.
        self.assertEqual(len(read_labels(self.out / "merged_labels.csv")), 10)

    def test_three_labellers_add_fleiss_kappa(self):
        for name in self.copies:
            fill(self.copies[name], {item: "safe" if index % 2 else "unsafe"
                                     for index, item in enumerate(self.items)})

        report, printed = self.run_agreement("alice", "bob", "carol")

        self.assertEqual(len(report["pairs"]), 3)
        self.assertAlmostEqual(report["fleiss_kappa"], 1.0)
        self.assertIn("Fleiss' kappa", printed)

    def test_blank_and_invalid_labels_are_reported_not_counted(self):
        alice = {item: "safe" for item in self.items}
        alice[self.items[0]] = "maybe"
        alice[self.items[1]] = ""
        fill(self.copies["alice"], alice)
        fill(self.copies["bob"], {item: "safe" for item in self.items})

        report, printed = self.run_agreement("alice", "bob")

        [labeller, _] = report["labellers"]
        self.assertEqual(labeller["blank"], 1)
        self.assertEqual([item["value"] for item in labeller["invalid"]], ["maybe"])
        self.assertEqual(report["items_compared"], 8)
        self.assertIn("invalid labels on rows", printed)

    def test_labellers_can_be_checked_against_ground_truth(self):
        rows = read_csv(self.master)
        copy = labeller_rows(rows, "dave", seed=7, include_prelabelled=True)
        path = self.master.with_name("sheet_dave.csv")
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=LABELLER_COLUMNS)
            writer.writeheader()
            writer.writerows(copy)
        fill(path, {row["item_id"]: "safe" for row in rows})
        fill(self.copies["alice"], {item: "safe" for item in self.items})
        self.copies["dave"] = path

        report, _ = self.run_agreement("alice", "dave")

        [truth] = report["versus_ground_truth"]
        self.assertEqual((truth["name"], truth["items"]), ("dave", 1))
        self.assertEqual(truth["percent_agreement"], 1.0)

    def test_one_labeller_or_a_sheet_without_item_ids_stops(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                agreement_script.main([str(self.master), str(self.copies["alice"])])

    def test_read_labeller_sheet_flags_duplicates(self):
        path = self.copies["alice"]
        rows = read_csv(path)
        rows.append(dict(rows[0]))
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        sheet = read_labeller_sheet(path, "alice")

        self.assertEqual(sheet.duplicates, [rows[0]["item_id"]])


class LabellerNameTests(unittest.TestCase):
    def test_bad_or_repeated_names_are_refused(self):
        with self.assertRaises(SystemExit):
            make_label_sheet.labeller_paths("labels/sheet.csv", ["al ice"])
        with self.assertRaises(SystemExit):
            make_label_sheet.labeller_paths("labels/sheet.csv", ["a", "a"])
        self.assertEqual(
            make_label_sheet.labeller_paths("labels/sheet.csv", ["alice"])["alice"],
            Path("labels/sheet_alice.csv"),
        )


if __name__ == "__main__":
    unittest.main()
