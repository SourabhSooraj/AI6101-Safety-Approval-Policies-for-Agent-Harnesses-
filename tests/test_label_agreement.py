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
        self.root = root
        self.run_id = run_id

    def run_agreement(self, *names, extra=()):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            report = agreement_script.main(
                [str(self.master), *[str(self.copies[name]) for name in names],
                 "--out", str(self.out), *extra]
            )
        return report, printed.getvalue()

    def call_ids(self):
        """item_id -> tool_call_id，用来写对应的 reviewer 日志。
        item_id -> tool_call_id, to write matching reviewer logs."""

        return {row["item_id"]: row["tool_call_id"] for row in read_csv(self.master)}

    def write_reviewer_log(self, name, verdicts, model="glm-5.1", failed=()):
        """verdicts: item_id -> ALLOW/DENY；failed 中的条目写成失败的审核。
        verdicts: item_id -> ALLOW/DENY; items in failed are written as failed
        reviews."""

        folder = self.root / name / "run"
        folder.mkdir(parents=True)
        call_ids = self.call_ids()
        records = [
            {
                "run_id": self.run_id,
                "tool_call_id": call_ids[item],
                "reviewer_verdict": "DENY" if item in failed else verdict,
                "reviewer_success": item not in failed,
                "reviewer_model": model,
                "judge_model": "qwen3.7-flash",
            }
            for item, verdict in verdicts.items()
        ]
        (folder / "reviewer_decisions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        return self.root / name

    def test_ground_truth_rows_stay_out_of_human_kappa(self):
        rows = read_csv(self.master)
        [truth_item] = [row["item_id"] for row in rows if row["label"]]
        for name in ("alice", "bob"):
            copy = labeller_rows(rows, name, seed=7, include_prelabelled=True)
            with self.copies[name].open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=LABELLER_COLUMNS)
                writer.writeheader()
                writer.writerows(copy)
        alice = {item: "safe" for item in self.items}
        bob = dict(alice)
        bob[self.items[0]] = "unsafe"
        # 两人在 ground truth 行上也意见不同：不应进入 κ 或 disagreements。
        # They also disagree on the ground-truth row: it must not reach kappa
        # or disagreements.
        alice[truth_item] = "safe"
        bob[truth_item] = "unsafe"
        fill(self.copies["alice"], alice)
        fill(self.copies["bob"], bob)

        report, _ = self.run_agreement("alice", "bob")

        self.assertEqual(report["rows_needing_a_human"], 10)
        [pair] = report["pairs"]
        self.assertEqual(pair["items"], 10)
        self.assertAlmostEqual(pair["percent_agreement"], 0.9)
        self.assertEqual(
            [row["items"] for row in report["versus_ground_truth"]], [1, 1]
        )
        self.assertEqual(report["labellers"][0]["ground_truth_rows"], 1)
        settle = [row["item_id"] for row in read_csv(self.out / "disagreements.csv")]
        self.assertEqual(settle, [self.items[0]])

    def test_labellers_are_compared_with_the_simulated_reviewer(self):
        labels = {item: "safe" if index < 6 else "unsafe" for index, item in enumerate(self.items)}
        fill(self.copies["alice"], labels)
        bob = dict(labels)
        bob[self.items[0]] = "unsafe"
        fill(self.copies["bob"], bob)
        verdicts = {item: "ALLOW" if index < 7 else "DENY" for index, item in enumerate(self.items)}
        [truth_item] = [row["item_id"] for row in read_csv(self.master) if row["label"]]
        verdicts[truth_item] = "ALLOW"
        logs = self.write_reviewer_log("reviewer_logs", verdicts, failed={self.items[9]})

        report, printed = self.run_agreement(
            "alice", "bob", extra=["--reviewer-logs", str(logs)]
        )

        reviewer = report["reviewer"]
        self.assertEqual(reviewer["models"], ["glm-5.1"])
        # 失败的审核不算；ground truth 行不比较。
        # The failed review is left out; the ground-truth row isn't compared.
        self.assertEqual(reviewer["reviews"], 10)
        self.assertEqual(reviewer["ground_truth_rows"], 1)
        self.assertEqual(reviewer["items_compared"], 9)
        alice_vs = reviewer["labellers_vs_reviewer"][0]
        self.assertEqual((alice_vs["a"], alice_vs["b"], alice_vs["items"]), ("alice", "reviewer", 9))
        # alice 与 reviewer 只在第 7 项不同（unsafe vs ALLOW）。
        # alice and the reviewer differ only on item 7 (unsafe vs ALLOW).
        self.assertAlmostEqual(alice_vs["percent_agreement"], 8 / 9)
        self.assertEqual(alice_vs["confusion_matrix"]["unsafe"]["safe"], 1)
        agreed = reviewer["agreed_humans_vs_reviewer"]
        self.assertEqual(agreed["items"], 8)
        [same_rows] = reviewer["human_pairs_same_items"]
        self.assertEqual(same_rows["items"], 9)
        self.assertIn("Simulated reviewer (glm-5.1)", printed)
        self.assertIn("Human-human on the same rows", printed)
        # reviewer 不参与 disagreements 和 merged_labels。
        # The reviewer takes no part in disagreements or merged_labels.
        self.assertNotIn("reviewer_label", read_csv(self.out / "disagreements.csv")[0])

    def test_reviewer_logs_from_two_models_are_refused(self):
        fill(self.copies["alice"], {item: "safe" for item in self.items})
        fill(self.copies["bob"], {item: "safe" for item in self.items})
        half = len(self.items) // 2
        self.write_reviewer_log(
            "mixed/a", {item: "ALLOW" for item in self.items[:half]}, model="glm-5.1"
        )
        self.write_reviewer_log(
            "mixed/b", {item: "ALLOW" for item in self.items[half:]}, model="deepseek-flash"
        )
        logs = str(self.root / "mixed")

        with contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit):
                self.run_agreement("alice", "bob", extra=["--reviewer-logs", logs])
        self.assertIn("more than one reviewer model", err.getvalue())
        report, _ = self.run_agreement(
            "alice", "bob", extra=["--reviewer-logs", logs, "--allow-mixed-models"]
        )
        self.assertEqual(report["reviewer"]["items_compared"], 10)

    def test_settled_labels_are_read_back(self):
        alice = {item: "safe" for item in self.items}
        bob = dict(alice)
        bob[self.items[0]] = "unsafe"
        bob[self.items[1]] = "unsafe"
        fill(self.copies["alice"], alice)
        fill(self.copies["bob"], bob)
        self.run_agreement("alice", "bob")
        settle_path = self.out / "disagreements.csv"
        rows = read_csv(settle_path)
        self.assertEqual(len(rows), 2)
        rows[0]["final_label"] = "unsafe"
        with settle_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        _, printed = self.run_agreement(
            "alice", "bob", extra=["--settled", str(settle_path)]
        )

        merged = {row["item_id"]: row for row in read_csv(self.out / "merged_labels.csv")}
        settled_item = rows[0]["item_id"]
        self.assertEqual(
            (merged[settled_item]["label"], merged[settled_item]["label_source"]),
            ("unsafe", "settled"),
        )
        self.assertEqual(merged[rows[1]["item_id"]]["label"], "")
        self.assertIn("still to settle: 1", printed)
        # 重新写出的 disagreements.csv 保留已填的 final_label。
        # The rewritten disagreements.csv keeps the filled final_label.
        rewritten = {row["item_id"]: row["final_label"] for row in read_csv(settle_path)}
        self.assertEqual(rewritten[settled_item], "unsafe")
        self.assertEqual(len(read_labels(self.out / "merged_labels.csv")), 10)

    def test_invalid_or_unknown_settled_labels_stop(self):
        fill(self.copies["alice"], {item: "safe" for item in self.items})
        fill(self.copies["bob"], {item: "safe" for item in self.items})
        settle_path = self.root / "settled.csv"
        [truth_item] = [row["item_id"] for row in read_csv(self.master) if row["label"]]
        for item, label, expected in (
            (self.items[0], "maybe", "must be safe or unsafe"),
            (truth_item, "safe", "not rows needing a human"),
        ):
            settle_path.write_text(f"item_id,final_label\n{item},{label}\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit):
                    self.run_agreement("alice", "bob", extra=["--settled", str(settle_path)])
            self.assertIn(expected, err.getvalue())

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

    def _copies_with_ground_truth_rows(self, names):
        rows = read_csv(self.master)
        for name in names:
            copy = labeller_rows(rows, name, seed=7, include_prelabelled=True)
            with self.copies[name].open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=LABELLER_COLUMNS)
                writer.writeheader()
                writer.writerows(copy)
        [truth_item] = [row["item_id"] for row in rows if row["label"]]
        return truth_item

    def test_all_rows_adds_kappa_with_ground_truth_rows(self):
        truth_item = self._copies_with_ground_truth_rows(("alice", "bob", "carol"))
        labels = {item: "safe" if index < 6 else "unsafe" for index, item in enumerate(self.items)}
        labels[truth_item] = "safe"
        fill(self.copies["alice"], labels)
        fill(self.copies["carol"], labels)
        bob = dict(labels)
        # bob 只在 ground truth 行上不同。 / bob differs only on the ground-truth row.
        bob[truth_item] = "unsafe"
        fill(self.copies["bob"], bob)

        report, printed = self.run_agreement("alice", "bob", "carol", extra=["--all-rows"])

        # 主要数字不变：只看需要人工判断的 10 行，三人完全一致。
        # The main figures are unchanged: only the 10 rows that needed a
        # human, where all three agree.
        self.assertEqual(report["items_compared"], 10)
        self.assertAlmostEqual(report["fleiss_kappa"], 1.0)
        every = report["all_rows"]
        self.assertEqual(every["items_compared"], 11)
        self.assertEqual(every["ground_truth_rows"], 1)
        alice_bob = every["pairs"][0]
        self.assertEqual((alice_bob["a"], alice_bob["b"], alice_bob["items"]), ("alice", "bob", 11))
        self.assertAlmostEqual(alice_bob["percent_agreement"], 10 / 11)
        self.assertLess(every["fleiss_kappa"], 1.0)
        self.assertIn("All rows everyone labelled: 11 (1 of them ground-truth rows)", printed)
        # ground truth 行已有定论，不进 disagreements。
        # Ground-truth rows are decided already and never reach disagreements.
        self.assertEqual(read_csv(self.out / "disagreements.csv"), [])

    def test_all_rows_is_off_by_default(self):
        labels = {item: "safe" for item in self.items}
        fill(self.copies["alice"], labels)
        fill(self.copies["bob"], labels)

        report, printed = self.run_agreement("alice", "bob")

        self.assertNotIn("all_rows", report)
        self.assertNotIn("All rows everyone labelled", printed)

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


class WritesOnlyTests(unittest.TestCase):
    def test_reads_are_left_out_of_the_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "batch" / "user_task_3__escalate_all__none"
            run_dir.mkdir(parents=True)
            run_id = f"batch/{run_dir.name}"
            records = [
                review("r1", run_id, "get_balance", {}, "user_task_3"),
                review("r2", run_id, "get_most_recent_transactions", {"n": 10}, "user_task_3"),
                review("w1", run_id, "send_money",
                       {"recipient": "GB29NWBK60161331926819", "amount": 4.0}, "user_task_3"),
            ]
            (run_dir / "reviewer_decisions.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            for flags, expected in (([], {"r1", "r2", "w1"}), (["--writes-only"], {"w1"})):
                sheet = root / f"sheet{len(flags)}.csv"
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    make_label_sheet.main(
                        [str(root / "batch"), "--output", str(sheet), "--ground-truth",
                         "--labellers", "alice", "--include-prelabelled", *flags]
                    )
                self.assertEqual({row["tool_call_id"] for row in read_csv(sheet)}, expected)
                self.assertEqual(
                    len(read_csv(sheet.with_name(f"{sheet.stem}_alice.csv"))), len(expected)
                )
            self.assertIn("Left out 2 read-only calls", printed.getvalue())


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
