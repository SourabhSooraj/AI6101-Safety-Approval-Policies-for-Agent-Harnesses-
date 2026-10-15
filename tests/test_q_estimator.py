import json
import tempfile
import unittest
from pathlib import Path

from harness.q_estimator import (
    estimate_q_details,
    estimate_q_table,
    load_q_table,
)


class QEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_directory.name) / "dev.jsonl"
        records = [
            {
                "decision": "REQUIRE_APPROVAL",
                "risk_score": 0.8,
                "reviewer_verdict": "ALLOW",
                "tool_name": "send_money",
            },
            {
                "decision": "REQUIRE_APPROVAL",
                "risk_score": 0.9,
                "reviewer_verdict": "DENY",
                "tool_name": "send_money",
            },
            {
                "decision": "REQUIRE_APPROVAL",
                "risk_score": 0.2,
                "reviewer_verdict": "DENY",
                "tool_name": "create_calendar_event",
            },
            {
                "decision": "ALLOW",
                "risk_score": 0.0,
                "reviewer_verdict": "ALLOW",
                "tool_name": "get_balance",
            },
            # reviewer 失败时的 fail-closed DENY 不能算作一次改变。
            {
                "decision": "REQUIRE_APPROVAL",
                "risk_score": 0.3,
                "reviewer_verdict": "DENY",
                "reviewer_success": False,
                "tool_name": "send_money",
            },
        ]
        self.log_path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_estimates_q_by_tool(self):
        q_table = estimate_q_table(
            self.log_path,
            group_by="tool_name",
            min_samples=2,
        )

        self.assertEqual(q_table, {"send_money": 0.5})

    def test_returns_sample_and_change_counts(self):
        details = estimate_q_details(self.log_path)

        self.assertEqual(details["send_money"].sample_count, 2)
        self.assertEqual(details["send_money"].changes, 1)
        self.assertAlmostEqual(
            details["create_calendar_event"].q,
            1.0,
        )

    def test_failed_reviews_are_not_counted(self):
        details = estimate_q_details(self.log_path)

        # 失败记录若被计入，send_money 会变成 3 个样本、2 次改变。
        self.assertEqual(details["send_money"].sample_count, 2)
        self.assertEqual(details["send_money"].changes, 1)

    def test_reads_log_and_q_table_with_bom(self):
        # Windows PowerShell 5.1 的 Set-Content -Encoding utf8 会写入 BOM。
        bom_log = Path(self.temp_directory.name) / "bom.jsonl"
        bom_log.write_text(
            self.log_path.read_text(encoding="utf-8"),
            encoding="utf-8-sig",
        )
        q_path = Path(self.temp_directory.name) / "bom_q_table.json"
        q_path.write_text(json.dumps({"send_money": 0.5}), encoding="utf-8-sig")

        self.assertEqual(
            estimate_q_table(bom_log, min_samples=2),
            {"send_money": 0.5},
        )
        self.assertEqual(load_q_table(q_path), {"send_money": 0.5})

    def test_load_q_table_round_trip(self):
        q_path = Path(self.temp_directory.name) / "q_table.json"
        q_path.write_text(
            json.dumps({"send_money": 0.5, "update_password": 1}),
            encoding="utf-8",
        )

        self.assertEqual(
            load_q_table(q_path),
            {"send_money": 0.5, "update_password": 1.0},
        )

    def test_load_q_table_rejects_out_of_range_values(self):
        q_path = Path(self.temp_directory.name) / "q_table.json"
        for payload in ({"send_money": 1.5}, {"send_money": "0.5"}, [0.5]):
            with self.subTest(payload=payload):
                q_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_q_table(q_path)

    def test_baseline_threshold_changes_automatic_decision(self):
        details = estimate_q_details(self.log_path, baseline_threshold=0.85)

        # 阈值 0.85 时 risk 0.8 的自动决定变为 ALLOW，与 reviewer 一致。
        self.assertEqual(details["send_money"].changes, 0)


if __name__ == "__main__":
    unittest.main()
