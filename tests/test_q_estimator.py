import json
import tempfile
import unittest
from pathlib import Path

from harness.q_estimator import estimate_q_details, estimate_q_table


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


if __name__ == "__main__":
    unittest.main()
