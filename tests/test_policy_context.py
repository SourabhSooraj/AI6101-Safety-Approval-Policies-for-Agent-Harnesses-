import unittest

from harness.policy_context import PolicyContextInitializer


class PolicyContextTests(unittest.TestCase):
    def test_initializes_task_budget_and_user_request(self):
        initializer = PolicyContextInitializer(
            task_id="task-1",
            escalation_budget=3,
            expected_remaining_risky_calls=2,
        )

        _, _, _, _, extra_args = initializer.query(
            query="Pay the bill.",
            runtime=object(),
        )

        self.assertEqual(extra_args["task_id"], "task-1")
        self.assertEqual(extra_args["user_request"], "Pay the bill.")
        self.assertEqual(extra_args["remaining_escalations"], 3)
        self.assertEqual(extra_args["expected_remaining_risky_calls"], 2)

    def test_does_not_overwrite_existing_state(self):
        initializer = PolicyContextInitializer(
            task_id="task-1",
            escalation_budget=3,
        )
        extra_args = {"remaining_escalations": 1, "task_id": "existing"}

        _, _, _, _, result = initializer.query(
            query="Pay the bill.",
            runtime=object(),
            extra_args=extra_args,
        )

        self.assertEqual(result["remaining_escalations"], 1)
        self.assertEqual(result["task_id"], "existing")


if __name__ == "__main__":
    unittest.main()
