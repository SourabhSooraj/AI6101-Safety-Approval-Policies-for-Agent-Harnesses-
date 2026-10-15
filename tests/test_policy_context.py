import unittest

from agentdojo.agent_pipeline import AgentPipeline, InitQuery
from agentdojo.functions_runtime import FunctionsRuntime

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

    def test_resets_state_left_by_a_previous_task(self):
        initializer = PolicyContextInitializer(
            task_id="task-1",
            escalation_budget=3,
        )
        stale_args = {
            "remaining_escalations": 0,
            "escalations_used": 3,
            "task_id": "previous-task",
            "user_request": "previous request",
            "initial_escalations": 3,
        }

        _, _, _, _, result = initializer.query(
            query="Pay the bill.",
            runtime=object(),
            extra_args=stale_args,
        )

        self.assertEqual(result["remaining_escalations"], 3)
        self.assertEqual(result["escalations_used"], 0)
        self.assertEqual(result["task_id"], "task-1")
        self.assertEqual(result["user_request"], "Pay the bill.")
        self.assertNotIn("initial_escalations", result)
        # 传入的字典不能被修改，否则共享的默认参数仍会被污染。
        self.assertEqual(stale_args["remaining_escalations"], 0)

    def test_back_to_back_pipelines_do_not_share_state(self):
        seen = []

        class UseUpBudget:
            name = "use-up-budget"

            def query(self, query, runtime, env, messages, extra_args):
                seen.append(dict(extra_args))
                # 模拟 gate 用完任务 A 的预算。
                extra_args["remaining_escalations"] = 0
                return query, runtime, env, messages, extra_args

        # 不传 extra_args，与 run_task_with_pipeline 的调用方式一致。
        for task_id, prompt in (("task_A", "Pay rent."), ("task_B", "Check balance.")):
            pipeline = AgentPipeline(
                [
                    PolicyContextInitializer(task_id, escalation_budget=3),
                    InitQuery(),
                    UseUpBudget(),
                ]
            )
            pipeline.query(prompt, FunctionsRuntime([]))

        task_b = seen[1]
        self.assertEqual(task_b["task_id"], "task_B")
        self.assertEqual(task_b["user_request"], "Check balance.")
        self.assertEqual(task_b["remaining_escalations"], 3)


if __name__ == "__main__":
    unittest.main()
