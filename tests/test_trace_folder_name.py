import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.cost import TokenUsage
from harness.episode import (
    add_episode_args,
    config_from_args,
    run_episode,
    trace_folder_name,
)


class TraceFolderNameTests(unittest.TestCase):
    def test_colon_in_ollama_model_id_is_replaced(self):
        self.assertEqual(trace_folder_name("qwen3:8b-hybrid"), "qwen3_8b-hybrid")

    def test_every_windows_reserved_character_is_replaced(self):
        self.assertEqual(trace_folder_name('a<b>c:d"e/f\\g|h?i*j'), "a_b_c_d_e_f_g_h_i_j")

    def test_trailing_dots_and_spaces_are_dropped(self):
        # Windows 会静默去掉目录名末尾的点和空格。
        self.assertEqual(trace_folder_name("model. "), "model")

    def test_plain_names_are_unchanged(self):
        self.assertEqual(
            trace_folder_name("deepseek-flash-p1_only"),
            "deepseek-flash-p1_only",
        )


class FinalAnswerAgent:
    """不调用工具、直接给出答案的 agent，替代 DeepSeekAgentLLM。"""

    name = "fake-agent"

    def __init__(self, client, model, **kwargs):
        self.total_usage = TokenUsage.zero(model)

    def query(self, query, runtime, env, messages=(), extra_args=None):
        answer = {
            "role": "assistant",
            "content": [{"type": "text", "content": "Done."}],
            "tool_calls": None,
        }
        return query, runtime, env, [*messages, answer], extra_args or {}


class UnusedJudgeClient:
    def __init__(self, **kwargs):
        self.last_usage = None

    def complete(self, prompt):
        raise AssertionError("no tool call should reach the Judge")


class RunEpisodeTraceTests(unittest.TestCase):
    def test_local_model_trace_folder_has_no_colon(self):
        parser = argparse.ArgumentParser()
        add_episode_args(parser)
        config = config_from_args(parser.parse_args(["--provider", "ollama"]))
        self.assertIn(":", config.agent_model)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "harness.episode.DeepSeekAgentLLM", FinalAnswerAgent
        ), patch("harness.episode.DeepSeekJudgeClient", UnusedJudgeClient):
            log_dir = Path(tmp)
            summary = run_episode(
                config,
                task_id="user_task_1",
                policy="hybrid",
                injection_task_id=None,
                run_id="trace-test",
                log_dir=log_dir,
            )

            traces = list((log_dir / "agentdojo_trace").rglob("*.json"))
            self.assertEqual(len(traces), 1)
            relative = traces[0].relative_to(log_dir)
            self.assertNotIn(":", str(relative))
            self.assertEqual(relative.parts[1], "qwen3_8b-hybrid")
            trace = json.loads(traces[0].read_text(encoding="utf-8"))
            self.assertEqual(trace["pipeline_name"], "qwen3_8b-hybrid")
            # 日志中的模型 id 不变。
            self.assertEqual(summary["agent_model"], "qwen3:8b")


if __name__ == "__main__":
    unittest.main()
