import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from harness.cost import (
    TokenUsage,
    estimate_deepseek_usage,
    estimate_usage,
    is_deepseek_peak,
    usage_from_openai_response,
)


class CostTests(unittest.TestCase):
    def test_peak_hours(self):
        self.assertTrue(
            is_deepseek_peak(datetime(2026, 10, 12, 2, tzinfo=timezone.utc))
        )
        self.assertFalse(
            is_deepseek_peak(datetime(2026, 10, 12, 5, tzinfo=timezone.utc))
        )
        self.assertFalse(
            is_deepseek_peak(datetime(2026, 10, 17, 2, tzinfo=timezone.utc))
        )

    def test_flash_off_peak_cost(self):
        usage = estimate_deepseek_usage(
            model="deepseek-flash",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            period="off_peak",
        )

        self.assertAlmostEqual(usage.estimated_cost_usd, 0.75)

    def test_cached_input_uses_cache_hit_rate(self):
        usage = estimate_deepseek_usage(
            model="deepseek-flash",
            input_tokens=1_000_000,
            cached_input_tokens=1_000_000,
            output_tokens=0,
            period="off_peak",
        )

        self.assertAlmostEqual(usage.estimated_cost_usd, 0.003)

    def test_usage_can_be_summed(self):
        first = TokenUsage("deepseek-flash", 10, 5, 2, 0.01)
        second = TokenUsage("deepseek-flash", 20, 8, 3, 0.02)

        total = first + second

        self.assertEqual(total.input_tokens, 30)
        self.assertEqual(total.output_tokens, 13)
        self.assertEqual(total.cached_input_tokens, 5)
        self.assertAlmostEqual(total.estimated_cost_usd, 0.03)

    def test_usage_from_openai_response(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            )
        )

        usage = usage_from_openai_response(
            response,
            model="deepseek-flash",
            period="off_peak",
        )

        self.assertEqual(usage.input_tokens, 1_000_000)
        self.assertEqual(usage.output_tokens, 1_000_000)
        self.assertAlmostEqual(usage.estimated_cost_usd, 0.75)

    def test_usage_from_response_without_usage_is_zero(self):
        usage = usage_from_openai_response(
            SimpleNamespace(),
            model="deepseek-flash",
        )

        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.estimated_cost_usd, 0.0)

    def test_qwen_flash_uses_short_input_tier(self):
        usage = estimate_usage(
            model="qwen3.7-flash",
            input_tokens=10_000,
            output_tokens=1_000,
        )

        self.assertAlmostEqual(
            usage.estimated_cost_usd,
            10_000 / 1e6 * 0.03 + 1_000 / 1e6 * 0.13,
        )

    def test_qwen_tier_depends_on_request_input_length(self):
        usage = estimate_usage(
            model="qwen3.7-flash",
            input_tokens=100_000,
            output_tokens=1_000_000,
        )

        # 整个请求都按 32K-256K 档计费，输出单价也随之变化。
        self.assertAlmostEqual(usage.estimated_cost_usd, 0.01 + 0.4)

    def test_qwen_cached_input_is_twenty_percent(self):
        usage = estimate_usage(
            model="qwen3.7-flash",
            input_tokens=20_000,
            cached_input_tokens=10_000,
            output_tokens=0,
        )

        self.assertAlmostEqual(
            usage.estimated_cost_usd,
            10_000 / 1e6 * 0.006 + 10_000 / 1e6 * 0.03,
        )

    def test_glm_reviewer_cost_ignores_deepseek_period(self):
        peak = estimate_usage(
            model="glm-5.1",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            period="peak",
        )
        off_peak = estimate_usage(
            model="glm-5.1",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            period="off_peak",
        )

        self.assertAlmostEqual(peak.estimated_cost_usd, 1.4 + 4.4)
        self.assertAlmostEqual(off_peak.estimated_cost_usd, 1.4 + 4.4)

    def test_unknown_model_costs_zero(self):
        usage = estimate_usage(
            model="llama3.1:8b",
            input_tokens=1_000,
            output_tokens=1_000,
        )

        self.assertEqual(usage.input_tokens, 1_000)
        self.assertEqual(usage.estimated_cost_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
