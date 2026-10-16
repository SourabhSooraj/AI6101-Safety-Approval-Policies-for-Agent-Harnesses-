import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from harness.cost import (
    TokenUsage,
    estimate_deepseek_usage,
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


if __name__ == "__main__":
    unittest.main()
