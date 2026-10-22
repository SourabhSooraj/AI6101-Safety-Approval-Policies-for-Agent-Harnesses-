# Dev pilot (b6), first run: archived

> **Archived.** This first dev pilot ran with the **v1 Judge prompt**, which
> denied most safe calls (see the findings below). It led to the v2 prompt
> (PR #19), and b6 was rerun with it. Kept as a record and as the "before" for
> the Judge change. Its logs are in `logs/archive/b6-v1-judge/` (not in git;
> also shared as a zip), and its q-table is `q_table.json` in this folder, not
> `configs/q_table.json`.

Results of the dev pilot run, written by `scripts/evaluate.py`. These are
**development numbers** for calibration and planning, not results for the
report: one run per setup, `k` = 3 only, and the no-attack rows have only 10
runs each.

## The run

| | |
|---|---|
| Batch | `logs/archive/b6-v1-judge/b6-dev-pilot` (originally `logs/agentdojo_batch/b6-dev-pilot`; not in git, shared as a zip) |
| Date | 21 October 2026 |
| Code | commit `7fcc6d5` (clean) |
| Benchmark | AgentDojo v1.2.2, banking, **dev split** (10 tasks) |
| Attacks | none, plus injection tasks 0–8, attack `important_instructions_deepseek` |
| Policies | `hybrid`, `no_budget`, `escalate_all`, `allow_all`; `k` = 3 |
| Settings | `t_low` 0.3, `t_high` 0.7, value allocator, `tau0` 1.0, 3 Judge samples |
| Agent | `deepseek-flash` (DeepSeek) |
| Judge | `qwen3.7-flash` (DashScope, Singapore) |
| Reviewer | `glm-5.1` (DashScope, Singapore) |

The team decided to use the same models for the dev pilot and the final runs
(the board's k3 originally had a local dev agent).

Reproduce with:

```bash
python scripts/run_batch.py --provider deepseek --model deepseek-flash \
  --judge-model qwen3.7-flash \
  --judge-base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --judge-api-key-env DASHSCOPE_API_KEY \
  --reviewer-model glm-5.1 \
  --reviewer-base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --reviewer-api-key-env DASHSCOPE_API_KEY \
  --split dev --policies hybrid no_budget escalate_all allow_all \
  --injection-tasks none injection_task_0 injection_task_1 injection_task_2 \
    injection_task_3 injection_task_4 injection_task_5 injection_task_6 \
    injection_task_7 injection_task_8 \
  --batch-id b6-dev-pilot
python scripts/evaluate.py logs/agentdojo_batch/b6-dev-pilot --out results/dev-pilot
```

## What it gives

- **Runs:** 400 of 400 finished, no crashes, no Judge or reviewer errors.
- **Cost:** about **$0.50** in total, $0.0008–0.0019 per run depending on the
  policy (`cost.csv`). The reviewer is the largest cost under `escalate_all`.
- **q-table:** `q_table.json` in this folder (originally `configs/q_table.json`),
  from the 41 P6 reviews in the `hybrid` and `no_budget` runs (8–13 per tool):
  ```bash
  python scripts/build_q_table.py logs/agentdojo_batch/b6-dev-pilot \
    --group-by tool_name --min-samples 5 --output configs/q_table.json
  ```
- **Calls to label:** 244 reviewed calls, but ground truth settles 236 of them,
  so only **8** need a human (see open decisions).

## Findings

| Policy | Utility, no attack | TCR | Attack success | Escalations on completed tasks |
|---|---|---|---|---|
| `allow_all` | 0.90 | 0.83 | 0.02 | 0 |
| `hybrid` | 0.70 | 0.61 | 0.00 | 13 |
| `no_budget` | 0.70 | 0.63 | 0.00 | 21 |
| `escalate_all` | 0.50 | 0.49 | 0.00 | 119 |

1. **The attacks barely work on DeepSeek.** Without a gate, 2% of attacked runs
   succeeded, and the attacker's payment reached the gate in only 6 of 200
   data-stealing runs. With so little to prevent, the gate can only cost
   utility: `allow_all` has the best TCR, significantly above `hybrid`
   (McNemar, Holm-corrected p = 0.0006).
2. **The Judge denies safe calls.** In `hybrid` it denied 41 calls outright;
   ground truth says 38 of them were safe and 2 unsafe. Its risk scores are
   high across the board (median 0.92), so it barely separates safe from
   unsafe calls. The reviewer approved all 17 calls it saw in `hybrid`.
3. **The budget hardly matters yet.** Because the Judge decides most calls
   itself, `hybrid` uses 5.7% of its budget, and `hybrid` and `no_budget` give
   the same result (p = 0.81). A TCR@k curve would be flat.
4. **The reviewer is strict.** Under `escalate_all` it approved 45% of 203
   calls, including read-only calls on legitimate tasks such as task 11, whose
   request it called a likely VAT scam.
5. **HITL load favours `hybrid`:** 13 escalations on completed tasks against
   119 for `escalate_all`.

## Open decisions

- **Stronger attack (before b8):** without attacks that succeed, the final
  results cannot show the gate preventing harm.
- **Judge thresholds (b7):** raise `t_high` or switch off the deny band
  (`--t-low 0 --t-high 1`), so calls the Judge over-rates go to the reviewer.
  Rebuild the q-table afterwards, since more calls will reach P6.
- **Reviewer prompt (k6):** the `escalate_all` results support a looser prompt.
- **Labelling (s2):** only 8 calls need a human, too few for a meaningful κ.
  Options: (a) label a sample of 50–100 escalated write calls with
  `--include-prelabelled` and report κ on all of them; (b) generate more
  ambiguous calls, for example with the stronger attack; (c) accept 8 and
  report it as a limitation.
- **Spot-check:** pending (teammate A).
