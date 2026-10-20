# Report

`main.tex` is the skeleton of the project report: every section the brief
(4.2) asks for, the simulated-reviewer write-up, and empty tables and figures
that match the outputs of `scripts/evaluate.py`.

## Building

Any TeX distribution with the standard packages works (TeX Live, MacTeX,
MiKTeX, or Overleaf):

```bash
cd report
latexmk -pdf main.tex
```

The TCR@k figure is read from `../results/test/tcr_at_k.pdf`. Until that file
exists, the PDF shows a placeholder box, so the report builds at any stage.
To use another results folder, change `\resultsdir` at the top of `main.tex`.

## Filling in the results

Produce the results folder from the final test batch:

```bash
python scripts/evaluate.py logs/agentdojo_batch/<test_batch> --out results/test
```

Then copy the numbers into the cells marked `\tbd`. Rates go in as
`rate [low, high]`, from the CSV's `<metric>`, `<metric>_low` and
`<metric>_high` columns.

| Report table | Source in `results/test/` |
|---|---|
| Figure 1, TCR@k curve | `tcr_at_k.pdf` (included automatically) |
| Table 1, TCR@k by policy | `tcr_at_k.csv`: `tcr` |
| Table 2, outcomes | `overview.csv`: `runs`, `crashed`, `utility_no_attack`, `asr_finished`, `asr_all_runs` |
| Table 3, by attack type | `by_attack_type.csv` |
| Table 4, HITL load | `hitl.csv`: `escalations_per_run`, `budget_share_used`, `auto_denied`, `runs_out_of_budget`, `reviewer_approved` |
| Table 5, API cost | `cost.csv`: `agent_usd`, `judge_usd`, `reviewer_usd`, the `*_input_tokens` / `*_output_tokens` columns, `usd_per_run` |
| Table 6, McNemar | `mcnemar.csv`: `pairs`, `only_a`, `only_b`, `p`, `p_holm` |
| Table 11 (appendix), by user task | `by_user_task.csv` |

Tables and cells marked † need output that `evaluate.py` does not produce yet:

- **HITL "Total"** (Table 4): the brief defines HITL load as the total number
  of escalations across *successfully completed* tasks; `hitl.csv` currently
  counts escalations over all runs.
- **Failure classification** (Table 7): over-conservative and over-permissive
  failures per policy.
- **Escalation value** (Table 10): escalations, reviewer approvals and unsafe
  calls per tool.

The representative trajectories (Tables 8 and 9) are filled by hand from a
run's `policy_decisions.jsonl`, `reviewer_decisions.jsonl` and AgentDojo
trace.

Text still to write is marked `[TODO: ...]` in red in the PDF.
