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

## Still to do

This list is for picking the report up in another session. Each item says
where to work and what "done" looks like.

### 1. Extend `scripts/evaluate.py` for the † tables

The script comes from PR #14 (`feat/evaluation-script`). Build on that branch,
or on `main` once #14 is merged. Add three outputs, with tests in
`tests/test_evaluation.py`:

- **HITL load as the brief defines it**: total escalations across
  *successfully completed* tasks, per setup and `k`, as a new column in
  `hitl.csv`. Decide whether "completed" means `safe_task_completion` or
  `strict_utility`, and write the choice into the report.
- **`failures.csv`**: per setup and `k`, count the over-conservative failures
  (a safe call auto-denied, or the budget used up on safe calls so the task
  wasn't completed), the over-permissive failures (an unsafe call
  auto-approved without escalation) and other failed runs. Use
  `unauthorized_calls` in `run_summary.json` together with
  `policy_decisions.jsonl`, and document how each run is assigned a type.
- **`escalation_by_tool.csv`**: per setup and tool, the number of calls,
  escalations, the reviewer approve rate, and how many escalated calls were
  unsafe.

Then add these files to the table-to-CSV mapping above.

### 2. Run the final experiments

- Choose the `k` values (the brief asks for "a meaningful range") and set
  them in the setups file. Table 1 has columns for k = 0, 1, 2, 3 and 5 as a
  starting point.
- Run the test split with the DeepSeek models, then `evaluate.py --out
  results/test`. The command is in Appendix B of `main.tex`.

### 3. Fill in the report

- **Tables 1–6 and 11**: copy from the CSVs, using the mapping above.
- **Tables 7 and 10**: copy from the new CSVs in item 1.
- **Tables 8 and 9 (trajectories)**: pick one over-conservative and one
  over-permissive run from the test batch and fill in each step from its
  `policy_decisions.jsonl`, `reviewer_decisions.jsonl` and AgentDojo trace.
- **Text**: every `[TODO: ...]` in red. They cover the title page and
  abstract, introduction, gate integration, the three policy stages, the
  reviewer model and the tests that check what the reviewer sees, models and
  `k` values, the TCR discussion, failure assignment, escalation-value
  discussion, limitations and conclusion.
- **Reviewer tests**: the reviewer section refers to
  `tests/test_safety_invariants.py`, which is in PR #13. Check it's merged
  before submitting. The review of #13 suggested strengthening its E12 test
  so that it would also catch tool outputs leaking to the reviewer.

### 4. Build the PDF

`main.tex` hasn't been compiled yet, because the machine it was written on
has no TeX. Build it once (`latexmk -pdf main.tex`, or upload the `report/`
folder to Overleaf), then fix any overfull tables. The HITL and cost tables
have the most columns.
