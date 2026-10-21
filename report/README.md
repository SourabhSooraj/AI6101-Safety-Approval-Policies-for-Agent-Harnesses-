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

| Report table or figure | Source in `results/test/` |
|---|---|
| TCR@k curve (figure) | `tcr_at_k.pdf` (included automatically) |
| TCR@k by policy | `tcr_at_k.csv`: `tcr` (k = 0, 1, 3, 5; `allow_all` takes one value for every k) |
| Outcomes | `overview.csv`: `runs`, `crashed`, `utility_no_attack`, `asr_finished`, `asr_all_runs` |
| Outcomes by attack type | `by_attack_type.csv` (`attack_mode`, `attack_type`) |
| HITL load | `hitl.csv`: `escalations_completed` (the brief's "Total"), `escalations_per_run`, `budget_share_used`, `auto_denied`, `runs_out_of_budget`, `reviewer_approved` |
| API cost | `cost.csv`: `agent_usd`, `judge_usd`, `reviewer_usd`, the `*_input_tokens` / `*_output_tokens` columns, `usd_per_run` |
| McNemar | `mcnemar.csv`: `pairs`, `only_a`, `only_b`, `p`, `p_holm` |
| Outcomes by user task (appendix) | `by_user_task.csv` |

Two tables in the reviewer section come from other scripts:

| Report table | Source |
|---|---|
| Reviewer agreement (humans vs reviewer) | `agreement.json` from `scripts/label_agreement.py --reviewer-logs` on the dev labels: `pairs` (A vs B), `reviewer.labellers_vs_reviewer`, `reviewer.agreed_humans_vs_reviewer` |
| Reviewer consistency and persuasion † | `scripts/replay_reviewer.py` (PR #18): a consistency run, plus the same denied calls with and without `--persuade` |

Tables marked † need output that isn't available yet:

- **Failure classification**: over-conservative and over-permissive failures
  per policy (`evaluate.py`).
- **Escalation value**: escalations, reviewer approvals and unsafe calls per
  tool (`evaluate.py`).
- **Reviewer consistency and persuasion**: `scripts/replay_reviewer.py`, once
  PR #18 is merged, with a control run for the persuasion test.

The representative trajectories (one table per failure type) are filled by
hand from a run's `policy_decisions.jsonl`, `reviewer_decisions.jsonl` and
AgentDojo trace.

Text still to write is marked `[TODO: ...]` in red in the PDF.

## Still to do

This list is for picking the report up in another session. Each item says
where to work and what "done" looks like.

### 1. Extend `scripts/evaluate.py` for the † tables

Add two outputs, with tests in `tests/test_evaluation.py`, then add them to the
mapping above:

- **`failures.csv`**: per setup and `k`, count the over-conservative failures
  (a safe call auto-denied, or the budget used up on safe calls so the task
  wasn't completed), the over-permissive failures (an unsafe call
  auto-approved without escalation) and other failed runs. Use
  `unauthorized_calls` in `run_summary.json` together with
  `policy_decisions.jsonl`, and document how each run is assigned a type.
- **`escalation_by_tool.csv`**: per setup and tool, the number of calls,
  escalations, the reviewer approve rate, and how many escalated calls were
  unsafe.

(HITL load over completed tasks is done: `escalations_completed` in
`hitl.csv`.)

### 2. Run the final experiments

- Create the final setups file with the agreed sweep, k = 0, 1, 3, 5, for
  every policy (`configs/setups/policy_comparison.json` only has k = 3), and
  put its name in Appendix B of `main.tex`.
- Decide the number of repeats; the run-to-run spread needs room in the
  tables.
- Run the test split with the agreed models (agent on DeepSeek; Judge
  `qwen3.7-flash` and reviewer `glm-5.1` on DashScope). The full command is in
  Appendix B. Then `evaluate.py --out results/test`.

### 3. Fill in the report

- **Tables from `evaluate.py`**: copy from the CSVs, using the mapping above.
- **Reviewer agreement**: run `label_agreement.py --reviewer-logs` on the
  dev labels.
- **Trajectories**: pick one over-conservative and one over-permissive run
  from the test batch and fill in each step from its logs.
- **Text**: every `[TODO: ...]` in red: title page and abstract,
  introduction, gate integration, the three policy stages, the reviewer
  boundary tests, the ToolEmu comparison, the TCR discussion, failure
  assignment, escalation-value discussion, limitations and conclusion.
- **Reviewer tests**: the reviewer section refers to
  `tests/test_safety_invariants.py` (merged with PR #13). The review of #13
  suggested strengthening its E12 test so that it would also catch tool
  outputs leaking to the reviewer.

### 4. Build the PDF

`main.tex` hasn't been compiled yet: nobody on the team has TeX installed.
Upload the `report/` folder to Overleaf (or run `latexmk -pdf main.tex`), then
fix any overfull tables. The HITL and cost tables have the most columns.
