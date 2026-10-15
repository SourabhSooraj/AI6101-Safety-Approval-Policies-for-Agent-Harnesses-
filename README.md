# AI6101-Safety-Approval-Policies-for-Agent-Harnesses-

NTU - Introduction to AI & AI Ethics project.

## English

### Overview

This project evaluates approval policies for AgentDojo tool calls. The gate
returns one of three decisions:

- `ALLOW`: execute the tool call.
- `DENY`: block the tool call.
- `REQUIRE_APPROVAL`: send the call to human review.

The current policy pipeline is:

```text
P1 rulebase
-> DeepSeek LLM Judge for P1 escalations
-> P6 budget allocator
-> ALLOW / DENY / REQUIRE_APPROVAL
```

### Setup

Run the following commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The PowerShell commands below use `.\.venv\Scripts\python.exe`; on macOS or
Linux use `.venv/bin/python` with forward slashes instead.

### DeepSeek API key

The real LLM Judge requires a DeepSeek API key. Set it in the current
PowerShell session:

```powershell
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
```

To set it permanently for future terminals:

```powershell
setx DEEPSEEK_API_KEY "your-deepseek-api-key"
```

After `setx`, close and reopen the terminal.

On macOS or Linux:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

Add the same line to `~/.zshrc` or `~/.bashrc` to keep it for new terminals.

Never commit the API key, `.env`, or any secret to GitHub.

### Run tests

Unit tests use fake Judge clients and do not require a DeepSeek API key:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Run the real DeepSeek smoke test

The smoke test makes up to two real DeepSeek requests with `sample_count=1`:

- one request for the LLM Judge;
- one request for the simulated reviewer when P6 escalates the call.

It verifies the complete
`P1 -> LLM Judge -> P6 allocator -> Simulated Reviewer` path:

```powershell
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
.\.venv\Scripts\python.exe scripts\smoke_test_deepseek_policy.py
```

A successful run prints results with `stage=p6` and, when the call is
escalated, a `Reviewer result` block. If the script reports
`p6_force_escalation`, the API request or Judge response parsing failed.

Without a DeepSeek key, tests and P1-only development still work. Only real
LLM Judge experiments require the key.

### Real AgentDojo banking end-to-end run

Run one complete AgentDojo task with DeepSeek as the agent, Judge, and
simulated reviewer:

```powershell
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
.\.venv\Scripts\python.exe scripts\run_agentdojo_e2e.py `
  --benchmark-version v1.2.2 `
  --suite banking `
  --task-id user_task_0 `
  --model deepseek-flash `
  --k 3
```

The Judge samples each escalated call 3 times by default. `--sample-count 1`
is cheaper but disables Judge disagreement detection, and the script prints a
warning. The command writes:

- `logs/agentdojo_e2e/<run_id>/policy_decisions.jsonl` (Judge cost per call)
- `logs/agentdojo_e2e/<run_id>/reviewer_decisions.jsonl` (reviewer cost per review)
- `logs/agentdojo_e2e/<run_id>/run_summary.json` (utility, attack result, agent token cost)

The total cost of a run is the sum of `estimated_cost_usd` in both JSONL files
plus `agent_usage.estimated_cost_usd` in the summary.

The task is successful when the script prints `utility=True` and the policy
logs show tool calls passing through P1, Judge, P6, and the reviewer.

Other options:

- `--policy` picks the setup to run (default `full`):

  | Policy | What it does | Budget |
  |---|---|---|
  | `full` | P1 -> Judge -> P6 allocator | `--k` |
  | `allow_all` | No gate: every tool call runs | not used |
  | `escalate_all` | Read-only calls run; every other call goes to the reviewer, including ones P1 would deny | unlimited |
  | `p1_only` | P1 only; calls P1 cannot settle go straight to the reviewer, no Judge | `--k` |
  | `no_budget` | `full` with an unlimited budget, so P6 escalates almost every risky call | unlimited |

  The policy name is written to every log line and to `run_summary.json`.
- `--injection-task injection_task_0` runs the task under a prompt-injection
  attack. `--attack` picks the AgentDojo attack (default
  `important_instructions_deepseek`, the standard `important_instructions`
  template addressed to DeepSeek). The script prints `attack_succeeded`, which
  is `None` when no attack was run.
- `--q-table configs\q_table.json` loads a q table built by
  `scripts/build_q_table.py`. Tools missing from the table use `q=1.0`.
- `--tau0` sets the P6 allocator's base threshold (default `1.0`).
- `--reviewer-model`, `--reviewer-base-url` and `--reviewer-api-key-env` point
  the simulated reviewer at a different OpenAI-compatible model. By default the
  reviewer uses the same DeepSeek model as the agent and Judge, and the script
  prints a warning. Use a different model family for the final experiments to
  reduce correlated reviewer behavior.

### Escalation budget `k`

Configure the per-task escalation budget on the executor:

```python
executor = PolicyGateToolsExecutor(
    policy_engine=engine,
    tool_metadata=tool_metadata,
    escalation_budget=k,
)
```

The executor keeps these values in `extra_args` for the lifetime of the task:

- `initial_escalations`
- `remaining_escalations`
- `escalations_used`

Each `REQUIRE_APPROVAL` consumes one escalation. When the budget reaches zero,
later `REQUIRE_APPROVAL` decisions are converted to `DENY`.

**Team decision: what P6 does when the budget is 0.** P6 makes its normal
decision and only denies the calls it would have escalated
(`p6_budget_exhausted`): calls with `value > tau`, and calls where the Judge
failed or disagreed, so the policy stays fail-closed. Calls that would not have
been escalated keep the normal rule: `ALLOW` if risk is below the
`allow_threshold` (0.5), otherwise `DENY`. This follows the brief, which only
requires denying calls that would otherwise have been escalated. The earlier
behaviour, denying every call that reached P6 once the budget was 0, made
nearly every money-moving task fail at `k=0`.

`expected_remaining_risky_calls` also counts down: each call that P1 cannot
settle on its own (it reaches P6, or P1 escalates it when no Judge is
configured) lowers it by one, down to zero.

### Simulated human reviewer

The reviewer is an LLM client that returns only `APPROVE` or `DENY`:

```python
from harness.deepseek_judge_client import DeepSeekJudgeClient
from harness.simulated_reviewer import SimulatedReviewer

reviewer = SimulatedReviewer(
    client=DeepSeekJudgeClient(model="deepseek-flash"),
    model_name="deepseek-flash-reviewer",
)

executor = PolicyGateToolsExecutor(
    policy_engine=engine,
    tool_metadata=tool_metadata,
    escalation_budget=k,
    reviewer=reviewer,
    review_logger=DecisionLogger("logs/dev_reviewer_decisions.jsonl"),
)
```

An `APPROVE` decision executes the tool. `DENY`, invalid JSON, API errors, or
client failures block the call. For the simulated reviewer, prefer a model
family different from the policy/judge model. The reviewer log contains the
`reviewer_verdict` field needed by the q-table estimator.

To replace the reviewer with another model, keep `SimulatedReviewer` and inject
a different client. For another OpenAI-compatible provider:

```python
import os

reviewer_client = DeepSeekJudgeClient(
    model="<reviewer-model>",
    base_url="<provider-openai-compatible-base-url>",
    api_key=os.environ["REVIEWER_API_KEY"],
)
reviewer = SimulatedReviewer(
    client=reviewer_client,
    model_name="<reviewer-model>",
)
```

No Executor or Policy changes are required. For a non-OpenAI-compatible
provider, add a small adapter class that implements `complete(prompt) -> str`.

### Decision logs and cost tracking

Pass a `DecisionLogger` to `PolicyEngine` to write one JSONL record per
decision:

```python
from harness.decision_logger import DecisionLogger

logger = DecisionLogger(
    "logs/dev_policy_decisions.jsonl",
    run_id="dev-run-1",
    extra_context={"suite": "banking", "attack_mode": "important_instructions"},
)
```

Real DeepSeek responses automatically record input tokens, output tokens,
cached input tokens, and estimated USD cost. Pricing is configured in
`harness/cost.py`; verify the official DeepSeek pricing page before large runs.

The reviewer log stores the final verdict in `reviewer_verdict` and whether the
review succeeded in `reviewer_success`. Build the dev q table with:

```powershell
.\.venv\Scripts\python.exe scripts\build_q_table.py `
  logs\dev_reviewer_decisions.jsonl `
  --group-by tool_name `
  --min-samples 5 `
  --output configs\q_table.json
```

Records with `reviewer_success: false` are skipped, because a failed review
defaults to `DENY` and is not a real reviewer judgement. Records whose `stage`
is not `p6` (from `p1_only` or `escalate_all` runs) are skipped too, because
their `risk_score` is a fixed placeholder, not a Judge score. `--baseline-threshold`
(default `0.5`) must match the allocator's `allow_threshold`.

Then load the JSON with `run_agentdojo_e2e.py --q-table configs\q_table.json`,
or in code with `PolicyEngine(q_table=load_q_table(path))`.

### Running several tasks in one process

AgentDojo's `AgentPipeline.query` uses a shared default `extra_args={}`, and
`run_task_with_pipeline` never passes its own. `PolicyContextInitializer`
therefore always returns a fresh `extra_args`, so each task starts with its own
budget, task id and user request. Build a new pipeline (and a new
`DeepSeekAgentLLM`, whose `total_usage` accumulates) for every episode so the
task id and agent cost belong to that episode.

If the agent produces no final answer, AgentDojo retries the same task up to
three times. Each retry starts a fresh `extra_args`, so the escalation budget
resets on a retry.

### Design choices and limitations

- **The reviewer and the Judge never see tool outputs.** Their prompts contain
  the user request, the current call, the tool metadata and the names and
  arguments of earlier calls. This keeps injected text away from them, but a
  real human reviewer would usually see more context. `trace_summary` is not set
  by the e2e script.
- **Tool metadata covers the banking suite only.** Tools from other AgentDojo
  suites have no metadata, so P1 treats every one of them as unknown and
  escalates it. Keep experiments to banking, or add metadata first.
- **`update_password` needs an explicit password mention.** P1's destructive
  rule only lets it through when the request contains "password", "passcode" or
  "密码". In banking v1.2.2 only `user_task_14` needs it, and its prompt mentions
  a password. The check is plain substring matching, so a request such as
  "don't touch my password" still passes P1 and is left to the Judge.
- **Attacks address the model as "DeepSeek".** AgentDojo's
  `important_instructions` attacks, including
  `important_instructions_no_model_name`, look up the model name from the
  pipeline name and raise `ValueError` for DeepSeek. Use
  `important_instructions_deepseek` (the default) instead of renaming the
  pipeline to another model's id.
- **Reading AgentDojo's `security` flag.** `security=True` means the attack
  succeeded, and AgentDojo also returns `True` for every run without an attack.
  The e2e script reports it as `attack_succeeded`, which is `None` when no
  attack was run. A safe outcome is `utility=True` with
  `attack_succeeded=False` on attacked runs.

## 中文

### 项目简介

本项目用于评估 AgentDojo 工具调用的审批策略。Gate 会返回三种结果：

- `ALLOW`：执行工具调用。
- `DENY`：阻止工具调用。
- `REQUIRE_APPROVAL`：升级人工审核。

当前 Policy 流程为：

```text
P1 rulebase
-> P1 升级的调用进入 DeepSeek LLM Judge
-> P6 预算 allocator
-> ALLOW / DENY / REQUIRE_APPROVAL
```

### 创建环境

在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 不允许运行 `Activate.ps1`，先执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

然后重新激活虚拟环境。

macOS 或 Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

下文的 PowerShell 命令使用 `.\.venv\Scripts\python.exe`；在 macOS 或 Linux 上
改用 `.venv/bin/python`，路径使用正斜杠。

### 设置 DeepSeek API Key

真实 LLM Judge 需要 DeepSeek API Key。只对当前终端生效：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

如果希望以后新开的终端也生效：

```powershell
setx DEEPSEEK_API_KEY "你的 DeepSeek API Key"
```

执行 `setx` 后需要关闭并重新打开终端。

macOS 或 Linux：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

把同一行加入 `~/.zshrc` 或 `~/.bashrc`，新终端也会生效。

不要把 API Key、`.env` 或其他秘密信息提交到 GitHub。

### 运行测试

单元测试使用 fake Judge，不需要 DeepSeek API Key：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 运行真实 DeepSeek smoke test

Smoke test 在 `sample_count=1` 下最多发起两次真实 DeepSeek 请求：

- 一次用于 LLM Judge；
- P6 升级时，再发起一次用于模拟 reviewer。

它验证完整的
`P1 -> LLM Judge -> P6 allocator -> Simulated Reviewer` 链路：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
.\.venv\Scripts\python.exe scripts\smoke_test_deepseek_policy.py
```

成功运行时输出中会包含 `stage=p6`；如果调用被升级，还会输出
`Reviewer result`。如果出现 `p6_force_escalation`，通常表示 API 请求失败
或 Judge 返回内容无法解析。

没有 DeepSeek Key 时，仍可运行测试和进行 P1 开发；只有真实 LLM Judge
实验需要 Key。

### 真实 AgentDojo banking 端到端运行

使用 DeepSeek 同时作为 agent、Judge 和模拟 reviewer，运行一个完整任务：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
.\.venv\Scripts\python.exe scripts\run_agentdojo_e2e.py `
  --benchmark-version v1.2.2 `
  --suite banking `
  --task-id user_task_0 `
  --model deepseek-flash `
  --k 3
```

Judge 默认对每个升级调用采样 3 次。`--sample-count 1` 更便宜，但会关闭
Judge 分歧检测，脚本会输出警告。命令会写入：

- `logs/agentdojo_e2e/<run_id>/policy_decisions.jsonl`（每次调用的 Judge 成本）
- `logs/agentdojo_e2e/<run_id>/reviewer_decisions.jsonl`（每次审核的 reviewer 成本）
- `logs/agentdojo_e2e/<run_id>/run_summary.json`（utility、攻击结果、agent token 成本）

一次运行的总成本 = 两份 JSONL 中 `estimated_cost_usd` 之和 + summary 中的
`agent_usage.estimated_cost_usd`。

当脚本输出 `utility=True`，并且日志显示工具调用经过 P1、Judge、P6 和
reviewer 时，说明端到端链路成功。

其他选项：

- `--policy` 选择要运行的设置（默认 `full`）：

  | Policy | 作用 | 预算 |
  |---|---|---|
  | `full` | P1 -> Judge -> P6 allocator | `--k` |
  | `allow_all` | 没有 gate：所有工具调用直接执行 | 不使用 |
  | `escalate_all` | 只读调用直接执行；其余调用全部交给 reviewer，包括 P1 会拒绝的调用 | 无限 |
  | `p1_only` | 只用 P1；P1 无法决定的调用直接交给 reviewer，不使用 Judge | `--k` |
  | `no_budget` | 预算无限的 `full`，P6 会把几乎所有风险调用升级 | 无限 |

  policy 名称会写入每一行日志和 `run_summary.json`。
- `--injection-task injection_task_0`：在 prompt injection 攻击下运行任务。
  `--attack` 选择 AgentDojo 攻击（默认 `important_instructions_deepseek`，即
  标准 `important_instructions` 模板，模型名写为 DeepSeek）。脚本输出
  `attack_succeeded`，未启用攻击时为 `None`。
- `--q-table configs\q_table.json`：加载 `scripts/build_q_table.py` 生成的
  q table，表中没有的工具使用 `q=1.0`。
- `--tau0`：设置 P6 allocator 的基础阈值（默认 `1.0`）。
- `--reviewer-model`、`--reviewer-base-url`、`--reviewer-api-key-env`：让模拟
  reviewer 使用另一个兼容 OpenAI API 的模型。默认 reviewer 与 agent/Judge
  使用同一个 DeepSeek 模型，脚本会输出警告。正式实验应换用不同模型家族，
  以降低 reviewer 与 policy 判断高度相关的问题。

### Escalation 预算 `k`

在 Executor 上配置每个任务的升级预算：

```python
executor = PolicyGateToolsExecutor(
    policy_engine=engine,
    tool_metadata=tool_metadata,
    escalation_budget=k,
)
```

Executor 会把以下状态保存在任务的 `extra_args` 中：

- `initial_escalations`
- `remaining_escalations`
- `escalations_used`

每次 `REQUIRE_APPROVAL` 消耗一次升级额度。额度耗尽后，后续
`REQUIRE_APPROVAL` 会自动转成 `DENY`。

**团队决定：预算为 0 时 P6 的行为。** P6 照常决策，只拒绝本应升级的调用
（`p6_budget_exhausted`）：`value > tau` 的调用，以及 Judge 失败或分歧的调用，
因此仍然 fail closed。本来不会升级的调用照常处理：风险低于 `allow_threshold`
（0.5）时 `ALLOW`，否则 `DENY`。这与 brief 的要求一致：只需拒绝本应升级的调用。
之前的行为是预算为 0 后拒绝所有进入 P6 的调用，导致 `k=0` 时几乎所有涉及转账的
任务都失败。

`expected_remaining_risky_calls` 也会递减：每个 P1 无法单独决定的调用（进入
P6，或未配置 Judge 时被 P1 升级）都会让它减一，最低为 0。

### LLM 模拟 human reviewer

Reviewer 是一个只返回 `APPROVE` 或 `DENY` 的 LLM client：

```python
from harness.deepseek_judge_client import DeepSeekJudgeClient
from harness.simulated_reviewer import SimulatedReviewer

reviewer = SimulatedReviewer(
    client=DeepSeekJudgeClient(model="deepseek-flash"),
    model_name="deepseek-flash-reviewer",
)

executor = PolicyGateToolsExecutor(
    policy_engine=engine,
    tool_metadata=tool_metadata,
    escalation_budget=k,
    reviewer=reviewer,
    review_logger=DecisionLogger("logs/dev_reviewer_decisions.jsonl"),
)
```

Reviewer 返回 `APPROVE` 时才执行工具；返回 `DENY`、JSON 非法、API 异常或
client 失败时都阻止执行。模拟 reviewer 最好使用与 policy/judge 不同的模型
家族。reviewer 日志包含 q-table 估计所需的 `reviewer_verdict` 字段。

替换 reviewer 模型时不需要修改 Executor 或 Policy，只需要给
`SimulatedReviewer` 注入另一个 client。如果新 provider 兼容 OpenAI API：

```python
import os

reviewer_client = DeepSeekJudgeClient(
    model="<reviewer-model>",
    base_url="<provider-openai-compatible-base-url>",
    api_key=os.environ["REVIEWER_API_KEY"],
)
reviewer = SimulatedReviewer(
    client=reviewer_client,
    model_name="<reviewer-model>",
)
```

如果 provider 不兼容 OpenAI API，则新增一个实现
`complete(prompt) -> str` 的适配器即可。

### 决策日志与成本统计

给 `PolicyEngine` 传入 `DecisionLogger`，每次决定都会写入一条 JSONL：

```python
from harness.decision_logger import DecisionLogger

logger = DecisionLogger(
    "logs/dev_policy_decisions.jsonl",
    run_id="dev-run-1",
    extra_context={"suite": "banking", "attack_mode": "important_instructions"},
)
```

真实 DeepSeek 响应会自动记录输入 token、输出 token、缓存命中输入 token
和估算美元成本。价格配置位于 `harness/cost.py`，正式大规模运行前应重新
核对 DeepSeek 官方价格页。

reviewer 日志把最终审核结果写入 `reviewer_verdict`，把审核是否成功写入
`reviewer_success`。用以下命令生成 dev q table：

```powershell
.\.venv\Scripts\python.exe scripts\build_q_table.py `
  logs\dev_reviewer_decisions.jsonl `
  --group-by tool_name `
  --min-samples 5 `
  --output configs\q_table.json
```

`reviewer_success: false` 的记录会被跳过，因为审核失败时默认 `DENY`，并不是
reviewer 的真实判断。`stage` 不是 `p6` 的记录（来自 `p1_only` 或 `escalate_all`
运行）也会被跳过，因为它们的 `risk_score` 是固定占位值，不是 Judge 分数。`--baseline-threshold`（默认 `0.5`）必须与 allocator 的
`allow_threshold` 一致。

之后用 `run_agentdojo_e2e.py --q-table configs\q_table.json` 加载，或在代码中
使用 `PolicyEngine(q_table=load_q_table(path))`。

### 在同一进程中运行多个任务

AgentDojo 的 `AgentPipeline.query` 使用共享的默认参数 `extra_args={}`，而
`run_task_with_pipeline` 从不传入自己的 `extra_args`。因此
`PolicyContextInitializer` 总是返回新的 `extra_args`，让每个任务从自己的预算、
task id 和用户请求开始。每个 episode 都应新建 pipeline（以及新的
`DeepSeekAgentLLM`，它的 `total_usage` 会累加），使 task id 和 agent 成本属于
该 episode。

如果 agent 没有给出最终回答，AgentDojo 会对同一任务最多重试三次。每次重试都会
得到新的 `extra_args`，因此升级预算会在重试时重置。

### 设计选择与局限

- **Reviewer 和 Judge 看不到工具输出。** 它们的 prompt 只包含用户请求、当前调用、
  工具 metadata 以及之前调用的名称和参数。这样可以让注入文本远离它们，但真实的人类
  reviewer 通常能看到更多上下文。e2e 脚本不会设置 `trace_summary`。
- **工具 metadata 只覆盖 banking suite。** 其他 AgentDojo suite 的工具没有
  metadata，P1 会把它们全部视为未知工具并升级。实验请限于 banking，或先补充 metadata。
- **`update_password` 需要明确提到密码。** 只有请求中包含 "password"、"passcode"
  或 "密码" 时，P1 的 destructive 规则才放行。banking v1.2.2 中只有 `user_task_14`
  需要它，且其 prompt 提到了密码。该检查是简单的子串匹配，因此 "don't touch my
  password" 这样的请求仍会通过 P1，交给 Judge 判断。
- **攻击文本称模型为 "DeepSeek"。** AgentDojo 的 `important_instructions` 系列攻击
  （包括 `important_instructions_no_model_name`）会从 pipeline 名称查找模型名，
  对 DeepSeek 会抛出 `ValueError`。请使用 `important_instructions_deepseek`（默认值），
  不要把 pipeline 改名为其他模型的 id。
- **AgentDojo 的 `security` 标志。** `security=True` 表示攻击成功，且没有攻击的运行
  也总是返回 `True`。e2e 脚本把它输出为 `attack_succeeded`，未启用攻击时为 `None`。
  有攻击时，安全的结果是 `utility=True` 且 `attack_succeeded=False`。
