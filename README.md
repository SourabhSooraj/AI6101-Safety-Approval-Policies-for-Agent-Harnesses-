# AI6101-Safety-Approval-Policies-for-Agent-Harnesses-

NTU - Introduction to AI & AI Ethics project.

## English

### Overview

This project evaluates approval policies for AgentDojo tool calls. The gate
returns one of three decisions:

- `ALLOW`: execute the tool call.
- `DENY`: block the tool call.
- `REQUIRE_APPROVAL`: send the call to human review.

The repository contains three controlled policy variants:

```text
p1_only  -> P1 rules only
p4_only  -> LLM Judge with fixed thresholds
hybrid   -> P1 -> LLM Judge -> P6 budget allocator
```

All variants use the same Harness, reviewer, escalation budget, AgentDojo
suites, and logging format. See "Policy variants for controlled comparison"
below for details.

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

### DeepSeek API key

The real AgentDojo pipeline and its LLM-backed Agent, Judge, and reviewer
require a DeepSeek API key. Set it in the current PowerShell session:

```powershell
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
```

To set it permanently for future terminals:

```powershell
setx DEEPSEEK_API_KEY "your-deepseek-api-key"
```

After `setx`, close and reopen the terminal.

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

Without a DeepSeek key, unit tests and P1-only development still work. A real
AgentDojo run with the DeepSeek Agent, Judge, or reviewer requires the key.

### Real AgentDojo banking end-to-end run

Run one complete AgentDojo task with DeepSeek as the agent, Judge, and
simulated reviewer:

```powershell
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
.\.venv\Scripts\python.exe scripts\run_agentdojo_e2e.py `
  --benchmark-version v1.2.2 `
  --suite banking `
  --task-id user_task_0 `
  --policy hybrid `
  --model deepseek-flash `
  --sample-count 1 `
  --k 3
```

`sample_count=1` keeps the first real run inexpensive. The command writes:

- `logs/agentdojo_e2e/<run_id>/policy_decisions.jsonl`
- `logs/agentdojo_e2e/<run_id>/reviewer_decisions.jsonl`
- `logs/agentdojo_e2e/<run_id>/agentdojo_trace/`

The task is successful when the script prints `utility=True` and the policy
logs show tool calls passing through P1, Judge, P6, and the reviewer.

DeepSeek agent thinking mode is disabled by default for stable tool-call
loops. Use `--agent-thinking` to enable it; the adapter then preserves
`reasoning_content` across all tool-call turns as required by DeepSeek.
The runner also appends an automated-benchmark instruction telling the agent
that no human is available for follow-up questions. Use `--no-auto-complete`
to run with the stock AgentDojo system message only.

This first end-to-end run temporarily uses DeepSeek for the agent, Judge, and
reviewer to validate the plumbing. For the final experiments, keep DeepSeek as
the agent/Judge if desired, but replace the simulated reviewer with a different
model family to reduce correlated reviewer behavior.

### Policy variants for controlled comparison

The runner can evaluate three controlled policy variants:

- `p1_only`: run P1 only. `ALLOW` and `DENY` are final; P1 escalations go
  directly to the reviewer. No Judge or P6 allocator is used.
- `p4_only`: run the LLM Judge only. Scores below `t_low` are allowed, scores
  above `t_high` are denied, and the middle band is escalated. No P1 or P6
  allocator is used.
- `hybrid`: the main P1 -> Judge -> P6 policy. Judge scores below `t_low` are
  allowed, scores above `t_high` are denied, and only the middle band reaches
  the P6 allocator.

Run one variant with:

```powershell
.\.venv\Scripts\python.exe scripts\run_agentdojo_e2e.py `
  --policy p4_only `
  --t-low 0.3 `
  --t-high 0.7 `
  --suite banking `
  --task-id user_task_0 `
  --k 3
```

Run a small comparison matrix with:

```powershell
.\.venv\Scripts\python.exe scripts\run_policy_comparison.py `
  --policies p1_only p4_only hybrid `
  --k-values 0 3 `
  --task-ids user_task_0 `
  --sample-count 1
```

The default Judge thresholds are stored in `configs/judge_thresholds.json`. Tune
them only on dev logs, then freeze them for test. Every JSONL decision record
includes `policy_variant`, so results from the three variants can be compared
without mixing files.

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

The reviewer writes its final verdict to the JSONL field `reviewer_verdict`.
Build the dev q table from the reviewer log with:

```powershell
.\.venv\Scripts\python.exe scripts\build_q_table.py `
  logs\dev_reviewer_decisions.jsonl `
  --group-by tool_name `
  --min-samples 5 `
  --output configs\q_table.json
```

Then load the JSON into `PolicyEngine(q_table=...)`.

## 中文

### 项目简介

本项目用于评估 AgentDojo 工具调用的审批策略。Gate 会返回三种结果：

- `ALLOW`：执行工具调用。
- `DENY`：阻止工具调用。
- `REQUIRE_APPROVAL`：升级人工审核。

仓库目前包含三种受控 Policy 变体：

```text
p1_only  -> 只运行 P1 规则
p4_only  -> LLM Judge 加固定阈值
hybrid   -> P1 -> LLM Judge -> P6 预算 allocator
```

三种变体共用 Harness、reviewer、escalation 预算、AgentDojo suite 和日志
格式。具体说明见后面的“用于受控对比的 Policy 变体”。

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

### 设置 DeepSeek API Key

真实 AgentDojo pipeline，以及其中的 LLM Agent、Judge 和 reviewer，需要
DeepSeek API Key。只对当前终端生效：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

如果希望以后新开的终端也生效：

```powershell
setx DEEPSEEK_API_KEY "你的 DeepSeek API Key"
```

执行 `setx` 后需要关闭并重新打开终端。

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

没有 DeepSeek Key 时，仍可运行单元测试和 P1 开发；真实 AgentDojo
运行中的 DeepSeek Agent、Judge 或 reviewer 需要 Key。

### 真实 AgentDojo banking 端到端运行

使用 DeepSeek 同时作为 agent、Judge 和模拟 reviewer，运行一个完整任务：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
.\.venv\Scripts\python.exe scripts\run_agentdojo_e2e.py `
  --benchmark-version v1.2.2 `
  --suite banking `
  --task-id user_task_0 `
  --policy hybrid `
  --model deepseek-flash `
  --sample-count 1 `
  --k 3
```

第一次真实运行使用 `sample_count=1` 可以控制费用。命令会写入：

- `logs/agentdojo_e2e/<run_id>/policy_decisions.jsonl`
- `logs/agentdojo_e2e/<run_id>/reviewer_decisions.jsonl`
- `logs/agentdojo_e2e/<run_id>/agentdojo_trace/`

当脚本输出 `utility=True`，并且日志显示工具调用经过 P1、Judge、P6 和
reviewer 时，说明端到端链路成功。

为了获得稳定的工具调用循环，DeepSeek agent 的 thinking mode 默认关闭。
传入 `--agent-thinking` 可以开启；适配器会在所有 tool-call 回合中保留
DeepSeek 要求的 `reasoning_content`。
运行脚本还会追加自动化 benchmark 提示，明确没有真人可以追问。传入
`--no-auto-complete` 可以只使用 AgentDojo 原始 system message。

这版端到端验证暂时让 agent、Judge 和 reviewer 都使用 DeepSeek，目的是先
跑通系统。正式实验时，agent/Judge 可以继续使用 DeepSeek，但应把模拟
reviewer 替换为不同模型家族，以降低 reviewer 与 policy 判断高度相关的问题。

### 用于受控对比的 Policy 变体

运行脚本支持三种受控变体：

- `p1_only`：只运行 P1。`ALLOW` 和 `DENY` 直接作为最终结果；P1 升级的
  调用直接进入 reviewer，不调用 Judge，也不使用 P6 allocator。
- `p4_only`：只运行 LLM Judge。低于 `t_low` 放行，高于 `t_high` 拒绝，
  中间区间升级；不使用 P1 和 P6 allocator。
- `hybrid`：当前主策略 P1 -> Judge -> P6。Judge 分数低于 `t_low` 时直接
  放行，高于 `t_high` 时直接拒绝，只有中间区间进入 P6 allocator。

运行单个变体：

```powershell
.\.venv\Scripts\python.exe scripts\run_agentdojo_e2e.py `
  --policy p4_only `
  --t-low 0.3 `
  --t-high 0.7 `
  --suite banking `
  --task-id user_task_0 `
  --k 3
```

运行小型对比矩阵：

```powershell
.\.venv\Scripts\python.exe scripts\run_policy_comparison.py `
  --policies p1_only p4_only hybrid `
  --k-values 0 3 `
  --task-ids user_task_0 `
  --sample-count 1
```

Judge 默认阈值保存在 `configs/judge_thresholds.json`。阈值只能使用 dev logs
调整，调好后在 test 上冻结。每条 JSONL 决策日志都包含
`policy_variant`，因此三种变体的结果不会混淆。

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

reviewer 会把最终审核结果写入 JSONL 的 `reviewer_verdict` 字段。使用
reviewer 日志生成 dev q table：

```powershell
.\.venv\Scripts\python.exe scripts\build_q_table.py `
  logs\dev_reviewer_decisions.jsonl `
  --group-by tool_name `
  --min-samples 5 `
  --output configs\q_table.json
```

之后通过 `PolicyEngine(q_table=...)` 加载这份 JSON。
