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
  --sample-count 1 `
  --k 3
```

`sample_count=1` keeps the first real run inexpensive. The command writes:

- `logs/agentdojo_e2e/<run_id>/policy_decisions.jsonl`
- `logs/agentdojo_e2e/<run_id>/reviewer_decisions.jsonl`

The task is successful when the script prints `utility=True` and the policy
logs show tool calls passing through P1, Judge, P6, and the reviewer.

This first end-to-end run temporarily uses DeepSeek for the agent, Judge, and
reviewer to validate the plumbing. For the final experiments, keep DeepSeek as
the agent/Judge if desired, but replace the simulated reviewer with a different
model family to reduce correlated reviewer behavior.

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

When the reviewer flow exists, store its final verdict in the JSONL field
`reviewer_verdict`. Build the dev q table with:

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
  --sample-count 1 `
  --k 3
```

第一次真实运行使用 `sample_count=1` 可以控制费用。命令会写入：

- `logs/agentdojo_e2e/<run_id>/policy_decisions.jsonl`
- `logs/agentdojo_e2e/<run_id>/reviewer_decisions.jsonl`

当脚本输出 `utility=True`，并且日志显示工具调用经过 P1、Judge、P6 和
reviewer 时，说明端到端链路成功。

这版端到端验证暂时让 agent、Judge 和 reviewer 都使用 DeepSeek，目的是先
跑通系统。正式实验时，agent/Judge 可以继续使用 DeepSeek，但应把模拟
reviewer 替换为不同模型家族，以降低 reviewer 与 policy 判断高度相关的问题。

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

后续 reviewer 流程完成后，把最终审核结果写入 JSONL 的
`reviewer_verdict` 字段。然后可以用以下命令生成 dev q table：

```powershell
.\.venv\Scripts\python.exe scripts\build_q_table.py `
  logs\dev_reviewer_decisions.jsonl `
  --group-by tool_name `
  --min-samples 5 `
  --output configs\q_table.json
```

之后通过 `PolicyEngine(q_table=...)` 加载这份 JSON。
