# AI 自主选基实验台 · Agent Harness 架构设计

> 版本 v1.0 · 2026-09-23
> 设计目标：让「AI 拿一笔钱自主选基、跑一个月、期间持续训练、月末判赚亏」这件事
> **可复现、可归因、可对照**——而不是只得到一个无法解释的数字。
>
> 本文档状态标注：`[已确认]` = 用户明确要求；`[默认假设]` = 架构师建议，可推翻。

---

## 1. 需求规格

| 维度 | 内容 | 状态 |
|---|---|---|
| 目标与用途 | 给 AI 一笔资金，自主选择基金建仓/调仓，运行约一个月，判定盈亏 | 已确认 |
| 成功标准 | ①实验可复现（同输入→同结果）②结果可归因（赚亏知道为什么）③相对基准有超额才算"AI 有效"④期间训练的效果可被单独度量 | 默认假设 |
| 任务结构 | 周期驱动：每个决策日 取数→筛候选→AI 决策→约束裁剪→下单→留痕；月末结算 | 默认假设 |
| 数据/资源边界 | 复用现有 akshare/东财数据源；LLM 走 OpenAI 兼容协议（DeepSeek）；本地 SQLite | 已确认（沿用项目既有） |
| 硬约束 | ①不破坏现有 predictions/CalibSample 评估链路的纯洁性<br>②AI 不得凭空指定未在候选池中的标的<br>③所有决策必须过确定性约束校验后才可下单<br>④交易成本必须真实建模（含 ETF 免印花税、场外赎回费阶梯） | 默认假设 |
| 用户与场景 | 单人（用户自己），本地/自用服务器运行，非对外荐股 | 已确认 |
| 边界（不做什么） | 不做实盘对接、不做对外推荐、不追求高频交易、不承诺准确率 | 默认假设 |
| 技术栈约束 | Python 3.11+ / SQLAlchemy / APScheduler / 现有 core 包结构，不引入新框架 | 已确认 |

---

## 2. 架构范式 P

### 2.1 正交子维度推导

| 子维度 | 决策 | 理由 |
|---|---|---|
| **扩展方式** | 模块内聚 + 策略注册（轻量） | 对照组策略（AI/随机/等权/指数）是同接口的不同实现，用 dict 注册即可，不需要插件内核。项目体量不值得上 DI 容器。 |
| **配置方式** | 声明式 TOML（沿用现有 `config.toml`）+ 每实验一份冻结 JSON 快照 | 实验协议必须**冻结**才能复现；TOML 管全局默认，实验自己的规则存 DB 的 JSON 字段（`universe_rule`/`constraints`/`policy`）。 |
| **部署拓扑** | 单机进程内，复用现有 `jobs/scheduler.py` 常驻调度 | 无需新拓扑。实验日循环注册成一个 cron job，与现有盘后任务串行（在对账之后、重训之前）。 |
| **编排模式** | **固定管线（确定性）+ 单点 AI 决策** | 关键决策：编排**不做成 agent 自由循环**，而是固定 6 步管线，只在第 4 步让 LLM 介入。理由见 §4.1。 |

### 2.2 范式关键抉择（架构师建议 + 依据）

**抉择 A：编排要不要 agent 化？→ 不要，用固定管线。**

对照知识库 `patterns/graph-state-machine`（用图显式声明状态流转）与
`frameworks/core/Codex-Harness`（thread/turn 显式边界）：
两者都指向同一个经验——**当任务的正确性可被明确定义时，显式编排优于自由循环**。

本项目的正确性定义得很清楚（候选池合法、权重和为1、约束不被突破），
自由循环只会引入非确定性而不带来能力增益。**AI 的自主性体现在"选谁、配多少"，
而不是"下一步干什么"。** 后者用固定管线。

**抉择 B：实验配置要不要冻结？→ 必须冻结，且冻结到"代码版本"级别。**

对照 `patterns/immutable-state`：实验开始时的
`universe_rule` / `constraints` / `policy` / 关键 config 项 / 代码 commit hash
全部写入 `Experiment.config_snapshot`，实验期间不可改。
否则"AI 这次赚了"无法回答"是用哪套规则赚的"。

**抉择 C：评估与决策要不要走同一套读写路径？→ 分离（读模型独立）。**

对照 `patterns/cqrs`（命令与查询分离）：
决策链路（写：下单、留痕）与评估链路（读：算收益、比基准）必须解耦。
**评估器禁止调用决策链路的任何函数**，只能读 `ExperimentRun` / `NavHistory` 的落库数据。
这样评估结果不会被执行细节污染，也保证"事后复盘"和"当时决策"看到的是同一份事实。

---

## 3. 技术栈选型

| 项 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | 与现有项目一致，akshare 生态 |
| 数据源 | 复用 `core.data.stock/fund/market`；**新增** `fetch_fund_universe`（全市场名录）+ `fetch_fund_metrics`（单基特征） | 现有 fund.py 只有「给代码→查档案」，缺候选池发现能力 |
| 数值计算 | numpy + pandas（ETF 特征/回撤/夏普）；复用 `core.predict.quant` 作免费预筛 | 量化预筛把付费 LLM 调用从 800 只压到 12 只 |
| 存储 | SQLite（沿用），新增 2 张表 | 复用现有 session_scope / 轻量加列迁移 |
| 调度 | APScheduler（复用 `jobs/scheduler.py`） | 无需新组件 |
| 交易内核 | **复用** `core.paper.engine`（需修正费用模型，见 §12-1） | 撮合/持仓/净值已具备，不重造 |
| API | FastAPI（复用 `api/main.py`），新增 `/experiment/*` 端点 | 便于前端展示与手动触发 |
| LLM | 复用 `core.predict.llm.LLMClient`（chat 用于预测，reasoner 用于组合决策） | 已具备重试/JSON模式/token 计量 |

**关键依赖不新增**——本设计的全部能力都建立在现有依赖之上（numpy/pandas/sqlalchemy/apscheduler/openai/akshare）。

---

## 4. 六层设计

### 4.1 E 执行循环

**设计决策**

- **执行模式**：周期驱动的**固定 6 步管线**，非自由循环。每个决策日跑一次，跑完即退出。
- **终止条件**：三重——① 日历到期（`as_of > end_date`）② 熔断（单日 LLM 调用超预算 / 候选池为空 / 连续 N 次下单失败）③ 人工 abort。
- **迭代上限**：**每决策日内部，LLM 调用硬上限 = `candidates_limit + 1` 次**（N 次标的预测 + 1 次组合决策）。**不设重试循环**——约束不满足时用确定性裁剪，不让 LLM 重来。这是防止成本失控的核心闸门。

**每日 6 步管线**

```
run_day(experiment_id, as_of):
  Step 0  读实验协议 → 校验状态（非 running 直接退出）
  Step 1  settle_t1 + snapshot_nav          [确定性·复用 engine]
  Step 2  判断是否调仓日（rebalance_rule）   [确定性]
          —— 非调仓日：只做估值快照，写 ExperimentRun 后退出（零 LLM 成本）
  Step 3  build_candidates(universe_rule)    [确定性·三层筛] → Top 12
  Step 4  对候选逐个跑 predict()             [LLM·复用现有 pipeline]
          → 得到结构化概率预测（同时落 predictions 表，不额外污染样本池）
  Step 5  decide() → 目标权重                [LLM·唯一自主决策点]
          validate_and_clip()               [确定性·约束裁剪]
  Step 6  rebalance() 下单 + 写 ExperimentRun [确定性·复用 engine]
          → 若 adaptive 组：触发 calibration 增量重训
```

**失败处理**

| 失败点 | 处理 |
|---|---|
| 某候选标的取数/预测失败 | 从候选池剔除，继续（不中断整轮）；若剩余候选 < `min_positions`，本轮**跳过调仓**并记录 `error` |
| 候选池为空（全市场筛不出） | 本轮跳过，`ExperimentRun.error` 记录，**不降级为"随便买"** |
| LLM 调用失败（重试后仍失败） | 本轮跳过调仓，保留现有持仓，记录 error。**绝不用随机权重兜底** |
| 下单失败（资金不足/不足一手） | 按权重降序尝试，失败者权重转现金，记录 `constraint_hits` |

> 设计立场：**宁可这一轮不动，也不动得不明不白。** 实验的价值在于可归因，
> 一次"兜底成交"会让整段归因链失效。

### 4.2 T 工具注册

**设计决策**

- **接入方式**：**不给 LLM 暴露工具调用（function calling）**。
  原因是本设计的编排是固定管线，LLM 的输入是被代码准备好的候选池文本，
  输出是结构化权重 JSON。引入 tool_calls 只会增加不确定性与成本。

  > 如果你未来想要"AI 自己决定还要看什么数据"，那才是 T 层真正介入的时点
  > ——那属于 §10 阶段 2，不在本设计范围。

- **工具清单（内部函数，由管线调用而非 LLM 调用）**：
  | 工具 | 读写 | 归属 |
  |---|---|---|
  | `fetch_fund_universe` | 只读（+磁盘缓存） | 新增 |
  | `fetch_fund_metrics` | 只读 | 新增 |
  | `build_candidates` | 只读（+可选持久化） | 新增 |
  | `quant_engine.predict_next` | 只读（本地模型） | 复用 |
  | `pipeline.predict` | 只读数据 + **写** predictions | 复用 |
  | `engine.buy/sell` | **写**（唯一资金变动入口） | 复用 |
  | `calibration.train_all` | 写（校准包） | 复用 |

- **读写分离与权限**：**资金变动只能通过 `engine.buy/sell`**。
  `allocator` 只产出"目标权重"，**不直接下单**；`runner` 负责把权重翻译成订单。
  这条边界保证"AI 的意图"与"实际成交"永远分离，事后可对比二者差异。

### 4.3 C 上下文管理

**设计决策**

- **预算**：喂给组合决策模型（reasoner）的上下文硬上限 **约 12k token**，组成固定五块：
  1. 实验协议摘要（约束 + 基准 + 已运行天数）～400 tok
  2. 候选池 Top 12：每只一行，含代码/名称/类型/规模/阶段收益/回撤/夏普/量化预筛分 ～1.5k tok
  3. 每只候选的结构化预测概率（复用 predict 输出，**要压成一行**，不贴完整报告）～1.2k tok
  4. 当前持仓 + 现金 + 浮动盈亏 ～600 tok
  5. 上一轮决策与后续表现（**仅自适应组**，用于反思）～800 tok

- **采样/清理策略**：
  - 候选池**只喂 Top 12**（`candidates_limit`），绝不把 800 只精选股全塞进去。
  - 标的预测结果**只取 `next_day` 与 `one_week` 两个 horizon**（月度实验用不到 quarter）。
  - 舆情明细**不进上下文**——舆情的价值已经通过 `predict` 的内部链路体现在概率里，
    再贴一遍是重复计费。若模型需要解释，带一句 `key_factors` 即可。
  - 历史决策**只保留最近 3 轮**（第 5 块），更早的折叠成"累计收益 + 换手率"两个数。

- **落地规格**：新增 `core/experiment/context.py::build_decision_context()`，
  签名与输出见 §6.2。函数内做 token 粗估（字符数/2），超预算时按上述优先级**从尾部截断**
  （历史决策 → 候选特征细节 → 舆情解释），**永不截断约束块**。

### 4.4 S 状态存储

**设计决策**

- **结构**：新增 2 张表（**复用**现有 `paper_accounts`/`paper_positions`/`paper_orders`/`nav_history` 记账），
  不动现有任何表结构——避免影响已跑通的对账链路。
- **备份/回滚**：实验配置（`config_snapshot`）落库即不可变；
  每个 `ExperimentRun` 只追加不修改（同日重跑覆盖，见"幂等"）；`EventSourcing` 式留痕。
  对照 `patterns/event-sourcing`：`ExperimentRun` 序列本身就是该实验的单一真相源。
- **治理**：`ExperimentRun` 只保留决策日 + 结算日（非调仓日仅存净值行，字段精简）；
  `candidates_snapshot` 只存 TopN 而非全量候选，附 `screen_trace` 统计量。
  > **权衡**：这牺牲了"事后回看被淘汰的 788 只是什么"的能力，换取 DB 体积可控。
  > 若你希望全量留痕，把 `candidates_snapshot` 拆成独立表并加 TTL 清理——列入 §12 遗留问题。

**两张新表字段定义（可直接建表）**

```python
class Experiment(Base):
    """一次受控实验：冻结协议 + 周期 + 组别。"""
    __tablename__ = "experiments"
    id: int                       # PK
    name: str                     # 实验名（人类可读）
    group: str                    # ai_frozen / ai_adaptive / baseline_random
                                  # / baseline_equal / baseline_index
    account_id: int               # FK -> paper_accounts
    start_date: date              # 实验起始（决策从次日起）
    end_date: date                # 到期日（= start + horizon_days）
    horizon_days: int = 30        # 周期天数
    rebalance_rule: str = "weekly_mon"   # daily / weekly_mon / monthly
    universe_rule: dict           # JSON: 候选池筛选规则（冻结）
    constraints: dict             # JSON: 组合约束（冻结）
    policy: dict                  # JSON: 决策策略参数（adaptive 组会更新）
    policy_version: int = 1       # 策略版本（adaptive 组递增）
    config_snapshot: dict         # JSON: 关键 config + git_rev + 代码 hash
    benchmark_symbol: str = "沪深300"
    initial_cash: float
    status: str = "running"       # running / finished / aborted
    frozen_at: datetime | None    # 冻结组：训练/策略冻结时点
    created_at: datetime
    notes: str = ""


class ExperimentRun(Base):
    """每个决策日的留痕：可复现『AI 当时看到了什么、决定了什么』。"""
    __tablename__ = "experiment_runs"
    id: int                       # PK
    experiment_id: int            # FK, index
    run_date: date
    is_rebalance_day: bool
    nav: float                    # 当日总资产
    cash: float
    positions_snapshot: list      # JSON [{symbol,name,volume,avg_cost,price,weight}]
    candidates_snapshot: list     # JSON TopN [{symbol,name,score,features,forecast}]
    screen_trace: dict            # JSON {scanned,passed_hard,per_layer_dropped}
    decision: dict                # JSON LLM 原始输出 {targets,rationale,cash_weight}
    final_targets: list           # JSON 过裁剪后的实际目标权重
    constraint_hits: list         # JSON 被触发的约束 [{rule,action,detail}]
    orders: list                  # JSON 实际成交
    benchmark_nav: float
    excess_return_pct: float      # 相对基准超额（当日）
    llm_tokens: int = 0
    llm_cost_est: float = 0.0
    error: str = ""               # 非空表示本轮未正常完成
    __table_args__ = (Index("ix_exprun_exp_date", "experiment_id", "run_date",
                            unique=True),)   # 幂等键
```

### 4.5 L 生命周期钩子

**设计决策**

- **拦截点**（4 个，全部为**硬门禁**，不可绕过）：

  | 钩子 | 时机 | 拦截内容 |
  |---|---|---|
  | `pre_universe` | 筛候选前 | 校验 `universe_rule` 合法（规模下限>0、类型非空）；非法则中止本轮 |
  | `pre_alloc` | AI 决策后、下单前 | **核心门禁**：候选合法性（symbol ∈ 候选池）+ 权重和为 1±ε + 单一上限 + 持仓数 + 现金下限；违规则调用 `clip` 而非放行 |
  | `pre_trade` | 每笔下单前 | 资金/整手/T+1 可卖量校验（复用 engine 既有抛错）+ **交易成本预检**（估算费用占比，超阈值告警） |
  | `post_run` | 本轮结束 | 写 ExperimentRun；若 adaptive 组则触发增量重训；写审计日志 |

- **审批策略**：**默认全自动**（实验要连续跑一个月，人工审批会断链）。
  但保留**软熔断**：单轮 LLM 成本超 `max_cost_per_run` → 本轮不调仓并置 `error`，
  等人工确认（查 `/experiment/{id}/runs` 可见）。这是"自动但不失控"。

- **落地规格**：钩子以**函数列表**实现，`protocol.py::run_hooks(stage, ctx)` 顺序执行。
  每个钩子签名 `hook(ctx: dict) -> dict`，返回 `{"ok": bool, "action": str, "detail": str}`。
  `ok=False` 的钩子在 `pre_alloc` 阶段不阻断，而是把 action 交给 `clip` 执行。

> 为什么钩子放在这里而不是 agent 里：这是**确定性护栏**。
> 对照 `patterns/capability-security`（最小权限）——AI 的能力边界不该靠提示词约束，
> 该靠代码在资金变动入口前拦截。

### 4.6 V 评估接口

**设计决策**

- **验证方式**：三层，逐层回答不同问题。
  | 层 | 问题 | 指标 |
  |---|---|---|
  | 单实验 | 这个月赚了没？ | 月末总收益、相对基准超额、最大回撤、夏普、换手率、费用占比 |
  | 组间对照 | AI 比瞎选强吗？ | AI 组 vs 随机组 vs 等权组 vs 指数组 的超额差、IR |
  | 训练对照 | 在线学习有用吗？ | `ai_adaptive` vs `ai_frozen` 的收益差 + 命中率差 + 校准改善 |
- **独立读路径**（`cqrs` 落地）：`experiment/report.py` **只读 DB 落库数据**，
  不 import `allocator` / `runner` 的任何函数。禁止"重算一遍"式的评估。
- **可复现判据**：给定 `config_snapshot` + `universe_rule` + 同一 `as_of`，
  确定性部分（Step 1/2/3/6）应产出**完全一致**的结果；
  非确定性部分（Step 4/5，LLM）允许差异，但必须可通过 `candidates_snapshot` + `decision` 回溯。
  → 新增自检：`verify_reproducible(experiment_id)` 重跑 Step 3 并与库中快照比对。
- **结算双口径**（必须同时给，否则结论会被费用问题误导）：
  - **口径 A 浮动估值**：月末按市价估值，反映"如果不赎回"的账面结果
  - **口径 B 清仓实现**：月末全部清仓，扣真实赎回费/佣金，反映"真落袋"的结果
  - 两个口径的差额 = **交易成本拖累**，这个数字本身对优化策略很有价值

---

## 5. 模块 / 目录结构（代码骨架）

```
core/
├── experiment/                    # 【新增】AI 自主选基实验台
│   ├── __init__.py
│   ├── protocol.py                # 实验协议：约束定义、校验、钩子执行 run_hooks()
│   ├── universe.py                # 候选池发现：确定性三层筛 build_candidates()
│   ├── context.py                 # 决策上下文组装（token 预算控制）
│   ├── allocator.py               # AI 组合决策 decide() + 确定性裁剪 validate_and_clip()
│   ├── baseline.py                # 对照组策略：随机/等权/指数（同接口）
│   ├── runner.py                  # 每日管线 run_day() / create_experiment() / settle_experiment()
│   ├── report.py                  # 评估与报告（独立读路径）experiment_report() / compare()
│   └── selfcheck.py               # 可复现自检 verify_reproducible()
├── data/
│   └── fund.py                    # 【扩展】+fetch_fund_universe() +fetch_fund_metrics()
├── paper/
│   └── engine.py                  # 【修正】费用模型：ETF 免印花税；场外赎回费阶梯
├── store/
│   └── models.py                  # 【扩展】+Experiment +ExperimentRun
api/
└── main.py                        # 【扩展】+/experiment/* 端点
jobs/
└── scheduler.py                   # 【扩展】+job_experiment_daily（挂在对账与重训之间）
tests/
└── test_experiment.py             # 【新增】无网络/无 LLM 的冒烟测试（合成数据跑通全管线）
```

---

## 6. 关键接口签名

### 6.1 数据源接口（`core/data/fund.py` 扩展）

```python
def fetch_fund_universe(
    types: tuple[str, ...] = ("fund", "ofund"),   # fund=场内ETF/LOF, ofund=场外
    max_age_days: int = 7,                         # 磁盘缓存有效期
) -> list[dict]:
    """全市场基金名录（含规模/成交额，用于硬筛）。

    数据源：ak.fund_etf_spot_em()（场内，带规模/成交额/涨跌幅）
            + ak.fund_name_em()（场外名录，规模需后续补）
    返回：[{"symbol": "510300", "name": "沪深300ETF", "type": "fund",
            "size": 12_000_000_000.0, "turnover": 8.5e8,
            "tradeable": "etf"}]
    失败：返回 []，并由调用方记录 screen_trace 中的失败原因。
    """

def fetch_fund_metrics(
    symbol: str,
    asset_type: str = "fund",
    lookback_days: int = 250,
) -> dict:
    """单只基金的特征快照（候选池打分用）。纯本地计算，不调 LLM。

    返回：{
      "symbol": str, "name": str, "nav": float,
      "stage_returns": {"1m": float, "3m": float, "6m": float, "1y": float},
      "max_drawdown_pct": float,      # lookback 窗口内最大回撤（正数）
      "volatility_pct": float,        # 年化波动率
      "sharpe": float,                # 年化夏普（无风险利率取 0.02）
      "top10_concentration": float,   # 重仓股合计占净值比（0~1）
      "size": float, "turnover": float,
      "manager": str, "company": str, "type": str,
      "data_days": int,               # 实际拿到多少天净值（判断数据质量）
    }
    失败：{"symbol": symbol, "error": str}（调用方剔除该候选）
    """
```

### 6.2 工具接口

```python
# === core/experiment/universe.py ===
def build_candidates(
    rule: dict,
    as_of: date | None = None,
    limit: int = 12,
    persist: bool = False,
) -> dict:
    """确定性三层筛，产出候选池。不调用 LLM。

    rule = {
      "include_types": ["fund", "ofund"],
      "min_fund_size": 2e8,
      "min_age_days": 365,
      "max_drawdown_pct": 35.0,
      "exclude_keywords": ["货币", "债券", "定开", "持有期"],
      "score_weights": {"ret_1y": 0.3, "ret_3m": 0.2. "sharpe": 0.3, "dd": -0.2}
    }

    三层：
      L1 硬筛（名录字段）：类型 / 规模 / 名称关键词  → ~800 只
      L2 指标筛（本地计算）：回撤 / 波动 / 数据天数   → ~200 只
      L3 打分排序：加权得分 TopN；同分以规模降序稳定排序（保证可复现）
      量化预筛（可选）：对 TopN×3 跑 quant 免费打分，再取 TopN → 省 LLM 调用

    返回：{
      "as_of": "2026-09-23",
      "candidates": [{"symbol","name","type","score","features":{...}} × limit],
      "screen_trace": {"scanned": int, "L1_passed": int, "L2_passed": int,
                       "llm_forecast_count": int, "errors": [...]}
    }
    """

# === core/experiment/context.py ===
def build_decision_context(
    experiment: Experiment,
    candidates: list[dict],
    forecasts: dict[str, dict],     # symbol -> 结构化预测摘要
    portfolio: dict,
    history: list[dict],            # 最近 3 轮 ExperimentRun 摘要
    token_budget: int = 12000,
) -> str:
    """按五块固定结构组装决策上下文，超预算从尾部截断（永不截断约束块）。"""

# === core/experiment/allocator.py ===
def decide(
    experiment: Experiment,
    context: str,
    client: LLMClient,
) -> dict:
    """AI 组合决策（唯一自主决策点）。输出目标权重，不直接下单。

    返回：{
      "targets": [{"symbol": str, "weight": float, "reason": str}],
      "cash_weight": float,
      "rationale": str,
      "llm_tokens": int,
    }
    失败：raise DecisionError（调用方跳过本轮）
    """

def validate_and_clip(
    targets: list[dict],
    constraints: dict,
    candidates: list[dict],
) -> dict:
    """确定性约束校验与裁剪。不做 LLM 重试。

    constraints = {
      "max_positions": 6, "min_positions": 3,
      "max_weight_per_symbol": 0.25,
      "min_cash_weight": 0.05,
      "max_asset_type_weight": {"ofund": 0.5},
    }
    裁剪次序（确定性，可预测）：
      1. 剔除不在候选池的 symbol（AI 幻觉防线）
      2. 权重归一化到 1 - min_cash_weight
      3. 超上限者削至上限，超出部分按得分顺序分配给未超限标的
      4. 超过 max_positions 则按权重截断至允许多数
      5. 不足 min_positions 则按候选得分顺序补齐至最小数
    返回：{"targets": [...], "cash_weight": float,
           "hits": [{"rule": str, "action": str, "detail": str}]}
    """

# === core/experiment/baseline.py ===
def decide_baseline(
    group: str,
    candidates: list[dict],
    portfolio: dict,
    experiment: Experiment,
    seed: int = 42,
) -> dict:
    """对照组策略，返回结构与 decide() 完全一致（同接口，便于统一评估）。
    group:
      baseline_random  → 用 seed 固定随机选 min_positions 只等权（可用不同 seed 重复）
      baseline_equal   → 候选池 TopN 等权
      baseline_index   → 全仓基准指数（benchmark_symbol），不调仓
    """
```

### 6.3 钩子接口

```python
# === core/experiment/protocol.py ===
def run_hooks(stage: str, ctx: dict) -> dict:
    """按序执行某阶段的全部钩子。

    stage ∈ {"pre_universe", "pre_alloc", "pre_trade", "post_run"}
    返回：{"ok": bool, "actions": [{"hook": str, "action": str, "detail": str}],
           "ctx": dict}   # ctx 可被钩子修正（如 clip 后的 targets）
    """

def check_allocation(ctx: dict) -> dict:   # pre_alloc 主力钩子
    """候选合法性 + 权重和 + 各上限校验；不通过则返回 action="clip"。"""

def check_trade_cost(ctx: dict) -> dict:   # pre_trade 钩子
    """估算本轮费用占资产比；超阈值（默认 1.5%）返回 action="warn"。

    注意：场外基金短期赎回费可达 1.5%，月度调仓时该钩子会频繁告警——
    这不是 bug，是真实成本在提示你降低换手。
    """
```

### 6.4 管线与评估接口

```python
# === core/experiment/runner.py ===
def create_experiment(
    name: str,
    group: str,
    initial_cash: float,
    start_date: date,
    horizon_days: int = 30,
    rebalance_rule: str = "weekly_mon",
    universe_rule: dict | None = None,     # None → 取 config [experiment.universe]
    constraints: dict | None = None,       # None → 取 config [experiment.constraints]
    benchmark_symbol: str | None = None,
    notes: str = "",
) -> int:
    """建实验：建 PaperAccount(mode='experiment') + 冻结 protocol + 落 Experiment。
    返回 experiment_id。
    """

def run_day(experiment_id: int, as_of: date | None = None) -> dict:
    """执行一个决策日。幂等：同日重复调用覆盖当日 ExperimentRun（不重复下单）。

    幂等实现：进入时先查当日 ExperimentRun，若已存在且无 error → 直接返回；
    若存在但有 error → 先按 orders 回滚当日本账户成交（复用反向订单），再重跑。
    返回：{"experiment_id", "run_date", "is_rebalance_day", "nav",
           "orders": int, "skipped": bool, "reason": str, "error": str}
    """

def settle_experiment(experiment_id: int) -> dict:
    """到期结算：写最终指标、置 status='finished'。
    同时计算口径 A（浮动估值）与口径 B（清仓实现），后者会真实下单平仓。
    """

# === core/experiment/report.py ===
def experiment_report(experiment_id: int) -> dict:
    """单实验报告（独立读路径，只读 DB）。
    返回：{"experiment": {...}, "nav_curve": [...],
           "metrics": {"total_return", "excess_return", "ir",
                       "max_drawdown", "sharpe", "turnover", "cost_ratio"},
           "settlement_a": {...}, "settlement_b": {...},
           "constraint_hits_summary": {...},
           "attribution": {"by_symbol": [...], "by_rebalance": [...]}}
    """

def compare_experiments(experiment_ids: list[int]) -> dict:
    """多组对照表：同起点不同组的指标并排 + 相对差值。
    用于回答『AI 比随机强吗』『在线学习有用吗』。
    """

# === core/experiment/selfcheck.py ===
def verify_reproducible(experiment_id: int) -> dict:
    """重跑确定性部分（Step 3）并与库中 candidates_snapshot 比对。
    返回：{"reproducible": bool, "diff": [...]}
    """
```

---

## 7. 核心数据结构

### 7.1 状态结构（实验运行时状态，不落库）

```python
ExperimentContext = {
    "experiment": Experiment,          # ORM 对象
    "run_date": date,
    "is_rebalance_day": bool,
    "portfolio": {                     # 来自 engine.portfolio()
        "cash": float, "market_value": float, "total_value": float,
        "positions": [{"symbol","name","volume","avg_cost","price",
                       "market_value","pnl","pnl_pct"}],
    },
    "candidates": list[dict],          # build_candidates 输出
    "forecasts": dict[str, dict],      # symbol -> {"next_day":{...},"one_week":{...}}
    "context_text": str,               # 组装后的 LLM 输入
    "decision": dict,                  # decide() 原始输出
    "final_targets": list[dict],       # clip 后的目标
    "hook_actions": list[dict],
    "orders": list[dict],
    "llm_tokens": int,
    "errors": list[str],
}
```

### 7.2 决策结构（LLM 输出契约）

```python
# PREDICTION 之外的第二个 JSON Schema —— 组合决策输出
DECISION_JSON_SCHEMA = {
  "type": "object",
  "required": ["targets", "rationale"],
  "properties": {
    "targets": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["symbol", "weight", "reason"],
        "properties": {
          "symbol": {"type": "string"},        # 必须来自候选池
          "weight": {"type": "number"},        # 0~1 目标权重
          "reason": {"type": "string"},        # ≤60 字
        }
      }
    },
    "cash_weight": {"type": "number"},         # 现金目标占比
    "rationale": {"type": "string"},           # 整体逻辑 ≤300 字
  }
}
```

### 7.3 配置结构

见 §8。

---

## 8. 配置文件格式（`config.toml` 新增段）

```toml
[experiment]
enabled = true
default_cash     = 1000000.0     # 默认起始资金
horizon_days     = 30            # 实验周期（天）
rebalance_rule   = "weekly_mon"  # daily / weekly_mon / monthly
candidates_limit = 12            # 喂给 LLM 的候选数（成本主控阀）
benchmark_symbol = "沪深300"     # 基准（用于超额收益）
adaptive_retrain = true          # ai_adaptive 组是否每日增量重训
max_cost_per_run = 5.0           # 单轮 LLM 成本上限（元），超则软熔断
run_time         = "18:40"       # 日循环执行时刻（须晚于 evaluate_time）

[experiment.universe]            # 候选池筛选规则（实验开始时冻结进 DB）
include_types    = ["fund", "ofund"]
min_fund_size    = 200000000.0   # 规模下限 2 亿（低于此流动性风险高）
min_age_days     = 365           # 成立满 1 年（避免新基金无数据）
max_drawdown_pct = 35.0          # 近 1 年最大回撤上限
exclude_keywords = ["货币", "债券", "定开", "持有期", "发起式", "联接"]
score_weights    = { ret_1y = 0.30, ret_3m = 0.20, sharpe = 0.30, dd = -0.20 }
use_quant_prescreen = true       # 用免费量化模型预筛，压 LLM 调用量

[experiment.constraints]         # 组合约束（冻结进 DB）
max_positions           = 6      # 最多持有几只
min_positions           = 3      # 最少持有几只（避免 AI 全押一只）
max_weight_per_symbol   = 0.25   # 单一标的上限 25%
min_cash_weight         = 0.05   # 现金下限 5%
max_asset_type_weight   = { ofund = 0.50 }   # 场外基金上限（费用考虑）

[experiment.cost]                # 交易成本模型（真实化）
etf_no_stamp_tax   = true        # ETF 交易免印花税（A股法规）
ofund_buy_fee      = 0.0015      # 场外申购费（1折后约 0.15%）
ofund_redeem_tiers = [           # 场外赎回费阶梯（按持有天数）
  { days = 7,   rate = 0.015 },
  { days = 30,  rate = 0.005 },
  { days = 365, rate = 0.0025 },
  { days = 99999, rate = 0.0  },
]
```

---

## 9. 层间交叉检查

| 关系 | 说明 | 是否自洽 |
|---|---|---|
| E 的终止条件 ↔ V 的评估 | E 让"跳过调仓"成为合法终态，V 必须能解释"这段空窗期"——因此 `ExperimentRun.error` 参与归因，评估时按"有效决策日/总日"计算覆盖率 | ✅ 已设计（`experiment_report` 含 `coverage`） |
| E 的迭代上限 ↔ C 的预算 | E 限定 LLM 调用 ≤ N+1 次/日，C 的 12k 预算必须覆盖 N+1 次调用的输入总量；C 的截断策略保证不会因上下文膨胀而突破 E 的成本闸 | ✅ 自洽（截断优先级明确） |
| S 的写操作 ↔ L 的拦截 | 所有资金写操作都过 `pre_trade` 钩子；`L` 是 `S` 的唯一门禁，不存在绕过路径（因为 `allocator` 不直接下单） | ✅ 自洽 |
| C 的清理策略 ↔ E 的成本 | C 只喂 Top12 + 两个 horizon，直接把 Step4 的付费调用从 800 压到 12 | ✅ 成本可预测 |
| T 的工具权限 ↔ L 的审批 | T 层无 LLM 工具调用 → L 的拦截点全在代码路径上，无"AI 绕过钩子"风险 | ✅ 自洽 |
| V 的独立读路径 ↔ S 的事件溯源 | V 只读 `ExperimentRun` 序列，该序列只追加 → 评估永远基于冻结事实 | ✅ 自洽 |

---

## 10. 六层决策表

| 层 | 决策 | 依据 | 状态 |
|---|---|---|---|
| E | 固定 6 步管线，非自由循环；LLM 调用 ≤ N+1 次/日 | `graph-state-machine` 模式 + Codex 显式 turn 边界 | 默认假设 |
| E | 失败时"跳过本轮"而非兜底成交 | 可归因性优先于连续性 | 默认假设 |
| T | **不**给 LLM 暴露工具调用；工具由管线调用 | 固定编排下 tool_calls 只增不确定性 | 默认假设 |
| T | 资金变动唯一入口 = `engine.buy/sell` | 意图与成交分离 | 默认假设 |
| C | 12k 预算 / 五块固定结构 / 尾部截断 / 永不截断约束 | 成本可预测 + 约束不可被挤掉 | 默认假设 |
| C | 舆情明细不进决策上下文（已体现在概率里） | 避免重复计费 | 默认假设 |
| S | 新增 2 表，**不动**现有任何表 | 保护已跑通的对账链路 | 默认假设 |
| S | 候选池只存 TopN + 统计量 | DB 体积可控（牺牲全量回看） | 默认假设 |
| L | 4 个硬门禁钩子；默认全自动 + 成本软熔断 | 连续一个月不能断链，但也不能失控 | 默认假设 |
| L | `pre_alloc` 违规走确定性 clip，不让 LLM 重试 | 成本闸门 + 结果可预测 | 默认假设 |
| V | 三层评估（单实验/组间/训练对照） | 单次一个月结果无统计意义，必须对照 | **建议重点** |
| V | 独立读路径（cqrs），禁止重算式评估 | 评估不被执行细节污染 | 默认假设 |
| V | 结算双口径（浮动 / 清仓实现） | 费用是月度实验的主要杀伤因素 | **建议重点** |
| P | 配置冻结到代码 commit 级 | `immutable-state` | 默认假设 |

---

## 11. 参考架构图

```
                    ┌─────────────────────────────────────────┐
                    │  实验协议 Experiment（冻结·不可变）        │
                    │  universe_rule / constraints / policy    │
                    │  config_snapshot + git_rev               │
                    └────────────────┬────────────────────────┘
                                     │ 每个决策日
        ┌────────────────────────────▼────────────────────────────┐
        │  Step1  settle_t1 + snapshot_nav        [确定性]         │
        │  Step2  是否调仓日？ ──否──► 仅写净值行，退出（0 成本）    │
        │  Step3  build_candidates  三层筛→Top12   [确定性]         │
        │            L1 硬筛 → L2 指标筛 → L3 打分 → 量化预筛        │
        │  Step4  对候选逐个 predict()            [LLM·复用]        │
        │  Step5  decide() → 目标权重             [LLM·唯一自主点]  │
        │         validate_and_clip()             [确定性裁剪]      │
        │  Step6  rebalance() → engine.buy/sell   [确定性·唯一写口] │
        │         写 ExperimentRun（事件溯源）                      │
        └────────────────────────────┬────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │  L 钩子：pre_alloc 硬门禁                │
                    │  候选合法性 / 权重和 / 上限 / 现金下限     │
                    └────────────────┬────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────┐
        │  V 评估（独立读路径，只读 DB）                            │
        │  单实验：收益/超额/IR/回撤/换手/费用占比                   │
        │  组间对照：ai_frozen │ ai_adaptive │ random │ equal │ idx │
        │  训练对照：adaptive − frozen = 在线学习的边际价值          │
        └─────────────────────────────────────────────────────────┘
```

**四组对照的设计意图**

| 组 | 做什么 | 回答什么问题 |
|---|---|---|
| `ai_frozen` | AI 选基，训练层冻结在实验开始前 | 当前方法本身好不好 |
| `ai_adaptive` | AI 选基，每日吸收新样本重训 | **在线学习有用吗** |
| `baseline_random` | 固定种子随机选 N 只等权 | **AI 是不是还不如瞎选**（monkey baseline） |
| `baseline_index` | 满仓基准指数不调仓 | 这个月 beta 本身是多少 |

> 若 `ai_frozen` 跑不过 `baseline_random`，那问题不在训练策略而在预测信号本身——
> 这个结论比"赚了/亏了"有价值得多。

---

## 12. 遗留问题与未知区（诚实标注）

1. **现有费用模型有正确性缺陷（必须先修，否则所有实验结论失真）**
   `core/paper/engine.py::sell` 对所有标的**无条件收取印花税**：
   ```python
   fee = _commission(revenue) + _stamp_tax(revenue)   # 第 144 行
   ```
   但 **ETF/LOF 卖出免征印花税**（A股法规）。这会让 ETF 卖出的成本被高估 0.05%，
   在月度高频调仓实验中是系统性偏差。
   **编码时需临时采用的占位方案**：按 `asset_type == "stock"` 或 `"bse"` 才收印花税。

2. **场外基金申赎的持有期阶梯费率未建模**
   现状：`engine` 对 `ofund` 是"T+0 确认、不计申购费"。真实是 T+1 确认份额，
   且赎回费按持有天数阶梯（<7天 1.5%，7~30天 0.5%…）。
   **这是本设计最大的未知区**：月度实验若做场外基金周频调仓，
   费用可能吃掉全部收益。§8 已给出费率配置结构，但**是否需要"确认延迟"建模**
   （即下单日与份额到账日不同，影响权重计算）需要你决定——
   若接受简化，需在报告里明确标注"ofund 按 T+0 简化"。

3. **`min_age_days` 依赖成立日期数据，而 akshare 的场外名录不带该字段**
   **占位方案**：用"净值序列长度 ≥ 240 天"作为成立年限的代理指标
   （`fetch_fund_metrics` 的 `data_days`），这是可实现的近似。

4. **单次一个月实验的统计意义极低（架构能做的极限）**
   框架能保证"结果可归因"，但**不能创造信息量**。一个月 ≈ 20 个交易日，
   基金收益的方差远大于 AI 可能产生的 alpha。
   → 建议：若要结论可信，用 `repeat_starts` 跑多个不同起始月（如连续 6 个月各跑一组），
   看**超额收益的分布**而非单次数值。这会让成本从 ¥37 量级升到 ¥200 量级，
   仍完全可控。
   > 这是框架无法替代的判断：**任何声称"一个月验证 AI 选基能力"的结论都不可信**，
   > 包括本框架跑出来的单次结果。

5. **`score_weights` 中的特征权重是拍定的，无依据来源**
   我给的 `{ret_1y:0.3, ret_3m:0.2, sharpe:0.3, dd:-0.2}` 属于**默认假设**。
   更严谨的做法是用 `walk_forward` 样本回测不同权重组合，但这本身是独立课题。
   编码时先用默认值，把权重写进 `universe_rule` 便于事后调参。

6. **未验证：akshare 全市场基金接口的稳定性与耗时**
   `ak.fund_etf_spot_em()` 与 `ak.fund_name_em()` 的真实响应时间/限流行为我未实测。
   若单次调用超过 30 秒，需要改为"每日一次性拉取并落盘缓存"（`max_age_days=7` 已预留）。
   另：`fetch_fund_universe` 的 `size` 字段对场外基金可能取不到，
   届时 L1 硬筛需降级为"仅场内按规模筛，场外按数据天数筛"。

7. **框架未覆盖的维度：融资/杠杆、可转债、跨市场（港股通基金）、申赎套利**
   本设计只处理"现金 + 基金多头"。任何加杠杆或做空的需求都需要新设计。

---

## 13. 成本估算（单月 · 四组对照）

| 项 | 计算 | 金额 |
|---|---|---|
| 单次调仓（周频） | 12 只候选 × fast 预测 ≈ ¥1.8 + 1 次 reasoner 决策 ≈ ¥0.5 | ≈ ¥2.3 |
| 单实验一个月 | 4 次调仓 × ¥2.3 | ≈ ¥9.2 |
| 四组对照 | ai_frozen + ai_adaptive + random + index（index 零 LLM 成本） | ≈ ¥28 |
| 若重复 6 个起始月 | ¥28 × 6 | ≈ ¥170 |
| 数据抓取 | akshare 免费 | ¥0 |

> 成本远低于我原先担心的水平，**这直接说明"多起始日重复实验"在经济上完全可行**——
> 而这正是让结论可信的唯一办法。建议你直接从 `repeat_starts ≥ 3` 起步。

---

## 14. 待用户拍板的决策点

| # | 决策 | 我的建议 | 影响面 |
|---|---|---|---|
| 1 | 标的范围：只做 ETF / 只做场外 / 混合 | **以 ETF 为主，场外受限 ≤50%** | 决定费用模型复杂度与实验可信度——场外赎回费会吃掉月度收益 |
| 2 | 调仓频率 | **周频（每周一）** | 决定成本与噪声；日频在月度实验里几乎纯噪声 |
| 3 | 是否接受四组对照（含 monkey baseline） | **必须包含** | 决定结论是否可信；不设 random 组则无法排除运气 |
| 4 | 实验重复次数 | **≥3 个起始日** | 决定结论有没有统计意义；成本仍可控 |
| 5 | 场外基金是否建模"份额确认延迟" | **先简化（T+0），报告标注** | 影响引擎改动量与真实度 |
| 6 | 是否现在就修 ETF 印花税缺陷 | **必须修，先于实验开发** | 影响所有历史模拟盘数据的一致性 |

---

**本设计交付状态**：可编码。已过可编码三门槛自查——
类型完整（§7 全部字段已定义）、API 真实（全部复用项目既有真实函数 + akshare 真实接口名）、
标准可达（§13 成本逐项预算 + §4.1 失败路径明确）。

**未做知识库扩展**：本次为现状评估型设计，无需网络检索新框架案例，
仅对照了本地知识库既有模式（`graph-state-machine` / `immutable-state` / `cqrs` /
`event-sourcing` / `capability-security`）。
