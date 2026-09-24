# AI 交易员 Harness 架构设计（v2.1）

> 版本 v2.0 · 2026-09-23 · **取代 v1.0「受控实验台」的形态判断**
>
> v1.0 文档（`AI自主选基实验台-架构设计.md`）仍然有效，但**降级为本设计的 V 层**
> ——它负责"验证交易员学得好不好"，而本设计负责"让交易员真的会交易"。

---

## 0. 形态更正：为什么场景变了，判断就变了

v1.0 我判定「编排不做成 agent 自由循环」，理由是**确定性保证可复现**。
那个判断在**"验证某个策略好不好"**的场景下是对的。

但你要的是另一件事：**让 AI 在一个真实交易台上，自己试、自己错、自己改、并且记住。**
这个场景下，我的原判断**必须推翻**，理由有三条：

| 维度 | v1.0 受控实验台 | v2.0 AI 交易员 |
|---|---|---|
| 目标 | 验证"这套策略有效吗" | 让"AI 学会交易"这件事发生 |
| 学习来源 | 只在**实验外**训练，实验内冻结或对照 | 学习就发生在**交易过程中** |
| 编排要求 | 固定管线（保证可复现） | **自主循环**（学习需要试错） |
| 非确定性 | 是污染源，要压制 | 是**学习介质**，要保留但要可追溯 |
| 记忆 | 不需要（配置冻结） | **核心组件** |

**关键差异**：学习本身依赖试错循环。如果你把循环固定死了，AI 就没有"试试另一个做法"的空间，
也就学不到东西。所以 **E 层必须是自主循环**。

> 但有一条底线不能破：**自主的是"做什么决定"，不自主的是"能不能下单"。**
> 见 §4.5 L 层风控——这是资金安全的硬门禁，不交给模型判断。

---

## 1. 需求规格

| 维度 | 内容 | 状态 |
|---|---|---|
| 目标与用途 | 给 AI 一个真实交易台（虚拟资金），它每个交易日自行决策买卖；发现预测与实际不符时纠正认知并写入记忆，持续迭代 | 已确认 |
| 成功标准 | ①AI 能自主完成"感知→决策→下单"闭环 ②错误能触发纠错并沉淀为记忆 ③记忆能被后续决策读取生效 ④学到的规律**可验证是否真的有用** ⑤资金安全由确定性风控保证 | 默认假设 |
| 任务结构 | 周期驱动自主循环（每交易日一轮），轮内含工具调用与反思 | 已确认 |
| 数据/资源边界 | 复用现有 akshare/东财数据源与 `core.paper.engine` 记账内核；LLM 走 OpenAI 兼容协议 | 已确认 |
| 硬约束 | ①**下单必过确定性风控**（模型不可绕过）②AI 不能直接改写生效记忆（只能提候选，需证据升级）③每轮 LLM 调用与成本有硬上限 ④纠错必须归因，不得把市场 beta 当自己的错 ⑤所有决策与工具调用全留痕 | 默认假设 |
| 用户与场景 | 单人自用，本地/自有服务器，非对外荐股 | 已确认 |
| 边界（不做什么） | 不接实盘、不做高频、不承诺收益、AI 不得触碰风控阈值本身 | 默认假设 |
| 技术栈约束 | Python 3.11+ / SQLAlchemy / APScheduler / 现有 core 包，不引入新框架 | 已确认 |

---

## 2. 架构范式 P

| 子维度 | 决策 | 理由 |
|---|---|---|
| **扩展方式** | 工具注册表 + 模块内聚 | T 层工具以 dict 注册（name → callable + JSON schema），新增工具不改循环代码 |
| **配置方式** | TOML（风控/预算/记忆规则）+ 记忆用 Markdown 落盘（人类可审计） | 对照 Hermes Agent「文件即记忆」：记忆用可读文件比向量库更利于审计"AI 到底学了什么" |
| **部署拓扑** | 单机进程内，挂到现有 `jobs/scheduler.py`（在 evaluate 之后、retrain 之前） | 复用现有常驻调度，无新组件 |
| **编排模式** | **自主循环（ReAct 式：思考→调用工具→观察→再思考）+ 确定性护栏** | 学习需要试错空间；护栏保证不失控。对照 `actor-model`（隔离实体间消息传递）：交易台与决策循环通过工具接口隔离 |

### 2.1 三个关键抉择（附依据）

**抉择 A：E 层用「自主循环」还是「固定管线」？→ 自主循环。**

依据：Heremes Agent 的自进化闭环（E = 自进化循环）与 Codex Harness 的 thread/turn
都表明——**当目标包含"学会某件事"时，循环必须给模型留出"自主尝试"的空间**，
否则学习无从发生。

但循环必须带三重护栏（照抄 Hermes 的成熟做法）：
`max_turns` 硬上限 + `tool_loop_guardrails`（检测"重复失败/无进展"循环）+ 成本上限。

**抉择 B：记忆用「Markdown 文件」还是「向量库」？→ 分层：结构化入库 + Markdown 镜像。**

依据：Hermes 用"文件即记忆"（可审计、容量无限、跨会话）；
但本项目需要**按标的/按类型检索**（"我对 510300 的所有教训"），
纯文件检索效率低。故做成双写：
- **SQLite 为真相源**（可查询、可关联、可统计）
- **Markdown 镜像为审计面**（`data/trader_memory/*.md`，人类可读、可 diff、可手工修正）

> 权衡：双写有轻微不一致风险。缓解：Markdown 为**只读派生视图**，
> 任何写入以 DB 为准、由 DB 重新渲染 MD，不允许反向修改。

**抉择 C：AI 能不能直接改自己的策略？→ 不能，必须过「证据门」。**

依据：对照 `patterns/capability-security`（最小权限）。
如果 AI 写一条教训就立即改变行为，而该教训来自单次运气，
它会**自我强化错误**（买了A亏→"以后不买A"→再也不买A→永远不知道自己错）。
故设记忆状态机：`candidate`（候选）→ `active`（生效）→ `refuted`（被反驳）/ `archived`。
**AI 只能写 candidate，升 active 需要 ≥N 条独立证据支撑。**

---

## 3. 技术栈选型

| 项 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | 与项目一致 |
| LLM | `core.predict.llm.LLMClient`，**新增 function calling 支持** | 现有 LLMClient 只有 chat/ask_json，需要扩展 tools 参数 |
| 交易内核 | **复用** `core.paper.engine`（buy/sell/portfolio/snapshot_nav/perform_stats） | 撮合/持仓/T+1/佣金已具备 |
| 数据源 | 复用 `core.data.stock/fund/market` + 新增 `search_funds`（候选发现） | 现有能力之上补"发现" |
| 预测 | 复用 `core.predict.pipeline.predict`（作为 AI 的一个工具） | AI 可主动调用预测 |
| 存储 | SQLite（沿用），新增 2 张表 | 复用 session_scope |
| 记忆镜像 | 本地 Markdown（`data/trader_memory/`） | 人类可审计 |
| 调度 | APScheduler（复用） | — |
| 测试 | 纯合成数据 + FakeLLM + 假行情，无网络无真实 LLM | 与现有 `test_smoke.py` 风格一致 |

---

## 4. 六层设计

### 4.1 E 执行循环

**每日一轮，七步**

```
run_day(trader_id, as_of):
  ┌─ Step 1 感知（Perceive）            [确定性]
  │   拉取：账户快照、持仓市值、当日行情、大盘环境摘要
  │
  ├─ Step 2 对账（Reconcile）            [确定性]
  │   取到期预测 → 调 evaluate_due 对账 → 得到「预测 vs 实际」记录
  │
  ├─ Step 3 归因（Attribute）            [确定性·核心]
  │   对每条偏差做四分类（attributor.py）：beta / noise / forecast_error / decision_error
  │   → 只把 forecast_error / decision_error 写成 fact
  │
  ├─ Step 4 反思（Reflect）              [LLM·可触发记忆写入]
  │   AI 看新 fact → 总结教训 → 写 lesson(candidate)
  │   → 系统按证据数决定是否升 active → 重建 belief
  │
  ├─ Step 5 决策（Decide）               [LLM·自主循环·核心]
  │   AI 带工具集自主决策：可以查行情、跑预测、翻记忆、看历史错误
  │   → 输出「今日动作」：买入/卖出/持有/观望 + 理由
  │
  ├─ Step 6 校验（Guard）                [确定性·硬门禁]
  │   每笔动作过 guard.check_order()；不通过则拒单并回灌给 AI 说明原因
  │
  └─ Step 7 执行与留痕（Act & Record）   [确定性]
      下单 → snapshot_nav → 写 TraderRun（含完整工具调用轨迹）
```

**终止/熔断条件（照抄 Hermes 的三层硬上限）**

| 熔断项 | 默认值 | 触发后行为 |
|---|---|---|
| `max_turns`（单轮循环内 LLM 调用上限） | 8 | 强制收束，用已有信息出决策；若仍无决策 → 本轮观望 |
| `max_tool_calls`（单轮工具调用上限） | 20 | 同上 |
| `tool_loop_guardrail`（重复失败/无进展检测） | 同工具同参连续 3 次失败 | 熔断该工具，回灌提示 |
| `max_cost_per_day`（单日 LLM 成本上限） | ¥8 | 本轮强制观望，记录 `cost_fused` |
| 连续失败 | 3 轮 | 置 `status=paused`，等人工介入（不静默继续） |

**失败处理立场**：宁可这一轮"观望"，也不做没理由的交易。
`watch`（观望）是**一等公民动作**，不是失败。

### 4.2 T 工具注册

**与 v1.0 的根本区别：这里必须给 LLM 暴露 function calling。**
因为"AI 自己决策"的前提是它能自主获取信息。v1.0 说"不给工具"是对的（固定管线不需要），
本设计**必须给**。

**工具清单（注册表形式，每个工具 = callable + JSON schema + 读写标记）**

| 工具 | 权限 | 说明 |
|---|---|---|
| `get_market_snapshot` | 只读 | 大盘环境（指数、涨跌家数、资金流） |
| `search_funds` | 只读 | 按关键词/类型搜基金（**候选发现的入口**） |
| `get_fund_profile` | 只读 | 基金档案（类型/经理/规模/重仓股/跟踪指数） |
| `get_quote` | 只读 | 实时行情 |
| `get_kline` | 只读 | K线/净值走势 |
| `get_metrics` | 只读 | 基金特征（收益/回撤/波动/夏普） |
| `run_prediction` | 只读* | 跑一次结构化预测（*内部会写 predictions 表，但不算资金操作） |
| `get_news` | 只读 | 相关舆情 |
| `get_portfolio` | 只读 | 当前持仓 + 现金 + 浮盈 |
| `get_trade_history` | 只读 | 历史成交（供 AI 复盘自己的操作） |
| `recall_memory` | 只读 | **翻自己的记忆**（按 symbol / kind 检索） |
| `get_past_mistakes` | 只读 | **看自己犯过的错**（关键工具：让 AI 直面失败） |
| `place_buy_order` | **写** | 买入（必过 guard） |
| `place_sell_order` | **写** | 卖出（必过 guard） |

**关键边界**：
- **只有 2 个写工具**，且都过 §4.5 硬门禁。
- **AI 没有"写记忆"的工具**。记忆写入发生在 Step 4 反思阶段，
  由系统提供结构化接口（AI 输出教训内容，系统负责评级与入库）。
  这是刻意的权限收窄——防止 AI 在决策时顺手编造记忆来给自己找理由。
- `run_prediction` 标记为"只读*"，因为它的副作用（写 predictions 表）不影响资金，
  且能自动进入现有对账链路，是**有益副作用**。

### 4.3 C 上下文管理

**决策上下文预算：约 16k token**（比 v1.0 高，因为要带工具定义 + 记忆）

| 块 | 内容 | 预算 |
|---|---|---|
| 1 | 交易员人格 + 铁律（system） | ~1.2k |
| 2 | 工具定义（function calling schema） | ~2.5k |
| 3 | 账户现状（现金/持仓/浮盈/累计收益） | ~0.8k |
| 4 | **记忆注入**：active lessons（≤10 条）+ beliefs | ~2.5k |
| 5 | **近期错误回顾**：最近 5 条 forecast_error fact（带归因） | ~1.5k |
| 6 | 大盘与关注标的快照 | ~1.5k |
| 7 | 本轮工具调用轨迹（滚动） | ~剩余 |

**采样/清理策略**：
- 记忆注入**只给 active 的 lesson**（candidate 不进上下文，避免噪声影响决策）
- 每类记忆最多 10 条，超出按 `置信度 × 证据数` 排序截断
- 错误回顾**只给最近 5 条**，更早的折叠成统计（"近 30 日预测偏差 12 次，其中 beta 占 7 次"）
- 工具调用轨迹**保留最近 6 次**，更早的压成一行摘要
- 对照 Hermes 的双重压缩：本设计轻量得多（单轮短、不跨会话），无需 85%/50% 两级

### 4.4 S 状态存储

**新增 2 张表，不动现有任何表。**

```python
class TraderRun(Base):
    """交易员每个交易日一轮的完整留痕（含工具调用轨迹）。"""
    __tablename__ = "trader_runs"
    id: int
    trader_id: int                    # FK -> paper_accounts（mode='trader'）
    run_date: date
    status: str                       # ok / watched / fused / error
    perception: dict                  # JSON 感知快照（账户+行情+大盘）
    reconciliations: list             # JSON 到期对账结果
    attributions: list                # JSON 归因结论 [{pred_id, cls, evidence}]
    new_facts: int                    # 本轮新增 fact 数
    new_lessons: list                 # JSON 新增/更新的 lesson [{id, kind, statement}]
    memory_promotions: list           # JSON 本轮升级的记忆 [{id, from, to, reason}]
    tool_calls: list                  # JSON 完整轨迹 [{tool, args, ok, ms, result_digest}]
    decision: dict                    # JSON AI 决策 {actions, rationale, watch_reason}
    guard_results: list               # JSON 风控结论 [{action, allowed, rule, detail}]
    orders: list                      # JSON 实际成交
    nav: float
    llm_tokens: int
    llm_cost_est: float
    turns: int                        # 实际用了多少轮
    error: str
    __table_args__ = (Index("ix_trader_run", "trader_id", "run_date", unique=True),)

class TraderMemory(Base):
    """三层记忆：fact（经验）/ lesson（教训）/ belief（信念）。

    写入权限：
    - fact   : 系统自动写（对账+归因后），只追加
    - lesson : AI 在反思阶段写，但只能写 candidate；升 active 需证据数达标
    - belief : 系统从 active lesson 聚合生成，AI 不可直接写
    """
    __tablename__ = "trader_memories"
    id: int
    trader_id: int                    # FK, index
    kind: str                         # fact / lesson / belief
    scope: str                        # global / asset_type:fund / symbol:510300
    symbol: str                       # 关联标的（可空）
    statement: str                    # 记忆正文（fact=客观事实描述；lesson=规律；belief=当前策略）
    status: str                       # candidate / active / refuted / archived
    confidence: float                 # 0~1
    evidence_count: int               # 支撑证据数
    evidence_refs: list               # JSON 证据引用 [{run_date, symbol, detail}]
    counter_evidence: list            # JSON 反证 [{run_date, detail}]
    source: str                       # system / ai_reflection
    policy_patch: dict                # JSON 仅 belief 用：对策略参数的覆盖
    version: int                      # belief 版本号（可回滚）
    created_at: datetime
    updated_at: datetime
    __table_args__ = (Index("ix_mem_lookup", "trader_id", "kind", "scope", "status"),)
```

**治理规则**：
- `TraderRun` 只追加（同日重跑覆盖，幂等键 = trader_id + run_date）
- `TraderMemory` **不可硬删除**，只改 `status`（refuted/archived 保留反证痕迹）
- belief 每次重建 `version+1`，保留历史版本（可回滚）
- Markdown 镜像：`data/trader_memory/trader_{id}.md`，每次记忆变更后由 DB 重新渲染

### 4.5 L 生命周期钩子（资金安全的硬门禁）

**四个拦截点，AI 不可绕过**（因为这些钩子在代码路径上，不在模型可见范围内）：

| 钩子 | 时机 | 拦截内容 | 违规处理 |
|---|---|---|---|
| `pre_reflect` | 反思写记忆前 | 校验 lesson 不能凭空产生（必须有 fact 引用） | 丢弃该 lesson |
| `pre_decision` | 决策前 | 检查是否处于熔断状态（成本/连续失败） | 强制观望 |
| **`pre_order`** | **每笔下单前** | **单标的上限 / 总仓位 / 现金下限 / 单日交易次数 / 单日亏损熔断 / 标的合法性** | **拒单 + 回灌原因给 AI** |
| `post_run` | 本轮结束 | 写 TraderRun、渲染记忆镜像、更新 belief | — |

**风控规则默认值**

```toml
[trader.guard]
max_weight_per_symbol  = 0.30   # 单一标的上限 30%
max_total_position     = 0.95   # 总仓位上限（留 5% 现金）
min_cash_weight        = 0.05   # 现金下限
max_trades_per_day     = 3      # 单日交易笔数上限（防手痒）
max_daily_loss_pct     = 3.0    # 单日亏损熔断（相对日初净值）
max_positions          = 8      # 最多持仓数
min_order_amount       = 1000   # 单笔最小金额
banned_symbols         = []     # 黑名单
cost_warn_pct          = 1.5    # 交易成本占比告警阈值
```

**"拒单 + 回灌"是关键设计**：拒单不是静默失败，而是把原因作为 observation 返回给 AI，
让它知道"为什么不行"并重新决策。这既是安全机制，也是**教学机制**——
AI 会从"被拒"中学到约束边界。这对应 Hermes 的"工具返回 recovery_hint，模型自己推理修复"。

### 4.6 V 评估接口（三层，逐层回答不同问题）

| 层 | 问题 | 指标 | 独立读路径 |
|---|---|---|---|
| **单笔** | 这笔交易对了吗？ | 预测方向命中、区间覆盖、Brier、偏差归因分布 | `report.py::trade_review()` |
| **组合** | 这个月赚了吗？比基准强吗？ | 总收益、超额、回撤、夏普、换手率、费用占比 | 复用 `engine.perf_stats()` + 基准对比 |
| **记忆** | **学的东西真的有用吗？** | lesson 生效前后的**相关决策胜率差**、记忆命中率 | `report.py::memory_effectiveness()` |

**第三层是最关键的，也是 v1.0 完全没有的。**
逻辑：每条 active lesson 记录 `activated_at`；
统计该时点前后、**该 lesson 作用域内**的决策表现差异。
若生效后没有改善甚至变差 → 该 lesson 应被降级（这是**记忆自我纠正机制**）。

> 没有这一层，"不断训练"就是黑箱——AI 可能越学越糟而无人发现。

---

## 5. 模块 / 目录结构（代码骨架）

```
core/
├── trader/                        # 【新增】AI 交易员
│   ├── __init__.py
│   ├── memory.py                  # S 层：三层记忆读写 + 状态机 + 防污染 + MD 镜像
│   ├── attributor.py              # V/S 层：纠错四分类归因
│   ├── tools.py                   # T 层：交易台工具注册表（function calling schema）
│   ├── guard.py                   # L 层：确定性风控门禁
│   ├── loop.py                    # E 层：每日自主循环（感知→反思→决策→校验→执行→留痕）
│   ├── prompts.py                 # 交易员人格 + 铁律 + 决策契约
│   └── report.py                  # V 层：三层评估（单笔/组合/记忆有效性）
├── predict/
│   └── llm.py                     # 【扩展】+chat_with_tools() 支持 function calling
├── paper/
│   └── engine.py                  # 【修正】ETF 免印花税；+场外赎回费阶梯
├── data/
│   └── fund.py                    # 【扩展】+search_funds() 候选发现
└── store/
    └── models.py                  # 【扩展】+TraderRun +TraderMemory
api/main.py                        # 【扩展】+/trader/* 端点
jobs/scheduler.py                  # 【扩展】+job_trader_daily
tests/test_trader.py               # 【新增】合成数据冒烟测试
```

---

## 6. 关键接口签名

### 6.1 记忆层（`core/trader/memory.py`）

```python
VIEW_FACT, VIEW_LESSON, VIEW_BELIEF = "fact", "lesson", "belief"
ST_CANDIDATE, ST_ACTIVE, ST_REFUTED, ST_ARCHIVED = (
    "candidate", "active", "refuted", "archived")

def write_fact(
    trader_id: int, statement: str, *, scope: str = "global",
    symbol: str = "", detail: dict | None = None,
    run_date: date | None = None,
) -> int:
    """系统写经验事实（对账+归因后调用）。只追加，永不修改。
    返回 fact_id。"""

def write_lesson(
    trader_id: int, statement: str, *, scope: str = "global",
    symbol: str = "", evidence_refs: list[dict] | None = None,
    proposed_confidence: float = 0.5,
) -> int:
    """AI 反思阶段提教训。**只能写成 candidate**。
    evidence_refs 必须非空（pre_reflect 钩子校验）；否则 ValueError。
    返回 lesson_id。"""

def promote_lessons(
    trader_id: int, *, min_evidence: int = 3, min_confidence: float = 0.6,
) -> list[dict]:
    """评估所有 candidate → 达标者升 active。
    达标条件：evidence_count >= min_evidence 且 confidence >= min_confidence
             且 counter_evidence 不超过 evidence_count 的 1/3。
    返回：[{"id": int, "from": "candidate", "to": "active", "reason": str}]"""

def refute_lesson(
    trader_id: int, lesson_id: int, counter: dict, *,
    auto: bool = False,
) -> dict:
    """记一条反证；反证数超阈值（>1/3）则降级 refuted。
    auto=True 表示由 memory_effectiveness() 自动触发。
    返回：{"lesson_id", "counter_count", "new_status"}"""

def rebuild_beliefs(trader_id: int) -> dict:
    """从 active lessons 聚合生成 belief（version+1）。
    只读系统行为，AI 不可直接调用。
    返回：{"version": int, "beliefs": [...], "policy_patch": dict}"""

def recall(
    trader_id: int, *, kind: str | None = None, scope: str | None = None,
    symbol: str | None = None, status: str = ST_ACTIVE, limit: int = 10,
) -> list[dict]:
    """检索记忆（供 AI 的 recall_memory 工具 + 上下文注入）。
    默认只返回 active（candidate 不进上下文，避免噪声影响决策）。"""

def render_markdown(trader_id: int) -> str:
    """把 DB 中的记忆渲染成 Markdown（镜像到 data/trader_memory/trader_{id}.md）。
    DB 是真相源，MD 是只读派生视图。"""
```

### 6.2 归因器（`core/trader/attributor.py`）—— 纠错的核心

```python
CLS_BETA, CLS_NOISE, CLS_FORECAST_ERR, CLS_DECISION_ERR = (
    "market_beta", "noise", "forecast_error", "decision_error")

def attribute_one(
    prediction: dict,          # {"id","symbol","horizon","probs","predicted_direction",
                               #  "base_date","base_price"}
    actual: dict,              # {"price","change_pct","direction","due_date"}
    market: dict | None = None,  # {"change_pct": float} 大盘同期涨跌
    *,
    flat_threshold_pct: float = 1.0,
    beta_ratio: float = 0.6,   # 标的涨跌中"可由大盘解释"的比例阈值
) -> dict:
    """单条偏差的四分类归因。

    判定次序（互斥，短路）：
      1) |actual.change_pct| <= flat_threshold → noise（噪声带内，不学）
      2) 大盘同向 且 |标的涨跌| 与 |大盘涨跌| 同向
         且 标的跌幅中"大盘解释部分"占比 >= beta_ratio → market_beta（不学）
      3) 预测方向 != 实际方向（且非上述）→ forecast_error（该学）
      4) 方向对但幅度/时点显著偏离（|预测区间外|） → decision_error（学策略层）

    返回：{
      "cls": str, "should_learn": bool,
      "evidence": {                     # 归因依据，写进 fact 的 detail
        "actual_change_pct": float, "market_change_pct": float | None,
        "beta_explained_pct": float, "flat_threshold": float,
        "predicted_direction": str, "actual_direction": str,
      },
      "reason": str,                    # 人读说明
    }
    """

def attribute_batch(
    reconciliations: list[dict], market: dict | None = None, **kw,
) -> dict:
    """批量归因，产出本轮应写入的 fact 列表 + 统计。

    返回：{
      "items": [{"prediction": ..., "attribution": ...}],
      "stats": {"total": int, "noise": int, "market_beta": int,
                "forecast_error": int, "decision_error": int},
      "learnable": int,      # should_learn 为 True 的条数
    }
    """
```

### 6.3 工具层（`core/trader/tools.py`）

```python
ToolSpec = dict   # {"name","description","parameters"(JSON Schema),"handler","write":bool}

def register(spec: ToolSpec) -> None:
    """注册工具到全局表（模块导入时自动注册内置工具）。"""

def schemas(write_allowed: bool = True) -> list[dict]:
    """导出 OpenAI function calling 的 tools 参数。
    write_allowed=False 时剔除写工具（用于只读场景/测试）。"""

def call(name: str, args: dict, ctx: dict) -> dict:
    """统一调用入口。

    ctx = {"trader_id", "run_date", "portfolio", "guard", "cache"}
    写工具内部会先调 guard.check_order()；被拒则返回
        {"ok": False, "rejected": True, "rule": str, "hint": str}
    返回：{"ok": bool, "data": Any, "digest": str, "ms": int, "rejected": bool}
    失败：不抛异常，返回 ok=False（让 AI 自己推理修复，对照 Hermes 的 recovery_hint）
    """
```

**内置工具清单**（对应 §4.2 表格，14 个）：
只读 12 个：`get_market_snapshot` `search_funds` `get_fund_profile` `get_quote`
`get_kline` `get_metrics` `run_prediction` `get_news` `get_portfolio`
`get_trade_history` `recall_memory` `get_past_mistakes`
写 2 个：`place_buy_order` `place_sell_order`

### 6.4 风控门禁（`core/trader/guard.py`）

```python
def check_order(
    action: dict,          # {"side":"buy"/"sell","symbol":str,"amount"/"volume":...}
    portfolio: dict,       # engine.portfolio() 输出
    day_state: dict,       # {"trades_today":int,"day_start_nav":float,
                           #  "current_nav":float,"realized_pnl":float}
    rules: dict | None = None,   # None → 读 config [trader.guard]
) -> dict:
    """确定性风控校验。**AI 不可绕过**。

    返回：{
      "allowed": bool,
      "rule": str,        # 触发的规则名（allowed=True 时为 ""）
      "detail": str,      # 人读说明
      "hint": str,        # 回灌给 AI 的修复提示（关键：教学机制）
      "adjusted": dict | None,   # 可选：建议的合规替代量（如削至上限）
    }

    校验项（次序短路）：
      1. banned_symbols / symbol 合法性
      2. max_daily_loss_pct 熔断（当日已亏超阈值 → 禁止一切买入）
      3. max_trades_per_day
      4. min_order_amount
      5. max_positions（买入时检查持仓数）
      6. max_weight_per_symbol / max_total_position / min_cash_weight
    """
```

### 6.5 循环（`core/trader/loop.py`）

```python
def create_trader(
    name: str, initial_cash: float, *,
    universe_hint: list[str] | None = None,
    persona: str = "", notes: str = "",
) -> int:
    """建交易员：PaperAccount(mode='trader') + 可选初始记忆。
    返回 trader_id。"""

def run_day(trader_id: int, as_of: date | None = None) -> dict:
    """执行一个交易日。幂等（同 trader_id+run_date 重复调用覆盖）。

    返回：{"trader_id","run_date","status","orders":int,"nav":float,
           "new_facts":int,"new_lessons":int,"promotions":int,
           "turns":int,"cost":float,"error":str}
    status ∈ {ok, watched, fused, error}
    """

def reflect(trader_id: int, run_date: date, attributions: dict) -> dict:
    """反思阶段：把归因结果交给 AI 总结教训 → write_lesson(candidate)
           → promote_lessons() → 若 belief 变化则 rebuild_beliefs()。
    返回：{"facts":int,"lessons":[...],"promotions":[...],"belief_version":int}
    """

def decide(trader_id: int, ctx: dict) -> dict:
    """自主决策循环（ReAct）：带工具，最多 max_turns 轮。

    返回：{"actions":[{"side","symbol","amount"|"volume","reason"}],
           "rationale":str,"watch_reason":str,
           "tool_calls":[...],"turns":int,"tokens":int}
    """
```

### 6.6 评估（`core/trader/report.py`）

```python
def trade_review(trader_id: int, days: int = 30) -> dict:
    """单笔层：预测命中/归因分布/Brier。"""

def portfolio_review(trader_id: int, benchmark: str = "沪深300") -> dict:
    """组合层：收益/超额/回撤/夏普/换手/费用占比。"""

def memory_effectiveness(trader_id: int) -> dict:
    """记忆层：每条 active lesson 生效前后的相关决策胜率差。

    逻辑：lesson.activated_at 为分界，统计 scope 内决策在分界前后的表现。
    若生效后无改善 → 返回 {"lesson_id", "verdict": "harmful", "suggest": "refute"}
    并（可选）自动调用 memory.refute_lesson(auto=True)。

    返回：{"lessons": [{"id","statement","before_win_rate","after_win_rate",
                      "delta","verdict","suggest"}]}
    """
```

---

## 7. 核心数据结构

### 7.1 交易员人格与铁律（system prompt 骨架）

```
你是「@name」，一个自主交易员，管理一笔虚拟资金并对其表现负责。

## 铁律（不可违背）
1. 你有权自主决定买卖，但每一次下单都要过风控；被拒时读 hint 再决定，不要重复被拒的动作。
2. 观望是合法且常见的选择。没有把握时观望，比勉强交易更专业。
3. 你的记忆里可能有错误结论。当新证据与记忆冲突时，以新证据为准，并在反思中提出修正。
4. 不要为自己的持仓找理由。先问"如果我没持仓，我还会买吗"。
5. 归因要诚实：市场整体下跌导致的亏损不是你的错，但也不是你的功；只有可归因于判断的部分才值得学。

## 决策输出契约（JSON）
{
  "actions": [{"side":"buy|sell","symbol":"","amount":0,"volume":0,"reason":""}],
  "rationale": "整体判断 ≤300字",
  "watch_reason": "若无动作，说明为什么观望"
}
```

### 7.2 记忆结构（运行时视图）

```python
MemoryView = {
    "facts":   [{"id","statement","symbol","scope","created_at","detail"}],
    "lessons": [{"id","statement","status","confidence","evidence_count",
                 "scope","symbol","activated_at","counter_count"}],
    "beliefs": [{"id","statement","version","policy_patch","updated_at"}],
}
```

---

## 8. 配置文件格式（`config.toml` 新增段）

```toml
[trader]
enabled           = true
default_cash      = 1000000.0   # 默认初始资金
run_time          = "18:40"     # 日循环时刻（晚于 evaluate_time=18:00）
max_turns         = 8           # 单轮 LLM 调用上限
max_tool_calls    = 20          # 单轮工具调用上限
max_cost_per_day  = 8.0         # 单日 LLM 成本上限（元）
max_consecutive_failures = 3    # 连续失败轮数 → 暂停
benchmark_symbol  = "沪深300"
memory_mirror_dir = "data/trader_memory"

[trader.memory]
min_evidence_to_activate = 3    # lesson 升 active 所需最少证据数
min_confidence_to_activate = 0.6
max_counter_ratio        = 0.333  # 反证占比超此值 → 降级 refuted
ctx_max_lessons          = 10   # 上下文注入的 lesson 上限
ctx_max_mistakes         = 5    # 上下文注入的近期错误上限

[trader.guard]
max_weight_per_symbol = 0.30
max_total_position    = 0.95
min_cash_weight       = 0.05
max_trades_per_day    = 3
max_daily_loss_pct    = 3.0
max_positions         = 8
min_order_amount      = 1000.0
banned_symbols        = []
cost_warn_pct         = 1.5

[trader.attributor]
flat_threshold_pct = 1.0        # 噪声带
beta_ratio         = 0.6        # 大盘解释占比阈值
```

---

## 9. 层间交叉检查

| 关系 | 说明 | 自洽性 |
|---|---|---|
| E 循环 ↔ L 门禁 | 循环内 AI 可任意发起下单，但每笔必经 `pre_order`；被拒原因回灌为 observation 形成教学闭环 | ✅ |
| E 熔断 ↔ C 预算 | `max_turns`/`max_tool_calls` 与上下文 16k 预算同源——轨迹滚动压缩保证不因轮次增加而爆预算 | ✅ |
| S 记忆 ↔ C 注入 | `recall()` 默认只返 active；candidate 不进上下文 → 防止未经证实的教训影响决策 | ✅ |
| S 记忆 ↔ V 有效性 | `memory_effectiveness()` 读 `activated_at` 分界统计；差评自动 `refute_lesson(auto=True)` → 记忆自我纠正 | ✅ |
| T 写工具 ↔ L 门禁 | 写工具 handler 内部先调 `guard.check_order()`，**不存在绕过路径**（因为 engine 只被 handler 调用） | ✅ |
| T 权限 ↔ S 权限 | AI 无写记忆工具；记忆写入只能在反思阶段由系统接口完成 → 决策时无法编造记忆 | ✅ |
| V 归因 ↔ S fact | 只有 `should_learn=True` 的归因才写 fact → 记忆库不会被 beta/噪声污染 | ✅ |

---

## 10. 六层决策表

| 层 | 决策 | 依据 | 状态 |
|---|---|---|---|
| E | **自主循环（ReAct）**，非固定管线 | 学习需试错空间；Hermes 自进化循环 | 已确认 |
| E | 三重熔断 + 观望是一等公民 | Hermes 三层硬上限；宁可不动不可乱动 | 默认假设 |
| T | **给 LLM 暴露 function calling**（与 v1.0 相反） | 自主决策需自主取信息 | 已确认 |
| T | 只 2 个写工具，且必经 guard | `capability-security` 最小权限 | 默认假设 |
| T | **AI 无写记忆工具** | 防决策时编造记忆自证 | 默认假设 |
| C | 16k 预算；candidate 不进上下文 | 防未证实教训污染决策 | 默认假设 |
| S | 2 新表，不动现有表 | 保护已跑通的对账链路 | 默认假设 |
| S | 记忆不可硬删，只改状态；belief 版本化 | 保留反证痕迹，可回滚 | 默认假设 |
| S | DB 为真相源 + MD 只读镜像 | 可审计"AI 学了什么" | 默认假设 |
| L | 4 拦截点；拒单带 hint 回灌 | 安全 + 教学双重目的 | 默认假设 |
| L | 风控阈值不在模型可见范围 | 防止 AI 通过工具调用改阈值 | 默认假设 |
| V | **三层评估，第三层验证记忆有效性** | 没有它"不断训练"就是黑箱 | **建议重点** |
| V | **归因四分类，只有 2 类可学** | 防 AI 把 beta/噪声当成自己的错去学 | **核心创新** |
| P | 记忆双写（DB + MD） | 可查询 + 可审计 | 默认假设 |

---

## 11. 参考架构图

```
          ┌──────────────────────────────────────────────────────┐
          │  E 自主循环  run_day()                               │
          │                                                      │
   ┌──────▼──────┐   ┌──────────────┐   ┌──────────────────┐   │
   │ 1 感知       │──►│ 2 对账       │──►│ 3 归因 (四分类)  │   │
   │ 账户/行情/大盘│   │ 到期预测对账 │   │ β/噪声/预测/决策 │   │
   └─────────────┘   └──────────────┘   └────────┬─────────┘   │
                                                  │ 只有2类可学
                    ┌─────────────────────────────▼─────────┐ │
                    │ 4 反思 [LLM]                          │ │
                    │   写 fact(系统) → 提 lesson(candidate) │ │
                    │   → 证据达标升 active → 重建 belief    │ │
                    └─────────────┬─────────────────────────┘ │
   ┌─────────────┐   ┌────────────▼──────────┐   ┌──────────┐ │
   │ 7 执行留痕  │◄──│ 6 风控门禁 guard      │◄──│ 5 决策   │ │
   │ 下单/净值   │   │ 拒单+回灌 hint        │   │[LLM]自主 │ │
   │ TraderRun   │   │ (AI 不可绕过)         │   │ 带工具   │ │
   └─────────────┘   └───────────────────────┘   └──────────┘ │
          └──────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────▼──────────────────────────┐
        │  S 记忆三层（DB 真相源 + MD 可审计镜像）             │
        │    fact 经验   ──只追加──►  lesson 教训  ──聚合──►   │
        │    (系统写)                 (AI写·需证据升级)        │
        │                                          belief 信念 │
        │                                          (系统写·版本化)│
        └─────────────────────────┬──────────────────────────┘
                                  │ 反馈：active 记忆注入决策上下文
        ┌─────────────────────────▼──────────────────────────┐
        │  V 三层评估                                          │
        │   单笔命中/归因分布 → 组合收益/超额 → 记忆有效性      │
        │   记忆无效 → 自动 refute → 记忆自我纠正              │
        └────────────────────────────────────────────────────┘
```

---

## 12. 遗留问题与未知区（诚实标注）

1. **在线学习在月尺度上依然可能学到噪声（最重要的一条）**
   归因机制能过滤掉"beta"和"噪声带"，但**无法过滤"伪相关"**——
   比如 AI 恰好连续 3 次在周三买入赚钱，可能总结出"周三买入更好"。
   **缓解**：`min_evidence=3` 太低，建议后期提到 8~10；
   且 `memory_effectiveness()` 会持续监督，无效则自动反驳。
   但**架构无法根除这个问题**——这是统计学习的本性。
   > 诚实的说法：这个系统能保证"AI 会记住它以为学到的东西，并且系统会持续检验这些东西有没有用"，
   > 但**不能保证它学到的是真的**。

2. **"纠正"的三种可能含义，本设计只实现了两种**
   - 纠正「对某标的的看法」→ ✅ 记忆层
   - 纠正「交易策略偏好」→ ✅ belief 层
   - 纠正「底层预测模型参数」→ ❌ **未实现**
   若你要的是第三种（真的重新训练 quant/LLM），那是另一个课题：
   一个月内样本量不足以微调任何模型。现有 `calibration` 是最接近的部分（概率校准），
   建议先用它，不要轻易上微调。

3. **`engine.py` 费用模型缺陷必须先修**
   `sell()` 对所有标的收印花税（第 144 行），但 **ETF/LOF 免印花税**。
   交易员会做 ETF 交易，成本被高估 0.05% 会系统性影响结论。
   **占位方案**：仅 `asset_type in ("stock","bse")` 收印花税。

4. **场外基金赎回到账延迟未建模**
   现状 `ofund` 是 T+0 确认、不计申购费、无赎回费阶梯。
   若交易员频繁买卖场外基金，费用会被严重低估。
   **占位方案**：先在 guard 里对 `ofund` 追加"持有 <7 天禁止卖出"的硬规则
   （模拟惩罚性赎回费），比精确建模赎回费更简单且效果等价。

5. **function calling 的模型支持度未知**
   现有 config 用 `deepseek-flash` / `deepseek-reasoner`。
   **我未验证这两个模型对 OpenAI 风格 tools 参数的支持程度与稳定性。**
   → 编码时必须有降级路径：若模型不支持 tools，退化为
   "把所有工具结果预设进上下文 + JSON 输出决策"（能力弱但仍可跑）。
   这一点必须在首次真实运行时验证。

6. **记忆规模增长后的检索性能未验证**
   记忆按标的×类型增长，一年后可能上万条。
   SQLite + 索引在万级应该没问题，但 `memory_effectiveness()` 的
   "前后胜率对比"需要扫描历史决策，可能变慢。
   → 预留：加 `activated_at` 索引 + 结果按需缓存。

7. **框架未覆盖：短视频/情绪数据、雪球组合抄作业、跨市场套利**
   本设计只处理"公开数据 + 基金/ETF 多头"。

---

## 13. 与 v1.0 的关系

| v1.0（受控实验台） | v2.0（AI 交易员） |
|---|---|
| 目的是**验证策略** | 目的是**让 AI 学会交易** |
| 固定管线 + 四组对照 | 自主循环 + 记忆三层 |
| 配置冻结（immutable-state） | 记忆演化（版本化 + 状态机） |
| 评估：单实验/组间/训练对照 | 评估：单笔/组合/**记忆有效性** |
| **降级为 v2.0 的一部分**：其"组间对照"能力可作为 v2.0 的评估增强 | 主体 |

**两者可以并存**：跑一个 AI 交易员 + 一个随机交易员 + 一个指数账户，
三个月后对比，就能回答"AI 到底有没有学会"。
v1.0 的 `baseline.py`（对照组策略）在本设计中直接复用。

---

## 14. 待用户拍板的决策点

| # | 决策 | 我的建议 | 影响 |
|---|---|---|---|
| 1 | 交易员可选标的口径 | **ETF/LOF 为主，场外受限** | 场外赎回费会吃掉短周期收益 |
| 2 | 决策频率 | **每交易日一轮，但允许 AI 自主选择"观望"** | 日频决策但不必日频交易 |
| 3 | 记忆不生效时怎么处理 | **交给 `memory_effectiveness()` 自动反驳** | 需要跑够天数才能见效 |
| 4 | 是否要"多个交易员同跑对比" | **建议要**（AI / 随机 / 指数 各一个） | 否则无法判断 AI 是否真的学会了 |
| 5 | 记忆镜像目录 | `data/trader_memory/` | 便于你直接读文件看 AI 学了什么 |
| 6 | 是否先修 ETF 印花税 | **必须先修** | 否则所有成本统计失真 |

---

## 15. 实施与验收记录（v2.0 编码完成后回填）

### 15.1 已落地的文件

| 层 | 文件 | 说明 |
|---|---|---|
| 设计 | `docs/AI交易员Harness-架构设计.md` | 本文档 |
| S | `core/store/models.py` | 新增 `TraderRun`（只追加留痕）、`TraderMemory`（三层记忆） |
| E/S | `core/trader/loop.py` | 七步循环 + ReAct 决策 + 幂等 + 熔断 + 自动暂停 |
| T | `core/trader/tools.py` | 14 个工具（12 只读 + 2 写），写工具内部必经 guard |
| L | `core/trader/guard.py` | 确定性风控门禁 + 拒单回灌 hint |
| L/E | `core/trader/attributor.py` | 归因四分类（唯一允许学习的闸门） |
| S | `core/trader/memory.py` | fact/lesson/belief 三层 + 证据门 + 自动反驳 + MD 镜像 |
| C | `core/trader/prompts.py` | system prompt（铁律）+ 决策上下文 + JSON 契约 |
| V | `core/trader/report.py` | 三层评估（逐笔 / 组合 / 记忆有效性） |
| 依赖 | `core/predict/llm.py` | 新增 `supports_tools()` / `chat_with_tools()` |
| 接线 | `config.toml` `config.example.toml` | `[trader]` `[trader.memory]` `[trader.guard]` `[trader.attributor]` |
| 接线 | `jobs/scheduler.py` | `job_trader_daily`（工作日 14:40，早于 AI 调仓） |
| 接线 | `api/main.py` | `/trader` 系列 10 个端点 |
| 数据 | `core/data/fund.py` | 新增 `search_funds()`（候选发现）、`fetch_fund_metrics()`（技术特征） |
| 测试 | `tests/test_trader.py` | 7 个用例，合成数据 + 假 LLM，不依赖网络与真实模型 |

### 15.2 跑测试暴露的 4 个真实缺陷（已修）

这些都是"看着写完了、其实不对"的问题，靠**真的把测试跑起来**才暴露：

1. **`guard` 削额可能削出"下不出来的单"**（`core/trader/guard.py`）
   原实现只看**单一约束**：权重超了按权重削，仓位超了按仓位削。于是当"权重还剩 30 万、
   账户只有 10 万现金"时，会回灌 `adjusted=300000` —— 一个根本下不出来的假额度。
   而这个额度会被 `tools._do_trade` 直接灌给 `engine.buy`，也会作为 hint 回灌给 AI。
   **hint 是教学机制，喂假额度等于教错。**
   修法：逐项算出每个约束的剩余额度，取**最紧的那一项**作为合规额度，并按真正
   起约束作用的那一项归因（`clip_weight` / `clip_cash` / `clip_position`）。

2. **`loop` 漏了 `new_facts` 键，且被裸 `except` 静默吞掉**（`core/trader/loop.py`）
   `run` 字典初始化时漏了 `new_facts`，导致 `run["new_facts"] += 1` 抛 KeyError；
   而它被同层的 `except Exception: pass` 吃掉 → **事实已入库、计数永远是 0**，
   最后在返回处才炸。修法：补键，并把裸 except 改为写入 `run["error"]` 留痕。
   > 教训：`except: pass` 会把"逻辑崩了"伪装成"本来就没有"，是最贵的沉默。

3. **`tools._deps` 是"整体替换"而非"逐项覆盖"**（`core/trader/tools.py`）
   文档写的是"可被 ctx['deps'] 覆盖"，实现却是"deps 非空就整体替换"。于是调用方
   只传 `engine` 时，`stock`/`fund` 凭空消失 → `KeyError` → 又被 `_do_trade` 的裸
   except 伪装成"取价失败"。**一个依赖缺失被报成了行情问题。**
   修法：先铺默认真实实现，再用调用方给的键逐项盖上；并把"资产类型解析失败"
   与"取价失败"拆成两个不同的错误码（`deps_missing` / `no_price`）。

4. **ETF/LOF 卖出被误征印花税**（`core/paper/engine.py`）
   `fee = _commission(revenue) + _stamp_tax(revenue)` 对卖出**无条件**计税。
   实际上印花税只对股票类卖出单边征收，场内 ETF/LOF 与场外基金卖出**免征**。
   后果是系统性低估基金策略收益，会让 AI 学到"交易越少越好"这种由错误成本
   造成的**伪规律**。
   修法：引入 `_STAMP_TAX_TYPES = ("stock", "bse")` 白名单。

### 15.3 §14 决策点的当前状态

| # | 决策 | 状态 |
|---|---|---|
| 1 | 标的口径 ETF 为主 | 已按建议实现：`search_funds()` 把场内 ETF/LOF 排前（可即时成交），场外作次级候选 |
| 2 | 每日一轮、允许观望 | 已实现：调度器工作日 14:40 一轮；`watched` 是一等公民状态，不计为失败 |
| 3 | 记忆不生效自动反驳 | 已实现：`memory_effectiveness(auto_refute=True)`，并暴露 API |
| 4 | 多交易员同跑对比 | 已支持（`run_all` 跑所有 `mode='trader'` 账户）；**对比视图待前端** |
| 5 | 记忆镜像目录 | 已按建议：`data/trader_memory/`（DB 是真相源，MD 为只读派生） |
| 6 | 先修 ETF 印花税 | **已修**（见 15.2-4） |

### 15.4 验收结论与剩余工作

**已验收**：
- `tests/test_trader.py` 7 个用例全绿（归因四分类 / 记忆证据门 / 记忆有效性自动反驳 /
  风控拦截与 hint / 端到端每日循环 / 观望非失败 / 无 tools 能力降级）。
- 接线验证 8 项全绿（配置读取 / guard 读配置 / 新数据函数 / deps 合并 /
  调度任务注册 / 印花税白名单 / 模块导入）。
- API 10 条 `/trader` 路由注册成功并返回 200。

**已验收（真实 LLM 补充，2026-09-23 18:41）**：
- **已用真实 LLM（deepseek-flash，`supports_tools()=True`）跑通 1 个真实交易日**
  （trader #2「试点交易员」，初始 100 万）。结果：`status=watched`、`turns=5`、
  `tokens=24644`、`cost=¥0.026`、`nav=1000000`。真实模型的 function calling、
  真实行情抓取、决策 JSON 解析、落库留痕**全部可用**。
- **归因链路已在真实数据上跑通**：`evaluate_due` 对账出 2 条到期预测，归因器正确判为
  `noise`（000997 −0.30%、016665 +0.00%，均落在 ±1.0% 噪声带内）。
  因此 `new_facts=0` 是**正确的"不该学就不学"**，不是失败。
- **学习闭环已就位**：本轮落下 `base_date=2026-09-23`、含 `next_day` 周期的预测，
  次一交易日运行即触发「对账→归因→写 fact→反思→升 belief」完整链路。
- 7 条读接口在**真实非空数据**下全部 200（见 §15.5）。

**尚未做（诚实标注）**：
- **"可学误差"分支未在真实数据上观察到**：真实跑到的 2 条到期预测都被判为 `noise`，
  故「forecast_error/decision_error → 写 fact → 升 belief」这条**真实**分支尚未走通；
  该分支现由 `test_run_day_end_to_end` 以合成数据覆盖，需等真实市场出现可学误差再确认。
- **本 harness 是"向前做"而非回放/回测**：`engine.current_price(symbol, asset_type)`
  无 `as_of` 参数，永远取最新价 → 把一个月压成一次历史回放会**用现价成交**，结果无意义。
  详见 §15.5-3。
- 前端页面未做（当前只有 API，靠 `/trader/{tid}` + `/trader/{tid}/runs?detail=true` 取数）。
- 多交易员对比视图（§15.3-4）待前端。
- 港股的印花税税率与规则未细化（当前沿用 A 股 0.05%，港股实际为 0.1%）。

### 15.5 本轮新增修复与设计发现

**新增修复（第 5 个缺陷）—— 对账失败被静默吞掉**

`_reconcile_and_attribute()` 里原 `except Exception: pass`（L375）与 15.2-2 是同一类缺陷：
对账一旦抛错就直接返回空 `items`，外部只看到"本轮没有可归因项"，
**无法区分"本就没到期"与"对账崩了"**——这正是自进化系统最怕的"学不到东西却查不出原因"。
已改为记录 `eval_err`、回传 `eval_error` 字段，由 `run_day` 写入 `run["error"]` 留痕
（**非致命**，不阻断本轮决策）。

**新增修复 —— pandas 弃用告警**

`core/data/fund.py:256` 的 `df["date"].iloc[-1] - pd.Timedelta(days=days)` 触发
numpy `generic` timedelta 单位弃用告警（未来版本将直接报错）。
已改为 `pd.Timestamp(...) - pd.Timedelta(days=int(days))` 显式归一。

**新增设计发现 —— 取价无时间维度（重要：决定"一个月模拟"怎么跑）**

- 事实：`core/paper/engine.py::current_price(symbol, asset_type)` **没有 `as_of`**，
  `buy()`/`sell()` 都走它。而**归因链路的行情是 point-in-time 正确的**——
  它复用 `core.verify.evaluator.evaluate_due()`，按 `Prediction.base_date + horizon` 回查历史净值。
  > ⚠️ **本节此处的"归因链路 point-in-time 正确"是错误结论，已在 §16.1 更正。**
  > `evaluate_due()` 实际用 `iloc[-1]`（最新价）当到期价，并非到期日价。
  > 保留原文不作删除，是为了留下"推断未落到代码行去核"这个教训本身。
- 结论：**成交用现价（只能向前做），预测对账用历史价（可回看）**。两条路径的时间语义不同，
  这是刻意的：交易不可穿越，评估必须能回看。
- 用户"给一组资金、为期一个月"因此有两种跑法，性质不同：
  1. **向前实盘（推荐，当前即可用）**：每交易日 14:40 跑一轮，跑满一个月。真实、无前视偏差，
     代价是必须等真实日历时间。
  2. **历史回放 / 回测（当前不可用）**：须先给 `current_price` 增补 `as_of`（取该日收盘价）
     + 交易日历；否则会用现价成交，结果失真。
- **建议**：v2.1 增补 `as_of` 支持，使同一 harness 既能向前实盘、也能快速回放验证策略。

**本轮接口验证（真实非空数据，7/7 全部 200）**

| 接口 | 返回要点 |
|---|---|
| `GET /trader` | trader #2（cash 100 万、runs 1、last_status=watched） |
| `GET /trader/2` | overview（run_stats + memory 统计） |
| `GET /trader/2/runs` | 含 2 条归因明细（`cls=noise`） |
| `GET /trader/2/review` | 三层评估（trade / prediction / learnable_signal） |
| `GET /trader/2/memory` | 4 条记忆（2 lesson + 2 belief，均 active） |
| `GET /trader/2/memory/markdown` | 只读 MD 镜像正文 |
| `GET /trader/2/tools` | 工具清单 + description |

另外 `create_trader(seed_beliefs=[...])` 的种子信条已**实测落库为 active 的 lesson+belief**
（memory_effectiveness 对它们返回 `verdict=insufficient` 并诚实说明样本不足）。

---

**交付状态**：可编码，且**已编码并跑通测试**。已过可编码三门槛——
类型完整（§4.4 表结构 + §7 全部结构字段已定义）、
API 真实（全部复用项目既有真实函数 + 已确认的 akshare 接口名 + OpenAI tools 参数格式）、
标准可达（§10 全部决策可验证；§12 已诚实标注 7 项未知区及占位方案）。

**知识库对照**：本设计对照了 `frameworks/core/hermes-agent.md`（自进化循环 + 文件即记忆 +
三层硬上限 + recovery_hint 自修复）、`patterns/capability-security.md`（最小权限）、
`patterns/actor-model.md`（实体隔离）、`patterns/event-sourcing.md`（只追加留痕）、
`patterns/cqrs.md`（评估独立读路径）。未做网络检索（无需扩展）。

---

## 16. v2.1：时点取价与历史回放（已编码并验收）

§15.5 末尾提出的 v2.1 建议（给取价链路增补 `as_of`）已落地。本节记录**做了什么、
为什么这样做、以及一条必须先更正的旧结论**。

### 16.1 先更正 §15.5 的一处错误结论（重要）

§15.5 曾写：

> 「而**归因链路的行情是 point-in-time 正确的**——它复用 `evaluate_due()`，
> 按 `Prediction.base_date + horizon` 回查历史净值。」

**这条是错的，本次核查后更正**。实际 `evaluate_due()` 取"实际价"用的是
`iloc[-1]`（**最新一根**），不是到期日那根。后果：

- 实盘下看不太出来（到期日就在眼前，"最新价"≈"到期价"），所以长期被掩盖；
- 一旦做回放就致命：三个月后到期的 `next_day` 预测，会拿"回放终点当天"的价去判对错，
  写进 `CalibSample` 就是**脏样本**，且校准模型会据此学出错误规律。

⇒ 这正好印证 §15.5 的分工结论（"交易不可穿越、评估必须能回看"）——
但当时把"应该怎样"当成了"已经怎样"。**设计文档里的断言必须逐个落到代码行去核**，
否则一处善意的推断会变成下游全部结论的地基裂缝。

### 16.2 新增模块：`core/data/price.py`（时点取价原语）

| 函数 | 语义 | 关键点 |
|---|---|---|
| `to_date(v)` | date/datetime/Timestamp/str → `date` | 三类数据源 dtype 不同，统一入口 |
| `bars_needed(as_of, lead=250)` | 估算要请求多少根 K 线 | 按 250 向上取整 → 相邻回放日**共享同一缓存键**（实测整月稳定 500 根） |
| `price_on(symbol, type, as_of)` | as_of（含）之前最近交易日收盘价 | 取不到未来价 → `None` |
| `prev_close_on(...)` | 严格早于 as_of 的最近收盘价 | 算"当日涨跌幅"用，回放不能用实时涨跌幅 |
| `snapshot_on(...)` | 历史口径行情快照 | 返回 `note="历史口径（截至 X 收盘）"`，**避免 AI 误当实时价** |
| `closes_upto(..., limit)` | 截至 as_of 的收盘序列 | `get_kline` / 技术特征计算用 |
| `trading_days_ex(start, end)` | 交易日 + 口径来源 | `source ∈ {"index","weekday"}`；`weekday` 是降级兜底 |

**交易日历口径**：以沪深300指数日 K 的真实日期为准（自动含节假日），
而不是"排除周末"——后者会把国庆/春节算成交易日，净值曲线凭空多出几天。
指数不可达时降级为 `weekday` 并在返回里**显式标注 source**，让上层必须处理，
而不是静默用一个偏多的日历往下跑。

**dtype 陷阱（踩过）**：股票/ETF 的 K 线 `date` 是 `object`（字符串）、
指数 K 线是 `object`（`datetime.date` 对象）、场外基金净值是 `datetime64[ns]`。
直接拿原始列比较，基金那条路径能过、指数那条会静默错。
一律 `pd.to_datetime(..., errors="coerce")` 归一后再比。

### 16.3 分层改动（回放的正确性靠"处处截断"，不是靠某个开关）

| 文件 | 改动 | 为什么必须 |
|---|---|---|
| `core/paper/engine.py` | `current_price/buy/sell/portfolio/snapshot_nav` 增 `as_of` | `snapshot_nav` 原来**硬编码 `date.today()`** → 20 个回放日全部挤成同一天并互相覆盖，净值曲线只剩一根点 |
| `core/verify/evaluator.py` | 新增 `_actual_price_on(pred, on_date)`；`evaluate_due(..., replay=)` | 修 §16.1 的脏样本缺陷。**实盘路径逐字节不变**（`_actual_price` 签名保留，避免破坏既有测试与外部调用） |
| `core/trader/tools.py` | 回放时按 as_of 截断各工具；`_REPLAY_UNAVAILABLE` 隐藏工具 | 见 16.4 |
| `core/trader/loop.py` | `replay_preflight()` + `replay()` 驱动；`_day_state` 回放分支；`_market_brief_on()`；`_call_evaluate_due()` | 见 16.5 |

### 16.4 回放的铁律：宁可少给工具，也不给一个能偷看未来的工具

这是整个 v2.1 最重要的一条设计判断。回放里一旦有任何一处泄漏未来数据，
AI 等于开了天眼，回放结果**全部作废且看起来完全正常**（有净值、有曲线、有胜率），
是最难人工发现的一类错误。所以采用**三档处理**，不做"尽力而为"：

1. **能按时点截断的 → 严格截断**
   `get_quote` / `get_kline` / `get_metrics` / `get_portfolio` / `get_fund_profile`（剥掉实时收益字段）、
   `get_news`（按 `publish_time <= as_of` 当日 23:59 过滤）；
   AI 上下文里的"大盘简报"也换成 `_market_brief_on(as_of)`（只用基准指数 as_of 之前的涨跌幅）。
2. **做不到的 → 明确拒绝，绝不静默降级**
   `get_market_snapshot`（涨跌家数/资金流向没有历史回溯数据源）、
   `run_prediction`（依赖实时行情 + 面向未来的校准模型）→ 返回 `rule="replay_unavailable"`
   并附 `hint` 说明"请改用 get_kline / get_metrics / get_quote（已按时点截断）"。
   这两个还**同时从 AI 可见的工具 schema 里摘掉**（`tools.schemas(replay=True)`）——
   拒绝是二道防线，不给才是第一道。
3. **刻意不截断的 → 说明理由**
   `get_trade_history` 不过滤 as_of。因为回放写入的 `traded_at` 是**墙钟时间**而非模拟日，
   按 as_of 过滤反而会把回放自己刚下的单藏起来。正确性由"全新交易员 + 按日推进"保证：
   当日看到的成交必然只来自本日及之前。**这是"看起来该过滤但故意不过滤"的少数派**，
   必须有注释，否则下一个人会以为是漏了。

### 16.5 回放驱动：`replay_preflight()` + `replay()`

拆成两个函数是刻意的：API 需要**先知道要跑几个交易日**才能决定同步跑还是丢后台。
若把判断同时写在 API 和 `replay()` 里，两边迟早不一致——
典型症状：API 放行了 90 天、`replay()` 内部又拒了，用户看到"已接受"却永远拿不到结果。

**四条正确性前提**（`replay_preflight` 逐条把关，任一不满足直接拒绝并说明原因）：

1. **全新交易员**：账户不得已有成交/净值。否则回放持仓与真实持仓叠加，净值与 T+1 可卖量全错。
   需绕过时 `allow_existing=True`，并在 `note` 里显式标注"结果不可信，仅供调试"。
2. **按日推进**：交易日升序，`for ds in pre["days_list"]` 逐日 `run_day(..., replay=True)`。
   跳日会让 T+1 与净值曲线错位。
3. **只用时点价**：成交与估值全部走 `price_on`，回放路径**不得**碰 `current_price()`。
4. **成本闸门**：每个交易日 = 一次完整 LLM 循环。默认 `max_days=60`，超出直接拒绝；
   确认要跑长区间传 `allow_long=True`。**这条是省钱闸门，不是技术限制**。

长跑（后台任务）用 `on_day(progress)` 回调上报进度，回调自身出错被吞掉——进度上报不该拖垮回放。

`_day_state()` 的回放分支还有一个隐蔽坑：回放写入的 `traded_at` 是墙钟时间，
若按 `traded_at >= date.today()` 统计"今日成交"，会把之前所有回放日的成交都算进来，
几轮之后就误触 `max_trades_per_day`、交易被判超额。
→ 回放改为"本日从 0 起算 + 成交时自增"，日初净值取 `as_of` 之前最近一条净值。

### 16.6 API

```
POST /trader/{tid}/replay?start=YYYY-MM-DD&end=YYYY-MM-DD
     [&allow_existing=false][&max_days=60][&allow_long=false][&background=true]
```

默认 `background=true`：同步跑二十天必然撞 HTTP 超时，故立即返回受理信息
（`days` / `calendar_source` / `runs_endpoint`），逐日进度落在 `TraderRun`，
用 `GET /trader/{tid}/runs` 观察。短区间调试可传 `background=false` 同步拿完整汇总。

`replay()` 返回：`{trader_id, start, end, calendar_source, days, ok, watched, skipped,
errors, nav_start, nav_end, total_return_pct, cost_total, runs[], error, note}`。

### 16.7 验收（26/26 通过，含 6 条新增回放测试）

| 测试 | 守的是什么 |
|---|---|
| `test_price_point_in_time_truncation` | 时点取价严格截断；周末退到前一交易日；早于数据返回 `None`（不许倒推）；空数据不抛 |
| `test_replay_hides_realtime_only_tools` | 泄漏未来的工具被摘出 schema；硬调也明确拒绝而非静默返回实时数据 |
| `test_replay_requires_fresh_trader` | 非全新账户被拒；`allow_existing` 放行并在 note 标注不可信 |
| `test_replay_drives_trading_days_in_order` | 严格升序、每日都走回放路径、净值/成本汇总正确、日历降级必须告警 |
| `test_replay_rejects_bad_range_and_long_span` | 日期非法/起晚于止/超 max_days 全部明确拒绝；超限时**一次 run_day 都不许调** |
| `test_replay_day_end_to_end_point_in_time_only` | **不 mock run_day** 的真实全链路：每次取价都带 `as_of`、成交价=历史收盘价、净值快照落在 as_of、不可用工具已摘除 |

最后一条是关键：它从 `_perceive` 到成交到净值快照走的是与实盘**完全相同**的代码路径，
只差 `replay=True`。只要有一处忘了传 `as_of`，成交价就变成"今天的价"，
而回放结果看起来仍然正常——人工几乎发现不了。

### 16.8 回放验不了什么（诚实标注，别拿回放当实盘验收）

- **验不了「预测驱动的自我纠错」**：`run_prediction` 在回放中被隐藏，
  所以"看到预测→按预测下注→对账→归因→改进"这条学习闭环在回放里是断的。
  回放能验的是：**成交价口径 / 估值 / 风控闸门 / 留痕 / 记忆写入**。
- ⇒ **回放与实盘必须都跑**：回放用来快速积累"决策-结果"样本、验证风控与账务；
  学习闭环必须靠实盘向前跑。这条限制无法用工程手段绕过（除非重建一套历史预测快照库）。
- 交易日历降级为 `weekday` 时，天数会偏多、净值曲线含非交易日点 → 返回里显式告警。
- **`_actual_price` 的双口径**：实盘用"最新价"、回放用"到期日价"。
  这意味着**同一历史预测，实盘与回放判出的对错可能不同**——回放那条才是对的。
  若某天发现校准样本口径可疑，先查这个。

### 16.9 本轮修复的缺陷（第 6、7 个）

6. **`snapshot_nav` 硬编码 `date.today()`** → 多日回放净值全部覆盖成一天，净值曲线失效。
7. **`evaluate_due` 的"最新价当到期价"**（§16.1）→ 静默污染 `CalibSample`。

另有 1 个健壮性补强：`_call_evaluate_due()` 兼容**不认识 `replay` 参数**的自定义
`evaluate_due` 实现（`deps` 注入的可能是旧签名）。原先直接 `fn(as_of, replay=...)`
会 `TypeError` 让整轮对账作废，而失败被记成 `eval_error`，
让人误判成"对账逻辑坏了"而不是"签名不匹配"。现在改为**签名探测 → 降级并显式留痕**，
说明"回放对账已降级为默认口径，样本口径不一致，慎用该回放结果"。

### 16.10 遗留问题（v2.1 新增）

1. **回放无历史预测快照**：`run_prediction` 依赖实时行情，回放中不可用 →
   学习闭环无法在回放中验证（§16.8）。若要做，需要先建立"预测入库时同时快照所有输入"的机制。
2. **`daily_ret` 依赖 `NavHistory` 前一条**：跳日回放时"日收益"跨越多个自然日，
   语义上更像"区间收益"。当前按顺序跑不受影响，但**跳日回放不可用**。
3. **`get_trade_history` 依赖"全新交易员"前提**：这是契约级约束而非代码强制，
   只由 `replay_preflight` 把关。若有人绕过 API 直接调 `run_day(..., replay=True)`，约束失效。
4. **回放不真实的部分**：T+1 用"下一交易日结算"简化；涨跌停/停牌未模拟；
   成交按收盘价全额成交、无滑点与冲击成本。**回放收益必然偏乐观**，
   适合验证机制正确性，不适合评估策略盈利能力。
