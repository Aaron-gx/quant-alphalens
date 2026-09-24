"""数据库表结构：预测记录、对账结果、情报库、模拟盘等。"""
from __future__ import annotations

from datetime import datetime, date

from sqlalchemy import (
    String, Float, Integer, Text, DateTime, Date, Boolean, ForeignKey, JSON, Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """一次结构化预测：概率分布必须机器可读，这是验证闭环的前提。"""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    asset_type: Mapped[str] = mapped_column(String(16), default="stock")  # stock/fund/index/hk
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    base_date: Mapped[date] = mapped_column(Date)          # 预测基准日（收盘后数据）
    base_price: Mapped[float] = mapped_column(Float, default=0.0)
    engine: Mapped[str] = mapped_column(String(32), default="llm")  # llm / quant / fused
    mode: Mapped[str] = mapped_column(String(16), default="fast")   # fast / deep

    # 各周期预测: {"next_day": {"up":.5,"flat":.3,"down":.2,"range":"±1.5%"}, ...}
    horizons: Mapped[dict] = mapped_column(JSON, default=dict)
    scenarios: Mapped[dict] = mapped_column(JSON, default=dict)      # 乐观/基准/悲观情景
    confidence: Mapped[int] = mapped_column(Integer, default=50)     # 0-100
    key_factors: Mapped[list] = mapped_column(JSON, default=list)
    action: Mapped[dict] = mapped_column(JSON, default=dict)         # 建议/买卖点/止损
    risks: Mapped[list] = mapped_column(JSON, default=list)
    report_text: Mapped[str] = mapped_column(Text, default="")       # 人读报告
    input_snapshot: Mapped[dict] = mapped_column(JSON, default=dict) # 输入数据摘要(复盘用)
    llm_model: Mapped[str] = mapped_column(String(64), default="")

    evals: Mapped[list["PredictionEval"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan")


class PredictionEval(Base):
    """到期对账结果：每个 prediction × horizon 一行。"""
    __tablename__ = "prediction_evals"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    horizon: Mapped[str] = mapped_column(String(16))       # next_day/one_week/one_month/quarter
    due_date: Mapped[date] = mapped_column(Date)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    actual_price: Mapped[float] = mapped_column(Float, default=0.0)
    actual_change_pct: Mapped[float] = mapped_column(Float, default=0.0)
    actual_direction: Mapped[str] = mapped_column(String(8), default="")   # up/flat/down
    predicted_direction: Mapped[str] = mapped_column(String(8), default="")
    hit: Mapped[bool] = mapped_column(Boolean, default=False)
    brier_score: Mapped[float] = mapped_column(Float, default=0.0)  # 概率校准误差,越小越好

    prediction: Mapped[Prediction] = relationship(back_populates="evals")

    __table_args__ = (Index("ix_eval_pred_horizon", "prediction_id", "horizon", unique=True),)


class NewsItem(Base):
    """情报库：新闻原文 + LLM 情绪打分，供预测与复盘检索。"""
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    symbol: Mapped[str] = mapped_column(String(20), default="", index=True)  # 空=宏观/市场
    source: Mapped[str] = mapped_column(String(32), default="")   # eastmoney/caixin/cls/...
    category: Mapped[str] = mapped_column(String(16), default="news")  # news/notice/research/flash
    title: Mapped[str] = mapped_column(String(256), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(String(512), default="")
    publish_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    title_hash: Mapped[str] = mapped_column(String(64), index=True, default="")  # 去重

    # LLM 情绪三维度（xystock 方案）+ 关联度（本项目新增，用于舆情筛选）
    sentiment: Mapped[str] = mapped_column(String(8), default="")   # 乐观/中性/悲观
    intensity: Mapped[int] = mapped_column(Integer, default=0)      # 1-5
    deviation: Mapped[int] = mapped_column(Integer, default=0)      # 1-5 预期偏离度
    relevance: Mapped[int] = mapped_column(Integer, default=0)      # 1-5 与标的的关联度
    scored: Mapped[bool] = mapped_column(Boolean, default=False)


class CalibSample(Base):
    """自学习样本：一次"引擎概率 → 实际方向"的观测，用于校准与元模型训练。

    两个来源：
    - eval：真实归档预测到期对账的结果（LLM/融合引擎）
    - walk_forward：量化引擎在历史数据上做样本外滚动预测的结果（冷启动即可用）
    """
    __tablename__ = "calib_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    source: Mapped[str] = mapped_column(String(16), default="eval", index=True)
    engine: Mapped[str] = mapped_column(String(16), default="llm", index=True)
    horizon: Mapped[str] = mapped_column(String(16), default="next_day", index=True)
    symbol: Mapped[str] = mapped_column(String(20), default="", index=True)
    asset_type: Mapped[str] = mapped_column(String(16), default="stock")
    base_date: Mapped[date] = mapped_column(Date, default=date.today)
    p_up: Mapped[float] = mapped_column(Float, default=0.0)
    p_flat: Mapped[float] = mapped_column(Float, default=0.0)
    p_down: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[int] = mapped_column(Integer, default=50)
    senti_score: Mapped[float] = mapped_column(Float, default=0.0)     # 标的舆情分 -100~100
    universe_senti: Mapped[float] = mapped_column(Float, default=0.0)  # 全市场情绪分
    relevance_cov: Mapped[float] = mapped_column(Float, default=0.0)   # 舆情相关度覆盖率
    actual_direction: Mapped[str] = mapped_column(String(8), default="")   # up/flat/down
    actual_change_pct: Mapped[float] = mapped_column(Float, default=0.0)
    flat_threshold: Mapped[float] = mapped_column(Float, default=1.0)
    prediction_id: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (Index("ix_calib_dedup", "source", "engine", "horizon",
                            "symbol", "base_date", unique=False),)


class ModelTrainingRun(Base):
    """自学习训练记录：每轮"用新样本重训校准层"的效果留痕，构成学习曲线。"""
    __tablename__ = "model_training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    scope: Mapped[str] = mapped_column(String(32), default="global", index=True)  # global/quant/llm
    horizon: Mapped[str] = mapped_column(String(16), default="next_day")
    method: Mapped[str] = mapped_column(String(32), default="")   # identity/temperature/meta_logit
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_test: Mapped[int] = mapped_column(Integer, default=0)
    hit_rate: Mapped[float] = mapped_column(Float, default=0.0)          # 样本外方向命中率
    base_hit_rate: Mapped[float] = mapped_column(Float, default=0.0)     # 基准（多数类）命中率
    brier_before: Mapped[float] = mapped_column(Float, default=0.0)      # 校准前样本外 Brier
    brier_after: Mapped[float] = mapped_column(Float, default=0.0)       # 校准后样本外 Brier
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(String(256), default="")


class PaperAccount(Base):
    """虚拟账户。"""
    __tablename__ = "paper_accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(16), default="manual")  # manual / ai_signal
    initial_cash: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    note: Mapped[str] = mapped_column(String(256), default="")

    positions: Mapped[list["PaperPosition"]] = relationship(
        back_populates="account", cascade="all, delete-orphan")
    orders: Mapped[list["PaperOrder"]] = relationship(
        back_populates="account", cascade="all, delete-orphan")


class PaperPosition(Base):
    __tablename__ = "paper_positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("paper_accounts.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(20))
    name: Mapped[str] = mapped_column(String(64), default="")
    asset_type: Mapped[str] = mapped_column(String(16), default="stock")
    volume: Mapped[int] = mapped_column(Integer, default=0)          # 总持仓
    available: Mapped[int] = mapped_column(Integer, default=0)       # 可卖（T+1 当日买入不可用）
    avg_cost: Mapped[float] = mapped_column(Float, default=0.0)

    account: Mapped[PaperAccount] = relationship(back_populates="positions")


class PaperOrder(Base):
    __tablename__ = "paper_orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("paper_accounts.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(20))
    name: Mapped[str] = mapped_column(String(64), default="")
    side: Mapped[str] = mapped_column(String(8))         # buy / sell
    price: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    amount: Mapped[float] = mapped_column(Float)          # 成交额
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    traded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    reason: Mapped[str] = mapped_column(String(256), default="")  # 下单理由/AI信号来源

    account: Mapped[PaperAccount] = relationship(back_populates="orders")


class NavHistory(Base):
    """账户每日净值快照。"""
    __tablename__ = "nav_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("paper_accounts.id"), index=True)
    date: Mapped[date] = mapped_column(Date)
    cash: Mapped[float] = mapped_column(Float)
    market_value: Mapped[float] = mapped_column(Float)
    total_value: Mapped[float] = mapped_column(Float)
    daily_return: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (Index("ix_nav_acc_date", "account_id", "date", unique=True),)


class Watchlist(Base):
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    asset_type: Mapped[str] = mapped_column(String(16), default="stock")
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    auto_predict: Mapped[bool] = mapped_column(Boolean, default=True)  # 每日自动跑预测


class AnalysisReport(Base):
    """分析报告存档（含 prompt 摘要，可复盘）。"""
    __tablename__ = "analysis_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    report_type: Mapped[str] = mapped_column(String(32), default="comprehensive")
    content: Mapped[str] = mapped_column(Text)
    prediction_id: Mapped[int] = mapped_column(Integer, default=0)


class UserProfile(Base):
    """用户画像：单例表，持久化偏好并注入每次分析。"""
    __tablename__ = "user_profile"

    id: Mapped[int] = mapped_column(primary_key=True)
    risk_preference: Mapped[str] = mapped_column(String(16), default="neutral")
    custom_principles: Mapped[str] = mapped_column(Text, default="")
    position_text: Mapped[str] = mapped_column(Text, default="")   # 当前持仓描述
    trade_style: Mapped[str] = mapped_column(String(64), default="")   # 左/右侧、长/短线
    common_mistakes: Mapped[str] = mapped_column(Text, default="")     # 常犯错误
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class TokenUsage(Base):
    """LLM token 用量流水，用于成本统计。"""
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    called_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    model: Mapped[str] = mapped_column(String(64), default="")
    caller: Mapped[str] = mapped_column(String(32), default="")  # predict/sentiment/brief...
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)


# ---------------------------------------------------------------------------
# AI 交易员（core/trader）：自主循环 + 三层记忆
# 设计文档见 docs/AI交易员Harness-架构设计.md
# ---------------------------------------------------------------------------


class TraderRun(Base):
    """交易员每个交易日一轮的完整留痕（含工具调用轨迹）。

    只追加；同日重跑覆盖（幂等键 = trader_id + run_date）。
    这张表是该交易员的"事件流"——单一真相源，评估只读它。
    """
    __tablename__ = "trader_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trader_id: Mapped[int] = mapped_column(Integer, index=True)   # = paper_accounts.id
    run_date: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    # ok / watched / fused / error

    perception: Mapped[dict] = mapped_column(JSON, default=dict)      # 感知快照
    reconciliations: Mapped[list] = mapped_column(JSON, default=list)  # 到期对账结果
    attributions: Mapped[list] = mapped_column(JSON, default=list)     # 归因结论
    new_facts: Mapped[int] = mapped_column(Integer, default=0)
    new_lessons: Mapped[list] = mapped_column(JSON, default=list)      # 本轮新增/更新教训
    memory_promotions: Mapped[list] = mapped_column(JSON, default=list)  # 记忆升/降级

    tool_calls: Mapped[list] = mapped_column(JSON, default=list)       # 完整工具轨迹
    decision: Mapped[dict] = mapped_column(JSON, default=dict)         # AI 决策
    guard_results: Mapped[list] = mapped_column(JSON, default=list)    # 风控结论
    orders: Mapped[list] = mapped_column(JSON, default=list)           # 实际成交

    nav: Mapped[float] = mapped_column(Float, default=0.0)
    llm_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_cost_est: Mapped[float] = mapped_column(Float, default=0.0)
    turns: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(String(512), default="")

    __table_args__ = (Index("ix_trader_run_date", "trader_id", "run_date", unique=True),)


class TraderMemory(Base):
    """三层记忆：fact（经验）/ lesson（教训）/ belief（信念）。

    写入权限（硬规则）：
    - fact   : 系统自动写（对账 + 归因后），只追加、永不修改
    - lesson : AI 在反思阶段写，但只能写 candidate；升 active 需证据数达标
    - belief : 系统从 active lesson 聚合生成，AI 不可直接写

    不可硬删除：只改 status（refuted/archived 保留反证痕迹）。
    """
    __tablename__ = "trader_memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    trader_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(8), index=True)      # fact/lesson/belief
    scope: Mapped[str] = mapped_column(String(48), default="global")
    # global / asset_type:fund / symbol:510300
    symbol: Mapped[str] = mapped_column(String(20), default="")
    statement: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="candidate", index=True)
    # candidate / active / refuted / archived

    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    evidence_refs: Mapped[list] = mapped_column(JSON, default=list)    # 证据引用
    counter_evidence: Mapped[list] = mapped_column(JSON, default=list)  # 反证

    source: Mapped[str] = mapped_column(String(24), default="system")  # system/ai_reflection
    policy_patch: Mapped[dict] = mapped_column(JSON, default=dict)     # 仅 belief
    version: Mapped[int] = mapped_column(Integer, default=1)           # 仅 belief
    activated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    __table_args__ = (Index("ix_mem_lookup", "trader_id", "kind", "scope", "status"),)
