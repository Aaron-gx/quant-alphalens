"""标签口径的唯一权威：周期 → 前向交易日步数、平盘阈值、方向判定。

为什么必须单独成一个模块
------------------------
这三个口径原先散落在三处，各写一份：

    - 量化引擎的训练标签      core/predict/quant.py    （只认 next_day，阈值不缩放）
    - 到期对账的"真实方向"    core/verify/evaluator.py （单个阈值套所有周期）
    - 交易员归因的噪声带      core/trader/attributor.py（默认 1.0，与预测口径不同源）

后果不是"不够优雅"，而是**标签错位**：对账用 ±1.0% 判出来的"真实方向"，
和模型训练时用 ±0.5% 学的"方向"，本质上是两个不同的问题。
自学习校准层于是拿着错标的样本去调概率——越调越偏，而从任何指标上都看不出来。

现在这三个口径**只有一份实现**，谁要判方向都必须调这里，改一处即全局一致。
"""
from __future__ import annotations

import math

from core.config import get

# 四个预测周期（顺序即展示顺序）
HORIZONS = ("next_day", "one_week", "one_month", "quarter")
# 周期 → 前向收益的**交易日**步数
HORIZON_STEPS = {"next_day": 1, "one_week": 5, "one_month": 20, "quarter": 60}
# 对账用的到期**自然日**折算 —— ⚠️ **已废弃，不要在新代码里用**。
#
# 它只是"交易日 → 自然日"的粗略换算（_TRADING_PER_CAL ≈ 0.70，即 244/365）。
# 留着这个名字只为兼容旧引用（core/predict/schema.py 仍在 import 它）。
# 真正算到期日请用 `due_dates_for()` —— 它按真实交易日历数**交易日**。
#
# 为什么必须换掉：模型训练学的是「T+N 个**交易日**后的方向」（HORIZON_STEPS），
# 对账却按自然日到期。两者在每个周期上都不等，且差值不固定：
#     next_day   1 交易日 = 1~3 自然日（跨周末就漂 2 天）
#     one_week   5 交易日 = 5~9 自然日
#     one_month 20 交易日 = 26~32 自然日
#     quarter   60 交易日 = 82~94 自然日
# 于是"到期日"取到的收盘价和模型要预测的那一天不是同一天，
# direction_of 判出的"真实方向"与训练标签是两回事 —— 写进 CalibSample 就是脏样本。
HORIZON_DAYS = {"next_day": 1, "one_week": 7, "one_month": 30, "quarter": 90}
DEFAULT_HORIZON = "next_day"
PROB_KEYS = ("up", "flat", "down")

# 类别索引顺序：0=跌 1=平 2=涨。
# ⚠️ **凡是要把概率排成 numpy 数组的地方，列顺序必须与此一致**——
# Brier / 命中率都是拿列下标直接和这个索引比较的。
# 本项目真实踩过：特征向量按 [涨,平,跌] 排，标签索引按 [跌,平,涨]，
# 于是温度校准档位会把**涨/跌概率对调**后输出，而所有指标看起来都"正常"。
LABELS = ("down", "flat", "up")
LABEL_IDX = {"down": 0, "flat": 1, "up": 2}


def probs_to_vector(probs: dict) -> list[float]:
    """{up,flat,down} → [跌, 平, 涨]（列顺序 = LABEL_IDX，唯一正确的排法）。"""
    return [float(probs.get(k, 0) or 0) for k in LABELS]


def vector_to_probs(vec) -> dict:
    """[跌, 平, 涨] → {down, flat, up}。"""
    return {k: float(vec[i]) for i, k in enumerate(LABELS)}


DEFAULT_BASE_THRESHOLD = 0.5


def base_threshold() -> float:
    """基准（次日）平盘阈值（%），唯一来源 [predict].flat_threshold_pct。"""
    try:
        return float(get("predict", "flat_threshold_pct", DEFAULT_BASE_THRESHOLD))
    except Exception:
        return DEFAULT_BASE_THRESHOLD


def flat_threshold_for(horizon: str, base: float | None = None) -> float:
    """该周期的"平盘"阈值（%）。

    随机游走下波动率随 √t 增长，所以阈值也按 √step 缩放：
    次日 ±0.5% → 一周 ±1.12% → 一月 ±2.24% → 一季 ±3.87%（base=0.5）。

    不缩放的两种翻车方式：
      - 用 ±0.5% 判一季度 → 几乎一切都算涨/跌，中间类被抹掉；
      - 用 ±3.9% 判次日   → 几乎一切都是平盘，退化成"多数类猜谜"。
    """
    if base is None:
        base = base_threshold()
    step = HORIZON_STEPS.get(horizon, 1)
    return round(float(base) * math.sqrt(step), 4)


def direction_of(change_pct: float, threshold: float) -> str:
    """涨跌幅（%）+ 阈值（%）→ up / flat / down。

    全项目唯一的判定实现——训练标签、到期对账、交易员归因共用。
    """
    try:
        c = float(change_pct)
    except (TypeError, ValueError):
        return "flat"
    th = abs(float(threshold or 0.0))
    if c > th:
        return "up"
    if c < -th:
        return "down"
    return "flat"


def direction_for(change_pct: float, horizon: str) -> str:
    """按**该周期自己的**阈值判方向（对账 / 归因的推荐入口）。"""
    return direction_of(change_pct, flat_threshold_for(horizon))


def argmax_direction(probs: dict) -> str:
    """概率分布 → 方向（取最大者）。**全项目唯一实现。**

    原先 evaluator / schema / attributor 各写一份 `max(...)`，平手时的结果
    取决于字典字面顺序（都落到 "up"）——三路均等的"没判断"会被记成看涨，
    长期累积出系统性多头偏差。这里钉死规则：**平手取 flat**（视为无观点）。
    """
    if not probs:
        return ""
    vals = {k: float(probs.get(k, 0) or 0) for k in PROB_KEYS}
    top = max(vals.values())
    return next(k for k in ("flat", "up", "down") if vals[k] >= top - 1e-12)


def labels_table() -> dict:
    """各周期的步数 / 阈值一览（诊断与 UI 展示用）。"""
    return {h: {"step": HORIZON_STEPS[h], "flat_pct": flat_threshold_for(h)}
            for h in HORIZONS}


# ---------------------------------------------------------------------------
# 到期日：base_date 之后第 N 个交易日
# ---------------------------------------------------------------------------

# 向前取日历的窗口（自然日）。取**固定值**而不是"按需精确"是为了让缓存键稳定：
# price.trading_days_ex 的落盘缓存键含 [start, end]，若 end 随调用浮动，
# 同一个 base_date 会散出多个键，缓存命中率直接归零（每次都要联网取指数 K 线）。
# 一季 = 60 个交易日 ≈ 90 自然日，200 天留足节假日冗余。
_DUE_WINDOW_DAYS = 200


def _coerce_date(v):
    """把 date/datetime/ISO 字符串统一成 datetime.date；不合法返回 None。"""
    from datetime import date as _date
    from datetime import datetime as _datetime
    if v is None:
        return None
    # datetime 是 date 的子类，必须先判 datetime，否则拿到的是 date 而不是 .date()
    if isinstance(v, _datetime):
        return v.date()
    if isinstance(v, _date):
        return v
    try:
        return _datetime.fromisoformat(str(v)[:10]).date()
    except Exception:
        return None


def due_dates_for(base_date, horizons=None) -> tuple[dict, str]:
    """各周期的**到期日** = base_date 之后第 N 个**交易日**（N = HORIZON_STEPS）。

    返回 `({"next_day": "2026-09-25", "one_week": "2026-10-09", ...}, source)`。

    source 取值（与 `data.price.trading_days_ex` 同义，是**口径留痕**，不要丢掉）：
      - `"index"`   ：日期来自沪深300 日 K 的真实交易日（含节假日，严格口径）
      - `"weekday"` ：指数日历不可达 / 尚未覆盖到那么远，降级为"排除周末"
                      （**不排节假日，到期日可能比真实交割日早 1~几天**）
      - `"empty"`   ：日历完全不可用，返回空 dict

    调用方拿到 `"empty"` 时应**放弃该周期**，而不是退回自然日折算：
    按错的到期日判出来的"真实方向"会污染 CalibSample，且从任何指标上都看不出来。

    关于 `"weekday"`：预测的 base_date 很近（今天/明天）时，未来那几天
    根本还没有指数 K 线，"严格日历"**原理上**取不到，此时排除周末就是唯一诚实的
    最好努力。所以这不是"降级得有罪"，而是"必须把口径标出来"。
    """
    from datetime import timedelta as _timedelta

    hs = tuple(horizons or HORIZONS)
    bd = _coerce_date(base_date)
    if bd is None:
        return {}, "empty"
    try:
        from core.data import price as price_data
    except Exception:
        return {}, "empty"

    try:
        days, source = price_data.trading_days_ex(bd, bd + _timedelta(days=_DUE_WINDOW_DAYS))
    except Exception:
        return {}, "empty"
    if not days:
        return {}, "empty"

    # base_date 本身是交易日 → 从它往后数 N 个；
    # 否则（周末/节假日发的研判）→ 从它之后第一个交易日算起，取第 N 个。
    pos = next((i for i, d in enumerate(days) if d >= bd), None)
    if pos is None:
        return {}, "empty"
    same_day = days[pos] == bd

    need = max((HORIZON_STEPS.get(h, 0) for h in hs), default=0) + 1
    if len(days) - pos < need:
        # 指数日历还没覆盖到那么远（base_date 在近未来时必然如此）：
        # 后半段用"排除周末"补足，并把口径降级标成 weekday（不排节假日，可能偏早）。
        # 注意：只补内存里的副本，不把补出来的日期写回日历缓存 ——
        # 缓存一旦被"排除周末"的结果污染，将来真实日历到了也读不到。
        cur = days[-1]
        while len(days) < pos + need:
            cur = cur + _timedelta(days=1)
            if cur.weekday() < 5:
                days.append(cur)
        source = "weekday"

    out: dict[str, str] = {}
    for h in hs:
        step = HORIZON_STEPS.get(h)
        if not step:
            continue
        idx = (pos + step) if same_day else (pos + step - 1)
        if 0 <= idx < len(days):
            out[h] = str(days[idx])
    return out, source
