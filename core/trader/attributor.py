"""纠错归因：把「预测 vs 实际」的偏差做四分类，决定**什么值得学**。

设计文档：docs/AI交易员Harness-架构设计.md §6.2

这是本设计里最重要的一个模块。原因：

用户说"几天后发现它跌得跟之前预测的不一样，那么就纠正"。
但**跌了 ≠ 判错了**。如果不做归因，AI 会把三种东西全当成自己的错去学：

  1. 大盘整体下跌拖累（beta）——这是市场，不是 AI 的判断
  2. 随机波动（噪声）——本来就在不可预测范围内
  3. 真正的判断失误——这才是该学的

把 1、2 当成 3 去学，AI 会总结出"跌了就该卖"这种追涨杀跌的伪规律，
并在之后自我强化（买了A亏→"再也不买A"→永远不知道自己错在哪）。

所以归因是**纠错机制的前置闸门**：
只有 forecast_error 与 decision_error 才允许写入记忆（fact）。
"""
from __future__ import annotations

import re
from typing import Any

# 口径同源：归因的"噪声带"必须与预测/对账/量化标签完全一致，
# 否则会出现"对账认定它猜错了、归因却说落在噪声带内不用学"的自相矛盾
from core.predict.labels import argmax_direction, base_threshold, direction_of

# --- 归因分类 ---
CLS_BETA = "market_beta"          # 大盘同向拖累 → 不学
CLS_NOISE = "noise"               # 噪声带内 → 不学
CLS_FORECAST_ERR = "forecast_error"   # 方向判错且非 beta → 学
CLS_DECISION_ERR = "decision_error"   # 方向对但幅度/仓位严重偏离 → 学（策略层）

LEARNABLE = (CLS_FORECAST_ERR, CLS_DECISION_ERR)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def direction_from_change(change_pct: float, flat_threshold: float | None = None) -> str:
    """涨跌幅 → 方向（含平盘带）。

    默认阈值不再写死 1.0（与 [predict] 的 0.5 不同源），
    而是继承 `[predict].flat_threshold_pct`，保证三个环节同一把尺子。
    """
    if flat_threshold is None:
        flat_threshold = base_threshold()
    return direction_of(change_pct, flat_threshold)


def direction_from_probs(probs: dict) -> str:
    """概率分布 → 预测方向（取最大者；平手视为 flat）。

    统一走 core.predict.labels.argmax_direction：原先这里、schema、evaluator
    各写一份 max(...)，平手结果都落到 "up"，等于给"没判断"偷偷记成看涨。
    """
    return argmax_direction(probs or {})


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_range_pct(range_text: str | None, default: float = 0.0) -> float:
    """从 "±1.5%" / "1.5" 这类文本里提取幅度百分比（取绝对值）。

    取不到时返回 default（调用方会退化为 flat_threshold 近似）。
    """
    if not range_text:
        return default
    nums = re.findall(r"\d+(?:\.\d+)?", str(range_text))
    if not nums:
        return default
    try:
        return max(abs(float(n)) for n in nums)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# 单条归因
# ---------------------------------------------------------------------------

def attribute_one(
    prediction: dict,
    actual: dict,
    market: dict | None = None,
    *,
    flat_threshold_pct: float | None = None,
    beta_ratio: float = 0.6,
) -> dict:
    """单条偏差的四分类归因。判定次序互斥、短路。

    prediction 契约：
      {"id": int, "symbol": str, "horizon": str, "probs": {"up","flat","down"},
       "base_price": float, "base_date": str}
    actual 契约：
      {"price": float, "change_pct": float, "due_date": str}
    market 契约：
      {"change_pct": float}   # 同期大盘涨跌幅（None 表示取不到 → 退化为不判 beta）

    flat_threshold_pct 传 None 时继承 [predict].flat_threshold_pct（推荐）。

    返回：
      {"cls": str, "should_learn": bool, "evidence": {...}, "reason": str}
    """
    if flat_threshold_pct is None:
        flat_threshold_pct = base_threshold()
    actual_chg = _num(actual.get("change_pct"))
    mkt_chg = None if market is None else _num(market.get("change_pct"), None) \
        if market.get("change_pct") is not None else None
    pred_probs = prediction.get("probs") or {}
    pred_dir = direction_from_probs(pred_probs)
    actual_dir = direction_from_change(actual_chg, flat_threshold_pct)

    ev: dict[str, Any] = {
        "predicted_direction": pred_dir,
        "actual_direction": actual_dir,
        "actual_change_pct": round(actual_chg, 4),
        "market_change_pct": None if mkt_chg is None else round(mkt_chg, 4),
        "beta_explained_pct": 0.0,
        "flat_threshold": flat_threshold_pct,
        "probs": {k: round(_num(v), 4) for k, v in pred_probs.items()},
        "horizon": prediction.get("horizon", ""),
    }

    # 1) 噪声带内 —— 不学
    if abs(actual_chg) <= flat_threshold_pct:
        return {
            "cls": CLS_NOISE, "should_learn": False, "evidence": ev,
            "reason": f"实际涨跌 {actual_chg:+.2f}% 落在 ±{flat_threshold_pct}% "
                      f"噪声带内，不构成可学信号",
        }

    # 2) 大盘同向且可解释占比达标 —— 不学
    if mkt_chg is not None and abs(mkt_chg) > 1e-9:
        same_dir = (actual_chg > 0) == (mkt_chg > 0)
        if same_dir:
            ratio = min(1.0, abs(mkt_chg) / abs(actual_chg))
            ev["beta_explained_pct"] = round(ratio, 4)
            if ratio >= beta_ratio:
                return {
                    "cls": CLS_BETA, "should_learn": False, "evidence": ev,
                    "reason": f"标的 {actual_chg:+.2f}% 与大盘 {mkt_chg:+.2f}% 同向，"
                              f"其中 {ratio:.0%} 可由市场解释(≥{beta_ratio:.0%})，"
                              f"归因市场而非判断失误",
                }

    # 3) 方向判错且非 beta —— 该学
    if pred_dir and actual_dir and pred_dir != actual_dir:
        return {
            "cls": CLS_FORECAST_ERR, "should_learn": True, "evidence": ev,
            "reason": f"预测 {pred_dir} 而实际 {actual_dir}"
                      f"（{actual_chg:+.2f}%，大盘 "
                      f"{'—' if mkt_chg is None else f'{mkt_chg:+.2f}%'}），"
                      f"方向判断失误，值得学习",
        }

    # 4) 方向对但幅度严重偏离 —— 学策略层
    pred_band = parse_range_pct(
        ((prediction.get("horizons") or {}).get(prediction.get("horizon", ""))
         or {}).get("range"), default=flat_threshold_pct)
    if abs(actual_chg) > max(pred_band * 2, flat_threshold_pct * 2):
        return {
            "cls": CLS_DECISION_ERR, "should_learn": True, "evidence": ev,
            "reason": f"方向判断正确（{actual_dir}），但实际幅度 {actual_chg:+.2f}% "
                      f"远超预期区间 ±{pred_band}%，说明幅度/仓位估计偏差，"
                      f"属策略层可学点",
        }

    # 5) 方向对、幅度也在预期内 —— 判断有效，不产生"教训"
    return {
        "cls": "as_expected", "should_learn": False, "evidence": ev,
        "reason": f"预测 {pred_dir} 与实际 {actual_dir} 一致且幅度在预期内，判断有效",
    }


# ---------------------------------------------------------------------------
# 批量归因
# ---------------------------------------------------------------------------

def attribute_batch(
    reconciliations: list[dict],
    market: dict | None = None,
    *,
    flat_threshold_pct: float | None = None,
    beta_ratio: float = 0.6,
) -> dict:
    """批量归因，产出本轮应写入的 fact 清单 + 统计。

    reconciliations 每项契约：
      {"prediction": {见 attribute_one}, "actual": {见 attribute_one}}

    返回：
      {"items": [{"prediction_id","symbol","cls","should_learn","reason","evidence"}],
       "stats": {"total","noise","market_beta","forecast_error",
                 "decision_error","as_expected"},
       "learnable": int,
       "facts": [{"statement","symbol","scope","detail"}]}   # 可直接喂 memory.write_fact
    """
    stats = {"total": 0, "noise": 0, CLS_BETA: 0, CLS_FORECAST_ERR: 0,
             CLS_DECISION_ERR: 0, "as_expected": 0}
    items: list[dict] = []
    facts: list[dict] = []

    for r in reconciliations or []:
        pred = dict(r.get("prediction") or {})
        actual = dict(r.get("actual") or {})
        if not pred or not actual:
            continue
        mk = r.get("market", market)
        att = attribute_one(pred, actual, mk,
                            flat_threshold_pct=flat_threshold_pct,
                            beta_ratio=beta_ratio)
        cls = att["cls"]
        stats["total"] += 1
        stats[cls] = stats.get(cls, 0) + 1
        sym = str(pred.get("symbol") or "")
        horizon = str(pred.get("horizon") or "")
        items.append({
            "prediction_id": pred.get("id"), "symbol": sym,
            "horizon": horizon, "cls": cls,
            "should_learn": att["should_learn"],
            "reason": att["reason"], "evidence": att["evidence"],
        })
        if att["should_learn"]:
            a_chg = att["evidence"]["actual_change_pct"]
            facts.append({
                "statement": (
                    f"[{horizon}] 对 {sym} 的预测为"
                    f"{att['evidence']['predicted_direction']}，实际"
                    f"{a_chg:+.2f}%（{att['evidence']['actual_direction']}）："
                    f"{att['reason']}"
                ),
                "symbol": sym,
                "scope": f"symbol:{sym}" if sym else "global",
                "detail": {**att["evidence"], "cls": cls,
                           "prediction_id": pred.get("id")},
            })

    return {
        "items": items, "stats": stats,
        "learnable": len(facts), "facts": facts,
    }


def summarize_stats(stats: dict) -> str:
    """把统计转成一句人读摘要（供报告与上下文用）。"""
    total = int(stats.get("total", 0) or 0)
    if not total:
        return "本轮无到期对账"
    noise = int(stats.get(CLS_NOISE, 0) or 0)
    beta = int(stats.get(CLS_BETA, 0) or 0)
    fc = int(stats.get(CLS_FORECAST_ERR, 0) or 0)
    dc = int(stats.get(CLS_DECISION_ERR, 0) or 0)
    ne = int(stats.get("as_expected", 0) or 0)
    return (f"到期 {total} 条：噪声 {noise}、市场归因 {beta}、"
            f"判断失误 {fc}、幅度偏差 {dc}、判断有效 {ne}")
