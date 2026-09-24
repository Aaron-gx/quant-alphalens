"""预测验证：到期自动对账 + 准确率统计 + 自学习样本回填。

对每个预测 × 每个 horizon，到期后取实际收盘价计算方向，判定命中与否、
算 Brier score（概率校准误差，越小越好），并把这次"引擎概率 → 真实方向"
写成一条 CalibSample —— 校准层就是靠这些样本一轮轮把自己练准的。
"""
from __future__ import annotations

from datetime import date

from core.data import stock as stock_data
from core.data import fund as fund_data
from core.data import market as market_data
# 对账口径必须与量化训练标签**严格同源**，否则写进 CalibSample 的"真实方向"
# 和模型学的"方向"是两个不同问题（详见 core/predict/labels.py 的说明）
from core.predict.labels import (HORIZONS, argmax_direction, direction_of,
                                 due_dates_for, flat_threshold_for)
from core.store.db import session_scope
from core.store.models import Prediction, PredictionEval


def _actual_price(pred: Prediction) -> float | None:
    """取标的最新价（K线最后一根收盘）。**仅作时点取价失败时的实盘兜底**。

    ⚠️ 这是"评估时刻"的价，**不是"到期日"的价**，所以**不再是主取价路径**：
    对账一律先走 `_price_for` → `_actual_price_on`（按 due 取时点价），
    只有时点历史不可得且非回放模式才退化到这里（见 `_price_for` 的说明）。
    它同时是离线测试的注入点（tests/test_smoke.py 会替换本函数）。
    """
    try:
        if pred.asset_type == "index":
            df = market_data.fetch_index_kline(pred.symbol, count=5)
        elif pred.asset_type == "ofund":
            df = fund_data.fetch_fund_nav_history(pred.symbol, count=5)
        else:
            df = stock_data.fetch_kline(pred.symbol, count=5)
        if df is not None and not df.empty:
            return float(df.iloc[-1]["close"])
    except Exception:
        return None
    return None


def _actual_price_on(pred: Prediction, on_date: date) -> float | None:
    """**时点口径**：取 on_date（含）之前最近一个交易日的收盘价。

    回放 / 补账用。这样每个 horizon 都能用它**自己的到期日**去对账——
    next_day 与 quarter 不会共用同一个价。
    """
    try:
        from core.data import price as price_data
        return price_data.price_on(pred.symbol, pred.asset_type, on_date)
    except Exception:
        return None


def _price_for(pred: Prediction, due: date, memo: dict, *,
               strict: bool = False) -> float | None:
    """对账取价：**优先按到期日取价**（时点口径），失败才退化为最新价。

    为什么不再让实盘口径"共用一份最新价"
    ------------------------------------
    旧实盘口径给同一标的的四个 horizon 共用一个"评估时刻最新价"。
    只要评估日 ≠ 到期日（停机补账、手动触发、周末顺延、跑批跨天），
    短周期就会被长周期的价污染：next_day 的"实际涨跌"实际是三个月涨幅，
    而这条脏样本会**直接写进 CalibSample 教校准层**，从任何指标上都看不出来。
    改为逐 horizon 按 due 取价后，四周期各自独立，互不干扰。

    `memo` 按 (symbol, asset_type, due) 在同一批次内去重——同一标的的多条预测
    到期日相同，避免重复拉取行情。

    strict=True（回放/补账）：时点价不可得就**放弃该 horizon**，绝不用最新价兜底
        （回放里"最新价"与 due 相差可能几个月，兜底等于伪造标签）。
    strict=False（实盘）：时点价不可得时退化用最新价——调度器每日跑，
        due ≈ 今天，误差可忽略；有价总比丢掉对账好。
        ⚠️ 这里保留 `_actual_price` 作为兜底入口，同时也是离线测试的注入点。
    """
    key = (pred.symbol, pred.asset_type, due)
    if key in memo:
        return memo[key]
    px = _actual_price_on(pred, due)
    if px is None and not strict:
        px = _actual_price(pred)
    memo[key] = px
    return px


def _brier(probs: dict, actual: str) -> float:
    """三类 Brier score：sum((p_i - o_i)^2)，0 最好，2 最差。"""
    return round(sum((float(probs.get(k, 0)) - (1.0 if k == actual else 0.0)) ** 2
                     for k in ("up", "flat", "down")), 4)


def evaluate_due(today: date | None = None, *, replay: bool = False) -> int:
    """评估所有到期的预测。返回本次新评估条数。

    **实盘与回放现在都按"各自的到期日"取价**（见 _price_for）：
    这是本次修正的核心——旧实盘口径给同一标的的四个 horizon 共用一个
    "评估时刻最新价"，只要评估日 ≠ 到期日（停机补账 / 手动触发 / 周末顺延
    / 跑批跨天），next_day 的"实际涨跌"就会变成 three-month 涨幅，
    并被写进 CalibSample 去教校准层，从任何指标上都看不出来。

    replay 的含义因此收敛为"严格程度"：
      replay=False（实盘）：时点价取不到时**退化用最新价**（调度器每日跑，
                due ≈ 今天，误差可忽略；有价总比丢掉对账好）。
      replay=True （回放/补账）：时点价取不到就**放弃该 horizon**，绝不兜底
                （回放里"最新价"与 due 可能差几个月，兜底等于伪造标签）。

    ⚠️ 平盘阈值按周期取（`flat_threshold_for`），与量化引擎的训练标签同源。
    原先四个周期共用一个阈值，等于用"次日的问题"去给别人打分。

    ⚠️ **到期日按交易日数**（`due_dates_for`），不再按自然日折算。
    模型的 horizon 本来就是"T+N 个交易日"（HORIZON_STEPS），
    对账却用 HORIZON_DAYS 的自然日近似（20 交易日 ≈ 30 自然日这种），
    每个周期都会漂 1~几天 —— 等于拿**另一天**的收盘价判"真实方向"。
    这个偏差不会让任何指标变红，只会让校准层学到错的东西。
    """
    today = today or date.today()
    done = 0
    pending: list[dict] = []      # 自学习样本先攒着，等对账事务提交后再写
    memo: dict[tuple, float | None] = {}   # (symbol, asset_type, due) → 价，批次内去重
    # base_date → {horizon: "YYYY-MM-DD"}。一个批次里多个预测常共享 base_date，
    # 日历只算一次（trading_days_ex 有落盘缓存，但没必要每行都走一遍解析）。
    due_memo: dict = {}
    with session_scope() as s:
        preds = s.query(Prediction).all()
        for pred in preds:
            if not pred.base_price:
                continue
            evaluated = {e.horizon for e in
                         s.query(PredictionEval).filter_by(prediction_id=pred.id)}
            due_map = due_memo.get(pred.base_date)
            if due_map is None:
                due_map, _src = due_dates_for(pred.base_date, HORIZONS)
                due_memo[pred.base_date] = due_map
            for horizon in HORIZONS:
                if horizon in evaluated or horizon not in (pred.horizons or {}):
                    continue
                # **到期日按"base_date 之后的第 N 个交易日"算**，N = 训练时的 horizon 步数。
                # 旧实现在这里写 `base_date + timedelta(days=HORIZON_DAYS[horizon])`——
                # 用自然日折算交易日，每个周期都漂 1~几天，等于拿**另一天**的收盘价
                # 去判"真实方向"，与模型学的 horizon 不是同一个问题（脏样本）。
                iso = due_map.get(horizon)
                if not iso:
                    # 日历不可用：**宁可不评**。按错的到期日判出来的方向会被
                    # 写进 CalibSample 教校准层，而且从任何指标上都看不出来。
                    continue
                due = date.fromisoformat(iso)
                if due > today:
                    continue
                # **每个 horizon 用它自己的到期日取价**（见 _price_for）。
                # 取不到价只跳过这一个周期，不牵连同一标的的其他到期周期。
                px = _price_for(pred, due, memo, strict=replay)
                if px is None:
                    continue
                change_pct = (px - pred.base_price) / pred.base_price * 100
                # 每个周期用它**自己的**阈值判定真实方向（与量化引擎的训练标签同源）
                flat_th = flat_threshold_for(horizon)
                actual = direction_of(change_pct, flat_th)
                h = pred.horizons[horizon]
                predicted = argmax_direction(h)
                s.add(PredictionEval(
                    prediction_id=pred.id, horizon=horizon,
                    due_date=due,
                    actual_price=px, actual_change_pct=round(change_pct, 3),
                    actual_direction=actual, predicted_direction=predicted,
                    hit=(predicted == actual), brier_score=_brier(h, actual),
                ))
                pending.append(_sample_payload(pred, horizon, h, actual, change_pct, flat_th))
                done += 1
    for item in pending:
        _write_sample(item)
    return done


def _sample_payload(pred: Prediction, horizon: str, probs: dict, actual: str,
                    change_pct: float, flat_th: float) -> dict:
    """整理一条自学习样本（不落库，由调用方在事务外写入）。"""
    snap = pred.input_snapshot or {}
    signal = snap.get("news_signal") or {}
    return {
        "engine": pred.engine or "llm", "horizon": horizon, "symbol": pred.symbol,
        "asset_type": pred.asset_type, "base_date": pred.base_date,
        "probs": probs, "actual_direction": actual,
        "actual_change_pct": change_pct, "flat_threshold": flat_th,
        "source": "eval", "confidence": pred.confidence or 50,
        "senti_score": float(signal.get("score") or 0.0),
        "universe_senti": float(signal.get("universe_score") or 0.0),
        "relevance_cov": float(signal.get("coverage") or 0.0),
        "prediction_id": pred.id,
        "payload": {"predicted": argmax_direction(probs),
                    "engine": pred.engine, "mode": pred.mode,
                    "calib": snap.get("calibration")},
    }


def _write_sample(item: dict) -> None:
    """把对账结果写成自学习样本（校准与元模型的训练数据来源）。"""
    try:
        from core.predict import calibration
        calibration.record_sample(**item)
    except Exception:
        pass  # 样本回填失败不影响对账本身


def maybe_retrain(min_new: int = 20, force: bool = False) -> dict | None:
    """样本够了就重训校准层（调度器每日对账后调用，也可手动触发）。

    节流：距上次成功训练新增样本 < min_new 就跳过，避免每天白跑重训。
    """
    from core.predict import calibration
    from core.store.models import ModelTrainingRun
    stats = calibration.sample_stats()
    with session_scope() as s:
        last = (s.query(ModelTrainingRun).order_by(ModelTrainingRun.id.desc()).first())
        last_n = last.n_samples if last else 0
    if not force and stats["total"] - last_n < min_new and last_n > 0:
        return {"skipped": True, "reason": f"新增样本不足（{stats['total'] - last_n}<{min_new}）",
                "total": stats["total"]}
    return calibration.train_all(record=True)


def accuracy_report() -> dict:
    """准确率看板数据：总体 + 分 horizon/方向/置信度桶 + 分标的类型。"""
    with session_scope() as s:
        evals = s.query(PredictionEval).all()
        if not evals:
            return {"total": 0}
        total = len(evals)
        hits = sum(1 for e in evals if e.hit)
        pred_map = {p.id: p for p in s.query(Prediction)}
        by_horizon: dict = {}
        for e in evals:
            b = by_horizon.setdefault(e.horizon, {"n": 0, "hit": 0, "brier": []})
            b["n"] += 1
            b["hit"] += int(e.hit)
            b["brier"].append(e.brier_score)
        horizon_stats = {
            h: {"n": b["n"], "hit": b["hit"], "hit_rate": round(b["hit"] / b["n"], 4),
                "avg_brier": round(sum(b["brier"]) / len(b["brier"]), 4)}
            for h, b in by_horizon.items()
        }
        # 按预测置信度分桶
        conf_buckets: dict = {}
        asset_buckets: dict = {}
        engine_buckets: dict = {}
        for e in evals:
            p = pred_map.get(e.prediction_id)
            conf = p.confidence if p else 50
            bucket = f"{int(conf // 20 * 20)}-{int(conf // 20 * 20) + 19}"
            b = conf_buckets.setdefault(bucket, {"n": 0, "hit": 0})
            b["n"] += 1
            b["hit"] += int(e.hit)
            if p:
                a = asset_buckets.setdefault(p.asset_type or "stock", {"n": 0, "hit": 0})
                a["n"] += 1
                a["hit"] += int(e.hit)
                g = engine_buckets.setdefault(p.engine or "llm", {"n": 0, "hit": 0})
                g["n"] += 1
                g["hit"] += int(e.hit)
        conf_stats = {k: {"n": v["n"], "hit_rate": round(v["hit"] / v["n"], 4)}
                      for k, v in sorted(conf_buckets.items())}
        return {
            "total": total,
            "hit_rate": round(hits / total, 4),
            "avg_brier": round(sum(e.brier_score for e in evals) / total, 4),
            "by_horizon": horizon_stats,
            "by_confidence": conf_stats,
            "by_asset": {k: {"n": v["n"], "hit_rate": round(v["hit"] / v["n"], 4)}
                         for k, v in asset_buckets.items()},
            "by_engine": {k: {"n": v["n"], "hit_rate": round(v["hit"] / v["n"], 4)}
                          for k, v in engine_buckets.items()},
        }


def recent_evaluations(limit: int = 20) -> list[dict]:
    """最近的对账流水（真实核销记录，供"履约核销对账流水"表展示）。

    为什么要单独开这个函数
    --------------------
    前端那张表原先读的是一个硬编码的 `MOCK_AUDIT_LOGS` 常量：6 条
    编造的核销单号（EVAL-20260922-01 之类）、编造的实际涨跌幅、编造的
    命中与 Brier，标题却写着「已对账前 6 笔记录」。这条链路上根本没有
    真实数据源 —— 因为 `accuracy_report()` 只返回聚合统计，不含明细。
    这里补上明细查询，让前端有真数据可读。
    """
    from sqlalchemy import desc
    with session_scope() as s:
        rows = (s.query(PredictionEval)
                .order_by(desc(PredictionEval.evaluated_at), desc(PredictionEval.id))
                .limit(max(1, min(int(limit), 200)))
                .all())
        pred_ids = [r.prediction_id for r in rows]
        pred_map = {p.id: p for p in s.query(Prediction).filter(Prediction.id.in_(pred_ids)).all()} if pred_ids else {}
        out: list[dict] = []
        for e in rows:
            p = pred_map.get(e.prediction_id)
            out.append({
                "id": e.id,
                "prediction_id": e.prediction_id,
                "symbol": p.symbol if p else "",
                "name": p.name if p else "",
                "confidence": p.confidence if p else None,
                "horizon": e.horizon,
                "due_date": str(e.due_date),
                "evaluated_at": str(e.evaluated_at)[:19],
                "actual_price": e.actual_price,
                "actual_change_pct": e.actual_change_pct,
                "actual_direction": e.actual_direction,
                "predicted_direction": e.predicted_direction,
                "hit": bool(e.hit),
                "brier_score": e.brier_score,
            })
        return out

