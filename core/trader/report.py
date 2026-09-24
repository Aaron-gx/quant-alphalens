"""V 层：三层评估（单笔 / 组合 / 记忆有效性）。

设计文档：docs/AI交易员Harness-架构设计.md §4.6 / §6.6

三层各回答一个问题：
  1. trade_review        —— 这笔交易对了吗？（预测命中、归因分布）
  2. portfolio_review    —— 这个月赚了吗？比基准强吗？（收益/回撤/夏普/成本）
  3. memory_effectiveness—— **学的东西真的有用吗？**（lesson 生效前后判断质量差）

第三层是本设计的关键。没有它，"不断训练"就是黑箱——
AI 可能越学越糟而无人发现。它的逻辑是：
以 lesson.activated_at 为分界，比较该 lesson 作用域内前后的判断命中率；
若生效后没有改善甚至变差 → 判定 harmful → 自动给该 lesson 记反证（记忆自我纠正）。

**独立读路径**（对照 patterns/cqrs）：
本模块只读 DB 落库数据，不 import allocator/loop 的任何决策函数，
禁止"重算一遍"式评估——保证评估不被执行细节污染。
"""
from __future__ import annotations

from datetime import date, timedelta

from core.config import load_config
from core.store.db import session_scope
from core.store.models import (
    PaperOrder, Prediction, PredictionEval, TraderMemory, TraderRun,
)
from core.trader import attributor as attr_mod
from core.trader import memory as memory_mod


def _benchmark() -> str:
    return str((load_config().get("trader") or {}).get("benchmark_symbol", "沪深300"))


# ---------------------------------------------------------------------------
# 第 1 层：单笔
# ---------------------------------------------------------------------------

def trade_review(trader_id: int, days: int = 30) -> dict:
    """单笔层：预测命中率 + 归因分布 + Brier。

    归因分布来自 TraderRun.attributions（每轮归因结论的留痕）。
    """
    cutoff = date.today() - timedelta(days=int(days))
    runs: list[dict] = []
    with session_scope() as s:
        rows = (s.query(TraderRun)
                .filter(TraderRun.trader_id == trader_id,
                        TraderRun.run_date >= cutoff)
                .order_by(TraderRun.run_date).all())
        for r in rows:
            runs.append({"run_date": str(r.run_date), "status": r.status,
                         "attributions": list(r.attributions or []),
                         "orders": list(r.orders or []),
                         "rejected": sum(1 for g in (r.guard_results or [])
                                         if not g.get("allowed"))})
        symbols = sorted({str(o.get("symbol")) for r in rows
                          for o in (r.orders or []) if o.get("symbol")})
        evals = []
        preds = s.query(Prediction).all()
        pmap = {p.id: p for p in preds}
        for e in s.query(PredictionEval).all():
            p = pmap.get(e.prediction_id)
            if not p:
                continue
            if symbols and p.symbol not in symbols:
                continue
            evals.append({"horizon": e.horizon, "hit": bool(e.hit),
                          "brier": float(e.brier_score or 0),
                          "predicted": e.predicted_direction,
                          "actual": e.actual_direction})

    stats: dict = {"total": 0}
    for k in (attr_mod.CLS_NOISE, attr_mod.CLS_BETA,
              attr_mod.CLS_FORECAST_ERR, attr_mod.CLS_DECISION_ERR,
              "as_expected"):
        stats[k] = 0
    for r in runs:
        for a in r["attributions"]:
            cls = a.get("cls")
            if cls:
                stats["total"] = stats.get("total", 0) + 1
                stats[cls] = stats.get(cls, 0) + 1

    n_eval = len(evals)
    return {
        "days": int(days),
        "runs": len(runs),
        "orders": sum(len(r["orders"]) for r in runs),
        "guard_rejections": sum(r["rejected"] for r in runs),
        "attribution": stats,
        "prediction": {
            "n": n_eval,
            "hit_rate": round(sum(1 for e in evals if e["hit"]) / n_eval, 4)
            if n_eval else None,
            "avg_brier": round(sum(e["brier"] for e in evals) / n_eval, 4)
            if n_eval else None,
            "by_horizon": _group_rate(evals, "horizon"),
            "by_actual_direction": _group_rate(evals, "actual"),
        },
        "learnable_signal": {
            "forecast_error": stats[attr_mod.CLS_FORECAST_ERR],
            "decision_error": stats[attr_mod.CLS_DECISION_ERR],
            "filtered_out": stats[attr_mod.CLS_NOISE] + stats[attr_mod.CLS_BETA],
        },
    }


def _group_rate(rows: list[dict], key: str) -> dict:
    g: dict = {}
    for r in rows:
        k = str(r.get(key) or "")
        b = g.setdefault(k, {"n": 0, "hit": 0})
        b["n"] += 1
        b["hit"] += int(bool(r.get("hit")))
    return {k: {"n": v["n"], "hit_rate": round(v["hit"] / v["n"], 4)}
            for k, v in sorted(g.items()) if v["n"]}


# ---------------------------------------------------------------------------
# 第 2 层：组合
# ---------------------------------------------------------------------------

def portfolio_review(trader_id: int, benchmark: str | None = None) -> dict:
    """组合层：收益/超额/回撤/夏普/换手/费用占比 + 相对基准。"""
    from core.paper import engine
    try:
        perf = engine.perf_stats(trader_id)
        pf = engine.portfolio(trader_id)
        curve = engine.nav_curve(trader_id)
    except Exception as e:
        return {"error": str(e)}

    # 交易成本占比：累计手续费 / 期初资金
    fee_total = 0.0
    turnover = 0.0
    initial = 0.0
    with session_scope() as s:
        from core.store.models import PaperAccount
        acc = s.get(PaperAccount, trader_id)
        initial = float(acc.initial_cash) if acc else 0.0
        for o in s.query(PaperOrder).filter_by(account_id=trader_id):
            fee_total += float(o.fee or 0)
            turnover += float(o.amount or 0)

    bench = benchmark or _benchmark()
    bench_ret, bench_note = _benchmark_return(bench, len(curve))
    total_ret = perf.get("total_return")
    excess = None
    if total_ret is not None and bench_ret is not None:
        excess = round(float(total_ret) - float(bench_ret), 2)

    return {
        "initial_cash": round(initial, 2),
        "total_value": pf.get("total_value"),
        "metrics": perf,
        "cost": {
            "fee_total": round(fee_total, 2),
            "fee_ratio_pct": round(fee_total / initial * 100, 4) if initial else None,
            "turnover": round(turnover, 2),
            "turnover_pct": round(turnover / initial * 100, 2) if initial else None,
        },
        "benchmark": {"symbol": bench, "return_pct": bench_ret, "note": bench_note},
        "excess_return_pct": excess,
        "nav_curve": curve[-60:],
    }


def _benchmark_return(symbol: str, days: int) -> tuple[float | None, str]:
    """基准区间收益（%）。取不到时如实返回 None 与原因。"""
    if days <= 1:
        return None, "净值天数不足，无法计算基准"
    try:
        from core.data import market as market_data
        if symbol in market_data.INDEX_CODE_MAP:
            df = market_data.fetch_index_kline(symbol, count=max(days + 5, 30))
        else:
            from core.data import stock as stock_data
            df = stock_data.fetch_kline(symbol, count=max(days + 5, 30))
        if df is None or df.empty or len(df) < 2:
            return None, "基准数据不足"
        closes = [float(x) for x in df["close"].tolist()]
        n = min(days, len(closes) - 1)
        if n <= 0:
            return None, "基准数据不足"
        return round((closes[-1] / closes[-1 - n] - 1) * 100, 2), "已获取"
    except Exception as e:
        return None, f"基准获取失败：{e}"


# ---------------------------------------------------------------------------
# 第 3 层：记忆有效性（本设计的关键）
# ---------------------------------------------------------------------------

def memory_effectiveness(
    trader_id: int, *, auto_refute: bool = True,
    min_samples_per_side: int = 3, delta_threshold: float = 0.05,
) -> dict:
    """每条 active lesson 生效前后的判断质量差 → 判定有用/有害。

    方法：
      1. 对每条 active lesson，取 activated_at 为分界
      2. 取该 lesson 作用域内（symbol:X 或 global）的所有 PredictionEval
      3. 比较分界前后的命中率（hit_rate）
      4. 判定：
         - 任一侧样本 < min_samples_per_side → insufficient（无法判断）
         - after - before > delta_threshold      → helpful
         - after - before < -delta_threshold     → harmful（建议反驳）
         - 其余                                   → neutral

    auto_refute=True 时对 harmful 的 lesson 自动记一条反证（记忆自我纠正）。
    返回：{"lessons": [...], "summary": {...}}
    """
    with session_scope() as s:
        lessons = memory_mod.recall(trader_id, kind=memory_mod.KIND_LESSON,
                                    status=memory_mod.ST_ACTIVE, limit=500)
        if not lessons:
            return {"lessons": [], "summary": {"total": 0}}
        preds = s.query(Prediction).all()
        pmap = {p.id: p for p in preds}
        evals = []
        for e in s.query(PredictionEval).all():
            p = pmap.get(e.prediction_id)
            if not p:
                continue
            evals.append({"symbol": p.symbol, "hit": bool(e.hit),
                          "at": e.evaluated_at})

    out: list[dict] = []
    n_help = n_harm = n_insuff = n_neutral = 0
    for les in lessons:
        sc = str(les.get("scope") or "global")
        act = les.get("activated_at")
        if not act:
            out.append({"id": les["id"], "statement": les["statement"],
                        "verdict": "insufficient",
                        "note": "该 lesson 无生效时点，无法评估"})
            n_insuff += 1
            continue
        try:
            from datetime import datetime as _dt
            act_dt = _dt.fromisoformat(str(act)[:19])
        except Exception:
            act_dt = None

        pool = [e for e in evals
                if (sc.startswith("symbol:") and e["symbol"] == sc.split(":", 1)[1])
                or sc == "global"]
        before = [e for e in pool if act_dt and e["at"] and e["at"] < act_dt]
        after = [e for e in pool if act_dt and e["at"] and e["at"] >= act_dt]

        if (len(before) < min_samples_per_side
                or len(after) < min_samples_per_side):
            verdict, note = "insufficient", (
                f"样本不足（生效前 {len(before)} / 生效后 {len(after)}，"
                f"各需 ≥{min_samples_per_side}），暂无法判断该教训是否有效")
            n_insuff += 1
            br = ar = None
            delta = None
        else:
            br = sum(1 for e in before if e["hit"]) / len(before)
            ar = sum(1 for e in after if e["hit"]) / len(after)
            delta = ar - br
            if delta > delta_threshold:
                verdict = "helpful"
                note = f"生效后命中率提升 {delta:+.1%}，教训有效"
                n_help += 1
            elif delta < -delta_threshold:
                verdict = "harmful"
                note = f"生效后命中率下降 {delta:+.1%}，该教训可能有害"
                n_harm += 1
            else:
                verdict = "neutral"
                note = f"命中率变化 {delta:+.1%}，未达到显著改善"
                n_neutral += 1

        entry = {
            "id": les["id"], "statement": les["statement"], "scope": sc,
            "confidence": les.get("confidence"),
            "before_n": len(before), "after_n": len(after),
            "before_hit_rate": round(br, 4) if br is not None else None,
            "after_hit_rate": round(ar, 4) if ar is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "verdict": verdict, "note": note,
        }
        if verdict == "harmful" and auto_refute:
            try:
                r = memory_mod.refute_lesson(
                    trader_id, les["id"],
                    {"reason": note, "before_hit_rate": br, "after_hit_rate": ar,
                     "before_n": len(before), "after_n": len(after)},
                    auto=True, mirror=False)
                entry["refuted"] = True
                entry["new_status"] = r.get("new_status")
            except Exception as e:
                entry["refuted"] = False
                entry["refute_error"] = str(e)
        out.append(entry)

    try:
        memory_mod.render_markdown(trader_id)
    except Exception:
        pass
    return {
        "lessons": out,
        "summary": {"total": len(lessons), "helpful": n_help, "harmful": n_harm,
                    "neutral": n_neutral, "insufficient": n_insuff},
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def overview(trader_id: int, days: int = 30) -> dict:
    """交易员全景报告（供 API/前端一键查看）。"""
    mem = memory_mod.view(trader_id, limit_lessons=50, limit_mistakes=20)
    with session_scope() as s:
        runs = (s.query(TraderRun)
                .filter_by(trader_id=trader_id)
                .order_by(TraderRun.run_date.desc()).limit(60).all())
        run_rows = [{"run_date": str(r.run_date), "status": r.status,
                     "orders": len(r.orders or []), "nav": r.nav,
                     "new_facts": r.new_facts,
                     "lessons": len(r.new_lessons or []),
                     "promotions": len(r.memory_promotions or []),
                     "rejected": sum(1 for g in (r.guard_results or [])
                                     if not g.get("allowed")),
                     "turns": r.turns, "cost": r.llm_cost_est,
                     "error": r.error} for r in runs]
        mem_log = (s.query(TraderMemory)
                   .filter_by(trader_id=trader_id)
                   .order_by(TraderMemory.id.desc()).all())
        by_status: dict = {}
        for m in mem_log:
            key = f"{m.kind}:{m.status}"
            by_status[key] = by_status.get(key, 0) + 1

    return {
        "trader_id": trader_id,
        "runs": run_rows,
        "run_stats": {
            "total": len(run_rows),
            "ok": sum(1 for r in run_rows if r["status"] == "ok"),
            "watched": sum(1 for r in run_rows if r["status"] == "watched"),
            "errors": sum(1 for r in run_rows if r["status"] == "error"),
            "total_cost": round(sum(r["cost"] or 0 for r in run_rows), 4),
        },
        "memory": {"stats": mem.get("stats"), "by_kind_status": by_status,
                   "active_lessons": mem.get("lessons"),
                   "recent_facts": mem.get("facts")},
        "trade": trade_review(trader_id, days=days),
        "portfolio": portfolio_review(trader_id),
        "memory_effectiveness": memory_effectiveness(trader_id, auto_refute=False),
    }
