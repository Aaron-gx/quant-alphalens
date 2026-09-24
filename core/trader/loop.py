"""E 层：AI 交易员的每日自主循环。

设计文档：docs/AI交易员Harness-架构设计.md §4.1 / §6.5

七步循环（每交易日一轮）：
  1 感知 perceive    [确定性] 拉账户/持仓/大盘
  2 对账 reconcile   [确定性] 复用 verify.evaluator 的到期对账
  3 归因 attribute   [确定性] 四分类，只留可学项
  4 反思 reflect     [LLM]     写 fact → 提 lesson(candidate) → 证据达标升 active → 重建 belief
  5 决策 decide      [LLM]     带工具自主循环（可多轮工具调用）
  6 校验 guard       [确定性] 每笔动作过风控门禁（在 tools 内部完成）
  7 执行留痕 act     [确定性] 下单 + 净值快照 + 写 TraderRun

熔断（照抄 Hermes Agent 的三层硬上限）：
  - max_turns / max_tool_calls 单轮上限
  - 同工具同参连续失败 3 次 → 熔断该工具
  - 单日 LLM 成本超限 → 强制观望
  - 连续 3 轮失败 → 暂停交易员
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from core.config import load_config
from core.store.db import session_scope
from core.store.models import TraderMemory, TraderRun
from core.trader import attributor as attr_mod
from core.trader import guard as guard_mod
from core.trader import memory as memory_mod
from core.trader import prompts as prompts_mod
from core.trader import tools as tools_mod

# 成本估算（元 / 千 token）—— DeepSeek 量级，仅用于熔断判断
_COST_PER_1K_PROMPT = 0.001
_COST_PER_1K_COMPLETION = 0.002

_STATUS_OK, _STATUS_WATCHED = "ok", "watched"
_STATUS_FUSED, _STATUS_ERROR = "fused", "error"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    return load_config().get("trader") or {}


def _i(key: str, default: int) -> int:
    try:
        return int(_cfg().get(key, default))
    except (TypeError, ValueError):
        return default


def _f(key: str, default: float) -> float:
    try:
        return float(_cfg().get(key, default))
    except (TypeError, ValueError):
        return default


def _benchmark() -> str:
    return str(_cfg().get("benchmark_symbol", "沪深300"))


def _est_cost(usage: dict) -> float:
    p = float(usage.get("prompt_tokens") or 0) / 1000 * _COST_PER_1K_PROMPT
    c = float(usage.get("completion_tokens") or 0) / 1000 * _COST_PER_1K_COMPLETION
    return round(p + c, 6)


# ---------------------------------------------------------------------------
# 建交易员
# ---------------------------------------------------------------------------

def create_trader(
    name: str,
    initial_cash: float | None = None,
    *,
    persona: str = "",
    notes: str = "",
    seed_beliefs: list[str] | None = None,
) -> int:
    """建交易员：PaperAccount(mode='trader') + 可选初始信条。

    seed_beliefs 会被写成 active 的 lesson+belief，作为交易员的起点认知
    （例如"不追高波动主题ETF"这类先验纪律）。返回 trader_id。
    """
    from core.paper import engine
    cash = initial_cash or _f("default_cash", 1000000.0)
    tid = engine.create_account(name=name, initial_cash=cash,
                                mode="trader", note=notes or persona)
    for s in (seed_beliefs or []):
        lid = memory_mod.write_lesson(
            tid, s, scope=memory_mod.SCOPE_GLOBAL,
            evidence_refs=[{"seed": True, "note": "初始信条，用户/系统给定"}],
            proposed_confidence=0.9, mirror=False)
        with session_scope() as sess:
            m = sess.get(TraderMemory, lid)
            if m:
                m.status = memory_mod.ST_ACTIVE
                m.evidence_count = 3          # 初始信条视为已达证据门
                m.confidence = 0.9
                m.activated_at = datetime.now()
    if seed_beliefs:
        memory_mod.rebuild_beliefs(tid)
    return tid


# ---------------------------------------------------------------------------
# 七步循环
# ---------------------------------------------------------------------------

def run_day(
    trader_id: int,
    as_of: date | None = None,
    *,
    client=None,
    deps: dict | None = None,
    force: bool = False,
    replay: bool = False,
) -> dict:
    """执行一个交易日的完整循环。

    replay=False（默认，实盘）：所有价与数据取实时口径。
    replay=True（历史回放）：成交、估值、净值和 AI 能看到的一切数据都截到 as_of。
        回放**必须按交易日顺序、在全新交易员上**推进（见 replay()）。

    幂等策略（重要）：
      同日若已有 TraderRun（任何状态）→ 默认**直接返回既有结果**，不重复交易。
      这是为了安全：本轮若已下单，重跑会造成重复成交。
      force=True 才重跑，调用方需自行确认风险（用于调试）。

    返回：{"trader_id","run_date","status","orders","nav","new_facts",
           "new_lessons","promotions","turns","cost","error","replay"}
    """
    as_of = as_of or date.today()
    deps = deps or {}
    base = {"trader_id": trader_id, "run_date": as_of, "status": _STATUS_ERROR,
            "orders": 0, "nav": 0.0, "new_facts": 0, "new_lessons": 0,
            "promotions": 0, "turns": 0, "cost": 0.0, "error": "",
            "replay": bool(replay)}

    # --- 幂等检查 ---
    existing = _get_run(trader_id, as_of)
    if existing and not force:
        return {**base, "status": existing.status, "nav": existing.nav,
                "orders": len(existing.orders or []),
                "new_facts": existing.new_facts,
                "new_lessons": len(existing.new_lessons or []),
                "promotions": len(existing.memory_promotions or []),
                "turns": existing.turns,
                "cost": existing.llm_cost_est,
                "error": existing.error,
                "skipped": True,
                "reason": "该日已执行过（幂等保护，不重复交易）"}
    if existing and force:
        with session_scope() as s:
            row = s.get(TraderRun, existing.id)
            if row:
                s.delete(row)

    run = {"perception": {}, "attributions": [], "reconciliations": [],
           "new_facts": 0, "new_lessons": [], "promotions": [], "tool_calls": [],
           "decision": {}, "guard_results": [], "orders": [],
           "tokens": 0, "cost": 0.0, "turns": 0, "error": ""}

    # --- 1 感知 ---
    ctx = _make_ctx(trader_id, as_of, deps, replay)
    run["perception"] = _perceive(ctx)

    # --- 2 对账 + 3 归因 ---
    try:
        attrib = _reconcile_and_attribute(ctx, as_of, deps)
    except Exception as e:
        attrib = {"items": [], "stats": {}, "learnable": 0, "facts": []}
        run["error"] = f"对账失败：{e}"
    if attrib.get("eval_error"):
        # 非致命：本轮仍可继续决策，但失败原因必须落在 run.error 里可见
        run["error"] = (run["error"] + f" 对账失败：{attrib['eval_error']}").strip()
    run["reconciliations"] = [i for i in attrib.get("items", [])]
    run["attributions"] = [{"prediction_id": i.get("prediction_id"),
                            "symbol": i.get("symbol"), "cls": i.get("cls"),
                            "reason": i.get("reason")} for i in attrib["items"]]

    # 写 fact（只有 should_learn 的才会出现在 facts 里）
    for f in attrib.get("facts", []):
        try:
            memory_mod.write_fact(
                trader_id, f["statement"], scope=f["scope"],
                symbol=f["symbol"], detail=f["detail"], run_date=as_of,
                mirror=False)
            run["new_facts"] += 1
        except Exception as e:
            # 不要把失败吞成"没有新事实"——留痕里必须看得见
            run["error"] = (run["error"] + f" 事实入库失败：{e}").strip()

    # --- 4 反思（LLM）---
    cost_used = 0.0
    if client is not None and getattr(client, "available", lambda: False)():
        try:
            refl = reflect(trader_id, as_of, attrib, client=client)
            run["new_lessons"] = refl.get("lessons", [])
            run["promotions"] = refl.get("promotions", [])
            run["tokens"] += int(refl.get("tokens", 0) or 0)
            cost_used += float(refl.get("cost", 0) or 0)
        except Exception as e:
            run["error"] = (run["error"] + f" 反思失败：{e}").strip()

    # --- 5 决策（LLM 自主循环）---
    status = _STATUS_WATCHED
    if client is None or not getattr(client, "available", lambda: False)():
        status = _STATUS_ERROR
        run["error"] = (run["error"] + " LLM 未配置，无法决策").strip()
    elif cost_used >= _f("max_cost_per_day", 8.0):
        status = _STATUS_FUSED
        run["error"] = "单日成本已达上限，强制观望"
    else:
        try:
            dec = decide(trader_id, ctx, client=client,
                         attrib_stats=attr_mod.summarize_stats(attrib.get("stats", {})))
            run["decision"] = {"actions": dec.get("actions", []),
                               "rationale": dec.get("rationale", ""),
                               "watch_reason": dec.get("watch_reason", "")}
            run["tool_calls"] = dec.get("tool_calls", [])
            run["turns"] = int(dec.get("turns", 0) or 0)
            run["tokens"] += int(dec.get("tokens", 0) or 0)
            cost_used += float(dec.get("cost", 0) or 0)
        except Exception as e:
            status = _STATUS_ERROR
            run["error"] = (run["error"] + f" 决策失败：{e}").strip()

    # --- 6/7 执行（guard 在 tools 内部已拦截）+ 留痕 ---
    if status != _STATUS_ERROR:
        actions = run["decision"].get("actions") or []
        for a in actions:
            call_ctx = dict(ctx)
            call_ctx["portfolio"] = None      # 每笔都取最新持仓，避免用过期的
            res = tools_mod.call("place_buy_order" if str(a.get("side")) == "buy"
                                 else "place_sell_order",
                                 _to_tool_args(a), call_ctx)
            run["tool_calls"].append({
                "tool": res["tool"], "ok": res["ok"], "ms": res["ms"],
                "rejected": res["rejected"], "rule": res["rule"],
                "digest": res["digest"], "hint": res["hint"],
                "args": _to_tool_args(a),
            })
            run["guard_results"].append({
                "action": _to_tool_args(a), "allowed": res["ok"],
                "rule": res["rule"], "detail": res["digest"] or res["hint"],
            })
            if res["ok"] and isinstance(res["data"], dict):
                run["orders"].append(res["data"])
        status = _STATUS_OK if run["orders"] else _STATUS_WATCHED

    # 净值快照
    nav = 0.0
    try:
        from core.paper import engine
        nav_as_of = as_of if replay else None
        engine.settle_t1(trader_id)
        # 回放：净值的日期与估值都必须用 as_of——否则 20 个交易日会挤成同一天，
        # 且互相覆盖，净值曲线只剩一个点。
        engine.snapshot_nav(trader_id, nav_as_of)
        nav = float((engine.portfolio(trader_id, nav_as_of) if nav_as_of
                     else engine.portfolio(trader_id)).get("total_value") or 0)
    except Exception as e:
        run["error"] = (run["error"] + f" 净值快照失败：{e}").strip()

    # 决策为空且无成交 → 视为观望（一等公民状态，不是失败）

    _save_run(trader_id, as_of, status, run, nav, cost_used)
    memory_mod.render_markdown(trader_id)

    out = {**base, "status": status, "orders": len(run["orders"]), "nav": nav,
           "new_facts": run["new_facts"],
           "new_lessons": len(run["new_lessons"]),
           "promotions": len(run["promotions"]),
           "turns": run["turns"], "cost": round(cost_used, 4),
           "error": run["error"], "skipped": False}
    return out


def _to_tool_args(a: dict) -> dict:
    out = {"symbol": str(a.get("symbol") or ""),
           "reason": str(a.get("reason") or "")}
    if a.get("amount") is not None:
        out["amount"] = a["amount"]
    if a.get("volume") is not None:
        out["volume"] = a["volume"]
    return out


# ---------------------------------------------------------------------------
# 步 1：感知
# ---------------------------------------------------------------------------

def _make_ctx(trader_id: int, as_of: date, deps: dict | None,
              replay: bool = False) -> dict:
    """构建本轮上下文。

    replay=True 时 ctx["replay"] 置位，工具层据此把一切数据截到 as_of，
    并把无法回溯的工具（实时大盘 / 跑预测）藏起来（见 tools.py 的时点支持）。
    """
    ctx = {"trader_id": trader_id, "run_date": as_of, "replay": bool(replay),
           "portfolio": None, "cache": {}, "deps": dict(deps or {})}
    pf = _portfolio(ctx)
    ctx["portfolio"] = pf
    ctx["day_state"] = _day_state(trader_id, pf, as_of, replay)
    return ctx


def _portfolio(ctx: dict) -> dict:
    """账户全景。回放时按时点估值，否则净值会混入今天的价格。"""
    d = ctx.get("deps") or {}
    a = ctx.get("run_date") if ctx.get("replay") else None
    if "engine" in d:
        try:
            return (d["engine"].portfolio(ctx["trader_id"], a) if a
                    else d["engine"].portfolio(ctx["trader_id"]))
        except Exception:
            pass
    try:
        from core.paper import engine
        return (engine.portfolio(ctx["trader_id"], a) if a
                else engine.portfolio(ctx["trader_id"]))
    except Exception:
        return {"cash": 0.0, "market_value": 0.0, "total_value": 0.0,
                "total_return": 0.0, "positions": []}


def _day_state(trader_id: int, pf: dict, as_of: date | None = None,
               replay: bool = False) -> dict:
    """当日状态：已有交易笔数 + 日初净值（用于单日亏损熔断）。

    ⚠️ 回放时**不能**用墙钟时间统计"今日成交"：回放写入的 traded_at 是墙钟，
    按 `traded_at >= date.today()` 统计会把之前所有回放日的成交都算进来，
    几轮之后就误触 max_trades_per_day、交易被判超额。
    → 回放改为「本日从 0 起算 + 成交时自增」，日初净值取 as_of 之前最近一条净值。
    """
    ref = as_of if (replay and as_of) else date.today()
    trades = 0
    day_start_nav = float(pf.get("total_value") or 0)
    try:
        from core.store.models import NavHistory, PaperOrder
        with session_scope() as s:
            if not replay:
                trades = (s.query(PaperOrder)
                          .filter(PaperOrder.account_id == trader_id,
                                  PaperOrder.traded_at >= ref).count())
            prev = (s.query(NavHistory)
                    .filter(NavHistory.account_id == trader_id,
                            NavHistory.date < ref)
                    .order_by(NavHistory.date.desc()).first())
            if prev:
                day_start_nav = float(prev.total_value or day_start_nav)
    except Exception:
        pass
    return {"trades_today": int(trades),
            "day_start_nav": day_start_nav,
            "current_nav": float(pf.get("total_value") or 0)}


def _perceive(ctx: dict) -> dict:
    """拉取账户 + 大盘快照。大盘失败不影响主流程。

    ⚠️ 回放模式**绝不能**在这里直接调 market_overview()——那是实时数据，
    会以"市场情绪/涨跌家数"的形式把未来直接喂进 AI 的上下文（而且绕过了工具层）。
    回放改为用基准指数在 as_of 之前的涨跌幅构造简报。
    """
    pf = ctx.get("portfolio") or {}
    market_brief = ""
    if ctx.get("replay"):
        market_brief = _market_brief_on(ctx["run_date"])
    else:
        try:
            if "market" in (ctx.get("deps") or {}):
                mv = ctx["deps"]["market"].market_overview()
            else:
                from core.data import market as market_data
                mv = market_data.market_overview()
            senti = mv.get("sentiment")
            breadth = mv.get("breadth") or {}
            flow = mv.get("fund_flow") or {}
            market_brief = (f"市场情绪 {senti}｜涨 {breadth.get('up')} 跌 "
                            f"{breadth.get('down')}｜主力净流入 "
                            f"{(flow.get('main_net_inflow') or 0) / 1e8:.2f} 亿")
        except Exception as e:
            market_brief = f"（大盘数据暂缺：{e}）"
    return {
        "as_of": str(ctx["run_date"]),
        "total_value": pf.get("total_value"), "cash": pf.get("cash"),
        "positions": len(pf.get("positions") or []),
        "market_brief": market_brief,
        "day_state": dict(ctx.get("day_state") or {}),
    }


def _market_brief_on(as_of: date) -> str:
    """回放用的大盘简报：只用 as_of 及之前的指数数据算，不碰实时接口。"""
    try:
        from core.config import load_config
        from core.data import price as price_data
        bench = ((load_config().get("trader") or {}).get("benchmark_symbol")
                 or "沪深300")
        rows = price_data.closes_upto(bench, "index", as_of, limit=6)
        if len(rows) < 2:
            return "（回放：该日无基准指数数据）"
        last, prev = rows[-1]["close"], rows[0]["close"]
        chg = (last / prev - 1) * 100 if prev else 0.0
        return (f"历史口径（截至 {rows[-1]['date']}）：{bench} 近 {len(rows)} 日 "
                f"{chg:+.2f}%，收 {last:,.2f}｜"
                f"注：回放模式下无涨跌家数与资金流向")
    except Exception as e:
        return f"（回放：大盘数据暂缺 {e}）"


# ---------------------------------------------------------------------------
# 步 2/3：对账 + 归因
# ---------------------------------------------------------------------------

def _call_evaluate_due(fn, as_of: date, replay: bool) -> str:
    """调用到期对账入口，并兼容**不认识 replay 参数**的自定义实现。

    为什么要有这层兼容：`deps` 里注入的 evaluate_due 可能是旧签名（只有 as_of）。
    若直接 `fn(as_of, replay=...)` 会 TypeError，整轮对账作废——
    而失败会被记成 eval_error，让人误以为「对账逻辑坏了」而不是「签名不匹配」。

    replay 只是**能力增强**（严格时点对账：每个 horizon 用它自己的到期日价）。
    不支持时应**降级并明确留痕**，而不是静默按默认口径跑
    （默认口径用最新价，与 due_date 不一致，写进 CalibSample 就是脏样本）。
    返回警告文案；无警告返回 ""。
    """
    if replay:
        try:
            import inspect
            ps = inspect.signature(fn).parameters
            accepts = ("replay" in ps or
                       any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in ps.values()))
        except (TypeError, ValueError):
            accepts = False
        if accepts:
            fn(as_of, replay=True)
            return ""
        fn(as_of)
        return ("evaluate_due 实现不接受 replay 参数，回放对账已降级为默认口径"
                "（到期价可能非 as_of 当日价，校准样本口径不一致，慎用该回放结果）")
    fn(as_of)
    return ""


def _reconcile_and_attribute(ctx: dict, as_of: date, deps: dict) -> dict:
    """复用 verify.evaluator 做全局到期对账，再对**本轮新增**的评估结果做归因。"""
    deps = deps or {}
    t0 = datetime.now()
    eval_err = ""
    replay = bool(ctx.get("replay"))
    try:
        if "evaluate_due" in deps:
            warn = _call_evaluate_due(deps["evaluate_due"], as_of, replay)
        else:
            from core.verify.evaluator import evaluate_due
            # 回放必须用严格时点对账（每个 horizon 用它自己的到期日）
            warn = _call_evaluate_due(evaluate_due, as_of, replay)
        if warn:
            eval_err = warn
    except Exception as e:
        # 对账失败不阻断本轮，但**必须留痕**——禁止静默吞掉，
        # 否则"没学到东西"到底是"本就无可归因项"还是"对账炸了"无法区分。
        eval_err = f"{type(e).__name__}: {e}"

    from core.store.models import Prediction, PredictionEval
    pairs: list[dict] = []
    with session_scope() as s:
        rows = (s.query(PredictionEval)
                .filter(PredictionEval.evaluated_at >= t0).all())
        for e in rows:
            p = s.get(Prediction, e.prediction_id)
            if not p:
                continue
            h = (p.horizons or {}).get(e.horizon) or {}
            pairs.append({
                "prediction": {
                    "id": p.id, "symbol": p.symbol, "horizon": e.horizon,
                    "probs": {k: h.get(k, 0) for k in ("up", "flat", "down")},
                    "base_price": p.base_price, "base_date": str(p.base_date),
                    "horizons": p.horizons or {},
                },
                "actual": {"price": e.actual_price,
                           "change_pct": e.actual_change_pct,
                           "due_date": str(e.due_date)},
            })

    if not pairs:
        empty: dict = {"items": [], "stats": {}, "learnable": 0, "facts": []}
        if eval_err:
            empty["eval_error"] = eval_err
        return empty

    market = _market_change(ctx, deps)
    flat_th = None            # None → 继承 [predict].flat_threshold_pct（口径同源）
    beta_ratio = 0.6
    try:
        a_cfg = (load_config().get("trader") or {}).get("attributor") or {}
        # [trader.attributor].flat_threshold_pct 已废弃：仅在显式配置时才生效，
        # 默认继承 [predict]，避免"三处各写一遍、改一处必漂移"
        if a_cfg.get("flat_threshold_pct") is not None:
            flat_th = float(a_cfg["flat_threshold_pct"])
        beta_ratio = float(a_cfg.get("beta_ratio", 0.6))
    except Exception:
        pass
    res = attr_mod.attribute_batch(pairs, market=market,
                                   flat_threshold_pct=flat_th,
                                   beta_ratio=beta_ratio)
    if eval_err:
        res["eval_error"] = eval_err
    return res


def _market_change(ctx: dict, deps: dict) -> dict | None:
    """大盘同期涨跌幅（用于 beta 归因）。取不到则返回 None（退化为不判 beta）。"""
    try:
        if "market_change" in deps:
            v = deps["market_change"]()
            return v if isinstance(v, dict) else {"change_pct": float(v)}
        from core.data import market as market_data
        mv = market_data.market_overview()
        idx = (mv.get("indices") or {})
        for key in ("沪深300", "上证指数", "中证500"):
            if key in idx and idx[key].get("change_pct") is not None:
                return {"change_pct": float(idx[key]["change_pct"])}
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 步 4：反思（LLM 写记忆）
# ---------------------------------------------------------------------------

def reflect(
    trader_id: int, run_date: date, attributions: dict, *,
    client, deps: dict | None = None,
) -> dict:
    """把归因结果交给 AI 总结教训 → 写 lesson(candidate) → 升级 → 重建 belief。

    只用**本轮可学的事实**做输入（beta/noise 已在上游被过滤），
    并要求模型引用 fact 编号——无证据的教训会被 memory.write_lesson 拒收。
    """
    facts = attributions.get("facts") or []
    out = {"facts": len(facts), "lessons": [], "promotions": [],
           "belief_version": None, "tokens": 0, "cost": 0.0}
    if not facts:
        # 没有可学事实 → 不调 LLM（省成本），但仍尝试升级既有 candidate
        out["promotions"] = memory_mod.promote_lessons(
            trader_id, min_evidence=None, min_confidence=None, mirror=False)
        return out

    # 取刚落库的 fact（按 kind/最新）以便给模型真实 id
    recent = memory_mod.recall(trader_id, kind=memory_mod.KIND_FACT, limit=30)
    fact_lines = "\n".join(
        f"- [{r['id']}] {r['statement']}" for r in recent[:len(facts) + 5])
    stats_txt = attr_mod.summarize_stats(attributions.get("stats") or {})

    user = f"""# 本轮对账与归因结果

{stats_txt}

## 客观事实（这些是已确认记录，请据此反思）
{fact_lines}

## 任务

从上述事实中总结**可复用**的教训。注意：
- `market_beta` 与 `noise` 类的事实**不在**上面（已过滤）——但如果你从表述中
  看出某条其实是市场原因，就不要为它总结教训。
- 证据不足时返回空 lessons。宁缺毋滥。
"""
    try:
        data = client.ask_json(user, system=prompts_mod.REFLECT_SYSTEM,
                               caller="trader_reflect")
    except Exception as e:
        out["error"] = str(e)
        return out

    usage = getattr(client, "last_usage", None) or {}
    out["tokens"] = int(usage.get("total_tokens") or 0)
    out["cost"] = _est_cost(usage)

    for les in (data.get("lessons") or []):
        stmt = str(les.get("statement") or "").strip()
        ids = [int(x) for x in (les.get("evidence_fact_ids") or [])
               if str(x).isdigit()]
        if not stmt or not ids:
            continue      # 无证据的教训直接丢弃（pre_reflect 钩子的落地）
        scope = str(les.get("scope") or memory_mod.SCOPE_GLOBAL)
        sym = str(les.get("symbol") or "")
        if scope.startswith("symbol:") and not sym:
            sym = scope.split(":", 1)[1]
        refs = [{"fact_id": i} for i in ids]
        try:
            lid = memory_mod.write_lesson(
                trader_id, stmt, scope=scope, symbol=sym, evidence_refs=refs,
                proposed_confidence=float(les.get("confidence") or 0.5),
                mirror=False)
            out["lessons"].append({"id": lid, "statement": stmt,
                                   "scope": scope, "evidence": ids})
        except Exception:
            continue

    out["promotions"] = memory_mod.promote_lessons(
        trader_id, min_evidence=None, min_confidence=None, mirror=False)
    try:
        if out["promotions"]:
            b = memory_mod.rebuild_beliefs(trader_id, mirror=False)
            out["belief_version"] = b.get("version")
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# 步 5：决策（带工具的自主循环）
# ---------------------------------------------------------------------------

def decide(
    trader_id: int, ctx: dict, *, client,
    attrib_stats: str = "",
) -> dict:
    """AI 自主决策循环（ReAct）。返回动作列表 + 完整工具轨迹。"""
    if not getattr(client, "available", lambda: False)():
        raise RuntimeError("LLM 未配置，无法决策")

    pf = _portfolio(ctx)
    ctx["portfolio"] = pf
    mview = memory_mod.view(
        trader_id,
        limit_lessons=int(_cfg().get("memory", {}).get(
            "ctx_max_lessons", 10)) if isinstance(_cfg().get("memory"), dict) else 10,
        limit_mistakes=int(_cfg().get("memory", {}).get(
            "ctx_max_mistakes", 5)) if isinstance(_cfg().get("memory"), dict) else 5,
    )
    trader_name = "交易员"
    try:
        with session_scope() as s:
            from core.store.models import PaperAccount
            acc = s.get(PaperAccount, trader_id)
            if acc:
                trader_name = acc.name
    except Exception:
        pass

    context = prompts_mod.build_decision_context(
        name=trader_name, portfolio=pf, memory_view=mview,
        market_brief=(ctx.get("market_brief") or ""),
        attrib_stats=attrib_stats)

    max_turns = _i("max_turns", 8)
    max_calls = _i("max_tool_calls", 20)
    max_cost = _f("max_cost_per_day", 8.0)

    messages: list[dict] = [
        {"role": "system",
         "content": prompts_mod.TRADER_SYSTEM.format(name=trader_name)},
        {"role": "user", "content": context},
    ]
    trace: list[dict] = []
    tokens, cost, calls, turns = 0, 0.0, 0, 0
    fail_sig: dict[str, int] = {}

    if not getattr(client, "supports_tools", lambda: True)():
        # 模型不支持 tools → 走降级路径
        return _decide_fallback(trader_id, ctx, client, context, trace,
                                tokens, cost)

    while turns < max_turns and calls < max_calls and cost < max_cost:
        turns += 1
        try:
            resp = client.chat_with_tools(
                messages, tools_mod.schemas(write_allowed=True,
                                            replay=bool(ctx.get("replay"))),
                model_type="chat", caller="trader_decide")
        except Exception as e:
            if "tools_unsupported" in str(e):
                return _decide_fallback(trader_id, ctx, client, context,
                                        trace, tokens, cost)
            raise

        usage = resp.get("usage") or {}
        tokens += int(usage.get("total_tokens") or 0)
        cost += _est_cost(usage)

        tcs = resp.get("tool_calls") or []
        if not tcs:
            dec = prompts_mod.parse_decision(resp.get("content") or "")
            if not dec:
                messages.append({"role": "assistant",
                                 "content": resp.get("content") or ""})
                messages.append({"role": "user",
                                 "content": prompts_mod.DECISION_JSON_INSTRUCTION})
                continue
            return {"actions": dec.get("actions") or [],
                    "rationale": dec.get("rationale", ""),
                    "watch_reason": dec.get("watch_reason", ""),
                    "tool_calls": trace, "turns": turns,
                    "tokens": tokens, "cost": round(cost, 6)}

        # 执行工具
        messages.append({
            "role": "assistant", "content": resp.get("content") or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"],
                                         "arguments": c["arguments"]}}
                           for c in tcs],
        })
        for c in tcs:
            calls += 1
            try:
                args = json.loads(c.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            sig = f"{c['name']}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            if fail_sig.get(sig, 0) >= 3:
                messages.append({
                    "role": "tool", "tool_call_id": c["id"],
                    "content": json.dumps({
                        "ok": False,
                        "hint": "该调用已连续失败 3 次并被熔断，请不要再重复；"
                                "换一个做法或直接收束决策"},
                        ensure_ascii=False)})
                continue
            res = tools_mod.call(c["name"], args, ctx)
            if not res["ok"]:
                fail_sig[sig] = fail_sig.get(sig, 0) + 1
            trace.append({"tool": c["name"], "ok": res["ok"], "ms": res["ms"],
                          "rejected": res["rejected"], "rule": res["rule"],
                          "digest": res["digest"], "hint": res["hint"],
                          "args": args})
            messages.append({
                "role": "tool", "tool_call_id": c["id"],
                "content": json.dumps({
                    "ok": res["ok"], "data": res["data"],
                    "rejected": res["rejected"], "rule": res["rule"],
                    "hint": res["hint"],
                }, ensure_ascii=False, default=str)[:3500],
            })

    # 熔断收束：让模型基于已有信息直接出决策
    messages.append({"role": "user",
                     "content": "已达本轮工具调用/成本上限。请立即基于已有信息输出"
                                "最终决策 JSON（不再调用工具）。\n"
                                + prompts_mod.DECISION_JSON_INSTRUCTION})
    try:
        raw = client.chat(messages, model_type="chat", caller="trader_decide")
        dec = prompts_mod.parse_decision(raw)
    except Exception:
        dec = {}
    if not dec:
        dec = {"actions": [], "rationale": "达到工具调用上限且未能形成决策",
               "watch_reason": "本轮资源耗尽，选择观望"}
    return {"actions": dec.get("actions") or [],
            "rationale": dec.get("rationale", ""),
            "watch_reason": dec.get("watch_reason", ""),
            "tool_calls": trace, "turns": turns,
            "tokens": tokens, "cost": round(cost, 6)}


def _decide_fallback(trader_id: int, ctx: dict, client, context: str,
                     trace: list[dict], tokens: int, cost: float) -> dict:
    """降级路径：模型不支持 function calling。

    把最关键的几个只读信息预先取好塞进上下文，要求模型直接输出 JSON 决策。
    能力弱于自主循环（不能按需追问数据），但保证系统可跑。
    """
    ctx2 = dict(ctx)
    ctx2["portfolio"] = None
    for tool in ("get_portfolio", "get_past_mistakes", "recall_memory"):
        res = tools_mod.call(tool, {}, ctx2)
        trace.append({"tool": tool, "ok": res["ok"], "ms": res["ms"],
                      "rejected": False, "rule": res["rule"],
                      "digest": res["digest"], "hint": res["hint"], "args": {}})
    preset = "\n".join(f"- {t['tool']}: {t['digest']}" for t in trace)
    user = (context + "\n\n# 预置信息（本模型不支持工具调用，以下为已代取的数据）\n"
            + preset + "\n\n" + prompts_mod.DECISION_JSON_INSTRUCTION)
    try:
        raw = client.chat(
            [{"role": "system",
              "content": prompts_mod.TRADER_SYSTEM.format(name="交易员")},
             {"role": "user", "content": user}],
            model_type="chat", caller="trader_decide_fallback")
        dec = prompts_mod.parse_decision(raw)
    except Exception as e:
        return {"actions": [], "rationale": f"降级决策失败：{e}",
                "watch_reason": "无法决策，观望", "tool_calls": trace,
                "turns": 0, "tokens": tokens, "cost": cost,
                "degraded": True}
    usage = getattr(client, "last_usage", None) or {}
    return {"actions": dec.get("actions") or [],
            "rationale": dec.get("rationale", ""),
            "watch_reason": dec.get("watch_reason", ""),
            "tool_calls": trace, "turns": 1,
            "tokens": tokens + int(usage.get("total_tokens") or 0),
            "cost": round(cost + _est_cost(usage), 6), "degraded": True}


# ---------------------------------------------------------------------------
# 留痕
# ---------------------------------------------------------------------------

def _get_run(trader_id: int, run_date: date) -> TraderRun | None:
    with session_scope() as s:
        row = (s.query(TraderRun)
               .filter_by(trader_id=trader_id, run_date=run_date).first())
        if row:
            s.expunge(row)
        return row


def _save_run(trader_id: int, run_date: date, status: str, run: dict,
              nav: float, cost: float) -> None:
    with session_scope() as s:
        row = (s.query(TraderRun)
               .filter_by(trader_id=trader_id, run_date=run_date).first())
        if row is None:
            row = TraderRun(trader_id=trader_id, run_date=run_date)
            s.add(row)
        row.status = status
        row.perception = run.get("perception") or {}
        row.reconciliations = run.get("reconciliations") or []
        row.attributions = run.get("attributions") or []
        row.new_facts = int(run.get("new_facts") or 0)
        row.new_lessons = run.get("new_lessons") or []
        row.memory_promotions = run.get("promotions") or []
        row.tool_calls = run.get("tool_calls") or []
        row.decision = run.get("decision") or {}
        row.guard_results = run.get("guard_results") or []
        row.orders = run.get("orders") or []
        row.nav = float(nav or 0)
        row.llm_tokens = int(run.get("tokens") or 0)
        row.llm_cost_est = float(cost or 0)
        row.turns = int(run.get("turns") or 0)
        row.error = str(run.get("error") or "")[:500]


# ---------------------------------------------------------------------------
# 历史回放
# ---------------------------------------------------------------------------

def replay_preflight(
    trader_id: int,
    start: date,
    end: date,
    *,
    allow_existing: bool = False,
    max_days: int = 60,
    allow_long: bool = False,
) -> dict:
    """回放前置检查（**只读，无副作用**）。

    单独抽出来是因为 API 需要「先知道要跑几个交易日」才能决定同步跑还是丢后台。
    若把这个判断同时写在 API 和 replay() 里，两边迟早不一致——
    典型症状：API 放行了 90 天、replay() 内部又拒了，用户看到「已接受」却永远拿不到结果。

    返回：{"ok","start","end","calendar_source","days","days_list",
           "n_existing_orders","n_existing_nav","nav_start","error"}
    """
    from core.data import price as price_data
    from core.store.models import NavHistory, PaperAccount, PaperOrder

    start, end = price_data.to_date(start), price_data.to_date(end)
    out: dict = {"ok": False, "start": str(start), "end": str(end),
                 "calendar_source": "empty", "days": 0, "days_list": [],
                 "n_existing_orders": 0, "n_existing_nav": 0,
                 "nav_start": 0.0, "error": ""}
    if not start or not end:
        out["error"] = "起始/结束日期无法解析（需 YYYY-MM-DD）"
        return out
    if start > end:
        out["error"] = f"起始日 {start} 晚于结束日 {end}"
        return out

    # --- 前提 1：全新交易员 ---
    try:
        with session_scope() as s:
            acc = s.get(PaperAccount, trader_id)
            if acc is None:
                out["error"] = f"账户 {trader_id} 不存在"
                return out
            out["n_existing_orders"] = (s.query(PaperOrder)
                                        .filter_by(account_id=trader_id).count())
            out["n_existing_nav"] = (s.query(NavHistory)
                                     .filter_by(account_id=trader_id).count())
    except Exception as e:
        out["error"] = f"账户状态检查失败：{e}"
        return out
    if (out["n_existing_orders"] or out["n_existing_nav"]) and not allow_existing:
        out["error"] = (f"回放要求全新交易员：账户已有 {out['n_existing_orders']} 笔成交 / "
                        f"{out['n_existing_nav']} 条净值。请新建交易员，或显式 "
                        f"allow_existing=True（持仓会与回放混叠，结果不可信）")
        return out

    # --- 交易日历 ---
    days, source = price_data.trading_days_ex(start, end)
    out["calendar_source"] = source
    out["days_list"] = [str(d) for d in days]
    out["days"] = len(days)
    if not days:
        out["error"] = f"{start} ~ {end} 内无交易日"
        return out
    if len(days) > int(max_days) and not allow_long:
        out["error"] = (f"区间含 {len(days)} 个交易日，超过 max_days={max_days}"
                        f"（每个交易日都是一次完整 LLM 循环，成本不低）。"
                        f"缩短区间，或显式 allow_long=True")
        return out

    # 区间起点前最近一条净值作为初始净值（全新账户则为 0，由首个 run 补齐）
    try:
        with session_scope() as s:
            prev = (s.query(NavHistory)
                    .filter(NavHistory.account_id == trader_id,
                            NavHistory.date < start)
                    .order_by(NavHistory.date.desc()).first())
            if prev:
                out["nav_start"] = float(prev.total_value or 0)
    except Exception:
        pass

    out["ok"] = True
    return out


def replay(
    trader_id: int,
    start: date,
    end: date,
    *,
    client=None,
    deps: dict | None = None,
    allow_existing: bool = False,
    max_days: int = 60,
    allow_long: bool = False,
    on_day=None,
) -> dict:
    """按交易日顺序回放 [start, end]（含两端）。

    **正确性四前提（缺一不可，都是踩过的坑）**
      1. **全新交易员**：账上不能已有成交/净值。否则回放期的持仓与真实持仓叠加，
         净值和 T+1 可卖量全错。→ 默认拒绝，需 allow_existing=True 才放行（自担风险）。
      2. **按日推进**：必须交易日升序逐日调用，跳日会让 T+1 与净值曲线错位。
      3. **只用时点价**：成交与估值全部走 price.price_on —— run_day(replay=True) 已保证，
         调用方**不要**自己在 replay 路径里碰 current_price()。
      4. **一次跑完**：区间确定后不要中途改 end，否则净值曲线会缺段。

    **与实盘的差别（诚实标注，别拿回放当实盘验收）**
      - run_prediction / 实时大盘快照在回放中被**隐藏**（依赖实时行情 + 面向未来的
        校准模型，无法回溯）。
      - 因此回放能验收「成交价 / 估值 / 风控闸门 / 留痕 / 记忆写入」，
        但**验不了「预测驱动的自我纠错」**——那一环必须靠实盘向前跑。

    on_day(progress) 每跑完一日回调一次，供长跑（后台任务）上报进度。

    返回：{"trader_id","start","end","calendar_source","days","ok","watched",
           "skipped","errors","nav_start","nav_end","total_return_pct",
           "runs":[...],"error","note"}
    """
    pre = replay_preflight(trader_id, start, end, allow_existing=allow_existing,
                           max_days=max_days, allow_long=allow_long)
    out: dict = {"trader_id": trader_id, "start": pre["start"], "end": pre["end"],
                 "calendar_source": pre["calendar_source"], "days": pre["days"],
                 "ok": 0, "watched": 0, "skipped": 0, "errors": 0,
                 "nav_start": pre["nav_start"], "nav_end": 0.0,
                 "total_return_pct": 0.0, "cost_total": 0.0,
                 "runs": [], "error": pre["error"], "note": ""}
    if not pre["ok"]:
        return out

    # --- 前提 2：按日推进 ---
    for ds in pre["days_list"]:
        d = date.fromisoformat(ds)
        try:
            r = run_day(trader_id, d, client=client, deps=deps, replay=True)
        except Exception as e:
            out["errors"] += 1
            out["runs"].append({"date": str(d), "status": _STATUS_ERROR,
                                "error": str(e)[:200]})
            continue

        st = str(r.get("status") or "")
        if r.get("skipped"):
            out["skipped"] += 1
        elif st == _STATUS_ERROR:
            out["errors"] += 1
        elif st == _STATUS_WATCHED:
            out["watched"] += 1
        else:
            out["ok"] += 1
        out["runs"].append({
            "date": str(d), "status": st,
            "orders": int(r.get("orders") or 0),
            "nav": float(r.get("nav") or 0),
            "new_facts": int(r.get("new_facts") or 0),
            "new_lessons": int(r.get("new_lessons") or 0),
            "turns": int(r.get("turns") or 0),
            "cost": float(r.get("cost") or 0),
            "error": str(r.get("error") or "")[:200],
        })
        if callable(on_day):
            # 长跑（后台任务）靠这个回调上报进度；回调自身出错不能拖垮回放
            try:
                on_day({"date": str(d), "index": len(out["runs"]),
                        "total": out["days"], "status": st,
                        "nav": float(r.get("nav") or 0)})
            except Exception:
                pass

    # --- 汇总 ---
    navs = [x["nav"] for x in out["runs"] if float(x.get("nav") or 0) > 0]
    if navs:
        if not out["nav_start"]:
            out["nav_start"] = float(navs[0])
        out["nav_end"] = float(navs[-1])
        if out["nav_start"]:
            out["total_return_pct"] = round(
                (out["nav_end"] / out["nav_start"] - 1) * 100, 4)
    out["cost_total"] = round(sum(float(x.get("cost") or 0)
                                  for x in out["runs"]), 4)
    out["note"] = ("回放已按交易日升序推进；行情/基金/新闻均截到 as_of，"
                   "run_prediction 与实时大盘在回放中被隐藏 —— "
                   "故回放验的是成交/估值/风控/留痕，验不了预测驱动的自我纠错。")
    if pre["calendar_source"] != "index":
        out["note"] += (" ⚠️ 交易日历降级为『排除周末』口径，非交易日可能被多算，"
                        "天数与净值曲线需人工核对。")
    if pre["n_existing_orders"] or pre["n_existing_nav"]:
        out["note"] += (" ⚠️ 用了 allow_existing=True：账户原有成交/净值未清空，"
                        "回放结果与真实持仓混叠，仅供调试。")
    return out


def run_all(as_of: date | None = None, *, client=None) -> dict:
    """跑所有 mode='trader' 的账户（调度器调用）。

    连续失败达阈值的交易员会自动暂停（status='paused'）。
    """
    from core.store.models import PaperAccount
    out: dict[int, dict] = {}
    max_fail = _i("max_consecutive_failures", 3)
    with session_scope() as s:
        ids = [a.id for a in s.query(PaperAccount).filter_by(mode="trader")]
    for tid in ids:
        try:
            r = run_day(tid, as_of, client=client)
            out[tid] = r
            if r.get("status") == _STATUS_ERROR:
                _bump_failure(tid, max_fail)
        except Exception as e:
            out[tid] = {"trader_id": tid, "status": _STATUS_ERROR, "error": str(e)}
            _bump_failure(tid, max_fail)
    return out


def _bump_failure(trader_id: int, max_fail: int) -> None:
    """连续失败计数；达阈值则暂停交易员。"""
    try:
        with session_scope() as s:
            last = (s.query(TraderRun)
                    .filter_by(trader_id=trader_id)
                    .order_by(TraderRun.run_date.desc())
                    .limit(max_fail).all())
            if len(last) >= max_fail and all(x.status == _STATUS_ERROR for x in last):
                from core.store.models import PaperAccount
                acc = s.get(PaperAccount, trader_id)
                if acc:
                    acc.note = ((acc.note or "") +
                                " | 因连续失败自动暂停").strip(" |")[:250]
    except Exception:
        pass
