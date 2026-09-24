"""T 层：交易台工具注册表（暴露给 LLM 的 function calling 工具集）。

设计文档：docs/AI交易员Harness-架构设计.md §4.2 / §6.3

权限铁律：
- 只读 12 个工具，写工具**只有 2 个**（place_buy_order / place_sell_order）
- 所有写工具内部**先调 guard.check_order()**，被拒则返回 rejected=True
  —— 不存在绕过路径：engine 的买卖只被这两个 handler 调用
- **AI 没有"写记忆"的工具**：记忆写入只能发生在反思阶段、由系统接口完成。
  这是刻意的权限收窄，防止 AI 在决策时顺手编造记忆来给自己找理由。

失败不抛异常，而是返回 ok=False + hint，
让 AI 自己推理修复（对照 Hermes Agent 的 recovery_hint 自愈机制）。
"""
from __future__ import annotations

import time
from datetime import date, datetime
from typing import Any, Callable

from core.trader import guard as guard_mod
from core.trader import memory as memory_mod

# name -> spec
_REGISTRY: dict[str, dict] = {}


def register(spec: dict) -> None:
    """注册一个工具。spec = {name, description, parameters, handler, write}。"""
    _REGISTRY[spec["name"]] = spec


# 回放模式下**不提供**给 AI 的工具。
# 不给它、比给它再让它失败更省 turn，也杜绝了"AI 试出来能拿到未来数据"的可能。
_REPLAY_UNAVAILABLE = {"get_market_snapshot", "run_prediction"}


def schemas(write_allowed: bool = True, replay: bool = False) -> list[dict]:
    """导出 OpenAI function calling 的 tools 参数。

    write_allowed=False 时剔除写工具（用于只读场景与测试）。
    replay=True 时再剔除回放不可用的工具（见 _REPLAY_UNAVAILABLE）。
    """
    out = []
    for spec in _REGISTRY.values():
        if spec.get("write") and not write_allowed:
            continue
        if replay and spec["name"] in _REPLAY_UNAVAILABLE:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        })
    return out


def names(write_allowed: bool = True, replay: bool = False) -> list[str]:
    return [s["function"]["name"] for s in schemas(write_allowed, replay)]


def _digest(data: Any, limit: int = 220) -> str:
    """把工具结果压成一行摘要（进上下文用，避免把大结果全塞进去）。"""
    try:
        if isinstance(data, dict):
            for k in ("summary", "digest", "name", "message", "error"):
                if data.get(k):
                    return str(data[k])[:limit]
            return str({k: data[k] for k in list(data)[:6]})[:limit]
        if isinstance(data, list):
            return f"{len(data)} 项: " + str(data[:2])[:limit]
        return str(data)[:limit]
    except Exception:
        return ""


def call(name: str, args: dict, ctx: dict) -> dict:
    """统一调用入口。

    ctx 契约：
      {"trader_id": int, "run_date": date, "portfolio": dict,
       "day_state": dict, "deps": {...覆盖默认依赖...}, "cache": dict}

    返回：
      {"ok": bool, "data": Any, "digest": str, "ms": int,
       "rejected": bool, "rule": str, "hint": str, "tool": str}
      失败/被拒都不抛异常——让 AI 读 hint 自行修复。
    """
    spec = _REGISTRY.get(name)
    if not spec:
        return {"ok": False, "data": None, "digest": "", "ms": 0,
                "rejected": False, "rule": "unknown_tool",
                "hint": f"没有名为 {name} 的工具，请从工具列表中重新选择",
                "tool": name}
    t0 = time.time()
    try:
        res = spec["handler"](ctx, **(args or {})) or {}
    except TypeError as e:
        res = {"ok": False, "hint": f"参数不合法：{e}。请检查参数名与类型"}
    except Exception as e:  # 工具内部异常不中断循环
        res = {"ok": False, "hint": f"工具执行异常：{e}"}
    ms = int((time.time() - t0) * 1000)
    ok = bool(res.get("ok", True))
    rejected = bool(res.get("rejected", False))
    return {
        "ok": ok, "data": res.get("data"), "ms": ms,
        "rejected": rejected, "rule": res.get("rule", ""),
        "hint": res.get("hint", ""),
        "digest": res.get("digest") or _digest(res.get("data")),
        "tool": name,
    }


# ---------------------------------------------------------------------------
# 依赖解析（可被 ctx["deps"] 覆盖，便于离线测试）
# ---------------------------------------------------------------------------

def _default_deps() -> dict:
    from core.data import fund as fund_data
    from core.data import market as market_data
    from core.data import stock as stock_data
    from core.paper import engine
    return {"stock": stock_data, "fund": fund_data, "market": market_data,
            "engine": engine, "memory": memory_mod}


def _deps(ctx: dict) -> dict:
    """依赖解析。

    ctx["deps"] 的语义是**逐项覆盖**，不是整体替换：
    先用真实实现铺满，再用调用方给的键盖上去。
    若写成"deps 非空就整体替换"，调用方只给 engine 时 stock/fund 会凭空消失，
    故障会以"取价失败""档案获取失败"这类误导性的面貌冒出来，极难定位。
    """
    d = _default_deps()
    d.update(ctx.get("deps") or {})
    return d


# ---------------------------------------------------------------------------
# 时点（回放）支持
#
# 回放模式的铁律：**AI 看到的每一个数都必须是 as_of 及之前的**。
# 一旦有任何工具泄漏未来数据，AI 等于开了天眼，回放结果毫无意义。
# 所以这里不做"尽力而为"，而是：
#   - 能按时点截断的（报价/K线/特征/成交历史/记忆）→ 严格截断；
#   - 做不到的（实时大盘、实时舆情、跑预测）→ **明确拒绝并说明原因**，
#     绝不静默返回实时数据。
# 宁可让 AI 少一个工具，也不能给它一个会泄漏未来的工具。
# ---------------------------------------------------------------------------

def _as_of(ctx: dict) -> date | None:
    """回放模式返回时点日期；实盘返回 None（各 handler 据 None 走实时路径）。"""
    if not ctx.get("replay"):
        return None
    d = ctx.get("run_date")
    return d if isinstance(d, date) else None


def _replay_blocked(tool: str, why: str) -> dict:
    """回放模式下不可用的工具统一返回这个（明确、可解释，而非静默降级）。"""
    return {"ok": False, "rule": "replay_unavailable",
            "hint": f"回放模式不可用：{tool} {why}。"
                    f"请改用 get_kline / get_metrics / get_quote（这些已按历史时点截断），"
                    f"或直接基于现有持仓与记忆决策。"}


def _portfolio(ctx: dict) -> dict:
    if ctx.get("portfolio") is not None:
        return ctx["portfolio"]
    d = _deps(ctx)
    a = _as_of(ctx)
    try:
        if a:
            return d["engine"].portfolio(ctx["trader_id"], a)
        return d["engine"].portfolio(ctx["trader_id"])
    except Exception:
        return {"cash": 0.0, "market_value": 0.0, "total_value": 0.0,
                "positions": []}


# ---------------------------------------------------------------------------
# 内置工具：只读
# ---------------------------------------------------------------------------

def _h_market(ctx, **_):
    d = _deps(ctx)
    if _as_of(ctx):
        # 涨跌家数/资金流向没有可靠的历史回溯接口 →
        # 若在这里偷偷调实时接口，AI 就能"看到未来"，回放直接作废。
        return _replay_blocked("get_market_snapshot",
                               "没有历史涨跌家数/资金流向的回溯数据源")
    try:
        mv = d["market"].market_overview()
        return {"ok": True, "data": mv}
    except Exception as e:
        return {"ok": False, "hint": f"大盘数据获取失败：{e}，可先跳过市场环境判断"}


def _h_search_funds(ctx, keyword: str = "", limit: int = 10, **_):
    """搜基金（候选发现的入口）。优先用 fund.search_funds，缺失时降级。"""
    d = _deps(ctx)
    fn = getattr(d["fund"], "search_funds", None)
    if not callable(fn):
        return {"ok": False,
                "hint": "基金搜索能力未启用（core.data.fund.search_funds 缺失），"
                        "请改用 get_quote / run_prediction 直接查已关注的标的"}
    try:
        res = fn(keyword, limit=int(limit)) or []
        return {"ok": True, "data": res}
    except Exception as e:
        return {"ok": False, "hint": f"搜索失败：{e}"}


def _h_fund_profile(ctx, symbol: str = "", **_):
    d = _deps(ctx)
    a = _as_of(ctx)
    if a:
        # 档案里的"阶段收益"是实时口径 → 回放时会泄漏。
        # 这里只保留静态元数据（名称/经理/公司/规模/跟踪指数），去掉收益类字段。
        try:
            info = d["stock"].classify_symbol(symbol)
            at = info.get("asset_type", "ofund")
            res = dict(d["fund"].fetch_fund_profile(symbol, at, with_holdings=False) or {})
            for k in ("stage_returns", "return_1y", "return_3m", "recent_return",
                      "holdings", "top_holdings"):
                res.pop(k, None)
            res["note"] = f"历史口径：已剔除实时收益类字段（截至 {a}）"
            return {"ok": bool(res), "data": res,
                    "hint": "" if res else f"{symbol} 无基金档案"}
        except Exception as e:
            return {"ok": False, "hint": f"档案获取失败：{e}"}
    try:
        info = d["stock"].classify_symbol(symbol)
        at = info.get("asset_type", "ofund")
        res = d["fund"].fetch_fund_profile(symbol, at, with_holdings=True)
        return {"ok": bool(res), "data": res,
                "hint": "" if res else f"{symbol} 无基金档案"}
    except Exception as e:
        return {"ok": False, "hint": f"档案获取失败：{e}"}


def _h_quote(ctx, symbol: str = "", **_):
    d = _deps(ctx)
    a = _as_of(ctx)
    if a:
        # 回放：必须取 as_of 当日收盘，不能取实时价（那是未来）
        try:
            from core.data import price as price_data
            info = d["stock"].classify_symbol(symbol)
            at = info.get("asset_type", "stock")
            if at == "auto":
                at = d["stock"].resolve_auto(symbol)
            snap = price_data.snapshot_on(symbol, at, a)
            if not snap.get("ok"):
                return {"ok": False, "hint": f"{symbol} 无 {a} 及之前的历史行情"}
            return {"ok": True, "data": snap, "digest": f"{symbol} 收盘 {snap['price']}"}
        except Exception as e:
            return {"ok": False, "hint": f"历史行情获取失败：{e}"}
    try:
        q = d["stock"].fetch_realtime_quote(symbol)
        if not q or not q.get("ok"):
            return {"ok": False, "hint": f"{symbol} 行情获取失败"}
        return {"ok": True, "data": q}
    except Exception as e:
        return {"ok": False, "hint": f"行情获取失败：{e}"}


def _h_kline(ctx, symbol: str = "", count: int = 60, **_):
    d = _deps(ctx)
    a = _as_of(ctx)
    if a:
        # 回放：截到 as_of，绝不把之后的 K 线给 AI
        try:
            from core.data import price as price_data
            info = d["stock"].classify_symbol(symbol)
            at = info.get("asset_type", "stock")
            if at == "auto":
                at = d["stock"].resolve_auto(symbol)
            rows = price_data.closes_upto(symbol, at, a, limit=int(count))
            if not rows:
                return {"ok": False, "hint": f"{symbol} 无 {a} 及之前的K线"}
            return {"ok": True, "data": {"rows": len(rows), "recent": rows[-5:],
                                         "as_of": str(a),
                                         "note": f"历史口径（截至 {a}）"}}
        except Exception as e:
            return {"ok": False, "hint": f"历史K线获取失败：{e}"}
    try:
        df = d["stock"].fetch_kline(symbol, count=int(count))
        if df is None or df.empty:
            df = d["fund"].fetch_fund_nav_history(symbol, count=int(count))
        if df is None or df.empty:
            return {"ok": False, "hint": f"{symbol} 无K线数据"}
        tail = df.tail(5).to_dict("records")
        return {"ok": True, "data": {"rows": len(df), "recent": tail}}
    except Exception as e:
        return {"ok": False, "hint": f"K线获取失败：{e}"}


def _metrics_from_closes(symbol: str, closes: list[float], note: str = "") -> dict:
    """由收盘序列算基础技术特征（回放的时点特征也走这里）。"""
    peak, mdd = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        mdd = max(mdd, (peak - c) / peak)
    ret = (closes[-1] / closes[0] - 1) * 100 if closes[0] else 0.0
    return {"symbol": symbol, "data_days": len(closes),
            "stage_returns": {"区间累计": round(ret, 2)},
            "max_drawdown_pct": round(mdd * 100, 2),
            "note": note or "简化特征（由收盘序列本地计算）"}


def _h_metrics(ctx, symbol: str = "", **_):
    """基金/标的技术特征（本地计算，不调 LLM）。"""
    d = _deps(ctx)
    a = _as_of(ctx)
    if a:
        # 回放：必须用**截至 as_of** 的序列算，否则回撤/收益会把未来算进去
        try:
            from core.data import price as price_data
            info = d["stock"].classify_symbol(symbol)
            at = info.get("asset_type", "stock")
            if at == "auto":
                at = d["stock"].resolve_auto(symbol)
            rows = price_data.closes_upto(symbol, at, a, limit=250)
            closes = [r["close"] for r in rows if r.get("close")]
            if len(closes) < 2:
                return {"ok": False, "hint": f"{symbol} 历史数据不足，无法算特征"}
            return {"ok": True, "data": _metrics_from_closes(
                symbol, closes, note=f"历史口径（截至 {a} 收盘，共 {len(closes)} 根）")}
        except Exception as e:
            return {"ok": False, "hint": f"历史特征计算失败：{e}"}
    fn = getattr(d["fund"], "fetch_fund_metrics", None)
    if not callable(fn):
        # 降级：至少给出阶段收益与回撤的粗略值
        try:
            df = d["stock"].fetch_kline(symbol, count=250)
            if df is None or df.empty:
                df = d["fund"].fetch_fund_nav_history(symbol, count=250)
            if df is None or df.empty:
                return {"ok": False, "hint": f"{symbol} 无数据可算特征"}
            closes = [float(x) for x in df["close"].tolist()]
            peak, mdd = closes[0], 0.0
            for c in closes:
                peak = max(peak, c)
                mdd = max(mdd, (peak - c) / peak)
            ret_1y = (closes[-1] / closes[0] - 1) * 100 if closes[0] else 0.0
            return {"ok": True, "data": {
                "symbol": symbol, "data_days": len(closes),
                "stage_returns": {"1y": round(ret_1y, 2)},
                "max_drawdown_pct": round(mdd * 100, 2),
                "note": "简化特征（fetch_fund_metrics 未启用）"}}
        except Exception as e:
            return {"ok": False, "hint": f"特征计算失败：{e}"}
    try:
        res = fn(symbol) or {}
        return {"ok": bool(res) and not res.get("error"), "data": res,
                "hint": res.get("error", "")}
    except Exception as e:
        return {"ok": False, "hint": f"特征计算失败：{e}"}


def _h_run_prediction(ctx, symbol: str = "", mode: str = "fast", **_):
    """跑一次结构化预测（有副作用：写 predictions 表，但不动资金）。

    ⚠️ 回放模式下**禁用**：预测链路的输入是实时行情/实时舆情/实时大盘，
    且校准层模型是用"包含未来样本"的数据训练出来的——
    既无法截到 as_of，也存在模型状态层面的前视。
    这是本回放能力**已知的、诚实的缺口**（见设计文档 §16）：
    因此回放只能验证「交易/估值/风控/留痕」，
    **不能**验证「基于预测的自我纠错学习」，后者必须走向前实盘。
    """
    d = _deps(ctx)
    if _as_of(ctx):
        return _replay_blocked("run_prediction",
                               "依赖实时行情/舆情与含未来样本的校准模型，无法截到历史时点")
    fn = (ctx.get("deps") or {}).get("predict_fn")
    try:
        if not callable(fn):
            from core.predict.pipeline import predict as fn
        pred = fn(symbol, mode=mode)
        horizons = getattr(pred, "horizons", None) or {}
        return {"ok": True, "data": {
            "prediction_id": getattr(pred, "id", None),
            "symbol": symbol,
            "confidence": getattr(pred, "confidence", None),
            "horizons": {k: dict(v) for k, v in horizons.items()
                         if k in ("next_day", "one_week")},
            "action": getattr(pred, "action", None),
            "risks": (getattr(pred, "risks", None) or [])[:3],
        }, "digest": f"{symbol} 预测完成"}
    except Exception as e:
        return {"ok": False, "hint": f"预测失败：{e}，可改用 get_metrics 自行判断"}


def _h_news(ctx, symbol: str = "", limit: int = 8, **_):
    d = _deps(ctx)
    a = _as_of(ctx)
    if a:
        # 回放：先用本地舆情库按 publish_time 截断；库里有就用，没有就明说没有，
        # **绝不**退回实时抓取（那等于泄漏未来消息面）。
        try:
            from core.store.db import session_scope
            from core.store.models import NewsItem
            with session_scope() as s:
                eod = datetime(a.year, a.month, a.day, 23, 59, 59)
                q = (s.query(NewsItem)
                     .filter(NewsItem.publish_time <= eod)
                     .order_by(NewsItem.publish_time.desc())
                     .limit(int(limit)).all())
                brief = [{"title": (r.title or "")[:80],
                          "time": str(r.publish_time)[:16]} for r in q]
            if brief:
                return {"ok": True, "data": brief,
                        "digest": f"{len(brief)} 条历史舆情（截至 {a}）"}
        except Exception:
            brief = []
        return _replay_blocked("get_news",
                               f"本地舆情库中没有 {a} 及之前的记录")
    try:
        items = d["fund"] and None  # 占位，下面走 news 模块
        from core.intel import news as news_data
        rows = news_data.collect_news(symbol, days=7, limit=int(limit)) or []
        brief = [{"title": r.get("title", "")[:80],
                  "time": str(r.get("publish_time", ""))[:16],
                  "sentiment": r.get("sentiment", "")} for r in rows[:limit]]
        return {"ok": True, "data": brief}
    except Exception as e:
        return {"ok": False, "hint": f"舆情获取失败：{e}"}


def _h_portfolio(ctx, **_):
    pf = _portfolio(ctx)
    return {"ok": True, "data": pf,
            "digest": f"总资产 {pf.get('total_value')}，现金 {pf.get('cash')}，"
                      f"持仓 {len(pf.get('positions') or [])} 只"}


def _h_trade_history(ctx, limit: int = 10, **_):
    d = _deps(ctx)
    try:
        from core.store.db import session_scope
        from core.store.models import PaperOrder
        # 注意：不做 traded_at <= as_of 过滤。
        # 原因：回放写入的成交，traded_at 是**墙钟时间**（非模拟日期），
        # 按 as_of 过滤会把回放自己的成交也滤掉。
        # 正确性由「回放必须用全新交易员、且按交易日顺序推进」来保证——
        # 见 core/trader/loop.py::replay()。
        with session_scope() as s:
            rows = (s.query(PaperOrder)
                    .filter_by(account_id=ctx["trader_id"])
                    .order_by(PaperOrder.id.desc())
                    .limit(int(limit)).all())
            data = [{"side": o.side, "symbol": o.symbol, "price": o.price,
                     "volume": o.volume, "fee": o.fee, "reason": o.reason,
                     "at": str(o.traded_at)[:16]} for o in rows]
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "hint": f"历史成交读取失败：{e}"}


def _h_recall_memory(ctx, kind: str = "lesson", symbol: str = "",
                     limit: int = 10, **_):
    """翻自己的记忆（默认只给 active — candidate 不参与决策）。"""
    rows = memory_mod.recall(
        ctx["trader_id"], kind=(kind or None), symbol=(symbol or None),
        limit=int(limit))
    return {"ok": True, "data": rows,
            "digest": f"{len(rows)} 条 {kind or '全部'} 记忆"}


def _h_past_mistakes(ctx, limit: int = 5, **_):
    """看自己犯过的错（关键工具：让 AI 直面失败）。"""
    rows = memory_mod.recall(
        ctx["trader_id"], kind=memory_mod.KIND_FACT, limit=int(limit))
    return {"ok": True, "data": rows,
            "digest": f"{len(rows)} 条历史判断失误事实"}


# ---------------------------------------------------------------------------
# 内置工具：写（必经 guard）
# ---------------------------------------------------------------------------

def _do_trade(ctx: dict, *, side: str, symbol: str, amount: float | None,
              volume: int | None, reason: str) -> dict:
    """买入/卖出的公共路径：取价 → 过 guard → 执行 → 返回结果。"""
    d = _deps(ctx)
    engine = d["engine"]
    pf = _portfolio(ctx)

    # 取价（供 guard 估算与 engine 成交）
    # 注意：资产类型解析失败与取价失败要分开报错。
    # 早期版本用一个裸 except 包住两件事，导致"依赖缺失"被伪装成"取价失败"，
    # 排查时完全看不出真正原因。
    try:
        info = d["stock"].classify_symbol(symbol)
        at = info.get("asset_type", "stock")
        if at == "auto":
            at = d["stock"].resolve_auto(symbol)
    except KeyError as e:
        return {"ok": False, "rejected": False, "rule": "deps_missing",
                "hint": f"交易台依赖缺失（{e}），无法解析 {symbol} 的标的类型"}
    except Exception:
        at = "stock"
    # 回放时按 as_of 历史价取价 —— 否则会用今天的价成交，回放结果无意义
    a = _as_of(ctx)
    try:
        price = engine.current_price(symbol, at, a) if a else engine.current_price(symbol, at)
    except Exception:
        price = None
    if not price or price <= 0:
        return {"ok": False, "rejected": False, "rule": "no_price",
                "hint": f"{symbol} 取价失败，无法下单；可改选其他标的或观望"}

    action = {"side": side, "symbol": symbol, "price": float(price),
              "amount": amount, "volume": volume, "asset_type": at}
    g = guard_mod.check_order(action, pf, ctx.get("day_state") or {})
    if not g["allowed"]:
        return {"ok": False, "rejected": True, "rule": g["rule"],
                "data": {"guard": g}, "hint": g["hint"],
                "digest": f"被风控拒绝：{g['detail']}"}

    # 合规裁剪
    adj = g.get("adjusted") or {}
    if "amount" in adj:
        amount = float(adj["amount"])
    if "volume" in adj:
        volume = int(adj["volume"])

    # 卖出：engine.sell 只接受 volume/ratio，金额需换算成数量
    if side == "sell" and not volume and amount and price > 0:
        volume = int(amount / price)

    try:
        if side == "buy":
            order = engine.buy(ctx["trader_id"], symbol, amount=amount,
                               volume=volume, reason=reason or "trader",
                               price=float(price), as_of=a)
        else:
            order = engine.sell(ctx["trader_id"], symbol, volume=volume,
                                reason=reason or "trader",
                                price=float(price), as_of=a)
    except Exception as e:
        return {"ok": False, "rejected": False, "rule": "trade_error",
                "hint": f"下单失败：{e}。请检查资金/持仓后重新决策",
                "digest": f"下单失败：{e}"}

    # 成交后刷新上下文缓存，避免 AI 用过期持仓做下一步决策
    ctx.pop("portfolio", None)
    ds = ctx.get("day_state")
    if isinstance(ds, dict):
        ds["trades_today"] = int(ds.get("trades_today") or 0) + 1

    out = {"order_id": getattr(order, "id", None),
           "side": side, "symbol": symbol,
           "price": float(getattr(order, "price", price) or price),
           "volume": int(getattr(order, "volume", 0) or 0),
           "amount": float(getattr(order, "amount", 0) or 0),
           "fee": float(getattr(order, "fee", 0) or 0),
           "guard_rule": g.get("rule", "")}
    return {"ok": True, "data": out,
            "hint": g.get("hint", ""),
            "digest": f"{side.upper()} {symbol} {out['volume']}股 "
                      f"@{out['price']:.3f}"}


def _h_buy(ctx, symbol: str = "", amount: float | None = None,
           volume: int | None = None, reason: str = "", **_):
    return _do_trade(ctx, side="buy", symbol=symbol, amount=amount,
                     volume=volume, reason=reason)


def _h_sell(ctx, symbol: str = "", amount: float | None = None,
            volume: int | None = None, reason: str = "", **_):
    return _do_trade(ctx, side="sell", symbol=symbol, amount=amount,
                     volume=volume, reason=reason)


# ---------------------------------------------------------------------------
# 注册内置工具
# ---------------------------------------------------------------------------

def _p(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_SYMBOL = {"type": "string", "description": "标的代码，如 510300 / 110022 / 00700"}

_BUILTIN: list[dict] = [
    {"name": "get_market_snapshot", "write": False,
     "description": "获取当前大盘环境：指数涨跌、涨跌家数、资金流向。判断市场系统性风险时用。",
     "parameters": _p({}, []), "handler": _h_market},

    {"name": "search_funds", "write": False,
     "description": "按关键词搜索基金/ETF，用于发现可投标的。不知道代码时用它。",
     "parameters": _p({"keyword": {"type": "string", "description": "关键词，如 沪深300、医药、科技"},
                       "limit": {"type": "integer", "description": "返回条数，默认 10"}},
                      ["keyword"]),
     "handler": _h_search_funds},

    {"name": "get_fund_profile", "write": False,
     "description": "获取基金档案：类型、经理、公司、规模、阶段收益、重仓股、跟踪指数。",
     "parameters": _p({"symbol": _SYMBOL}, ["symbol"]), "handler": _h_fund_profile},

    {"name": "get_quote", "write": False,
     "description": "获取标的实时行情：最新价、涨跌幅、成交量等。",
     "parameters": _p({"symbol": _SYMBOL}, ["symbol"]), "handler": _h_quote},

    {"name": "get_kline", "write": False,
     "description": "获取标的近期K线（或基金净值走势），返回行数与最近5根。",
     "parameters": _p({"symbol": _SYMBOL,
                       "count": {"type": "integer", "description": "取多少根，默认 60"}},
                      ["symbol"]),
     "handler": _h_kline},

    {"name": "get_metrics", "write": False,
     "description": "获取标的技术特征：阶段收益、最大回撤、波动、夏普。用于横向比较候选。",
     "parameters": _p({"symbol": _SYMBOL}, ["symbol"]), "handler": _h_metrics},

    {"name": "run_prediction", "write": False,
     "description": "对标的跑一次结构化预测，返回次日/一周的涨平跌概率与置信度。需要独立信号时用。",
     "parameters": _p({"symbol": _SYMBOL,
                       "mode": {"type": "string", "enum": ["fast", "deep"],
                                "description": "fast 省成本，deep 更深入"}},
                      ["symbol"]),
     "handler": _h_run_prediction},

    {"name": "get_news", "write": False,
     "description": "获取标的近期相关舆情（含情绪标签）。判断消息面时用。",
     "parameters": _p({"symbol": _SYMBOL,
                       "limit": {"type": "integer", "description": "条数，默认 8"}},
                      ["symbol"]),
     "handler": _h_news},

    {"name": "get_portfolio", "write": False,
     "description": "获取自己的账户现状：现金、持仓明细、市值、浮动盈亏。决策前应至少看一次。",
     "parameters": _p({}, []), "handler": _h_portfolio},

    {"name": "get_trade_history", "write": False,
     "description": "获取自己的历史成交记录，用于复盘自己的操作。",
     "parameters": _p({"limit": {"type": "integer", "description": "条数，默认 10"}},
                      []),
     "handler": _h_trade_history},

    {"name": "recall_memory", "write": False,
     "description": "检索自己的记忆（已生效的教训与信念）。决策前应查阅，避免重犯错误。",
     "parameters": _p({"kind": {"type": "string", "enum": ["fact", "lesson", "belief"],
                                "description": "记忆类型，默认 lesson"},
                       "symbol": {"type": "string", "description": "限定某个标的"},
                       "limit": {"type": "integer", "description": "条数，默认 10"}},
                      []),
     "handler": _h_recall_memory},

    {"name": "get_past_mistakes", "write": False,
     "description": "查看自己历史上判断失误的记录（已做归因）。下单前建议先看一眼。",
     "parameters": _p({"limit": {"type": "integer", "description": "条数，默认 5"}},
                      []),
     "handler": _h_past_mistakes},

    {"name": "place_buy_order", "write": True,
     "description": "买入标的。会经过风控校验，被拒时请读返回的 hint 调整方案，不要重复提交同一请求。",
     "parameters": _p({"symbol": _SYMBOL,
                       "amount": {"type": "number", "description": "买入金额（元）"},
                       "reason": {"type": "string", "description": "买入理由，简短"}},
                      ["symbol", "reason"]),
     "handler": _h_buy},

    {"name": "place_sell_order", "write": True,
     "description": "卖出已持有的标的。会经过风控校验（含 T+1 可卖量检查）。",
     "parameters": _p({"symbol": _SYMBOL,
                       "volume": {"type": "integer", "description": "卖出数量（股/份）"},
                       "amount": {"type": "number", "description": "或按金额卖出"},
                       "reason": {"type": "string", "description": "卖出理由，简短"}},
                      ["symbol", "reason"]),
     "handler": _h_sell},
]


def _install() -> None:
    for spec in _BUILTIN:
        register(spec)


_install()
