"""L 层：确定性风控门禁。**AI 不可绕过**。

设计文档：docs/AI交易员Harness-架构设计.md §4.5 / §6.4

为什么这个模块必须存在且必须是"代码"：

AI 能自由决策下单，是这套设计的价值所在，也是最大风险点。
风控如果写在提示词里（"请你不要超过 30% 仓位"），模型随时可能违反；
写在代码里、位于所有写操作之前，模型就无法绕过——它连这些阈值的具体数值都看不到。

对照 patterns/capability-security（最小权限）：能力边界靠授予机制保证，不靠自律。

另一个关键点：**拒单不是静默失败**。
每个拒绝都返回 hint（修复建议），会被回灌给 AI 作为 observation。
这既是安全机制，也是教学机制——AI 会从"被拒"中学到约束边界。
"""
from __future__ import annotations

from core.config import load_config

DEFAULT_RULES: dict = {
    "max_weight_per_symbol": 0.30,   # 单一标的上限
    "max_total_position": 0.95,      # 总仓位上限（留现金）
    "min_cash_weight": 0.05,         # 现金下限
    "max_trades_per_day": 3,         # 单日交易笔数上限
    "max_daily_loss_pct": 3.0,       # 单日亏损熔断（相对日初净值）
    "max_positions": 8,              # 最多持仓数
    "min_order_amount": 1000.0,      # 单笔最小金额
    "banned_symbols": [],            # 黑名单
    "cost_warn_pct": 1.5,            # 交易成本占比告警阈值
}


def rules() -> dict:
    """读取 [trader.guard] 段，缺项用默认值补齐。"""
    cfg = (load_config().get("trader") or {}).get("guard") or {}
    out = dict(DEFAULT_RULES)
    out.update({k: v for k, v in cfg.items() if v is not None})
    return out


def _reject(rule: str, detail: str, hint: str,
            adjusted: dict | None = None) -> dict:
    return {"allowed": False, "rule": rule, "detail": detail,
            "hint": hint, "adjusted": adjusted}


def _allow(detail: str = "", warn: str = "") -> dict:
    return {"allowed": True, "rule": "", "detail": detail or "通过风控",
            "hint": warn, "adjusted": None}


def check_order(
    action: dict,
    portfolio: dict,
    day_state: dict | None = None,
    rules_override: dict | None = None,
) -> dict:
    """确定性风控校验。次序短路，第一项不过即返回。

    action 契约：
      {"side": "buy"|"sell", "symbol": str, "price": float,
       "amount": float|None, "volume": int|None, "asset_type": str}
    portfolio 契约：engine.portfolio() 的输出
      {"cash","market_value","total_value","positions":[{symbol,...,market_value}]}
    day_state 契约：
      {"trades_today": int, "day_start_nav": float, "current_nav": float}

    返回：
      {"allowed": bool, "rule": str, "detail": str, "hint": str,
       "adjusted": dict|None}   # adjusted 为建议的合规替代下单量
    """
    r = dict(rules())
    if rules_override:
        r.update(rules_override)
    ds = day_state or {}

    side = str(action.get("side") or "").lower()
    symbol = str(action.get("symbol") or "")
    if side not in ("buy", "sell"):
        return _reject("bad_side", f"非法交易方向 {side!r}",
                       "side 只能是 buy 或 sell")
    if not symbol:
        return _reject("no_symbol", "未指定标的代码", "请给出 symbol")

    price = float(action.get("price") or 0)
    amount = action.get("amount")
    volume = action.get("volume")
    if amount is None and volume is not None and price > 0:
        amount = float(volume) * price
    amount = float(amount or 0)
    if amount <= 0:
        return _reject("no_amount", "未给出有效下单金额/数量",
                       "请给出 amount（金额）或 volume（数量）")

    # 1) 黑名单
    banned = [str(b) for b in (r.get("banned_symbols") or [])]
    if symbol in banned:
        return _reject("banned", f"{symbol} 在黑名单中",
                       f"{symbol} 已被禁止交易，请换标的或观望")

    # 2) 单日亏损熔断 → 禁止一切买入（卖出仍允许，用于止损）
    if side == "buy":
        start_nav = float(ds.get("day_start_nav") or 0)
        cur_nav = float(ds.get("current_nav") or 0)
        if start_nav > 0 and cur_nav > 0:
            loss_pct = (cur_nav / start_nav - 1) * 100
            if loss_pct <= -abs(float(r["max_daily_loss_pct"])):
                return _reject(
                    "daily_loss_fuse",
                    f"当日已亏 {loss_pct:.2f}%，触发 {r['max_daily_loss_pct']}% 熔断",
                    "今日禁止买入；只能观望，或对亏损持仓做止损卖出")

    # 3) 单日交易笔数
    trades = int(ds.get("trades_today") or 0)
    if trades >= int(r["max_trades_per_day"]):
        return _reject(
            "max_trades_per_day",
            f"今日已交易 {trades} 笔，达到上限 {r['max_trades_per_day']}",
            "今日不再允许新的交易；请把想法留到下一个交易日，或直接观望")

    positions = list(portfolio.get("positions") or [])
    total_value = float(portfolio.get("total_value") or 0)
    cash = float(portfolio.get("cash") or 0)
    held = {str(p.get("symbol")): p for p in positions}
    held_mv = {str(p.get("symbol")): float(p.get("market_value") or 0)
               for p in positions}

    # ---------- 卖出分支 ----------
    if side == "sell":
        pos = held.get(symbol)
        if not pos:
            return _reject("no_position", f"未持有 {symbol}，无法卖出",
                           "只能卖出已持有的标的；若想反向操作请改为买入其他标的")
        avail = int(pos.get("available") or 0)
        if avail <= 0:
            return _reject(
                "t_plus_1", f"{symbol} 可卖数量为 0（T+1 当日买入不可卖）",
                "该持仓今日买入，明日才可卖出；今日请观望")
        want_v = int(volume or 0)
        if want_v <= 0 and amount > 0 and price > 0:
            want_v = int(amount / price)
        if want_v <= 0:
            return _reject(
                "no_volume", f"{symbol} 未给出有效的卖出数量",
                "请给出 volume（数量）或 amount（金额）")
        # 注意：卖出不设最小金额门槛。小额持仓必须允许清仓，
        # 否则 min_order_amount 会把"卖不掉的小仓位"永久锁死。
        if want_v > avail:
            adj = {"volume": avail}
            return {
                "allowed": True, "rule": "clip_volume",
                "detail": f"卖出量 {want_v} 超过可卖 {avail}，已建议削至 {avail}",
                "hint": f"最多只能卖 {avail} 股", "adjusted": adj,
            }
        mv = held_mv.get(symbol, 0.0)
        if total_value > 0 and mv / total_value > float(r["max_weight_per_symbol"]) * 1.5:
            return _allow(
                f"卖出 {symbol}（当前占比 {mv / total_value:.0%}，偏高）")
        return _allow(f"卖出 {symbol} {want_v} 股")

    # ---------- 买入分支 ----------
    min_amt = float(r["min_order_amount"])
    if amount < min_amt:
        return _reject(
            "min_order_amount",
            f"下单金额 {amount:,.0f} 低于最小额 {min_amt:,.0f}",
            f"请把金额提高到 {min_amt:,.0f} 以上，或不做这笔交易")

    # 逐项算出"本单最多还能买多少"，取**最紧**的那一项作为合规额度。
    #
    # 为什么必须取 min，而不是"哪个约束先触发就按哪个削"：
    # 若只看权重，可能削出一个仍然下不出来的金额（例：权重还剩 30 万，
    # 但账户只有 10 万现金 → adjusted=300000 是假额度）。这个假额度会
    # 经 tools._do_trade 直接灌给 engine.buy，也会作为 hint 回灌给 AI——
    # 而 hint 是教学机制，喂假额度等于教错。故必须取所有约束的交集上界。
    if total_value > 0:
        max_total = float(r["max_total_position"])
        min_cash_w = float(r["min_cash_weight"])
        mv = held_mv.get(symbol, 0.0)
        rooms = [
            ("max_weight_per_symbol", "clip_weight",
             float(r["max_weight_per_symbol"]) * total_value - mv,
             f"{symbol} 仓位将超过单一标的上限 "
             f"{float(r['max_weight_per_symbol']):.0%}"),
            ("insufficient_cash", "clip_cash",
             cash,
             f"可用现金仅 {cash:,.0f} 元"),
            ("position_limit", "clip_position",
             min(max_total * total_value
                 - float(portfolio.get("market_value") or 0),
                 cash - min_cash_w * total_value),
             f"总仓位上限 {max_total:.0%} / 现金下限 {min_cash_w:.0%}"),
        ]
        # 平局时 rooms 的顺序决定归因：权重 → 现金 → 仓位
        binding_rule, clip_rule, capacity, why = min(rooms, key=lambda x: x[2])
        capacity = max(0.0, capacity)
        if amount > capacity + 1e-9:
            if capacity < min_amt:
                return _reject(
                    binding_rule,
                    f"{why}；本单 {amount:,.0f} 元无法通过，"
                    f"该约束下的合规额度仅 {capacity:,.0f} 元",
                    f"此约束下已无可用空间，请降低金额、改买其他候选标的，"
                    f"或直接观望")
            return {
                "allowed": True, "rule": clip_rule,
                "detail": f"买入额由 {amount:,.0f} 削至 {capacity:,.0f}，"
                          f"以满足：{why}",
                "hint": f"合规可买 {capacity:,.0f} 元（受限于 {why}）",
                "adjusted": {"amount": round(capacity, 2)},
            }

    # 持仓数量
    if symbol not in held and len(positions) >= int(r["max_positions"]):
        return _reject(
            "max_positions",
            f"已持有 {len(positions)} 只，达到上限 {r['max_positions']}",
            "请先卖出一部分持仓，或改为加仓已持有的标的")

    warn = ""
    if total_value > 0:
        est_fee_ratio = 0.00025 + 0.0   # 佣金（印花税仅卖出收）
        if est_fee_ratio * 100 >= float(r["cost_warn_pct"]):
            warn = f"注意交易成本约占 {est_fee_ratio * 100:.2f}%"
    return _allow(f"买入 {symbol} {amount:,.0f} 元", warn)


def summarize(results: list[dict]) -> dict:
    """汇总一组风控结论（供留痕与报告）。"""
    rejected = [x for x in results if not x.get("allowed")]
    adjusted = [x for x in results if x.get("allowed") and x.get("rule")]
    return {"total": len(results), "rejected": len(rejected),
            "adjusted": len(adjusted),
            "rules_hit": sorted({x["rule"] for x in results if x.get("rule")})}
