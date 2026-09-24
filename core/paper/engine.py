"""模拟盘引擎：虚拟账户、下单、持仓、净值。

规则贴近实盘：A股 T+1（当日买入不可卖）、100 股整数倍、佣金+印花税；
场外基金按净值申赎（简化：T+0 确认、不计申购费折扣）。
"""
from __future__ import annotations

from datetime import datetime, date

from core.config import get
from core.data import stock as stock_data
from core.data import fund as fund_data
from core.data import market as market_data
from core.store.db import session_scope
from core.store.models import (
    PaperAccount, PaperPosition, PaperOrder, NavHistory,
)


class TradeError(RuntimeError):
    pass


def create_account(name: str, initial_cash: float | None = None,
                   mode: str = "manual", note: str = "") -> int:
    cash = initial_cash or float(get("paper", "default_cash", 1000000))
    with session_scope() as s:
        acc = PaperAccount(name=name, mode=mode, initial_cash=cash, cash=cash, note=note)
        s.add(acc)
        s.flush()
        return acc.id


def _commission(amount: float) -> float:
    rate = float(get("paper", "commission_rate", 0.00025))
    min_fee = float(get("paper", "commission_min", 5.0))
    return max(amount * rate, min_fee)


def _stamp_tax(amount: float) -> float:
    return amount * float(get("paper", "stamp_tax_rate", 0.0005))


# 印花税只对**股票**类卖出单边征收 0.05%。
# 场内 ETF/LOF（asset_type="fund"）与场外基金（"ofund"）卖出免征——
# 早先版本对卖出无条件计税，等于系统性低估基金策略的收益，
# 会让 AI 学到"交易越少越好"这种由错误成本造成的伪规律。
_STAMP_TAX_TYPES = ("stock", "bse")


def _get_position(s, account_id: int, symbol: str) -> PaperPosition | None:
    return (s.query(PaperPosition)
            .filter_by(account_id=account_id, symbol=symbol).first())


def current_price(symbol: str, asset_type: str = "stock",
                  as_of: date | None = None) -> float | None:
    """取价。

    - `as_of=None`（默认）：**实时**口径，优先实时行情、失败用最近收盘。
      用于向前实盘的成交与估值。
    - `as_of=<日期>`：**时点**口径，取该日或之前最近交易日的收盘价。
      用于历史回放——回放时若仍取实时价，等于用未来价成交，结果无意义。

    分工原则见 core/data/price.py 模块说明：交易不可穿越，评估必须能回看。
    """
    if as_of is not None:
        try:
            from core.data import price as price_data
            return price_data.price_on(symbol, asset_type, as_of)
        except Exception:
            return None
    try:
        if asset_type == "index":
            df = market_data.fetch_index_kline(symbol, count=2)
            return float(df.iloc[-1]["close"]) if len(df) else None
        if asset_type == "ofund":
            est = fund_data.fetch_fund_realtime_estimate(symbol)
            if est.get("ok") and est.get("estimate_nav"):
                return est["estimate_nav"]
            nav = fund_data.fetch_fund_nav_history(symbol, count=2)
            return float(nav.iloc[-1]["close"]) if len(nav) else None
        q = stock_data.fetch_realtime_quote(symbol)
        if q.get("ok") and q.get("price"):
            return float(q["price"])
        df = stock_data.fetch_kline(symbol, count=2)
        return float(df.iloc[-1]["close"]) if len(df) else None
    except Exception:
        return None


def buy(account_id: int, symbol: str, volume: int | None = None,
        amount: float | None = None, name: str = "", reason: str = "",
        price: float | None = None, asset_type: str | None = None,
        as_of: date | None = None) -> PaperOrder:
    """买入：给股数 volume 或金额 amount 二选一。A股取整 100 股。

    as_of 非空时按该日历史价成交（回放用），为空时按实时价（实盘用）。
    """
    info = stock_data.classify_symbol(symbol)
    asset_type = asset_type or info["asset_type"]
    if asset_type == "auto":
        asset_type = stock_data.resolve_auto(symbol)
    price = price or current_price(symbol, asset_type, as_of)
    if not price or price <= 0:
        raise TradeError(f"{symbol} 取价失败，无法下单")

    with session_scope() as s:
        acc = s.get(PaperAccount, account_id)
        if not acc:
            raise TradeError("账户不存在")
        if amount is not None:
            volume = int(amount / price)
        if asset_type in ("stock", "fund", "hk", "bse"):
            volume = int(volume // 100 * 100)  # 整手
        if not volume or volume <= 0:
            raise TradeError("买入数量过小（不足一手）")
        cost = price * volume
        fee = _commission(cost)
        if cost + fee > acc.cash:
            raise TradeError(f"资金不足：需 {cost + fee:,.2f}，可用 {acc.cash:,.2f}")

        acc.cash -= cost + fee
        pos = _get_position(s, account_id, symbol)
        if pos is None:
            pos = PaperPosition(account_id=account_id, symbol=symbol,
                                name=name or symbol, asset_type=asset_type,
                                volume=0, available=0, avg_cost=0)
            s.add(pos)
        # 均价 = 加权
        total_v = pos.volume + volume
        pos.avg_cost = (pos.avg_cost * pos.volume + cost) / total_v
        pos.volume = total_v
        # T+1：当日买入不可用（ofund 简化按可赎）
        if asset_type == "ofund":
            pos.available = total_v
        order = PaperOrder(account_id=account_id, symbol=symbol,
                           name=name or symbol, side="buy", price=price,
                           volume=volume, amount=cost, fee=fee, reason=reason)
        s.add(order)
        s.flush()
        s.refresh(order)
        s.expunge(order)
        return order


def sell(account_id: int, symbol: str, volume: int | None = None,
         ratio: float | None = None, reason: str = "",
         price: float | None = None,
         as_of: date | None = None) -> PaperOrder:
    """卖出：给股数 volume 或持仓比例 ratio（如 0.5=卖一半）。

    as_of 非空时按该日历史价成交（回放用），为空时按实时价（实盘用）。
    """
    info = stock_data.classify_symbol(symbol)
    asset_type = info["asset_type"]
    with session_scope() as s:
        acc = s.get(PaperAccount, account_id)
        pos = _get_position(s, account_id, symbol)
        if not acc or not pos or pos.volume <= 0:
            raise TradeError("无持仓可卖")
        if ratio is not None:
            volume = int(pos.available * ratio)
        volume = min(volume or 0, pos.available)
        if asset_type in ("stock", "fund", "hk", "bse"):
            volume = int(volume // 100 * 100)
        if volume <= 0:
            raise TradeError("可卖数量不足（T+1 或不足一手）")
        price = price or current_price(symbol, asset_type, as_of)
        if not price or price <= 0:
            raise TradeError(f"{symbol} 取价失败")

        revenue = price * volume
        fee = _commission(revenue)
        if asset_type in _STAMP_TAX_TYPES:
            fee += _stamp_tax(revenue)
        acc.cash += revenue - fee
        pos_name = pos.name
        pos.volume -= volume
        pos.available -= volume
        if pos.volume <= 0:
            s.delete(pos)
        order = PaperOrder(account_id=account_id, symbol=symbol, name=pos_name or symbol,
                           side="sell", price=price, volume=volume,
                           amount=revenue, fee=fee, reason=reason)
        s.add(order)
        s.flush()
        s.refresh(order)
        s.expunge(order)
        return order


def settle_t1(account_id: int | None = None):
    """每日收盘后调用：把当日买入的持仓转为可用（T+1 生效）。"""
    with session_scope() as s:
        q = s.query(PaperPosition)
        if account_id:
            q = q.filter_by(account_id=account_id)
        for pos in q.all():
            if pos.asset_type != "ofund":
                pos.available = pos.volume


def portfolio(account_id: int, as_of: date | None = None) -> dict:
    """账户全景：现金 + 持仓市值 + 盈亏。

    as_of 非空时按该日历史价估值（回放用）——否则回放中的"总资产"会混入
    今天的价格，净值曲线就不再是历史净值。
    """
    with session_scope() as s:
        acc = s.get(PaperAccount, account_id)
        if not acc:
            raise TradeError("账户不存在")
        positions = []
        mv = 0.0
        for pos in s.query(PaperPosition).filter_by(account_id=account_id):
            px = current_price(pos.symbol, pos.asset_type, as_of) or pos.avg_cost
            value = px * pos.volume
            mv += value
            positions.append({
                "symbol": pos.symbol, "name": pos.name, "asset_type": pos.asset_type,
                "volume": pos.volume, "available": pos.available,
                "avg_cost": round(pos.avg_cost, 4), "price": round(px, 4),
                "market_value": round(value, 2),
                "pnl": round((px - pos.avg_cost) * pos.volume, 2),
                "pnl_pct": round((px / pos.avg_cost - 1) * 100, 2) if pos.avg_cost else 0,
            })
        total = acc.cash + mv
        return {
            "id": acc.id, "name": acc.name, "mode": acc.mode,
            "cash": round(acc.cash, 2), "market_value": round(mv, 2),
            "total_value": round(total, 2),
            "total_return": round((total / acc.initial_cash - 1) * 100, 2),
            "positions": positions,
        }


def snapshot_nav(account_id: int, as_of: date | None = None) -> None:
    """每日净值快照（含当日收益率）。

    原先固定用 date.today()，回放时会把 20 个交易日全部写成同一天、
    且彼此互相覆盖 → 净值曲线只有一根点。现按 as_of 落库。
    """
    pf = portfolio(account_id, as_of)
    today = as_of or date.today()
    with session_scope() as s:
        prev = (s.query(NavHistory)
                .filter_by(account_id=account_id)
                .order_by(NavHistory.date.desc()).first())
        daily_ret = 0.0
        if prev and prev.date != today and prev.total_value:
            daily_ret = (pf["total_value"] / prev.total_value - 1) * 100
        existing = s.query(NavHistory).filter_by(account_id=account_id, date=today).first()
        if existing:
            existing.cash = pf["cash"]
            existing.market_value = pf["market_value"]
            existing.total_value = pf["total_value"]
        else:
            s.add(NavHistory(account_id=account_id, date=today, cash=pf["cash"],
                             market_value=pf["market_value"],
                             total_value=pf["total_value"],
                             daily_return=round(daily_ret, 4)))


def nav_curve(account_id: int) -> list[dict]:
    with session_scope() as s:
        rows = (s.query(NavHistory).filter_by(account_id=account_id)
                .order_by(NavHistory.date).all())
        return [{"date": str(r.date), "total_value": r.total_value,
                 "daily_return": r.daily_return} for r in rows]


def perf_stats(account_id: int) -> dict:
    """绩效指标：总收益/年化/最大回撤/夏普/胜率。"""
    import math
    curve = nav_curve(account_id)
    if len(curve) < 2:
        return {"days": len(curve)}
    values = [c["total_value"] for c in curve]
    rets = [values[i] / values[i - 1] - 1 for i in range(1, len(values))]
    total_ret = values[-1] / values[0] - 1
    days = len(values)
    annual = (1 + total_ret) ** (250 / days) - 1 if days else 0
    peak, mdd = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
    sharpe = mean / math.sqrt(var) * math.sqrt(250) if var > 0 else 0
    # 胜率：按成对买卖估算
    with session_scope() as s:
        orders = (s.query(PaperOrder).filter_by(account_id=account_id)
                  .order_by(PaperOrder.traded_at).all())
        buys, sells = {}, []
        for o in orders:
            if o.side == "buy":
                buys.setdefault(o.symbol, []).append(o.price)
            else:
                bp = buys.get(o.symbol, [])
                sells.append((o.symbol, bp.pop(0) if bp else None, o.price))
        win = sum(1 for _, b, sp in sells if b and sp > b)
        win_rate = win / len(sells) if sells else 0
    return {
        "days": days, "total_return": round(total_ret * 100, 2),
        "annual_return": round(annual * 100, 2), "max_drawdown": round(mdd * 100, 2),
        "sharpe": round(sharpe, 3), "closed_trades": len(sells),
        "win_rate": round(win_rate * 100, 2),
    }
