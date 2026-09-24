"""AI 信号自动交易：ai_signal 模式账户按最新预测执行调仓。

规则（保守默认）：
- 信号源：当日（base_date=today）已落库的预测
- 买入：next_day.up >= 0.55 且 confidence >= 60 → 用可用现金的 20% 建仓/加仓
- 卖出：next_day.down >= 0.55 且 confidence >= 60 → 清掉该持仓 50%
- 同一标的当日只动一次；现金不足/无持仓自动跳过
- 每次成交的 reason 记录预测 id，方便事后归因
"""
from __future__ import annotations

from datetime import date

from core.paper import engine
from core.paper.engine import TradeError
from core.store.db import session_scope
from core.store.models import PaperAccount, PaperOrder, PaperPosition, Prediction, Watchlist

UP_TH, DOWN_TH, CONF_TH = 0.55, 0.55, 60
BUY_CASH_RATIO = 0.20   # 单笔用现金的 20%
SELL_RATIO = 0.5        # 卖出持仓的 50%


def _today_predictions() -> dict[str, Prediction]:
    """当日预测：symbol -> 最新一条。"""
    with session_scope() as s:
        rows = (s.query(Prediction)
                .filter(Prediction.base_date == date.today())
                .order_by(Prediction.id.desc()).all())
        out: dict[str, Prediction] = {}
        for p in rows:
            out.setdefault(p.symbol, p)
            s.expunge(p)
        return out


def _traded_today(account_id: int, symbol: str) -> bool:
    with session_scope() as s:
        return (s.query(PaperOrder)
                .filter(PaperOrder.account_id == account_id,
                        PaperOrder.symbol == symbol,
                        PaperOrder.traded_at >= date.today())
                .count() > 0)


def run_account(account_id: int) -> list[str]:
    """对一个 ai_signal 账户执行一轮自动调仓。返回执行日志。"""
    with session_scope() as s:
        acc = s.get(PaperAccount, account_id)
        if not acc or acc.mode != "ai_signal":
            return [f"账户 {account_id} 非 ai_signal 模式，跳过"]
    preds = _today_predictions()
    if not preds:
        return ["今日无预测记录，跳过"]

    with session_scope() as s:
        held = {p.symbol for p in
                s.query(PaperPosition).filter_by(account_id=account_id)}
        watched = {w.symbol for w in s.query(Watchlist).filter_by(auto_predict=True)}
    universe = held | watched

    logs = []
    for sym in universe:
        p = preds.get(sym)
        if not p or _traded_today(account_id, sym):
            continue
        h = (p.horizons or {}).get("next_day", {})
        reason = f"ai_signal pred#{p.id} conf={p.confidence}"
        try:
            if h.get("down", 0) >= DOWN_TH and p.confidence >= CONF_TH and sym in held:
                o = engine.sell(account_id, sym, ratio=SELL_RATIO, reason=reason)
                logs.append(f"SELL {p.name or sym} {o.volume}股 @ {o.price:.2f}")
            elif h.get("up", 0) >= UP_TH and p.confidence >= CONF_TH:
                pf = engine.portfolio(account_id)
                budget = pf["cash"] * BUY_CASH_RATIO
                if budget < 600:  # 太少不够一手
                    continue
                o = engine.buy(account_id, sym, amount=budget,
                               name=p.name, reason=reason)
                logs.append(f"BUY {p.name or sym} {o.volume}股 @ {o.price:.2f}")
        except TradeError as e:
            logs.append(f"SKIP {sym}: {e}")
    return logs or ["无信号触发"]


def run_all() -> dict[int, list[str]]:
    """所有 ai_signal 账户跑一轮（调度器调用）。"""
    with session_scope() as s:
        ids = [a.id for a in s.query(PaperAccount).filter_by(mode="ai_signal")]
    return {aid: run_account(aid) for aid in ids}
