"""盘前简报与盘中异动归因。

- morning_brief: 隔夜新闻 + 外围行情 → LLM 一段式简报（推送/机器人可直接转）
- anomaly_scan: 自选标的涨跌幅超阈值 → 拉当日新闻让 LLM 归因
"""
from __future__ import annotations

from datetime import datetime

from core.config import get
from core.data import market as market_data
from core.data import stock as stock_data
from core.intel import news as news_data
from core.predict.llm import LLMClient
from core.store.db import session_scope
from core.store.models import Watchlist, AnalysisReport

BRIEF_SYSTEM = """你是一位资深的晨会分析师。基于给定的隔夜资讯与市场数据，写一份 300 字内的 A 股盘前简报：
1. 隔夜要闻（政策/宏观/外围，挑最重要的 2-3 条）
2. 今日关注点与潜在催化
3. 一句话情绪判断（偏多/偏空/中性 + 理由）
直接、聚焦，不堆术语。"""


def morning_brief() -> str:
    """生成盘前简报文本。"""
    flash = news_data.fetch_flash_news(limit=15)
    news_data.save_news(flash)
    market = {}
    try:
        market = market_data.market_overview()
    except Exception:
        pass
    lines = [f"- [{str(i.get('publish_time',''))[5:16]}] {i['title']}" for i in flash]
    data = "【隔夜快讯】\n" + "\n".join(lines)
    if market.get("indices", {}).get("ok"):
        idx = market["indices"]["indices"]
        data += "\n\n【指数收盘】\n" + "\n".join(
            f"- {n}: {d['price']:.2f} ({d['change_pct']:+.2f}%)" for n, d in idx.items())
    client = LLMClient()
    text = client.ask(data, system=BRIEF_SYSTEM, caller="brief")
    with session_scope() as s:
        s.add(AnalysisReport(symbol="", name="盘前简报", report_type="morning_brief",
                             content=text))
    return text


def anomaly_scan(threshold_pct: float = 5.0) -> list[dict]:
    """扫描自选标的：|涨跌幅| 超阈值 → 当日新闻归因。返回异动列表。"""
    with session_scope() as s:
        symbols = [(w.symbol, w.name) for w in s.query(Watchlist)]
    anomalies = []
    client = LLMClient()
    for sym, name in symbols:
        q = stock_data.fetch_realtime_quote(sym)
        if not q.get("ok"):
            continue
        chg = q.get("change_pct", 0)
        if abs(chg) < threshold_pct:
            continue
        item = {"symbol": sym, "name": q.get("name") or name or sym,
                "change_pct": chg, "price": q.get("price")}
        # 归因：拉当日新闻让 LLM 解释
        if client.available():
            news_items = news_data.fetch_stock_news(sym, days=1, limit=5)
            news_text = "\n".join(f"- {i['title']}" for i in news_items) or "（无当日新闻）"
            try:
                item["reason"] = client.ask(
                    f"{item['name']}({sym}) 今日{'大涨' if chg > 0 else '大跌'} "
                    f"{chg:+.2f}%。以下为当日新闻：\n{news_text}\n\n"
                    "请用100字内给出最可能的异动原因；若新闻无法解释，说明可能原因（如板块轮动/资金推动）。",
                    caller="anomaly")
            except Exception:
                item["reason"] = "归因生成失败"
        anomalies.append(item)
    return anomalies
