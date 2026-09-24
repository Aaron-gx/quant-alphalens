"""新闻情报采集：多源抓取 → 去重 → 时效过滤 → 落库。

数据源（全部免费，参考并扩展 xystock / fund_assistant）：
- 个股新闻：ak.stock_news_em（东方财富）
- 宏观政策：ak.stock_news_main_cx（财新网）
- 7×24 快讯：ak.stock_info_global_em（东财）/ ak.stock_info_global_cls（财联社）
- 研报评级：ak.stock_research_report_em
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta

from core.store.db import session_scope
from core.store.models import NewsItem


def _hash(title: str) -> str:
    return hashlib.sha1(title.strip().encode("utf-8")).hexdigest()[:32]


def _parse_time(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip()[:19], fmt)
        except (ValueError, TypeError):
            continue
    return None


def fetch_stock_news(symbol: str, days: int = 7, limit: int = 10) -> list[dict]:
    """东财个股新闻：按时间过滤，最新在前。"""
    from core.data._net import eastmoney_ok
    if not eastmoney_ok():
        return []  # 东财接口被风控/不可达时快速跳过，避免逐个等超时
    import akshare as ak
    items = []
    try:
        df = ak.stock_news_em(symbol)
        if df.empty:
            return items
        cutoff = datetime.now() - timedelta(days=days)
        rows = df.to_dict("records")
        rows.sort(key=lambda x: str(x.get("发布时间", "")), reverse=True)
        for r in rows:
            pt = _parse_time(r.get("发布时间", ""))
            if pt and pt < cutoff:
                continue
            items.append({
                "symbol": symbol, "source": "eastmoney", "category": "news",
                "title": str(r.get("新闻标题", "")),
                "content": str(r.get("新闻内容", "") or ""),
                "url": str(r.get("新闻链接", "")),
                "publish_time": pt,
            })
            if len(items) >= limit:
                break
    except Exception:
        pass
    return items


def fetch_market_news(limit: int = 15) -> list[dict]:
    """财新宏观/政策面新闻。"""
    import akshare as ak
    items = []
    try:
        df = ak.stock_news_main_cx()
        for r in df.to_dict("records"):
            summary = str(r.get("summary", "")).strip()
            if not summary:
                continue
            tag = str(r.get("tag", "") or "宏观")
            # tag 只是栏目名（"市场动态"等），真正的标题要从摘要里取
            head = re.split(r"[。！？；\n]", summary)[0].strip() or summary[:60]
            items.append({
                "symbol": "", "source": "caixin", "category": "macro",
                "title": f"【{tag}】{head[:70]}",
                "content": summary,
                "url": str(r.get("url", "")),
                "publish_time": _parse_time(r.get("pub_time", "")),
            })
            if len(items) >= limit:
                break
    except Exception:
        pass
    return items


def fetch_flash_news(limit: int = 30) -> list[dict]:
    """7×24 快讯：财联社为主，东财全球为辅。"""
    import akshare as ak
    items = []
    try:
        df = ak.stock_info_global_cls()
        for r in df.head(limit).to_dict("records"):
            title = str(r.get("标题", ""))
            items.append({
                "symbol": "", "source": "cls", "category": "flash",
                "title": title or str(r.get("内容", ""))[:60],
                "content": str(r.get("内容", "")),
                "url": "",
                "publish_time": _parse_time(f"{r.get('发布日期','')} {r.get('发布时间','')}"),
            })
    except Exception:
        pass
    if len(items) < limit // 2:  # 财联社失败时补东财
        try:
            df = ak.stock_info_global_em()
            for r in df.head(limit - len(items)).to_dict("records"):
                items.append({
                    "symbol": "", "source": "eastmoney_global", "category": "flash",
                    "title": str(r.get("标题", "")),
                    "content": str(r.get("摘要", "")),
                    "url": str(r.get("链接", "")),
                    "publish_time": _parse_time(r.get("发布时间", "")),
                })
        except Exception:
            pass
    return items[:limit]


def fetch_research_reports(symbol: str, limit: int = 5) -> list[dict]:
    """个股研报/评级。"""
    from core.data._net import eastmoney_ok
    if not eastmoney_ok():
        return []  # 东财接口被风控/不可达时快速跳过
    import akshare as ak
    items = []
    try:
        df = ak.stock_research_report_em(symbol)
        for r in df.head(limit).to_dict("records"):
            rating = str(r.get("最新评级", "") or r.get("评级", ""))
            rt = str(r.get("报告标题", "") or r.get("标题", "") or "").strip()
            content = str(r.get("内容摘要", "") or r.get("盈利预测", "") or "")
            if not rt:  # 报告标题字段为空时从摘要首句取
                rt = re.split(r"[。！？；\n]", content)[0].strip()[:60] or "研报摘要"
            title = f"[{r.get('机构','')}{'·'+rating if rating else ''}] {rt}"
            items.append({
                "symbol": symbol, "source": "research", "category": "research",
                "title": title,
                "content": str(r.get("内容摘要", "") or r.get("盈利预测", "") or ""),
                "url": "", "publish_time": _parse_time(r.get("日期", r.get("发布日期", ""))),
            })
    except Exception:
        pass
    return items


def save_news(items: list[dict]) -> int:
    """新闻落库，按 title_hash 去重。返回新入库条数。"""
    if not items:
        return 0
    saved = 0
    with session_scope() as s:
        existing = {h for (h,) in s.query(NewsItem.title_hash).all()}
        for it in items:
            h = _hash(it.get("title", "") + str(it.get("publish_time", "")))
            if h in existing:
                continue
            s.add(NewsItem(
                symbol=it.get("symbol", ""), source=it.get("source", ""),
                category=it.get("category", "news"), title=it.get("title", "")[:250],
                content=it.get("content", ""), url=it.get("url", ""),
                publish_time=it.get("publish_time"), title_hash=h,
            ))
            existing.add(h)
            saved += 1
    return saved


def collect_news(symbol: str = "", days: int = 7, limit: int = 10,
                 include_market: bool = True,
                 extra_symbols: list[str] | None = None,
                 extra_per_symbol: int = 4) -> list[dict]:
    """聚合采集：个股新闻 + 研报 + 宏观快讯，落库后返回本次条目。

    extra_symbols 用于"基金舆情范围"：基金本身没有个股新闻，改抓它重仓股的消息，
    这样基金的消息面才不是空的大盘情绪（见 core.intel.relevance）。
    """
    items: list[dict] = []
    if symbol:
        items += fetch_stock_news(symbol, days=days, limit=limit)
        items += fetch_research_reports(symbol, limit=5)
    for code in (extra_symbols or [])[:5]:
        if not code or code == symbol:
            continue
        items += fetch_stock_news(code, days=days, limit=extra_per_symbol)
        items += fetch_research_reports(code, limit=1)
    if include_market:
        items += fetch_market_news(limit=10)
        items += fetch_flash_news(limit=15)
    # 跨来源去重：同一标题可能被个股新闻与重仓股新闻重复抓回
    seen_titles: set[str] = set()
    deduped: list[dict] = []
    for it in items:
        t = re.sub(r"\s+", "", str(it.get("title", "")))
        if t and t in seen_titles:
            continue
        seen_titles.add(t)
        deduped.append(it)
    save_news(deduped)
    return deduped


def load_recent_news(symbol: str = "", days: int = 7, limit: int = 30) -> list[NewsItem]:
    """从情报库读近期条目（含已打分的情绪）。"""
    cutoff = datetime.now() - timedelta(days=days)
    with session_scope() as s:
        q = s.query(NewsItem).filter(NewsItem.fetched_at >= cutoff)
        if symbol:
            q = q.filter(NewsItem.symbol == symbol)
        return q.order_by(NewsItem.publish_time.desc().nullslast()).limit(limit).all()
