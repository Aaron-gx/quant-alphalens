"""新闻情绪打分流水线：LLM 逐条评估 → 相关性筛选 → 加权聚合成信号。

四维打分（在 xystock 三维方案上加一维）：
- sentiment 乐观/中性/悲观、intensity 强度1-5、deviation 预期偏离度1-5
- relevance 与标的的关联度1-5 —— 决定这条消息该不该算进"这个基金/股票的舆情"

聚合时先按相关度门槛筛选（见 core.intel.relevance），再按
强度 × 偏离度 × 相关度加权投票，避免"大盘快讯"绑架个股舆情。
"""
from __future__ import annotations

from core.intel.relevance import DEFAULT_MIN_RELEVANCE
from core.predict.llm import LLMClient, LLMError
from core.predict.prompts import SENTIMENT_PROMPT, BATCH_SENTIMENT_PROMPT
from core.store.db import session_scope
from core.store.models import NewsItem

_SENTI_SCORE = {"乐观": 1.0, "中性": 0.0, "悲观": -1.0}


def _apply(r: dict, it: dict) -> None:
    it["sentiment"] = str(r.get("sentiment", "中性"))
    it["intensity"] = int(r.get("intensity", 3))
    it["deviation"] = int(r.get("deviation", 3))
    rel = r.get("relevance")
    if rel not in (None, ""):
        try:
            it["relevance_llm"] = max(1, min(5, int(float(rel))))
        except (TypeError, ValueError):
            pass


def score_items(items: list[dict], target_desc: str, *, client: LLMClient | None = None,
                limit: int = 10) -> list[dict]:
    """对一批新闻 dict 打分（就地追加 sentiment/intensity/deviation/relevance_llm）。

    target_desc 应尽量写全"这个标的的真实舆情范围"（含重仓股/跟踪指数），
    否则 LLM 无法判断关联度，relevance 维度会退化成全 3 分。
    优先批量调用，失败回退逐条，保证稳定。
    """
    client = client or LLMClient()
    if not client.available():
        return items
    batch = [it for it in items[:limit] if it.get("title")]
    if not batch:
        return items

    numbered = "\n".join(
        f"{i + 1}. 标题：{it.get('title', '')}\n   内容：{(it.get('content') or it.get('title', ''))[:300]}"
        for i, it in enumerate(batch))
    try:
        r = client.ask_json(BATCH_SENTIMENT_PROMPT.format(
            count=len(batch), target_desc=target_desc, news=numbered), caller="sentiment")
        results = r.get("results") if isinstance(r, dict) else r
        if isinstance(results, list) and len(results) >= len(batch):
            for it, res in zip(batch, results):
                if isinstance(res, dict):
                    _apply(res, it)
            _fill_defaults(batch)
            return items
    except (LLMError, ValueError, KeyError, TypeError):
        pass

    # 回退：逐条打分
    for it in batch:
        try:
            r = client.ask_json(SENTIMENT_PROMPT.format(
                target_desc=target_desc, title=it.get("title", ""),
                content=(it.get("content") or it.get("title", ""))[:600]), caller="sentiment")
            _apply(r, it)
        except (LLMError, ValueError, KeyError):
            pass
    _fill_defaults(batch)
    return items


def _fill_defaults(items: list[dict]) -> None:
    for it in items:
        it.setdefault("sentiment", "中性")
        it.setdefault("intensity", 3)
        it.setdefault("deviation", 3)


def score_db_backlog(limit: int = 20) -> int:
    """给情报库里已入库未打分的条目补打分（后台任务用）。"""
    client = LLMClient()
    if not client.available():
        return 0
    done = 0
    with session_scope() as s:
        rows = (s.query(NewsItem).filter(NewsItem.scored.is_(False))
                .order_by(NewsItem.id.desc()).limit(limit).all())
        for row in rows:
            desc = f"股票/基金 {row.symbol}" if row.symbol else "A股市场整体"
            try:
                r = client.ask_json(SENTIMENT_PROMPT.format(
                    target_desc=desc, title=row.title,
                    content=(row.content or row.title)[:600]), caller="sentiment")
                row.sentiment = str(r.get("sentiment", "中性"))[:8]
                row.intensity = int(r.get("intensity", 3))
                row.deviation = int(r.get("deviation", 3))
                rel = r.get("relevance")
                row.relevance = int(float(rel)) if rel not in (None, "") else 0
                row.scored = True
                done += 1
            except (LLMError, ValueError, KeyError, TypeError):
                row.scored = True  # 打分失败也标记，避免反复重试烧钱
    return done


def _weight(it: dict) -> float:
    """投票权重 = 强度 × 预期偏离度放大 × 相关度折扣（低相关不零分，但压到 40%）。"""
    w = (it.get("intensity") or 3) * (1 + (it.get("deviation") or 3) * 0.15)
    rel = it.get("relevance")
    if isinstance(rel, (int, float)):
        w *= 0.4 + 0.6 * max(0.0, min(1.0, float(rel)))
    return w


def _vote(items: list[dict]) -> tuple[float, int, int, int, float]:
    total_w, wsum = 0.0, 0.0
    bull = bear = neutral = 0
    for it in items:
        w = _weight(it)
        v = _SENTI_SCORE.get(it.get("sentiment"), 0.0)
        wsum += v * w
        total_w += w
        if v > 0:
            bull += 1
        elif v < 0:
            bear += 1
        else:
            neutral += 1
    score = round(wsum / total_w * 100, 1) if total_w else 0.0   # -100 ~ 100
    return score, bull, bear, neutral, round(total_w, 2)


def _level(score: float) -> str:
    if score >= 45:
        return "strong_bullish"
    if score > 15:
        return "bullish"
    if score <= -45:
        return "strong_bearish"
    if score < -15:
        return "bearish"
    return "neutral"


def aggregate_sentiment(items: list[dict],
                        min_relevance: float = DEFAULT_MIN_RELEVANCE) -> dict:
    """把逐条情绪分聚合成消息面信号。

    同时给出两个口径：
    - score：只统计"与本标的相关"的情报（真正的标的舆情）
    - universe_score：全量情报（近似全市场情绪）
    两者之差 divergence 说明该标的舆情是"独立驱动"还是"随大盘漂"。
    """
    scored = [it for it in items if it.get("sentiment")]
    if not scored:
        return {"score": 0.0, "level": "neutral", "count": 0, "bull": 0, "bear": 0,
                "neutral": 0, "top": [], "coverage": 0.0, "universe_score": 0.0,
                "divergence": 0.0, "min_relevance": min_relevance,
                "relevant_count": 0, "dropped_count": 0}
    relevant = [it for it in scored if it.get("relevant", True) and
                float(it.get("relevance", 1.0) or 0) >= min_relevance]
    base = relevant or scored          # 全被筛掉时退回全量，避免信号直接归零
    score, bull, bear, neutral, _ = _vote(base)
    uni_score, *_ = _vote(scored)
    # 重点条目 = 相关度 × 强度 × 偏离度最高的前 3 条
    top = sorted(scored, key=lambda x: -(
        (x.get("intensity", 3) * x.get("deviation", 3)) * (0.5 + float(x.get("relevance", 0.5) or 0.5))
    ))[:3]
    return {
        "score": score, "level": _level(score), "count": len(base),
        "bull": bull, "bear": bear, "neutral": neutral,
        "coverage": round(len(relevant) / len(scored), 3) if scored else 0.0,
        "relevant_count": len(relevant), "dropped_count": len(scored) - len(relevant),
        "universe_score": uni_score,
        "divergence": round(score - uni_score, 1),
        "min_relevance": min_relevance,
        "top": [{"title": t.get("title", ""), "sentiment": t.get("sentiment"),
                 "intensity": t.get("intensity"), "deviation": t.get("deviation"),
                 "relevance": t.get("relevance")} for t in top],
    }
