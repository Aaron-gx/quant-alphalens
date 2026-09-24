"""舆情相关性筛选：判断一条情报到底与"这个标的"有没有关系，值不值得进模型。

为什么需要它
------------
新闻接口给的是"沾边就推"：个股新闻里混着大盘快讯，基金/ETF 干脆只能拿到宏观面，
所以旧版本的舆情信号其实是"市场情绪"而不是"这个基金的舆情"。本模块把舆情拆成
三层并给每条情报算一个 0~1 的相关度：

1. 直接相关（1.0/0.95）：命中标的代码或名称 —— 就是这条标的自己的消息；
2. 间接相关（0.5~0.85）：命中重仓股/成分股/基金公司 —— 基金的真实风险暴露；
3. 环境相关（0.15~0.35）：跟踪指数与主题词（0.6）、宏观政策（0.35）、泛市场快讯（0.15）。

相关度同时决定"是否入选"（门槛）与"投票权重"（加权聚合），
再由 LLM 情绪打分里的 relevance 维度做二次修正，两者取高（召回优先，宁可多留）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 默认入选门槛：低于此值的情报仍会留档，但不参与信号聚合
DEFAULT_MIN_RELEVANCE = 0.30
# 语域兜底分值
SCORE_MACRO = 0.35          # 财新宏观/政策面：影响所有标的的环境信息
SCORE_RESEARCH = 0.45       # 研报评级：通常与该标的强相关
SCORE_FLASH = 0.15          # 7×24 泛快讯：未命中关键词时视为噪声
SCORE_UNRELATED = 0.20


@dataclass
class TargetProfile:
    """标的画像：判断舆情相关性的依据。"""
    symbol: str = ""
    name: str = ""
    asset_type: str = "stock"
    label: str = ""
    keywords: list[str] = field(default_factory=list)        # 强关联词（主题/跟踪指数）
    related: dict[str, float] = field(default_factory=dict)  # 间接关联标的 -> 权重
    holdings: list[dict] = field(default_factory=list)       # 基金重仓股（用于展示）

    def describe(self) -> str:
        parts = [self.label or f"{self.name}({self.symbol})"]
        if self.keywords:
            parts.append(f"主题词：{'、'.join(self.keywords)}")
        if self.related:
            top = sorted(self.related.items(), key=lambda x: -x[1])[:6]
            parts.append("关联标的：" + "、".join(k for k, _ in top))
        return "；".join(parts)


def _norm(s) -> str:
    return re.sub(r"[\s\-—_·]+", "", str(s or "")).lower()


def build_profile(symbol: str, name: str = "", asset_type: str = "stock",
                  *, fund_profile: dict | None = None,
                  extra_keywords: list[str] | None = None) -> TargetProfile:
    """构造标的画像。

    fund_profile 传入 core.data.fund.fetch_fund_profile() 的结果时，
    会用重仓股与跟踪指数把"基金舆情范围"补齐 —— 这是基金舆情可信的关键。
    """
    p = TargetProfile(symbol=symbol, name=name, asset_type=asset_type,
                      label=f"{name}({symbol})" if name else symbol)
    if extra_keywords:
        for kw in extra_keywords:
            if kw and kw not in p.keywords:
                p.keywords.append(kw)

    if fund_profile:
        for kw in (fund_profile.get("tracking") or {}).get("themes", []):
            if kw not in p.keywords:
                p.keywords.append(kw)
        # 重仓股：按占净值比映射 0.55~0.85 的关联权重（越重仓越相关）
        for h in (fund_profile.get("holdings") or [])[:12]:
            hname, w = h.get("name", ""), float(h.get("weight") or 0)
            if hname and len(hname) >= 2:
                p.related[hname] = round(min(0.85, 0.55 + w / 100 * 3), 2)
                p.holdings.append({"name": hname, "code": h.get("code", ""), "weight": w})
        company = (fund_profile.get("basic") or {}).get("company", "")
        if company:
            p.related[company] = 0.5
    return p


def score_item(item: dict, profile: TargetProfile) -> float:
    """单条情报对标的的相关度 0~1。"""
    text = _norm(f"{item.get('title', '')} {str(item.get('content') or '')[:300]}")
    if not text:
        return 0.1
    if profile.symbol and profile.symbol in text:
        return 1.0
    name = _norm(profile.name)
    if len(name) >= 2 and name in text:
        return 0.95
    best = 0.0
    for kw, w in profile.related.items():
        if len(kw) >= 2 and _norm(kw) in text:
            best = max(best, float(w))
    for kw in profile.keywords:
        if len(kw) >= 2 and _norm(kw) in text:
            best = max(best, 0.6)
    if best:
        return round(best, 2)

    # 没命中任何词：按来源语域兜底判断
    if str(item.get("symbol") or "") and str(item.get("symbol")) == profile.symbol:
        return 0.5                     # 接口本身是按该代码取的个股新闻
    cat, src = str(item.get("category") or ""), str(item.get("source") or "")
    if cat == "research":
        return SCORE_RESEARCH
    if cat == "macro" or src in ("caixin",):
        return SCORE_MACRO
    if cat == "flash":
        return SCORE_FLASH
    return SCORE_UNRELATED


def llm_relevance(item: dict) -> float | None:
    """情绪打分里的 relevance 维度（1-5）换算成 0~1；未打分返回 None。"""
    v = item.get("relevance_llm")
    if v in (None, "", 0):
        return None
    try:
        return round(max(0.0, min(1.0, (float(v) - 1) / 4)), 2)
    except (TypeError, ValueError):
        return None


def annotate(items: list[dict], profile: TargetProfile,
             min_relevance: float = DEFAULT_MIN_RELEVANCE) -> dict:
    """就地给每条情报写入 relevance 与 relevant 标记，返回筛选统计。

    规则分与 LLM 分取高：任一维度认为相关就保留（召回优先），
    权重则取两者加权（0.6 规则 + 0.4 LLM），避免把噪声权重抬得太高。
    """
    kept, dropped = 0, 0
    for it in items:
        rule = score_item(it, profile)
        llm = llm_relevance(it)
        it["relevance_rule"] = rule
        it["relevance"] = round(max(rule, llm), 2) if llm is not None else rule
        it["relevant"] = bool(it["relevance"] >= min_relevance)
        if it["relevant"]:
            kept += 1
        else:
            dropped += 1
    return {"total": len(items), "kept": kept, "dropped": dropped,
            "coverage": round(kept / len(items), 3) if items else 0.0,
            "min_relevance": min_relevance, "profile": profile.describe()}


def filter_relevant(items: list[dict], profile: TargetProfile,
                    min_relevance: float = DEFAULT_MIN_RELEVANCE) -> list[dict]:
    """返回相关情报子集（仅筛选，不修改入参）。"""
    return [it for it in items if score_item(it, profile) >= min_relevance]
