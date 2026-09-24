"""预测引擎主流水线。

数据组装 →（可选分层 LLM 分析）→ 综合裁决输出结构化 JSON → 相关性/校准后处理 →
pydantic 校验 → 写入 predictions 表 → 返回 Prediction。

mode:
- fast: 数据摘要一次喂给综合裁决模型（1 次调用，省 token）
- deep: 技术面/新闻面先各自调 LLM 出小结，再综合（3 次调用，更准更贵）

后处理两层（本项目新增，解决"舆情不相关 + 概率不像概率"两个问题）：
- 舆情相关性筛选：基金/ETF 先推导真实舆情范围（重仓股 + 跟踪指数），再按相关度
  筛选与加权，避免用大盘情绪冒充基金舆情（core.intel.relevance）
- 自学习校准：用到期对账样本把概率校准到"可以当概率用"，并把证据留痕
  （core.predict.calibration），没有样本时如实标注未校准
"""
from __future__ import annotations

import json
from datetime import datetime, date

from core.config import get
from core.data import market as market_data
from core.data import stock as stock_data
from core.data import fund as fund_data
from core.data.indicators import latest_snapshot, recent_bars_text
from core.intel import news as news_data
from core.intel import relevance as relevance_mod
from core.intel.relevance import DEFAULT_MIN_RELEVANCE
from core.intel.sentiment import score_items, aggregate_sentiment
from core.predict.llm import LLMClient, LLMError
from core.predict.prompts import (
    COMPREHENSIVE_SYSTEM, COMPREHENSIVE_USER, TECH_ANALYSIS_PROMPT,
    NEWS_ANALYSIS_PROMPT, FUND_SCOPE_NOTE, get_core_principles, disclaimer_text,
)
from core.predict.schema import PredictionResult, PREDICTION_JSON_INSTRUCTION
from core.store.db import session_scope
from core.store.models import Prediction, AnalysisReport

ASSET_DESC = {
    "stock": "A股个股", "fund": "场内ETF/LOF", "ofund": "场外基金",
    "hk": "港股(港股通)", "index": "大盘指数",
}


def _resolve_name(symbol: str, quote: dict) -> str:
    return quote.get("name") or symbol


def build_news_scope(symbol: str, name: str, asset_type: str) -> dict:
    """推导该标的的真实舆情范围。

    - 股票/港股：就是它自己
    - 基金/ETF：先取基金档案（重仓股 + 跟踪指数 + 主题词），再看这些关联标的的消息
      —— 这是"舆情跟这个基金到底有没有关系"的技术实现
    """
    fund_profile = None
    if asset_type in ("fund", "ofund"):
        try:
            fund_profile = fund_data.fetch_fund_profile(symbol, asset_type, with_holdings=True)
        except Exception:
            fund_profile = None
    scope = (fund_data.fund_news_scope(fund_profile) if fund_profile
             else {"symbols": [], "keywords": [], "label": ""})
    profile = relevance_mod.build_profile(
        symbol, name, asset_type, fund_profile=fund_profile,
        extra_keywords=scope.get("keywords"))
    label = scope.get("label") or f"{ASSET_DESC.get(asset_type, '标的')} {name}({symbol})"
    return {
        "fund_profile": fund_profile, "scope": scope, "profile": profile,
        "label": label,
        "extra_symbols": [s["symbol"] for s in scope.get("symbols", [])][:4],
        "is_fund": asset_type in ("fund", "ofund"),
    }


def build_data_block(symbol: str, name: str, asset_type: str, quote: dict,
                     kline_df, news_items: list[dict], news_signal: dict,
                     market_ctx: dict | None, basics: dict | None,
                     news_scope: dict | None = None,
                     calib: dict | None = None) -> str:
    """把各维度数据拼成给 LLM 看的文本块。"""
    snap = latest_snapshot(kline_df)
    parts = [
        f"## 实时行情\n{json.dumps({k: v for k, v in quote.items() if k != 'ok'}, ensure_ascii=False, default=str)}",
        f"## 技术指标快照\n{json.dumps(snap, ensure_ascii=False)}",
        f"## 近15根K线\n{recent_bars_text(kline_df)}",
    ]
    if news_scope and news_scope.get("is_fund") and news_scope.get("fund_profile"):
        fp = news_scope["fund_profile"]
        brief = {
            "基金名称": fp.get("name"), "类型": (fp.get("basic") or {}).get("type"),
            "基金经理": (fp.get("basic") or {}).get("manager"),
            "基金公司": (fp.get("basic") or {}).get("company"),
            "规模": (fp.get("basic") or {}).get("size"),
            "阶段收益%": fp.get("stage_returns"),
            "重仓股占净值%": fp.get("top10_weight"),
            "跟踪指数/主题": (fp.get("tracking") or {}).get("themes"),
        }
        parts.append(f"## 基金档案\n{json.dumps(brief, ensure_ascii=False, default=str)}")
        parts.append(FUND_SCOPE_NOTE.format(scope=news_scope["label"]))
    if basics:
        parts.append(f"## 基本面\n{json.dumps(basics, ensure_ascii=False, default=str)[:2000]}")
    if news_items:
        lines = []
        for it in news_items[:12]:
            tag = (f"[{it.get('sentiment', '未评分')}|强度{it.get('intensity', '-')}"
                   f"|偏离{it.get('deviation', '-')}|相关{it.get('relevance', '-')}]")
            lines.append(f"- {tag} {it.get('title', '')} ({str(it.get('publish_time', ''))[:16]})")
        parts.append("## 近期舆情（按相关度排序，含情绪打分）\n" + "\n".join(lines))
        parts.append(f"## 消息面聚合信号（已按相关度筛选加权）\n"
                     f"{json.dumps(news_signal, ensure_ascii=False, default=str)[:1200]}")
    if market_ctx:
        brief = {
            "市场情绪": market_ctx.get("sentiment"),
            "涨跌家数": {k: market_ctx["breadth"].get(k) for k in ("up", "down", "limit_up", "limit_down")} if market_ctx.get("breadth") else None,
            "主力净流入(亿)": round(market_ctx["fund_flow"].get("main_net_inflow", 0) / 1e8, 2) if market_ctx.get("fund_flow", {}).get("ok") else None,
            "融资余额周变化(亿)": round(market_ctx["margin"].get("weekly_change", 0) / 1e8, 2) if market_ctx.get("margin", {}).get("ok") else None,
        }
        parts.append(f"## 大盘环境\n{json.dumps(brief, ensure_ascii=False)}")
    if calib:
        # 由 calibration.brief() 生产的证据块：命中率 vs 多数类/随机双基准、
        # 置信度分桶命中率、实际方向分布、校准后 BSS。
        # 刻意原样透传（含"无样本"的说明），不做美化——模型需要看到真实基准率。
        parts.append("## 引擎历史表现（真实到期对账统计，用于判断本次概率该信几分）\n"
                     + json.dumps(calib, ensure_ascii=False, default=str)[:1800])
    return "\n\n".join(parts)


def _layered_summaries(client: LLMClient, symbol: str, name: str,
                       data_block: str, news_items: list[dict]) -> str:
    """deep 模式：技术面、新闻面先各自出小结。"""
    layers = []
    try:
        tech = client.ask(TECH_ANALYSIS_PROMPT.format(name=name, symbol=symbol)
                          + "\n\n数据：\n" + data_block[:4000])
        layers.append(f"## 技术面小结\n{tech}")
    except LLMError as e:
        layers.append(f"## 技术面小结\n（生成失败: {e}）")
    if news_items:
        news_text = "\n".join(
            f"- [{it.get('sentiment','')}|{it.get('intensity','')}|{it.get('deviation','')}] "
            f"{it.get('title','')}" for it in news_items[:10])
        try:
            na = client.ask(NEWS_ANALYSIS_PROMPT.format(name=name, symbol=symbol)
                            + "\n\n新闻列表：\n" + news_text)
            layers.append(f"## 消息面小结\n{na}")
        except LLMError as e:
            layers.append(f"## 消息面小结\n（生成失败: {e}）")
    return "\n\n".join(layers)


# ---------------------------------------------------------------------------
# 编排层：把「数据组装 → 量化融合 → 自学习校准 → 落库」拆成具名步骤
#   原先 predict() 一个函数里揉着五件事、300+ 行，任一步出问题都难定位；
#   拆开后每个阶段可以单独看、单独测。
# ---------------------------------------------------------------------------

def build_calib_brief() -> dict:
    """给 LLM 的「引擎历史表现」证据块（无到期样本的周期直接省略，不硬凑）。

    这是学习闭环的最后一厘米：对账与校准的结果必须**回流到决策输入**，
    否则模型永远不知道"自己上次在同类情形下错在哪"。
    """
    try:
        from core.predict import calibration
        brief = calibration.brief()
        return {h: v for h, v in brief.items() if v.get("到期对账样本数")}
    except Exception:
        return {}


def _apply_quant_fusion(result: PredictionResult, symbol: str, kline) -> dict:
    """逐周期量化交叉验证，就地改写 result.horizons / confidence。

    返回 {"engine": "llm"|"fused", "quant": {...}, "fusion": {...}}，
    全部留痕进 input_snapshot，便于事后回答"这次到底融没融、凭什么这么融"。

    原实现写死 `if quant_out and "next_day" in result.horizons`，
    于是 one_week / one_month / quarter 三个周期从来没有量化交叉验证，
    UI 与库里那三格是 LLM 的裸主观概率且看不出区别。
    """
    out: dict = {"engine": "llm", "quant": {}, "fusion": {}}
    if not get("predict", "use_quant", True):
        return out
    try:
        from core.predict import quant as quant_engine
        quant_map = quant_engine.predict_multi(symbol, kline)
    except Exception:
        return out
    out["quant"] = quant_map

    diverged: list[str] = []
    applied = False
    for key, h in result.horizons.items():
        q = quant_map.get(key)
        if not q:
            continue
        before = {"up": h.up, "flat": h.flat, "down": h.down}
        fused = quant_engine.fuse(before, q, horizon=key)
        detail = dict(fused.get("_fusion") or {})
        detail["llm_probs"] = before
        detail["quant_probs"] = {k: q.get(k) for k in ("up", "flat", "down")}
        out["fusion"][key] = detail
        applied = applied or bool(detail.get("applied"))
        # 双引擎分歧本身就是信号（权重为 0 时 fused == before，分歧必然为 0）
        if abs(fused["up"] - before["up"]) + abs(fused["down"] - before["down"]) > 0.4:
            diverged.append(key)
        h.up, h.flat, h.down = fused["up"], fused["flat"], fused["down"]
    # 四周期累计扣分会让置信度直接归零，所以只扣一次，但把分歧周期如实记下来
    if diverged:
        result.confidence = max(0, result.confidence - 15)
        out["fusion"]["_diverged"] = diverged
    if applied:
        out["engine"] = "fused"
    return out


def _apply_calibration(result: PredictionResult, *, engine_tag: str,
                       news_signal: dict, symbol: str, asset_type: str) -> dict:
    """用到期对账样本把概率校准成"可当概率用"，并留下证据。

    ⚠️ symbol/asset_type 必须传：分层校准（单标的 → 资产类型 → 全局）
    全靠这两个参数选层。原实现没传，于是 `stratum_for()` 永远只返回全局池，
    分层校准写好了却从未生效——同质样本的精度优势一点没拿到。
    """
    if not get("predict", "use_calibration", True):
        return {}
    from core.predict import calibration
    reliability = {h: calibration.reliability_table(h) for h in result.horizons}
    calib_info: dict = {}
    for key, h in result.horizons.items():
        before = {"up": round(h.up, 3), "flat": round(h.flat, 3),
                  "down": round(h.down, 3)}
        cal = calibration.apply(
            before, horizon=key, engine=engine_tag,
            senti_score=news_signal.get("score", 0), confidence=result.confidence,
            relevance_cov=news_signal.get("coverage", 0),
            symbol=symbol, asset_type=asset_type)
        h.up, h.flat, h.down = (cal["probs"]["up"], cal["probs"]["flat"],
                                cal["probs"]["down"])
        bucket = f"{int(result.confidence // 20 * 20)}-{int(result.confidence // 20 * 20) + 19}"
        hit = next((r["实际命中率"] for r in reliability.get(key, [])
                    if r["置信度区间"] == bucket), None)
        calib_info[key] = {
            "method": cal["method"], "n": cal["n"], "scope": cal.get("scope"),
            "stratum": cal.get("stratum"), "trained_at": cal.get("trained_at", ""),
            "bss_after": cal.get("bss_after"),
            "probs_before": before,
            "probs_after": {"up": round(h.up, 3), "flat": round(h.flat, 3),
                            "down": round(h.down, 3)},
            "evidence": cal.get("evidence", {}),
            "reliability": hit,
        }
    return calib_info


def predict(symbol: str, *, user_opinion: str = "", risk_preference: str | None = None,
            custom_principles: str = "", mode: str | None = None,
            asset_type_hint: str | None = None,
            client: LLMClient | None = None) -> Prediction:
    """对单个标的跑一次完整预测，返回已落库的 Prediction 记录。

    symbol 支持：6位A股/基金代码、5位或HK前缀港股、指数名（如"上证指数"）。
    未显式指定风险偏好时，自动读取用户画像。
    """
    from core.profile import get_profile, profile_prompt_block
    profile = get_profile()
    risk_preference = risk_preference or profile.get("risk_preference") or "neutral"
    custom_principles = custom_principles or profile.get("custom_principles", "")
    mode = mode or get("predict", "mode", "fast")
    client = client or LLMClient()
    cfg_days = int(get("predict", "news_days", 7))
    cfg_limit = int(get("predict", "news_limit", 10))
    # K 线长度：**不是给 LLM 看的**（LLM 只看近 15 根），而是给量化引擎做滚动样本外
    # 验证用的。120 根时有效样本只有约 60 行（ma60 吃掉前 59 根），
    # walk-forward 的 min_train=120 永远进不去 → 样本外指标恒为空 →
    # 融合权重被门控恒判为 0：**量化引擎等于从没参与过预测**，而日志上一切正常。
    hist_count = max(120, int(get("predict", "kline_count", 400)))

    # ---- 1. 数据组装 ----
    if symbol in market_data.INDEX_CODE_MAP:      # 指数（按名称传）
        asset_type = "index"
    else:
        info = stock_data.classify_symbol(symbol)
        asset_type = info["asset_type"]
        if asset_type == "auto":                  # 号段重叠：探测后确定
            asset_type = stock_data.resolve_auto(symbol)
        # 场外基金与 A 股代码段重叠（如 005827 同时也是深市股票号段）：
        # 只有调用方明确提示是基金时才按基金处理，不能反向探测——
        # 探测会把真实股票（如 000997 撞名"南方双元A"基金）误判成基金。
        if asset_type_hint in ("fund", "ofund"):
            asset_type = asset_type_hint

    if asset_type == "index":
        spot = market_data.fetch_index_spot().get("indices", {}).get(symbol, {})
        quote = {"ok": bool(spot), "name": symbol, "price": spot.get("price", 0),
                 "change_pct": spot.get("change_pct", 0)}
        kline = market_data.fetch_index_kline(symbol, count=hist_count)
        basics = None
    elif asset_type == "ofund":
        quote = fund_data.fetch_fund_realtime_estimate(symbol)
        if not quote.get("ok"):
            quote = {"ok": True, "price": None}
        kline = fund_data.fetch_fund_nav_history(symbol, count=max(hist_count, 250))
        basics = fund_data.fetch_fund_basic_info(symbol)
    else:
        quote = stock_data.fetch_realtime_quote(symbol)
        kline = stock_data.fetch_kline(symbol, count=hist_count)
        basics = stock_data.fetch_stock_basics(symbol) if asset_type == "stock" else None

    name = _resolve_name(symbol, quote if quote.get("ok") else {})
    if not name or name == symbol:
        name = (basics or {}).get("name", symbol)

    # 大盘环境：后台线程并行拉取（与舆情链路重叠，避免其耗时叠加到总时长）
    from concurrent.futures import ThreadPoolExecutor
    _market_ex = ThreadPoolExecutor(max_workers=1)
    try:
        _market_fut = _market_ex.submit(market_data.market_overview)
    except Exception:
        _market_fut = None

    # 舆情范围推导（基金/ETF 会展开成"重仓股 + 跟踪指数"的舆情范围）
    news_scope = build_news_scope(symbol, name, asset_type)
    news_symbol = symbol if asset_type in ("stock", "hk") else ""
    news_items = news_data.collect_news(
        news_symbol, days=cfg_days, limit=cfg_limit,
        extra_symbols=news_scope["extra_symbols"])
    target_desc = news_scope["label"]

    # 舆情相关性筛选：先按规则算相关度 → 让 LLM 只给最相关的条目打情绪分 → 合并 → 加权聚合
    min_rel = float(get("intel", "min_relevance", DEFAULT_MIN_RELEVANCE))
    rel_stats = relevance_mod.annotate(news_items, news_scope["profile"], min_rel)
    news_items.sort(key=lambda it: (-float(it.get("relevance") or 0),
                                    str(it.get("publish_time") or "")))
    if client.available():
        score_items(news_items, target_desc, client=client, limit=cfg_limit)
        rel_stats = relevance_mod.annotate(news_items, news_scope["profile"], min_rel)
    news_signal = aggregate_sentiment(news_items, min_relevance=min_rel)

    # 收拢后台并行的大盘数据
    market_ctx = None
    if _market_fut is not None:
        try:
            market_ctx = _market_fut.result(timeout=30)
        except Exception:
            market_ctx = None
        _market_ex.shutdown(wait=False)

    calib_brief = build_calib_brief()
    data_block = build_data_block(symbol, name, asset_type, quote, kline,
                                  news_items, news_signal, market_ctx, basics,
                                  news_scope=news_scope, calib=calib_brief)

    # ---- 2. 分层小结（deep 模式） ----
    layers_desc = "行情、技术指标、新闻情绪、基本面和大盘环境"
    if mode == "deep" and client.available():
        layered = _layered_summaries(client, symbol, name, data_block, news_items)
        data_block += "\n\n# 前置专项分析\n" + layered
        layers_desc += "，以及技术面/消息面专项小结"

    # ---- 3. 综合裁决 ----
    user_block = profile_prompt_block(profile, user_opinion)
    if user_block:
        user_block = "\n" + user_block
    system = COMPREHENSIVE_SYSTEM.format(
        layers_desc=layers_desc, name=name, symbol=symbol,
        asset_desc=ASSET_DESC.get(asset_type, "标的"),
        core_principles=get_core_principles(risk_preference, custom_principles),
        disclaimer=disclaimer_text())
    user_msg = COMPREHENSIVE_USER.format(name=name, symbol=symbol,
                                       data_block=data_block, user_block=user_block)
    user_msg += "\n\n" + PREDICTION_JSON_INSTRUCTION

    if not client.available():
        raise LLMError("LLM 未配置，无法生成预测")
    raw_json = client.ask_json(user_msg, system=system, model_type="reasoner",
                               caller="predict")
    result = PredictionResult.model_validate(raw_json).normalized()

    # ---- 3.5 量化引擎交叉验证（可选，config predict.use_quant=true） ----
    qs = _apply_quant_fusion(result, symbol, kline)
    engine_tag = qs["engine"]

    # ---- 3.6 自学习校准：把概率校准到"可当概率用"，并留下证据 ----
    calib_info = _apply_calibration(result, engine_tag=engine_tag,
                                    news_signal=news_signal,
                                    symbol=symbol, asset_type=asset_type)

    # ---- 4. 落库 ----
    base_price = quote.get("price") or (float(kline.iloc[-1]["close"]) if len(kline) else 0.0)
    with session_scope() as s:
        pred = Prediction(
            symbol=symbol, name=name, asset_type=asset_type,
            base_date=date.today(), base_price=float(base_price or 0),
            engine=engine_tag, mode=mode,
            horizons={k: h.model_dump() for k, h in result.horizons.items()},
            scenarios={k: sc.model_dump() for k, sc in result.scenarios.items()},
            confidence=result.confidence,
            key_factors=result.key_factors, action=result.action.model_dump(),
            risks=result.risks, report_text=result.summary,
            input_snapshot={"quote": {k: v for k, v in quote.items() if k != "ok"},
                            "news_signal": news_signal,
                            "news_relevance": {k: v for k, v in rel_stats.items()
                                               if k != "profile"},
                            "news_scope": news_scope["label"],
                            "news_scope_symbols": news_scope["extra_symbols"],
                            "calibration": calib_info,
                            "quant": qs["quant"],          # 逐周期量化概率（含样本外 BSS）
                            "quant_fusion": qs["fusion"],  # 逐周期融合权重与门控依据
                            "tech": latest_snapshot(kline)},
            llm_model=client.reasoner_model,
        )
        s.add(pred)
        s.flush()
        s.add(AnalysisReport(symbol=symbol, name=name,
                             report_type="comprehensive",
                             content=result.summary or json.dumps(raw_json, ensure_ascii=False),
                             prediction_id=pred.id))
        s.refresh(pred)
        pred_id = pred.id
        # 让返回对象脱离 session 也可用
        s.expunge(pred)
    return get_prediction(pred_id)


def get_prediction(pred_id: int) -> Prediction | None:
    from sqlalchemy.orm import selectinload
    with session_scope() as s:
        p = (s.query(Prediction).options(selectinload(Prediction.evals))
             .filter_by(id=pred_id).first())
        if p:
            _ = list(p.evals)  # 触发加载后再脱钩
            s.expunge(p)
        return p


def list_predictions(symbol: str | None = None, limit: int = 50) -> list[Prediction]:
    with session_scope() as s:
        q = s.query(Prediction).order_by(Prediction.id.desc())
        if symbol:
            q = q.filter(Prediction.symbol == symbol)
        rows = q.limit(limit).all()
        for r in rows:
            s.expunge(r)
        return rows
