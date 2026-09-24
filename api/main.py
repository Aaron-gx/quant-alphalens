"""FastAPI 服务层：供聊天机器人/外部系统调用。

启动：uvicorn api.main:app --host 0.0.0.0 --port 8000
鉴权：config.toml [api].api_key 非空时，请求需带 X-API-Key 头。
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import get, load_config, save_llm_config
from core.predict.pipeline import predict, list_predictions, get_prediction
from core.predict.llm import LLMClient, usage_summary, test_connection
from core.verify.evaluator import evaluate_due, accuracy_report, recent_evaluations
from core.predict.labels import HORIZONS, due_dates_for
from core.paper import engine as paper
from core.paper import auto_trade
from core.intel import news as news_data
from core.data import market as market_data
from core.profile import get_profile, save_profile
from core.store.db import session_scope
from core.store.models import AnalysisReport, Watchlist, Prediction, PredictionEval, PaperAccount

app = FastAPI(title="基金股票AI预测引擎", version="0.1.0",
              description="核心引擎的 REST 封装，供 Web/机器人/定时任务调用")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_prefix_middleware(request, call_next):
    if request.url.path.startswith("/api/"):
        request.scope["path"] = request.url.path[4:]
    return await call_next(request)


def _auth(x_api_key: str | None = Header(default=None)):
    required = get("api", "api_key", "")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="invalid api key")


def _resolve_asset_type(symbol: str, hint: str | None = None) -> str:
    """解析资产类型。场外基金与股票代码段重叠（005827 等），
    显式 hint（前端自选池里存的 asset_type）优先；无 hint 时按代码规则识别。"""
    from core.data import stock as stock_data
    if hint in ("stock", "fund", "ofund", "hk", "bse", "index"):
        return hint
    t = stock_data.classify_symbol(symbol)["asset_type"]
    if t == "auto":
        t = stock_data.resolve_auto(symbol)
    return t


# ---------- 请求模型 ----------
class PredictReq(BaseModel):
    symbol: str
    user_opinion: str = ""
    risk_preference: str = "neutral"
    mode: str | None = None
    asset_type: str | None = None   # 自选池里存的类型提示，用于歧义代码段（如 005827）


class BuyReq(BaseModel):
    symbol: str
    volume: int | None = None
    amount: float | None = None
    name: str = ""
    reason: str = ""


class SellReq(BaseModel):
    symbol: str
    volume: int | None = None
    ratio: float | None = Field(default=None, ge=0, le=1)
    reason: str = ""


class AccountReq(BaseModel):
    name: str
    initial_cash: float | None = None
    mode: str = "manual"
    note: str = ""


class ProfileReq(BaseModel):
    risk_preference: str | None = None
    custom_principles: str | None = None
    position_text: str | None = None
    trade_style: str | None = None
    common_mistakes: str | None = None


class WatchlistReq(BaseModel):
    symbol: str
    name: str = ""
    asset_type: str = ""


class LLMConfigReq(BaseModel):
    base_url: str
    api_key: str
    model: str
    reasoner_model: str = ""
    temperature: float = 0.3


# ---------- 路由 ----------
@app.get("/health")
def health():
    return {"ok": True, "llm_configured": LLMClient().available()}


@app.get("/market/overview")
def market_overview(x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    return market_data.market_overview()


def _due_payload(base_date, horizons, memo: dict | None = None) -> dict:
    """把「各周期到期日」附在预测响应里，让前端**直接采用**，不必自己推日期。

    为什么这一步必须在后端做
    ------------------------
    到期日 = base_date 之后第 N 个**交易日**（N = 模型训练时的 horizon 步数），
    需要真实交易日历（含节假日）。前端只有"排除周末"这一档精度：
    遇到国庆/春节/调休就会算错，于是图上标的「一季 60日」落在一个日期、
    对账记录里的到期日却是另一个 —— 用户看到的是两份互相矛盾的口径。

    所以连 `due_source`（index / weekday / empty）一起给出去，
    前端可以照实披露"这个日期是严格日历算的，还是只排了周末"。
    """
    keys = [h for h in HORIZONS if h in (horizons or {})] or list(HORIZONS)
    cache_key = (str(base_date), tuple(keys))
    if memo is not None and cache_key in memo:
        return memo[cache_key]
    try:
        due, src = due_dates_for(base_date, keys)
    except Exception:
        due, src = {}, "empty"
    payload = {"due_dates": due, "due_source": src}
    if memo is not None:
        memo[cache_key] = payload
    return payload


@app.post("/predict")
def create_prediction(req: PredictReq, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    try:
        p = predict(req.symbol, user_opinion=req.user_opinion,
                    risk_preference=req.risk_preference, mode=req.mode,
                    asset_type_hint=req.asset_type)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"id": p.id, "symbol": p.symbol, "name": p.name,
            "horizons": p.horizons, "confidence": p.confidence,
            "action": p.action, "risks": p.risks, "summary": p.report_text,
            **_due_payload(p.base_date, p.horizons)}


@app.post("/predict/async")
def create_prediction_async(req: PredictReq, bg: BackgroundTasks,
                            x_api_key: str | None = Header(default=None)):
    """异步发起预测（机器人场景：先回执，结果稍后查）。"""
    _auth(x_api_key)
    bg.add_task(predict, req.symbol, user_opinion=req.user_opinion,
                risk_preference=req.risk_preference, mode=req.mode,
                asset_type_hint=req.asset_type)
    return {"accepted": True, "symbol": req.symbol}


@app.get("/predictions")
def predictions(symbol: str | None = None, limit: int = 20,
                x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    rows = list_predictions(symbol, limit)
    memo: dict = {}
    return [{"id": p.id, "symbol": p.symbol, "name": p.name, "base_date": str(p.base_date),
             "confidence": p.confidence, "horizons": p.horizons,
             "summary": p.report_text,
             **_due_payload(p.base_date, p.horizons, memo)} for p in rows]


@app.get("/predictions/{pid}")
def prediction_detail(pid: int, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    p = get_prediction(pid)
    if not p:
        raise HTTPException(404, "not found")
    return {"id": p.id, "symbol": p.symbol, "name": p.name,
            "base_date": str(p.base_date), "base_price": p.base_price,
            "horizons": p.horizons, "scenarios": p.scenarios,
            "confidence": p.confidence, "key_factors": p.key_factors,
            "action": p.action, "risks": p.risks, "summary": p.report_text,
            # 口径留痕：前端要据此判断"概率到底有没有被校准"，
            # identity 校准（无样本时的恒等映射）不能对外宣称"已校准"。
            "asset_type": p.asset_type, "engine": p.engine, "mode": p.mode,
            "llm_model": p.llm_model, "created_at": str(p.created_at),
            "input_snapshot": p.input_snapshot,
            # 各周期到期日（交易日口径）+ 日历来源，前端图上标注直接用它
            **_due_payload(p.base_date, p.horizons),
            "evals": [{"horizon": e.horizon, "actual_change_pct": e.actual_change_pct,
                       "hit": e.hit, "brier": e.brier_score} for e in p.evals]}


@app.post("/verify/run")
def run_evaluation(x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    return {"evaluated": evaluate_due()}


@app.get("/verify/report")
def verify_report(x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    return accuracy_report()


@app.get("/verify/logs")
def verify_logs(limit: int = 20, x_api_key: str | None = Header(default=None)):
    """最近的对账流水明细。前端「履约核销对账流水」表用真实记录渲染，
    没有记录时返回空列表，由前端显示空态（不再有硬编码的演示流水）。"""
    _auth(x_api_key)
    items = recent_evaluations(limit)
    return {"items": items, "total": len(items)}


_news_cache: dict[str, tuple[float, dict]] = {}
# 同 key 串行锁：/news 会采集新闻+调 LLM 打分（秒级），前端并发重复请求必须合并成一次计算
import threading as _threading
_news_locks: dict[str, _threading.Lock] = {}
_news_locks_guard = _threading.Lock()


@app.get("/news/{symbol}")
def news(symbol: str, days: int = 7, force_refresh: bool = False,
         asset_type: str | None = None, x_api_key: str | None = Header(default=None)):
    """标的舆情：自动按"基金舆情范围"抓取并标注相关度（含情绪打分，默认5分钟内存缓存）。"""
    _auth(x_api_key)
    cache_key = f"{symbol}:{days}:{asset_type or ''}"
    now = time.time()
    if not force_refresh and cache_key in _news_cache:
        ts, data = _news_cache[cache_key]
        if (now - ts) < 300:
            return data

    # 同 key 加锁重算：并发的重复请求在锁内命中缓存直接返回，不会重复跑采集+LLM
    with _news_locks_guard:
        lock = _news_locks.setdefault(cache_key, _threading.Lock())
    with lock:
        now = time.time()
        if not force_refresh and cache_key in _news_cache:
            ts, data = _news_cache[cache_key]
            if (now - ts) < 300:
                return data

        from core.intel import relevance as relevance_mod
        from core.intel.sentiment import aggregate_sentiment, score_items
        from core.predict.pipeline import build_news_scope

        if symbol in market_data.INDEX_CODE_MAP:
            at = "index"
        else:
            at = _resolve_asset_type(symbol, asset_type)
        scope = build_news_scope(symbol, "", at)
        items = news_data.collect_news(
            symbol if at in ("stock", "hk") else "",
            days=days, limit=10, extra_symbols=scope["extra_symbols"])
        score_items(items, scope["label"], limit=10)
        stats = relevance_mod.annotate(items, scope["profile"])
        result = {"count": len(items), "scope": scope["label"],
                  "relevance": {k: v for k, v in stats.items() if k != "profile"},
                  "signal": aggregate_sentiment(items),
                  "items": [{"title": i["title"], "source": i["source"],
                             "category": i.get("category"),
                             "sentiment": i.get("sentiment"),
                             "intensity": i.get("intensity"),
                             "deviation": i.get("deviation"),
                             "relevance": i.get("relevance"),
                             "relevant": i.get("relevant"),
                             "publish_time": str(i.get("publish_time"))} for i in items]}
        _news_cache[cache_key] = (now, result)
        return result


@app.get("/fund/profile")
def fund_profile(symbol: str, with_holdings: bool = True,
                 asset_type: str | None = None,
                 x_api_key: str | None = Header(default=None)):
    """基金档案：基本信息 + 净值/估值 + 阶段收益 + 重仓股 + 跟踪指数 + 舆情范围。"""
    _auth(x_api_key)
    from core.data import fund as fund_data
    from core.data import stock as stock_data
    at = _resolve_asset_type(symbol, asset_type)
    if at not in ("fund", "ofund"):
        if asset_type:
            # 调用方明确给了非基金类型（如 stock），尊重它，不做基金探测
            raise HTTPException(400, f"{symbol} 不是基金代码（识别为 {at}）")
        # 股票号段与场外基金重叠：调用意图是"查基金"，探测确认后放行；
        # 但同代码股票真实存在时以股票为准（000997=新大陆 ≠ 南方双元A 基金）
        if fund_data.probe_fund(symbol)["is_fund"]:
            sq = stock_data.fetch_realtime_quote(symbol)
            if sq.get("ok") and sq.get("name"):
                raise HTTPException(400, f"{symbol} 是股票 {sq['name']}（同名代码另有基金，需传 asset_type=ofund）")
            at = "ofund"
        else:
            raise HTTPException(400, f"{symbol} 不是基金代码（识别为 {at}）")
    pf = fund_data.fetch_fund_profile(symbol, at, with_holdings=with_holdings)
    pf["news_scope"] = fund_data.fund_news_scope(pf)
    return pf


@app.get("/quote/{symbol}")
def quote(symbol: str, asset_type: str | None = None,
          x_api_key: str | None = Header(default=None)):
    """实时行情 + 五档盘口（场外基金走盘中估值接口）。"""
    _auth(x_api_key)
    from core.data import fund as fund_data
    from core.data import stock as stock_data
    at = _resolve_asset_type(symbol, asset_type)
    if at == "ofund":
        q = fund_data.fetch_fund_realtime_estimate(symbol)
        if not q.get("ok"):
            q = fund_data.nav_quote_fallback(symbol)
        return q
    q = stock_data.fetch_depth_quote(symbol)
    if not q.get("ok"):
        # 歧义号段兜底：股票行情源全失败时探测是否其实是场外基金（如 005827）
        probe = fund_data.probe_fund(symbol)
        if probe["is_fund"]:
            f = fund_data.fetch_fund_realtime_estimate(symbol)
            if not f.get("ok"):
                f = fund_data.nav_quote_fallback(symbol, probe["name"])
            if f.get("ok"):
                f["asset_type"] = "ofund"
                return f
    return q


@app.get("/indicators/{symbol}")
def indicators(symbol: str, asset_type: str | None = None,
               x_api_key: str | None = Header(default=None)):
    """技术指标快照（MA/MACD/RSI/KDJ/BOLL/动量），基金按净值序列计算。"""
    _auth(x_api_key)
    from core.data import fund as fund_data
    from core.data import stock as stock_data
    from core.data.indicators import latest_snapshot
    at = _resolve_asset_type(symbol, asset_type)
    if at == "ofund":
        df = fund_data.fetch_fund_nav_history(symbol, count=250)
        if df.empty:
            df = stock_data.fetch_kline(symbol, count=120)
    else:
        df = stock_data.fetch_kline(symbol, count=120)
        if df.empty and fund_data.probe_fund(symbol)["is_fund"]:
            df = fund_data.fetch_fund_nav_history(symbol, count=250)
            at = "ofund"
    snap = latest_snapshot(df)
    snap["asset_type"] = at
    return snap


@app.get("/model/status")
def model_status(x_api_key: str | None = Header(default=None)):
    """自学习现状：样本量、校准包、学习曲线。"""
    _auth(x_api_key)
    from core.predict import calibration
    return calibration.status()


@app.post("/model/train")
def model_train(symbol: str | None = None, x_api_key: str | None = Header(default=None)):
    """触发重训：默认校准层全量重训；带 symbol 时重训该标的量化模型。

    ⚠️ K 线根数走 [predict].kline_count，**不要硬编码小数字**：
    purged walk-forward 需要 min_train=250 + purge(horizon) + embargo，
    根数不够时样本外指标恒为空 → 融合权重被门控恒判 0，
    而接口仍返回 ok=true —— 从外部完全看不出量化其实没生效。
    另：原实现只取股票 K 线，基金/指数标的重训必然拿不到数据。
    """
    _auth(x_api_key)
    from core.config import get
    from core.predict import calibration, quant as quant_engine
    if not symbol:
        return {"calibration": calibration.train_all(record=True)}
    from core.data import fund as fund_data
    from core.data import market as market_data
    from core.data import stock as stock_data
    count = max(400, int(get("predict", "kline_count", 700)))
    at = _resolve_asset_type(symbol, None)
    if at == "index":
        df = market_data.fetch_index_kline(symbol, count=count)
    elif at == "ofund":
        df = fund_data.fetch_fund_nav_history(symbol, count=count)
    else:
        df = stock_data.fetch_kline(symbol, count=count)
    return {"asset_type": at, "kline": (0 if df is None else int(len(df))),
            "quant": quant_engine.train(symbol, df, max_age_days=0)}


# ---------- 模拟盘 ----------
@app.post("/paper/accounts")
def create_account(req: AccountReq, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    return {"id": paper.create_account(req.name, req.initial_cash, req.mode, req.note)}


@app.get("/paper/accounts/{aid}")
def get_account(aid: int, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    try:
        pf = paper.portfolio(aid)
    except paper.TradeError as e:
        raise HTTPException(404, str(e))
    pf["stats"] = paper.perf_stats(aid)
    return pf


@app.post("/paper/accounts/{aid}/buy")
def paper_buy(aid: int, req: BuyReq, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    try:
        o = paper.buy(aid, req.symbol, volume=req.volume, amount=req.amount,
                      name=req.name, reason=req.reason)
    except paper.TradeError as e:
        raise HTTPException(400, str(e))
    return {"order_id": o.id, "side": o.side, "price": o.price,
            "volume": o.volume, "fee": o.fee}


@app.post("/paper/accounts/{aid}/sell")
def paper_sell(aid: int, req: SellReq, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    try:
        o = paper.sell(aid, req.symbol, volume=req.volume, ratio=req.ratio,
                       reason=req.reason)
    except paper.TradeError as e:
        raise HTTPException(400, str(e))
    return {"order_id": o.id, "side": o.side, "price": o.price,
            "volume": o.volume, "fee": o.fee}


@app.get("/paper/accounts/{aid}/nav")
def paper_nav(aid: int, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    return {"curve": paper.nav_curve(aid), "stats": paper.perf_stats(aid)}


@app.post("/paper/accounts/{aid}/auto_trade")
def trigger_auto_trade(aid: int, x_api_key: str | None = Header(default=None)):
    """手动触发一次 AI 信号调仓（调度器每日尾盘也会自动跑）。"""
    _auth(x_api_key)
    return {"logs": auto_trade.run_account(aid)}


# ---------- 用户画像 ----------
@app.get("/profile")
def profile_get(x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    return get_profile()


@app.put("/profile")
def profile_put(req: ProfileReq, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    save_profile(**req.model_dump(exclude_none=True))
    return {"ok": True, "profile": get_profile()}


# ---------- 报告与用量 ----------
@app.get("/reports")
def reports(limit: int = 20, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    with session_scope() as s:
        rows = (s.query(AnalysisReport).order_by(AnalysisReport.id.desc())
                .limit(limit).all())
        return [{"id": r.id, "symbol": r.symbol, "name": r.name,
                 "type": r.report_type, "created_at": str(r.created_at)[:19]}
                for r in rows]


@app.get("/reports/{rid}/markdown")
def report_markdown(rid: int, x_api_key: str | None = Header(default=None)):
    """导出 Markdown 报告。"""
    _auth(x_api_key)
    from fastapi.responses import PlainTextResponse
    with session_scope() as s:
        r = s.get(AnalysisReport, rid)
        if not r:
            raise HTTPException(404, "not found")
        md = f"# {r.name or r.symbol} 分析报告\n\n_{r.created_at:%Y-%m-%d %H:%M}_\n\n{r.content}\n"
        return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")


@app.get("/usage")
def llm_usage(days: int = 30, x_api_key: str | None = Header(default=None)):
    """LLM token 用量统计。"""
    _auth(x_api_key)
    return usage_summary(days)


# ---------- 自选池与标的搜索 ----------
@app.get("/watchlist")
def get_watchlist(x_api_key: str | None = Header(default=None)):
    """获取所有自选标的列表，并附带最新行情快照。"""
    _auth(x_api_key)
    from core.data import stock as stock_data
    from core.data import fund as fund_data
    with session_scope() as s:
        items = s.query(Watchlist).all()
        rows = []
        for w in items:
            price = None
            chg = None
            try:
                if w.asset_type == "ofund":
                    q = fund_data.fetch_fund_realtime_estimate(w.symbol)
                    if not q.get("ok"):
                        q = fund_data.nav_quote_fallback(w.symbol, w.name)
                    price = q.get("estimate_nav") or q.get("nav")
                    chg = q.get("estimate_pct")
                else:
                    q = stock_data.fetch_depth_quote(w.symbol)
                    price = q.get("last") or q.get("price")
                    chg = q.get("change_pct")
                    if price is None:
                        # 歧义号段：可能是被误判为股票的场外基金
                        if fund_data.probe_fund(w.symbol)["is_fund"]:
                            fq = fund_data.nav_quote_fallback(w.symbol)
                            price = fq.get("estimate_nav") or fq.get("nav")
                            chg = fq.get("estimate_pct")
            except Exception:
                pass
            rows.append({
                "id": w.id,
                "symbol": w.symbol,
                "name": w.name or w.symbol,
                "asset_type": w.asset_type,
                "auto_predict": w.auto_predict,
                "price": price,
                "change_pct": chg,
                "added_at": str(w.added_at)[:19] if w.added_at else "",
            })
        return rows


@app.post("/watchlist")
def add_watchlist(req: WatchlistReq, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    from core.data import stock as stock_data
    sym = req.symbol.strip()
    if not sym:
        raise HTTPException(400, "symbol is required")
    info = stock_data.classify_symbol(sym)
    code = info["symbol"]
    name = req.name.strip()
    asset_type = req.asset_type or info["asset_type"]
    if not name:
        # 自动联网解析真实名称（基金走档案，股票走实时行情），并校正资产类型
        try:
            if info["asset_type"] in ("fund", "ofund"):
                from core.data import fund as fund_data
                pf = fund_data.fetch_fund_profile(code, info["asset_type"], with_holdings=False)
                if pf.get("name"):
                    name = (pf["name"] or "").strip()
                    asset_type = pf.get("asset_type") or asset_type
            if not name:
                q = stock_data.fetch_realtime_quote(code)
                name = (q.get("name") or "").strip()
            if not name:
                # 歧义号段兜底：股票行情拿不到名字时探测场外基金（如 005827）
                from core.data import fund as fund_data
                probe = fund_data.probe_fund(code)
                if probe["is_fund"]:
                    name = probe["name"]
                    asset_type = "ofund"
        except Exception:
            name = ""
    name = name or code
    asset_type = req.asset_type or info["asset_type"]
    with session_scope() as s:
        existing = s.query(Watchlist).filter_by(symbol=code).first()
        if existing:
            existing.name = name
            existing.asset_type = asset_type
        else:
            s.add(Watchlist(symbol=code, name=name, asset_type=asset_type))
    return {"ok": True, "symbol": code, "name": name}


@app.delete("/watchlist/{symbol}")
def delete_watchlist(symbol: str, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    with session_scope() as s:
        item = s.query(Watchlist).filter_by(symbol=symbol).first()
        if item:
            s.delete(item)
            return {"ok": True, "deleted": symbol}
        return {"ok": False, "message": "Not found"}


@app.get("/search")
def search(q: str, limit: int = 10, x_api_key: str | None = Header(default=None)):
    """搜索 A 股 / ETF / 场外基金。"""
    _auth(x_api_key)
    from core.data.stock import search_stock
    return search_stock(q, limit=limit)


@app.get("/kline/{symbol}")
def get_kline(symbol: str, count: int = 120, asset_type: str | None = None,
              x_api_key: str | None = Header(default=None)):
    """获取 K 线历史数据或场外基金历史净值走势。"""
    _auth(x_api_key)
    from core.data import stock as stock_data
    from core.data import fund as fund_data
    at = _resolve_asset_type(symbol, asset_type)
    if at == "ofund":
        df = fund_data.fetch_fund_nav_history(symbol, count=count)
        if df.empty:
            df = stock_data.fetch_kline(symbol, count=count)
    else:
        df = stock_data.fetch_kline(symbol, count=count)
        if df.empty and fund_data.probe_fund(symbol)["is_fund"]:
            df = fund_data.fetch_fund_nav_history(symbol, count=count)
    if df.empty:
        return []
    # 保证包含 date, open, close, high, low, volume
    records = df.to_dict(orient="records")
    return records


# ---------- 模拟盘列表补充 ----------
@app.get("/paper/accounts")
def list_accounts(x_api_key: str | None = Header(default=None)):
    """获取所有模拟交易账户列表及概要。"""
    _auth(x_api_key)
    with session_scope() as s:
        accounts = s.query(PaperAccount).all()
        out = []
        for a in accounts:
            try:
                pf = paper.portfolio(a.id)
                stats = paper.perf_stats(a.id)
            except Exception:
                pf = {"cash": a.cash, "total_value": a.cash, "positions": []}
                stats = {"total_return": 0.0}
            out.append({
                "id": a.id,
                "name": a.name,
                "initial_cash": a.initial_cash,
                "cash": a.cash,
                "mode": a.mode,
                "note": a.note,
                "created_at": str(a.created_at)[:19] if a.created_at else "",
                "total_value": pf.get("total_value", a.cash),
                "total_return": stats.get("total_return", 0.0),
                "positions_count": len(pf.get("positions", [])),
            })
        return out


# ---------- 仪表盘概览与异动扫描 ----------
_overview_stats_cache: dict = {"data": None, "ts": 0.0}


@app.get("/overview/stats")
def overview_stats(force_refresh: bool = False, x_api_key: str | None = Header(default=None)):
    """首页聚合统计数据（默认 60 秒内存缓存）。"""
    _auth(x_api_key)
    now = time.time()
    if not force_refresh and _overview_stats_cache["data"] is not None and (now - _overview_stats_cache["ts"]) < 60:
        return _overview_stats_cache["data"]

    try:
        ov = market_data.market_overview()
    except Exception:
        ov = {}
    try:
        acc = accuracy_report()
    except Exception:
        acc = {}
    total_paper_val = 0.0
    with session_scope() as s:
        w_count = s.query(Watchlist).count()
        accs = s.query(PaperAccount).all()
        for a in accs:
            try:
                pf = paper.portfolio(a.id)
                total_paper_val += pf.get("total_value", a.cash)
            except Exception:
                total_paper_val += a.cash

    from core.predict import calibration as _cal
    cal_status = _cal.status()
    active_cal = sum(1 for pools in cal_status.get("bundles", {}).values()
                     for b in pools.values() if b.get("available"))
    res = {
        "market": ov,
        "accuracy": acc,
        "paper_total_value": total_paper_val,
        "watchlist_count": w_count,
        "active_calibrators": active_cal,
        "sample_stats": _cal.sample_stats(),
    }
    _overview_stats_cache["data"] = res
    _overview_stats_cache["ts"] = now
    return res


@app.get("/overview/briefing")
def overview_briefing(x_api_key: str | None = Header(default=None)):
    """获取最新早报简报。"""
    _auth(x_api_key)
    with session_scope() as s:
        rep = (s.query(AnalysisReport)
               .filter_by(report_type="morning_brief")
               .order_by(AnalysisReport.id.desc())
               .first())
        if rep:
            return {"content": rep.content, "created_at": str(rep.created_at)[:19]}
    from core.intel.brief import morning_brief
    try:
        text = morning_brief()
        return {"content": text, "created_at": "刚刚生成"}
    except Exception as e:
        return {"content": "暂无晨会简报，正在准备今日市场复盘。", "error": str(e)}


@app.get("/overview/alerts")
def overview_alerts(threshold: float = 3.0, x_api_key: str | None = Header(default=None)):
    """异动扫描。"""
    _auth(x_api_key)
    from core.intel.brief import anomaly_scan
    try:
        return anomaly_scan(threshold_pct=threshold)
    except Exception:
        return []


@app.get("/analysis/{symbol}/accuracy-history")
def accuracy_history(symbol: str, x_api_key: str | None = Header(default=None)):
    """获取标的历史对账命中记录。"""
    _auth(x_api_key)
    with session_scope() as s:
        evals = (s.query(PredictionEval)
                 .join(Prediction)
                 .filter(Prediction.symbol == symbol)
                 .order_by(PredictionEval.due_date.asc())
                 .all())
        return [{
            "date": str(e.due_date),
            "horizon": e.horizon,
            "actual_change_pct": e.actual_change_pct,
            "hit": e.hit,
            "brier_score": e.brier_score,
            "actual_direction": e.actual_direction,
            "predicted_direction": e.predicted_direction,
        } for e in evals]


# ---------- 系统设置与 LLM 配置 ----------
@app.get("/config/llm")
def get_llm_config_api(x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    cfg = load_config()
    llm = cfg.get("llm", {})
    return {
        "base_url": llm.get("base_url", ""),
        "api_key": llm.get("api_key", ""),
        "model": llm.get("model", ""),
        "reasoner_model": llm.get("reasoner_model", ""),
        "temperature": llm.get("temperature", 0.3),
    }


@app.post("/config/llm")
def save_llm_config_api(req: LLMConfigReq, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    save_llm_config(
        base_url=req.base_url.strip(),
        api_key=req.api_key.strip(),
        model=req.model.strip(),
        reasoner_model=req.reasoner_model.strip() or req.model.strip(),
        temperature=req.temperature,
    )
    return {"ok": True}


@app.post("/config/llm/test")
def test_llm_config_api(req: LLMConfigReq, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    test_c = LLMClient(
        base_url=req.base_url.strip(),
        api_key=req.api_key.strip(),
        model=req.model.strip(),
        reasoner_model=req.reasoner_model.strip() or req.model.strip(),
        temperature=req.temperature,
        timeout=15,
    )
    return test_connection(test_c, timeout=15)


# ---------- AI 交易员（agent harness）----------
class TraderReq(BaseModel):
    name: str
    initial_cash: float | None = None
    persona: str = ""
    notes: str = ""
    seed_beliefs: list[str] = Field(default_factory=list)


def _parse_day(s: str | None):
    """解析 YYYY-MM-DD；空值返回 None（由 loop 取今天）。"""
    if not s:
        return None
    from datetime import datetime as _dt
    try:
        return _dt.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, f"as_of 需为 YYYY-MM-DD，收到 {s!r}")


@app.post("/trader")
def trader_create(req: TraderReq, x_api_key: str | None = Header(default=None)):
    """建一个 AI 交易员（mode='trader' 的模拟账户，可选初始信条）。

    与 /paper/accounts 的区别：mode='trader' 的账户会被 AI 交易员循环接管，
    由 AI 自主选标的、下单、并从错误中总结教训写入记忆。
    """
    _auth(x_api_key)
    if not get("trader", "enabled", True):
        raise HTTPException(400, "[trader] enabled=false，AI 交易员已禁用")
    from core.trader import loop as trader_loop
    try:
        tid = trader_loop.create_trader(
            req.name, req.initial_cash, persona=req.persona,
            notes=req.notes, seed_beliefs=req.seed_beliefs or None)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "id": tid, "name": req.name}


@app.get("/trader")
def trader_list(x_api_key: str | None = Header(default=None)):
    """所有 AI 交易员及其运行概况。"""
    _auth(x_api_key)
    from core.paper import engine as paper_engine
    from core.store.models import TraderRun
    out = []
    with session_scope() as s:
        accs = s.query(PaperAccount).filter_by(mode="trader").all()
        for a in accs:
            try:
                pf = paper_engine.portfolio(a.id)
            except Exception:
                pf = {"total_value": a.cash, "cash": a.cash,
                      "total_return": 0.0, "positions": []}
            last = (s.query(TraderRun).filter_by(trader_id=a.id)
                    .order_by(TraderRun.run_date.desc()).first())
            n_runs = s.query(TraderRun).filter_by(trader_id=a.id).count()
            out.append({
                "id": a.id, "name": a.name, "note": a.note or "",
                "initial_cash": a.initial_cash,
                "cash": pf.get("cash"),
                "total_value": pf.get("total_value"),
                "total_return": pf.get("total_return"),
                "positions_count": len(pf.get("positions") or []),
                "runs": n_runs,
                "last_run": str(last.run_date) if last else "",
                "last_status": last.status if last else "",
                "last_nav": last.nav if last else None,
            })
    return out


@app.get("/trader/{tid}")
def trader_overview(tid: int, days: int = 30,
                    x_api_key: str | None = Header(default=None)):
    """交易员全景：运行统计 + 交易复盘 + 组合评估 + 记忆概况。"""
    _auth(x_api_key)
    from core.trader import report as trader_report
    try:
        return trader_report.overview(tid, days=days)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/trader/{tid}/runs")
def trader_runs(tid: int, limit: int = 20, detail: bool = False,
                x_api_key: str | None = Header(default=None)):
    """每日循环留痕（倒序）。detail=True 时附带完整工具轨迹与决策。"""
    _auth(x_api_key)
    from core.store.models import TraderRun
    with session_scope() as s:
        rows = (s.query(TraderRun).filter_by(trader_id=tid)
                .order_by(TraderRun.run_date.desc()).limit(int(limit)).all())
        out = []
        for r in rows:
            item = {
                "run_date": str(r.run_date), "status": r.status, "nav": r.nav,
                "orders": len(r.orders or []), "new_facts": r.new_facts,
                "new_lessons": len(r.new_lessons or []),
                "promotions": len(r.memory_promotions or []),
                "turns": r.turns, "llm_tokens": r.llm_tokens,
                "llm_cost_est": r.llm_cost_est,
                "attributions": r.attributions or [],
                "error": r.error or "",
            }
            if detail:
                item.update({
                    "perception": r.perception or {},
                    "reconciliations": r.reconciliations or [],
                    "tool_calls": r.tool_calls or [],
                    "decision": r.decision or {},
                    "guard_results": r.guard_results or [],
                    "orders_detail": r.orders or [],
                })
            out.append(item)
        return out


@app.post("/trader/{tid}/run")
def trader_run(tid: int, as_of: str | None = None, force: bool = False,
               x_api_key: str | None = Header(default=None)):
    """手动跑一个交易日的完整循环。

    幂等：同日已执行过时默认直接返回既有结果（不重复成交）。
    force=true 才会重跑，会**先删除当日留痕并重新下单**，仅用于调试。
    """
    _auth(x_api_key)
    if not get("trader", "enabled", True):
        raise HTTPException(400, "[trader] enabled=false，AI 交易员已禁用")
    if not LLMClient().available():
        raise HTTPException(400, "LLM 未配置，AI 交易员无法决策")
    from core.trader import loop as trader_loop
    try:
        return trader_loop.run_day(tid, _parse_day(as_of),
                                   client=LLMClient(), force=bool(force))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/trader/run-all")
def trader_run_all(as_of: str | None = None,
                   x_api_key: str | None = Header(default=None)):
    """跑所有 mode='trader' 的账户（调度器每日 14:40 也会自动跑）。"""
    _auth(x_api_key)
    if not get("trader", "enabled", True):
        raise HTTPException(400, "[trader] enabled=false，AI 交易员已禁用")
    if not LLMClient().available():
        raise HTTPException(400, "LLM 未配置，AI 交易员无法决策")
    from core.trader import loop as trader_loop
    try:
        return trader_loop.run_all(_parse_day(as_of), client=LLMClient())
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/trader/{tid}/replay")
def trader_replay(tid: int, start: str, end: str, bg: BackgroundTasks,
                  allow_existing: bool = False, max_days: int = 60,
                  allow_long: bool = False, background: bool = True,
                  x_api_key: str | None = Header(default=None)):
    """历史回放：在 [start, end] 的每个交易日按序跑一次完整循环（replay=True）。

    与 /trader/{tid}/run 的区别：run 是"向前跑今天"，replay 是"回到过去按日重跑"，
    成交/估值/净值和 AI 能看到的一切数据都截到当日 —— 用来在极短时间内
    积累几十个交易日的决策样本，而不必真的等一个月。

    ⚠️ 三条硬约束（都是踩过的坑）
      1. **必须用全新交易员**：否则回放期的持仓与真实持仓叠加，净值与 T+1 全错。
         确实要复用旧账户时传 allow_existing=true，但结果仅供调试。
      2. **每个交易日 = 一次真实 LLM 调用**：既慢又花钱，默认超过 60 天直接拒绝；
         确认要跑长区间传 allow_long=true。
      3. **回放验不了「预测驱动的自我纠错」**：run_prediction 依赖实时行情 +
         面向未来的校准模型，无法回溯，因此在回放中被**隐藏**。
         回放能验的是：成交价口径 / 估值 / 风控闸门 / 留痕 / 记忆写入。

    默认 background=true：同步跑二十天必然撞 HTTP 超时，故立即返回受理信息，
    逐日进度落在 TraderRun 里，用 GET /trader/{tid}/runs 观察。
    短区间调试可传 background=false 同步拿完整汇总。
    """
    _auth(x_api_key)
    if not get("trader", "enabled", True):
        raise HTTPException(400, "[trader] enabled=false，AI 交易员已禁用")
    if not LLMClient().available():
        raise HTTPException(400, "LLM 未配置，AI 交易员无法决策")
    d1, d2 = _parse_day(start), _parse_day(end)
    if not d1 or not d2:
        raise HTTPException(400, "start / end 必填，格式 YYYY-MM-DD")

    from core.trader import loop as trader_loop
    # 先做只读前置检查：把"要跑几天/为什么不能跑"在受理前就说清楚，
    # 免得后台静默失败、用户只看到"已接受"却永远等不到结果。
    pre = trader_loop.replay_preflight(
        tid, d1, d2, allow_existing=bool(allow_existing),
        max_days=int(max_days), allow_long=bool(allow_long))
    if not pre["ok"]:
        raise HTTPException(400, pre["error"])

    if not background:
        try:
            return trader_loop.replay(
                tid, d1, d2, client=LLMClient(),
                allow_existing=bool(allow_existing), max_days=int(max_days),
                allow_long=bool(allow_long))
        except Exception as e:
            raise HTTPException(500, str(e))

    def _job() -> None:
        try:
            trader_loop.replay(tid, d1, d2, client=LLMClient(),
                               allow_existing=bool(allow_existing),
                               max_days=int(max_days),
                               allow_long=bool(allow_long))
        except Exception:
            # 逐日失败已落在 TraderRun.error 里，不再抛进后台线程（会静默丢栈）
            pass

    bg.add_task(_job)
    return {"accepted": True, "trader_id": tid, "start": str(d1), "end": str(d2),
            "days": pre["days"], "calendar_source": pre["calendar_source"],
            "runs_endpoint": f"/trader/{tid}/runs",
            "note": "已受理。每交易日一次 LLM 循环，耗时以分钟计；"
                    "请轮询 runs 看逐日进度，勿重复提交。"}


@app.get("/trader/{tid}/review")
def trader_review(tid: int, days: int = 30, benchmark: str | None = None,
                  x_api_key: str | None = Header(default=None)):
    """三层评估：① 逐笔交易复盘 ② 组合 vs 基准 ③ 记忆有效性（自动判有害）。"""
    _auth(x_api_key)
    from core.trader import report as trader_report
    try:
        return {
            "trade": trader_report.trade_review(tid, days=days),
            "portfolio": trader_report.portfolio_review(tid, benchmark=benchmark),
            "memory_effectiveness": trader_report.memory_effectiveness(tid),
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/trader/{tid}/memory")
def trader_memory(tid: int, kind: str | None = None, symbol: str | None = None,
                  scope: str | None = None, status: str = "active",
                  limit: int = 50, x_api_key: str | None = Header(default=None)):
    """三层记忆查询。

    status 默认 'active'（只有生效的教训才参与决策）；
    传 'all' 可看全部状态，含 candidate / refuted / archived。
    """
    _auth(x_api_key)
    from core.trader import memory as mem
    st = None if str(status).lower() in ("all", "", "none") else status
    try:
        return {
            "counts": mem.view(tid)["stats"],
            "items": mem.recall(tid, kind=kind, scope=scope, symbol=symbol,
                                status=st, limit=int(limit)),
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/trader/{tid}/memory/markdown")
def trader_memory_markdown(tid: int, x_api_key: str | None = Header(default=None)):
    """导出记忆的 Markdown 镜像（只读派生视图，真相源在 DB）。"""
    _auth(x_api_key)
    from fastapi.responses import PlainTextResponse
    from core.trader import memory as mem
    try:
        path = mem.render_markdown(tid)
        from pathlib import Path as _P
        return PlainTextResponse(_P(path).read_text(encoding="utf-8"),
                                 media_type="text/markdown; charset=utf-8")
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/trader/{tid}/tools")
def trader_tools(tid: int, x_api_key: str | None = Header(default=None)):
    """AI 交易员可用的工具清单（透明化：方便核对它到底能做什么）。

    注意：**不含任何"写记忆"的工具**——记忆只在反思阶段由系统接口写入。
    """
    _auth(x_api_key)
    from core.trader import tools as trader_tools_mod
    schemas = trader_tools_mod.schemas(write_allowed=True)
    return [{"name": s["function"]["name"],
             "description": s["function"]["description"],
             "write": s["function"]["name"] in ("place_buy_order",
                                                "place_sell_order")}
            for s in schemas]


# ---------- SPA 静态前端托管 ----------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    if (_frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        target = _frontend_dist / full_path
        if full_path and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(_frontend_dist / "index.html"))

