"""冒烟测试：不依赖网络和 LLM，验证纯逻辑层。

运行：python -m pytest tests/ -x   （或 python tests/test_smoke.py）
"""
import os
import sys
from pathlib import Path

# 用内存库，避免污染真实数据
os.environ["JIJIN_DB_URL"] = "sqlite:///:memory:"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402


def test_indicators():
    from core.data.indicators import add_indicators, latest_snapshot
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=80),
        "open": range(80), "high": [x + 1 for x in range(80)],
        "low": [x - 1 for x in range(80)], "close": [x + 0.5 for x in range(80)],
        "volume": [1000 + x for x in range(80)],
    })
    snap = latest_snapshot(df)
    assert snap["close"] == 79.5
    assert snap["ma5"] is not None and snap["ma60"] is not None
    assert snap["ma_alignment"] == "多头排列"


def test_schema_validation():
    from core.predict.schema import PredictionResult
    raw = {
        "horizons": {"next_day": {"up": 0.6, "flat": 0.2, "down": 0.2, "range_pct": "±1%"}},
        "confidence": 72, "key_factors": ["a"], "risks": ["r"],
        "action": {"advice": "持有"}, "summary": "test",
    }
    r = PredictionResult.model_validate(raw)
    assert r.horizons["next_day"].direction == "up"
    # 不归一的输入会被归一化
    raw2 = {"horizons": {"next_day": {"up": 6, "flat": 2, "down": 2}}}
    r2 = PredictionResult.model_validate(raw2).normalized()
    assert abs(r2.horizons["next_day"].up - 0.6) < 1e-6


def test_paper_engine():
    from core.paper import engine
    from unittest.mock import patch
    aid = engine.create_account("测试", 100000)
    # 固定价格 10 元，绕过行情接口
    with patch.object(engine, "current_price", return_value=10.0):
        o = engine.buy(aid, "600519", amount=50000)
        assert o.volume == 5000  # 100股整手
        pf = engine.portfolio(aid)
        assert pf["positions"][0]["volume"] == 5000
        # T+1：当日不可卖
        try:
            engine.sell(aid, "600519", volume=100)
            assert False, "T+1 应禁止卖出当日买入"
        except engine.TradeError:
            pass
        engine.settle_t1(aid)
        o2 = engine.sell(aid, "600519", ratio=0.5)
        assert o2.volume == 2500
        pf = engine.portfolio(aid)
        assert pf["positions"][0]["volume"] == 2500


def test_symbol_classify():
    from core.data.stock import classify_symbol
    assert classify_symbol("600519")["asset_type"] == "stock"
    assert classify_symbol("510300")["asset_type"] == "fund"      # 场内ETF
    assert classify_symbol("110022")["asset_type"] == "ofund"     # 场外基金
    assert classify_symbol("00700")["asset_type"] == "hk"         # 港股
    assert classify_symbol("HK00700")["em_secid"] == "116.00700"


def test_news_dedup():
    from core.intel.news import save_news
    items = [{"title": "测试新闻A", "source": "t", "publish_time": None},
             {"title": "测试新闻A", "source": "t", "publish_time": None}]
    assert save_news(items) == 1  # 去重后只入 1 条
    assert save_news(items) == 0  # 再次入库 0 条


def test_fund_scope_and_stage_returns():
    """基金舆情范围推导 + 阶段收益（纯计算，不联网）。"""
    import pandas as pd
    from core.data.fund import compute_stage_returns, fund_news_scope, infer_tracked_index

    nav = pd.DataFrame({
        "date": pd.date_range(end="2026-09-22", periods=300, freq="D"),
        "close": [1.0 + i * 0.001 for i in range(300)],
    })
    r = compute_stage_returns(nav)
    assert r["近1月"] > 0 and r["今年以来"] > 0

    track = infer_tracked_index("华泰柏瑞沪深300ETF", "沪深300指数收益率")
    assert "沪深300" in track["themes"] and track["index"]

    profile = {
        "symbol": "510300", "name": "华泰柏瑞沪深300ETF",
        "tracking": {"index": "沪深300", "themes": ["沪深300", "大盘"]},
        "basic": {"company": "华泰柏瑞基金"},
        "holdings": [{"code": "600519", "name": "贵州茅台", "weight": 5.2},
                     {"code": "000858", "name": "五粮液", "weight": 2.1}],
    }
    scope = fund_news_scope(profile, max_holdings=3)
    assert [s["symbol"] for s in scope["symbols"]] == ["600519", "000858"]
    assert "沪深300" in scope["keywords"] and "贵州茅台" in scope["label"]


def test_relevance_scoring_and_aggregate():
    """舆情相关性：直接命中 / 重仓股间接命中 / 宏观 / 泛快讯 的分层与加权。"""
    from core.intel.relevance import build_profile, score_item, annotate
    from core.intel.sentiment import aggregate_sentiment

    fund_profile = {
        "symbol": "510300", "name": "沪深300ETF",
        "tracking": {"index": "沪深300", "themes": ["沪深300"]},
        "basic": {"company": "华泰柏瑞基金"},
        "holdings": [{"code": "600519", "name": "贵州茅台", "weight": 5.0}],
    }
    p = build_profile("510300", "沪深300ETF", "fund", fund_profile=fund_profile)
    own = score_item({"title": "510300 今日成交额创历史新高"}, p)
    indirect = score_item({"title": "贵州茅台三季度业绩超预期"}, p)
    index = score_item({"title": "沪深300指数样本股调整"}, p)
    macro = score_item({"title": "央行宣布降准", "category": "macro", "source": "caixin"}, p)
    noise = score_item({"title": "某地天气晴好", "category": "flash", "source": "cls"}, p)
    assert own == 1.0 and indirect >= 0.55 and index == 0.6
    assert macro == 0.35 and noise < 0.3

    items = [
        {"title": "贵州茅台三季度业绩超预期", "sentiment": "乐观", "intensity": 5,
         "deviation": 4, "category": "news"},
        {"title": "某地天气晴好", "sentiment": "悲观", "intensity": 5, "deviation": 5,
         "category": "flash"},
    ]
    stats = annotate(items, p)
    assert stats["kept"] == 1 and stats["dropped"] == 1
    sig = aggregate_sentiment(items)
    assert sig["score"] > 50 and sig["count"] == 1 and sig["coverage"] == 0.5
    # 全量口径被无关的空头噪声拖低，divergence 说明该标的有独立驱动
    assert sig["universe_score"] < sig["score"] and sig["divergence"] > 0
    # LLM 关联度高时，即使规则分低也保留
    item3 = {"title": "行业景气度跟踪报告", "sentiment": "悲观", "intensity": 4,
             "deviation": 3, "relevance_llm": 5, "category": "flash"}
    assert annotate([item3], p)["kept"] == 1


def test_calibration_learns_and_improves():
    """自学习校准：过度自信的引擎概率应被校准，样本外 Brier 变好。"""
    import tempfile
    from datetime import date, timedelta
    from pathlib import Path

    import numpy as np

    from core.predict import calibration as cal

    with tempfile.TemporaryDirectory() as td:
        cal.MODEL_DIR = Path(td)
        rng = np.random.default_rng(7)
        today = date.today()
        # 引擎总是说"涨 0.6/平 0.2/跌 0.2"，但真实只有 40% 概率上涨
        for i in range(200):
            r = rng.random()
            actual = "up" if r < 0.4 else ("flat" if r < 0.8 else "down")
            cal.record_sample(
                engine="llm", horizon="next_day", symbol="600519", asset_type="stock",
                base_date=today - timedelta(days=200 - i),
                probs={"up": 0.6, "flat": 0.2, "down": 0.2},
                actual_direction=actual, actual_change_pct=1.0, flat_threshold=1.0,
                confidence=70, senti_score=10.0, source="eval", payload={})
        assert cal.sample_stats()["total"] == 200
        res = cal.train("next_day", record=False)
        assert res["trained"] and res["method"] in ("temperature", "meta_logit")
        ev = res["evidence"]
        after = ev.get("brier_meta") if res["method"] == "meta_logit" else ev.get("brier_temp")
        assert after <= ev["brier_before"]
        # 校准后的"涨"概率应明显低于原始的 0.6
        out = cal.apply({"up": 0.6, "flat": 0.2, "down": 0.2}, horizon="next_day")
        assert out["method"] != "identity" and out["probs"]["up"] < 0.6
        assert abs(sum(out["probs"].values()) - 1) < 1e-6
        assert cal.load("next_day") is not None
    # 无样本的周期如实返回 identity，不假装有校准
    assert cal.apply({"up": 0.5, "flat": 0.3, "down": 0.2},
                     horizon="quarter")["method"] == "identity"


def test_quote_depth_and_trends_parsing():
    """盘口/分时字段解析（用假响应离线验证列名与量纲，不依赖网络）。"""
    import requests
    from core.data import stock as sd

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    depth = {"data": {"f43": 12.34, "f44": 12.5, "f45": 12.0, "f46": 12.1, "f47": 12345,
                      "f48": 15200000.0, "f50": 1.8, "f51": 13.2, "f52": 10.8,
                      "f57": "600519", "f58": "贵州茅台", "f60": 12.2, "f71": 12.28,
                      "f116": 1.5e12, "f117": 1.5e12, "f162": 30.1, "f167": 8.2,
                      "f168": 1.1, "f169": 0.14, "f170": 1.15, "f171": 4.1,
                      "f19": 12.33, "f20": 120, "f17": 12.32, "f18": 80,
                      "f15": 12.31, "f16": 60, "f13": 12.30, "f14": 40,
                      "f11": 12.29, "f12": 20,
                      "f39": 12.35, "f40": 110, "f37": 12.36, "f38": 90,
                      "f35": 12.37, "f36": 70, "f33": 12.38, "f34": 50,
                      "f31": 12.39, "f32": 30}}
    trends = {"data": {"trends": [
        "2026-09-22 09:30,12.10,12.12,12.15,12.08,1000,1212000,12.11",
        "2026-09-22 09:31,12.12,12.20,12.22,12.12,800,976000,12.13"]}}

    def _fake_get(url, **kw):
        return _Resp(trends if "trends2" in url else depth)

    from core.data import _net
    orig_get, orig_proxies = requests.get, _net.get_proxies
    requests.get = _fake_get
    _net.get_proxies = lambda: {}      # 测试环境不走真实代理
    try:
        q = sd.fetch_depth_quote("600519")
        assert q["ok"] and q["last"] == 12.34 and q["name"] == "贵州茅台"
        assert q["asks"][0]["level"] == 5 and q["bids"][0]["level"] == 1
        assert q["bids"][0]["price"] == 12.33 and q["asks"][-1]["price"] == 12.35
        assert q["asks"][0]["price"] == 12.39  # 卖五最远，价格最高
        df = sd.fetch_intraday_trends("600519")
        assert list(df.columns) == ["time", "open", "price", "high", "low",
                                    "volume", "amount", "avg"]
        assert float(df["price"].iloc[-1]) == 12.20
    finally:
        requests.get = orig_get
        _net.get_proxies = orig_proxies


def test_evaluator_writes_calib_samples():
    """到期对账应同时写入自学习样本，闭环才算真的闭合。"""
    from datetime import date
    from core.store.db import session_scope
    from core.store.models import Prediction, CalibSample
    from core.verify import evaluator

    with session_scope() as s:
        s.add(Prediction(symbol="000001", name="测试标的", asset_type="stock",
                         base_date=date.today(), base_price=10.0, engine="fused",
                         mode="fast",
                         horizons={"next_day": {"up": 0.6, "flat": 0.2, "down": 0.2}},
                         confidence=66,
                         input_snapshot={"news_signal": {"score": 12.0,
                                                         "universe_score": 4.0,
                                                         "coverage": 0.5}}))
    n = 0
    orig_price = evaluator._actual_price
    evaluator._actual_price = lambda pred: 11.0   # 离线给定"实际价格"，避开行情接口
    try:
        n = evaluator.evaluate_due(today=date(2026, 12, 31))
    finally:
        evaluator._actual_price = orig_price
    assert n >= 1
    with session_scope() as s:
        sample = s.query(CalibSample).filter_by(symbol="000001").first()
        assert sample is not None and sample.engine == "fused"
        assert sample.senti_score == 12.0 and sample.relevance_cov == 0.5
        for row in s.query(Prediction).filter_by(symbol="000001").all():
            s.delete(row)
        for row in s.query(CalibSample).filter_by(symbol="000001").all():
            s.delete(row)


def test_due_dates_use_trading_days_not_calendar_days():
    """到期日必须是「base_date 之后第 N 个**交易日**」，不是自然日折算。

    与前端 `scripts/verify_forecast_logic.mjs` 的第 ⑥ 组断言**共用同一组数字**：
    两侧各断言一遍，任何一侧偷偷改口径都会立刻变红。这正是防"两处实现各自漂移"。

    离线且确定：把交易日历接口换成**冻结夹具**（一段写死的连续工作日），
    而不是靠联网或"今天是几号"。否则这个测试会在真日历可用的那一天自己变红。
    """
    from datetime import date, timedelta
    from core.data import price as price_data
    from core.predict.labels import HORIZON_DAYS, HORIZON_STEPS, due_dates_for

    def weekdays(start: date, end: date):
        out, cur = [], start
        while cur <= end:
            if cur.weekday() < 5:
                out.append(cur)
            cur += timedelta(days=1)
        return out

    base_date = date(2026, 9, 24)                      # 周四
    fixture = weekdays(date(2026, 1, 1), date(2027, 12, 31))
    orig = price_data.trading_days_ex
    price_data.trading_days_ex = lambda s, e: (
        [d for d in fixture if s <= d <= e], "index")
    try:
        due, source = due_dates_for(
            base_date, ["next_day", "one_week", "one_month", "quarter"])

        # ① 与前端冻结表逐字相同（base_date = 2026-09-24，行情最后一日 = 2026-09-22）
        assert source == "index"
        assert due == {"next_day": "2026-09-25", "one_week": "2026-10-01",
                       "one_month": "2026-10-22", "quarter": "2026-12-17"}, due

        # ② 全部严格晚于 base_date。旧口径最刺眼的失败就是把 next_day 落到 09-23
        #    （对 base_date=09-24 的研判来说，那是"昨天"）。
        assert all(v > "2026-09-24" for v in due.values()), due

        # ③ 恰是"第 N 个交易日"，不是第 N 个自然日
        for h, n in HORIZON_STEPS.items():
            after = [d for d in fixture if d > base_date]
            assert due[h] == str(after[n - 1]), (h, due[h], after[n - 1])

        # ④ 与已废弃的自然日折算对照：必须至少有一个周期不同，
        #    否则说明这个函数其实没生效（或者 HORIZON_DAYS 被改成了等价表）。
        old = {h: str(base_date + timedelta(days=n)) for h, n in HORIZON_DAYS.items()}
        assert any(due[h] != old[h] for h in due), (due, old)

        # ⑤ base_date 落在周末时，从它之后第一个交易日算起（第 N 个），不能落在周末
        due_sat, _ = due_dates_for(date(2026, 9, 26),  # 周六
                                   ["next_day", "one_month"])
        assert due_sat == {"next_day": "2026-09-28", "one_month": "2026-10-23"}, due_sat
        assert all(date.fromisoformat(v).weekday() < 5 for v in due_sat.values())

        # ⑥ 日历覆盖不到那么远 → 后半段补"排除周末"并把口径降级标出来，
        #    绝不静默返回一个基于自然日的日期。
        short = lambda s, e: ([d for d in fixture if s <= d <= e][:3], "index")
        price_data.trading_days_ex = short
        due_short, src_short = due_dates_for(base_date, ["quarter"])
        assert src_short == "weekday", src_short
        assert due_short["quarter"] == "2026-12-17", due_short

        # ⑦ 日历完全不可用 → 空 dict + empty，调用方必须跳过该周期而不是瞎猜日期
        price_data.trading_days_ex = lambda s, e: ([], "empty")
        assert due_dates_for(base_date) == ({}, "empty")
    finally:
        price_data.trading_days_ex = orig


def test_evaluator_skips_horizon_when_calendar_unavailable():
    """日历拿不到到期日时，对账必须**跳过**该周期，而不是退回自然日折算。

    退回的代价不是"报错"，而是**静默写脏样本**：按错的到期日取到的收盘价，
    判出一个和模型训练标签不是同一个问题的"真实方向"，再拿去教校准层。
    从任何指标上都看不出来。
    """
    from datetime import date
    from core.data import price as price_data
    from core.store.db import session_scope
    from core.store.models import Prediction, PredictionEval
    from core.verify import evaluator

    with session_scope() as s:
        s.add(Prediction(symbol="000002", name="口径测试", asset_type="stock",
                         base_date=date(2026, 9, 24), base_price=10.0, engine="fused",
                         mode="fast",
                         horizons={"next_day": {"up": 0.6, "flat": 0.2, "down": 0.2}},
                         confidence=66))
    orig_cal, orig_px = price_data.trading_days_ex, evaluator._actual_price
    price_data.trading_days_ex = lambda s, e: ([], "empty")
    evaluator._actual_price = lambda pred: 11.0
    try:
        assert evaluator.evaluate_due(today=date(2026, 12, 31)) == 0
    finally:
        price_data.trading_days_ex, evaluator._actual_price = orig_cal, orig_px
    with session_scope() as s:
        assert s.query(PredictionEval).filter_by(horizon="next_day").count() == 0
        for row in s.query(Prediction).filter_by(symbol="000002").all():
            s.delete(row)


def test_fund_profile_assembly_offline():
    """基金档案组装（mock 数据源，验证字段换算与容错，不联网）。"""
    import pandas as pd
    from core.data import fund as fd

    nav = pd.DataFrame({"date": pd.date_range(end="2026-09-22", periods=60, freq="D"),
                        "close": [3.0 + i * 0.01 for i in range(60)],
                        "open": [3.0 + i * 0.01 for i in range(60)],
                        "high": [3.0 + i * 0.01 for i in range(60)],
                        "low": [3.0 + i * 0.01 for i in range(60)],
                        "volume": [0.0] * 60})
    orig = (fd.fetch_fund_basic_info, fd.fetch_fund_holdings,
            fd.fetch_fund_realtime_estimate)
    fd.fetch_fund_basic_info = lambda code: {
        "ok": True, "name": "华泰柏瑞沪深300ETF", "type": "指数型-股票",
        "manager": "柳军", "company": "华泰柏瑞基金", "size": "1200亿",
        "estab_date": "2012-05-04", "source": "mock"}
    fd.fetch_fund_holdings = lambda code, top=15: [
        {"code": "300750", "name": "宁德时代", "weight": 4.27, "shares": 1.0,
         "value": 2.5e8, "quarter": "2026Q2"},
        {"code": "600519", "name": "贵州茅台", "weight": 3.65, "shares": 1.0,
         "value": 2.1e8, "quarter": "2026Q2"}]
    fd.fetch_fund_realtime_estimate = lambda code: {"ok": True, "nav": 4.5,
                                                   "estimate_nav": 4.52,
                                                   "estimate_pct": 0.44}
    from core.data import stock as sd
    orig_kline = sd.fetch_kline
    sd.fetch_kline = lambda *a, **k: nav
    try:
        pf = fd.fetch_fund_profile("510300", "fund")
        assert pf["name"] == "华泰柏瑞沪深300ETF"
        assert pf["top10_weight"] == 7.92
        assert pf["stage_returns"]["近1月"] > 0
        assert pf["tracking"]["index"] == "沪深300"
        scope = fd.fund_news_scope(pf)
        assert [s["symbol"] for s in scope["symbols"]] == ["300750", "600519"]
        assert "宁德时代" in scope["label"]
    finally:
        (fd.fetch_fund_basic_info, fd.fetch_fund_holdings,
         fd.fetch_fund_realtime_estimate) = orig
        sd.fetch_kline = orig_kline


def test_pipeline_news_scope_block():
    """预测流水线的舆情范围落进 prompt 数据块（离线，mock 基金档案）。"""
    import pandas as pd
    from core.data import fund as fd
    from core.predict import pipeline

    orig = fd.fetch_fund_profile
    fd.fetch_fund_profile = lambda *a, **k: {
        "symbol": "510300", "name": "沪深300ETF", "asset_type": "fund", "ok": True,
        "basic": {"name": "沪深300ETF", "type": "指数型", "company": "华泰柏瑞基金",
                  "manager": "柳军", "size": "1200亿"},
        "tracking": {"index": "沪深300", "themes": ["沪深300"]},
        "holdings": [{"code": "300750", "name": "宁德时代", "weight": 4.27}],
        "stage_returns": {"近1月": 2.5}, "top10_weight": 22.9}
    try:
        ns = pipeline.build_news_scope("510300", "沪深300ETF", "fund")
        assert ns["is_fund"] and ns["extra_symbols"] == ["300750"]
        assert "重仓股" in ns["label"] and "沪深300" in ns["label"]
        kline = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=70),
                              "open": range(70), "high": [x + 1 for x in range(70)],
                              "low": [x - 1 for x in range(70)],
                              "close": [x + 0.5 for x in range(70)],
                              "volume": [1000 + x for x in range(70)]})
        news = [{"title": "宁德时代三季度业绩超预期", "sentiment": "乐观", "intensity": 5,
                 "deviation": 4, "relevance": 0.68, "publish_time": "2026-09-21 10:00"}]
        block = pipeline.build_data_block(
            "510300", "沪深300ETF", "fund", {"ok": True, "price": 4.5}, kline,
            news, {"score": 60.0, "coverage": 0.5, "count": 1}, None, None,
            news_scope=ns)
        assert "基金档案" in block and "舆情范围判定" in block
        assert "相关0.68" in block and "消息面聚合信号" in block
    finally:
        fd.fetch_fund_profile = orig


def test_calibration_status_and_reliability():
    """校准层状态与置信度可靠性表（依赖前面的样本）。"""
    from core.predict import calibration as cal
    from core.predict.schema import HORIZONS

    stt = cal.status()
    assert set(stt["bundles"].keys()) == set(HORIZONS)
    for pools in stt["bundles"].values():
        assert set(pools.keys()) == {"llm", "quant"}
    rows = cal.reliability_table("next_day")
    assert rows and all("实际命中率" in r and r["样本数"] > 0 for r in rows)
    assert stt["samples"]["total"] >= 200


def test_labels_single_source():
    """标签口径唯一来源：四个模块的阈值/方向判定必须完全一致。

    历史问题：训练用 ±0.5%、对账用 ±1.0%、归因默认 ±1.0%，三把尺子各量各的，
    写进校准样本的"真实方向"和模型学的方向是两个不同问题。
    """
    from core.predict import labels as L
    from core.predict import quant
    from core.trader.attributor import direction_from_change, direction_from_probs
    from core.verify import evaluator

    assert L.base_threshold() == 0.5
    # 各周期按 √step 缩放（0.5 / 1.118 / 2.236 / 3.873）
    assert L.flat_threshold_for("next_day") == 0.5
    assert abs(L.flat_threshold_for("quarter") - 0.5 * 60 ** 0.5) < 1e-3
    assert (L.flat_threshold_for("one_week") < L.flat_threshold_for("one_month")
            < L.flat_threshold_for("quarter"))
    # 量化引擎 / 对账器 / 归因器 全部指向同一实现
    for h in L.HORIZONS:
        assert quant.flat_threshold_for(h) == L.flat_threshold_for(h)
        assert evaluator.flat_threshold_for(h) == L.flat_threshold_for(h)
    assert direction_from_change(0.6) == "up"          # 默认继承 0.5，而非旧的 1.0
    assert direction_from_change(0.6, 1.0) == "flat"   # 显式传参仍可用
    assert direction_from_change(-0.6) == "down"
    # 平手取 flat（无观点），不能偷偷记成看涨
    assert L.argmax_direction({"up": 0.34, "flat": 0.33, "down": 0.33}) == "up"
    assert L.argmax_direction({"up": 0.333, "flat": 0.333, "down": 0.334}) == "down"
    assert L.argmax_direction({"up": 1 / 3, "flat": 1 / 3, "down": 1 / 3}) == "flat"
    assert L.argmax_direction({}) == ""
    assert direction_from_probs({"up": 0.2, "flat": 0.2, "down": 0.6}) == "down"
    # 概率数组列序 = LABEL_IDX，转换必须可逆
    assert L.probs_to_vector({"up": 1, "flat": 2, "down": 3}) == [3, 2, 1]
    assert L.vector_to_probs([3, 2, 1])["up"] == 1


def test_probability_scale_pinned():
    """概率口径钉死器：百分数/全零/混用都要被归一，且和恒为 1。"""
    from core.predict.schema import HorizonForecast

    hf = HorizonForecast(up=40, flat=30, down=30)          # 百分数口径
    assert abs(hf.up - 0.4) < 1e-9 and abs(hf.up + hf.flat + hf.down - 1) < 1e-9
    hf = HorizonForecast(up=0.4, flat=30, down=0.3)        # 混用口径
    assert abs(hf.up + hf.flat + hf.down - 1) < 1e-9 and abs(hf.flat - 0.3) < 1e-9
    hf = HorizonForecast(up=0, flat=0, down=0)             # 全零退化为均匀
    assert abs(hf.up - 1 / 3) < 1e-9 and hf.direction == "flat"
    hf = HorizonForecast(up=0.5, flat=0.3, down=0.2)
    assert hf.spread == 0.3 and hf.direction == "up"


def test_calibration_preserves_class_order():
    """回归：校准绝不能把涨/跌概率对调；2 类样本窗口也不能崩。

    根因是特征向量按 [涨,平,跌] 排列、标签索引按 [跌,平,涨]，
    温度档位输出时手写 `p[0]→down` 就换反了。原来的测试恰好测不出来
    （它只断言 up 变小，没断言 up 仍是最大）。
    """
    import tempfile
    from datetime import date, timedelta
    from pathlib import Path

    from core.predict import calibration as cal

    with tempfile.TemporaryDirectory() as td:
        cal.MODEL_DIR = Path(td)
        today = date.today()
        # 每 20 条一循环 → 跌 10% / 平 35% / 涨 55%，三类都在；
        # 引擎却报"涨 0.90"，典型过度自信 → 温度校准应当带来改善
        for i in range(120):
            r = i % 20
            actual = "down" if r < 2 else ("flat" if r < 9 else "up")
            cal.record_sample(
                engine="llm", horizon="one_week", symbol="ORDER", asset_type="stock",
                base_date=today - timedelta(days=120 - i),
                probs={"up": 0.90, "flat": 0.07, "down": 0.03},
                actual_direction=actual, actual_change_pct=2.0,
                flat_threshold=1.118, confidence=60, source="eval", payload={})

        res = cal.train("one_week", record=False)
        assert res["trained"] and res["method"] in ("temperature", "meta_logit"), res
        ev = res["evidence"]
        after = ev["brier_meta"] if res["method"] == "meta_logit" else ev["brier_temp"]
        assert after < ev["brier_before"], ev

        out = cal.apply({"up": 0.90, "flat": 0.07, "down": 0.03},
                        horizon="one_week")["probs"]
        assert out["up"] == max(out.values()), f"涨/跌被对调: {out}"
        assert out["up"] > out["down"]
        assert abs(sum(out.values()) - 1) < 1e-6

        # 2 类样本窗口：只有涨/平（无跌），元模型只输出 2 列 —— 以前直接 IndexError
        for i in range(90):
            cal.record_sample(
                engine="quant", horizon="one_month", symbol="ORDER", asset_type="stock",
                base_date=today - timedelta(days=90 - i),
                probs={"up": 0.6, "flat": 0.4, "down": 0.0},
                actual_direction="up" if i % 3 else "flat", actual_change_pct=2.0,
                flat_threshold=2.236, confidence=50, source="walk_forward", payload={})
        r2 = cal.train("one_month", scope="quant", record=False)
        assert r2["trained"], r2
        assert "brier_meta" in r2["evidence"] or r2["method"] == "temperature", r2
    cal.MODEL_DIR = Path(cal.PROJECT_ROOT) / "data" / "models"


def test_calibration_brief_shape():
    """引擎自评证据块（回灌 prompt）必须覆盖四周期并带基线对照。"""
    from core.predict import calibration as cal
    from core.predict.schema import HORIZONS

    brief = cal.brief()
    assert set(brief.keys()) == set(HORIZONS)
    for item in brief.values():
        assert "到期对账样本数" in item and "校准方法" in item
        if item["到期对账样本数"]:
            assert item["随机基准"] == 0.3333
            assert item["多数类基准"] >= 0.3333          # 多数类必然不弱于随机
            dist = item["实际方向分布"]
            assert sum(dist.values()) == item["到期对账样本数"]
            assert set(dist.keys()) == {"down", "flat", "up"}
        else:
            assert "说明" in item


def test_bss_zero_when_probs_equal_clim():
    """气候学基准必须**与模型同源**：概率 == 基准 ⇒ BSS 恰为 0。

    回归背景：基准原先取 `bincount(训练标签)`（离散类别频率），而模型概率由
    残差经验分布导出（连续）。两套口径不同 ⇒ λ=0（模型自认无择时能力）时
    仍能测出 BSS=+0.0138 的**假技能**，并据此给模型发了 0.0966 的融合权重。
    改成同源基准后，λ=0 ⇒ BSS ≡ 0。
    """
    from core.predict import quant
    from core.predict.labels import LABELS
    import numpy as np

    rng = np.random.default_rng(3)
    samples = []
    for _ in range(120):
        p = rng.dirichlet([2.0, 2.0, 2.0])
        vec = [float(p[0]), float(p[1]), float(p[2])]        # [跌, 平, 涨]
        samples.append({"probs": {k: vec[i] for i, k in enumerate(LABELS)},
                        "clim": vec,
                        "actual": LABELS[int(rng.integers(0, 3))]})
    m = quant._metrics(samples)
    assert m, m
    assert abs(float(m["oos_bss"])) < 1e-9, m
    assert abs(float(m["oos_brier"]) - float(m["oos_brier_clim"])) < 1e-12, m


def test_walk_forward_folds_need_history():
    """长周期需要更长历史；不足时应**不出折**，而不是伪造样本。

    回归背景：MIN_TRAIN=250 + purge(horizon) + embargo 的协议下，
    400 根 K 线在 h=60 时有效样本仅约 280 < 317 → 一条样本外样本都产不出，
    表现为"该周期有量化概率但 has_skill 恒 false、融合权重恒 0"。
    """
    from core.predict import quant

    assert list(quant._purged_splits(189, 1)) == []          # 250 根 → 全周期无折
    assert list(quant._purged_splits(280, 60)) == []         # 400 根 → 季度无折
    for n, h in ((639, 1), (635, 5), (620, 20), (580, 60)):  # 700 根 → 四周期都有折
        folds = list(quant._purged_splits(n, h))
        assert len(folds) >= 2, (n, h, len(folds))
        for tr, te in folds:
            assert tr.stop > 0 and te.stop > te.start, (n, h, tr, te)
            assert te.start - tr.stop >= h, (n, h, tr, te)   # purge 生效，无标签泄漏


def test_backend_probe_rejects_broken_factory():
    """后端可用性判据必须是"真能 fit"，不是"import 成功"。

    回归背景：本机出现过 lightgbm "import 成功、fit 抛 access violation"，
    旧实现只看 import 就选中 lgbm，导致四个周期全部训练失败、量化引擎整体缺席。
    """
    from core.ml import NumpyRidge
    from core.predict import quant

    quant._probe_fit(lambda: NumpyRidge(alpha=1.0))          # 好后端必须通过

    class _Boom:
        def fit(self, X, y):
            raise RuntimeError("access violation reading 0x0")

    try:
        quant._probe_fit(lambda: _Boom())
        raise AssertionError("探针必须拒绝 fit 会抛异常的后端")
    except RuntimeError:
        pass

    # 探针结论与对外声明必须一致（UI 上的"当前后端"就是真在训练的那个）
    assert quant.active_backend() in {"lgbm", "gbdt", "numpy_ridge"}
    diag = quant.backend_diagnostics()
    assert diag["active_name"] == quant.active_backend(), diag
    assert diag["active_name"] not in diag["probe"], diag     # 失败记录里不应有在用后端


def test_record_samples_batch_matches_single_and_dedups():
    """批量写样本必须与逐条 record_sample 语义一致，且天然去重、跳过非法行。

    回归背景：原先 walk-forward 留档对**每条样本**调一次 record_sample（各自开一个
    session），一次收盘预测要开合上千次数据库连接。本机实测这种"长负载 + 高频连接
    开合"会引发**进程级硬崩溃**（无 traceback 猝死，与 ML 后端无关）。
    改为 record_samples_batch（一批一个 session）后崩溃消失——本测试锁死其语义。
    """
    from datetime import date

    from core.predict import calibration as cal
    from core.store.db import session_scope
    from core.store.models import CalibSample

    sym = "_BATCHTEST"
    with session_scope() as s:                       # 先清场，避免上次失败残留干扰计数
        s.query(CalibSample).filter_by(symbol=sym).delete()

    good = {"engine": "quant", "horizon": "next_day", "symbol": sym,
            "asset_type": "stock", "base_date": date(2026, 1, 5),
            "probs": {"up": 0.5, "flat": 0.3, "down": 0.2},
            "actual_direction": "up", "actual_change_pct": 1.23,
            "flat_threshold": 1.0, "source": "walk_forward", "confidence": 50,
            "payload": {"k": 1}}
    items = [
        dict(good),
        dict(good),                                        # 同键 → 去重
        dict(good, base_date=date(2026, 1, 6)),            # 换基准日 → 另计一条
        dict(good, actual_direction="??"),                 # 非法方向 → 跳过
        dict(good, base_date=None),                        # 无基准日 → 跳过
    ]
    try:
        assert cal.record_samples_batch(items) == 2        # 只有 2 条合法且不重复
        assert cal.record_samples_batch([dict(good)]) == 0  # 已存在 → 幂等
        with session_scope() as s:
            rows = s.query(CalibSample).filter_by(symbol=sym).all()
            assert len(rows) == 2, len(rows)
            for r in rows:
                assert r.p_up == 0.5 and r.p_flat == 0.3 and r.p_down == 0.2
                assert r.engine == "quant" and r.source == "walk_forward"
        assert cal.record_samples_batch([]) == 0           # 空批次是安全 no-op
    finally:
        with session_scope() as s:
            s.query(CalibSample).filter_by(symbol=sym).delete()


if __name__ == "__main__":
    test_indicators()
    test_schema_validation()
    test_paper_engine()
    test_symbol_classify()
    test_news_dedup()
    test_fund_scope_and_stage_returns()
    test_relevance_scoring_and_aggregate()
    test_calibration_learns_and_improves()
    test_quote_depth_and_trends_parsing()
    test_evaluator_writes_calib_samples()
    test_fund_profile_assembly_offline()
    test_pipeline_news_scope_block()
    test_calibration_status_and_reliability()
    test_labels_single_source()
    test_probability_scale_pinned()
    test_calibration_preserves_class_order()
    test_calibration_brief_shape()
    test_bss_zero_when_probs_equal_clim()
    test_walk_forward_folds_need_history()
    test_backend_probe_rejects_broken_factory()
    test_record_samples_batch_matches_single_and_dedups()
    print("[OK] all smoke tests passed")
