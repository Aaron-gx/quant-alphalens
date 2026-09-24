"""AI 交易员冒烟测试：不依赖网络与真实 LLM，用合成数据 + 假模型跑通全链路。

运行：python -m pytest tests/test_trader.py -x
      或 python tests/test_trader.py

覆盖四件事（对应设计文档的核心机制）：
  1. 纠错归因的四分类是否正确（市场 beta / 噪声不被误学）
  2. 记忆防污染规则是否生效（无证据教训被拒、证据达标才升 active）
  3. 风控门禁是否拦得住（AI 不可绕过）
  4. 完整每日循环能否跑通（感知→对账→归因→反思→决策→风控→执行→留痕）
"""
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# 用内存库，避免污染真实数据
os.environ["JIJIN_DB_URL"] = "sqlite:///:memory:"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class FakeLLM:
    """假 LLM：第一轮发起一次工具调用，第二轮给出决策 JSON。

    反思阶段（ask_json）从 prompt 里正则提取 fact 编号作为证据引用，
    这样能真实地走通「AI 提教训 → 系统校验证据 → 入库」这条链路。
    """

    def __init__(self, decision=None, lessons=None, want_tool_call=True,
                 tool_unsupported=False):
        self.decision = decision or {
            "actions": [{"side": "buy", "symbol": "510300",
                         "amount": 200000, "reason": "测试买入"}],
            "rationale": "测试决策", "watch_reason": ""}
        self.lessons = lessons
        self.want_tool_call = want_tool_call
        self.tool_unsupported = tool_unsupported
        self._turn = 0
        self.last_usage = {"prompt_tokens": 80, "completion_tokens": 20,
                           "total_tokens": 100}

    def available(self):
        return True

    def supports_tools(self):
        return not self.tool_unsupported

    def chat_with_tools(self, messages, tools, **kw):
        self._turn += 1
        if self._turn == 1 and self.want_tool_call:
            return {"content": "", "tool_calls": [{
                "id": "c1", "name": "get_portfolio", "arguments": "{}"}],
                "usage": dict(self.last_usage), "raw_finish_reason": "tool_calls"}
        return {"content": json.dumps(self.decision, ensure_ascii=False),
                "tool_calls": [], "usage": dict(self.last_usage),
                "raw_finish_reason": "stop"}

    def chat(self, messages, **kw):
        return json.dumps(self.decision, ensure_ascii=False)

    def ask_json(self, prompt, system="", **kw):
        if self.lessons is None:
            return {"lessons": [], "notes": "无清晰规律"}
        ids = [int(x) for x in re.findall(r"- \[(\d+)\]", prompt)]
        out = []
        for les in self.lessons:
            item = dict(les)
            # 把占位符 __FACT__ 换成 prompt 里真实出现的 fact 编号
            if item.get("evidence_fact_ids") == "__FACT__":
                item["evidence_fact_ids"] = ids[:item.pop("_n", 3)]
            else:
                item["evidence_fact_ids"] = item.get("evidence_fact_ids", []) or ids[:3]
            out.append(item)
        return {"lessons": out, "notes": "测试反思"}

    def ask(self, prompt, **kw):
        return json.dumps(self.decision, ensure_ascii=False)


def _fake_predict(symbol, *, mode="fast", **kw):
    """假预测：返回带 horizons 的轻量对象。"""
    class _P:
        id = 9001
        confidence = 66
        action = {"advice": "持有"}
        risks = ["测试风险"]
        horizons = {"next_day": {"up": 0.6, "flat": 0.2, "down": 0.2,
                                 "range": "±1.5%"},
                    "one_week": {"up": 0.5, "flat": 0.3, "down": 0.2,
                                 "range": "±3%"}}
    return _P()


class FakeMarket:
    @staticmethod
    def market_overview():
        return {"sentiment": "中性", "breadth": {"up": 2000, "down": 2500,
                                                "limit_up": 30, "limit_down": 5},
                "fund_flow": {"ok": True, "main_net_inflow": -3.2e9},
                "margin": {"ok": True, "weekly_change": 1.1e9},
                "indices": {"沪深300": {"price": 4000.0, "change_pct": -0.8}}}


def _make_deps(market_change=-0.5, engine=None, evaluate_due=None):
    from core.paper import engine as real_engine
    return {
        "engine": engine or real_engine,
        "market": FakeMarket,
        "market_change": lambda: {"change_pct": market_change},
        "evaluate_due": evaluate_due or (lambda *a, **k: 0),
        "predict_fn": _fake_predict,
    }


# ---------------------------------------------------------------------------
# 1. 纠错归因四分类
# ---------------------------------------------------------------------------

def test_attribution_four_classes():
    """归因必须能把「大盘拖累」「噪声」与「判断失误」区分开——
    这是防止 AI 学到伪规律的第一道闸门。"""
    from core.trader import attributor as A

    pred_up = {"id": 1, "symbol": "510300", "horizon": "next_day",
               "probs": {"up": 0.6, "flat": 0.2, "down": 0.2},
               "base_price": 4.0, "base_date": "2026-09-01",
               "horizons": {"next_day": {"range": "±1.5%"}}}

    # ① 噪声带内（跌 0.4%，未超 1% 阈值）→ 不学
    r = A.attribute_one(pred_up, {"change_pct": -0.4},
                        {"change_pct": -0.3}, flat_threshold_pct=1.0)
    assert r["cls"] == A.CLS_NOISE and not r["should_learn"], r

    # ② 大盘同向且可解释占比 ≥60% → 归因市场，不学
    r = A.attribute_one(pred_up, {"change_pct": -3.0}, {"change_pct": -2.5},
                        flat_threshold_pct=1.0, beta_ratio=0.6)
    assert r["cls"] == A.CLS_BETA and not r["should_learn"], r
    assert r["evidence"]["beta_explained_pct"] >= 0.6

    # ③ 大盘跌得少（只解释 20%）→ 判断失误，该学
    r = A.attribute_one(pred_up, {"change_pct": -5.0}, {"change_pct": -1.0},
                        flat_threshold_pct=1.0, beta_ratio=0.6)
    assert r["cls"] == A.CLS_FORECAST_ERR and r["should_learn"], r

    # ④ 大盘反向涨 → 标的大跌，完全是自己的判断问题
    r = A.attribute_one(pred_up, {"change_pct": -4.0}, {"change_pct": 1.5},
                        flat_threshold_pct=1.0, beta_ratio=0.6)
    assert r["cls"] == A.CLS_FORECAST_ERR and r["should_learn"], r

    # ⑤ 方向对但幅度远超预期区间（±1.5% 却涨 6%）→ 策略层可学点
    r = A.attribute_one(pred_up, {"change_pct": 6.0}, {"change_pct": 0.2},
                        flat_threshold_pct=1.0, beta_ratio=0.6)
    assert r["cls"] == A.CLS_DECISION_ERR and r["should_learn"], r

    # ⑥ 方向对、幅度也在预期内 → 判断有效，不产生教训
    r = A.attribute_one(pred_up, {"change_pct": 1.6}, {"change_pct": 0.1},
                        flat_threshold_pct=1.0, beta_ratio=0.6)
    assert r["cls"] == "as_expected" and not r["should_learn"], r

    # 批量：只有可学项才产出 facts
    batch = A.attribute_batch(
        [{"prediction": pred_up, "actual": {"change_pct": -0.4}},
         {"prediction": pred_up, "actual": {"change_pct": -3.0}},
         {"prediction": pred_up, "actual": {"change_pct": -5.0}}],
        market={"change_pct": -2.5}, flat_threshold_pct=1.0, beta_ratio=0.6)
    assert batch["stats"]["total"] == 3
    assert batch["stats"][A.CLS_NOISE] == 1
    assert batch["stats"][A.CLS_BETA] == 1
    assert batch["learnable"] == len(batch["facts"]) == 1
    assert "判断失误" in A.summarize_stats(batch["stats"])


# ---------------------------------------------------------------------------
# 2. 记忆防污染规则
# ---------------------------------------------------------------------------

def _new_trader(name="记忆测试"):
    from core.paper import engine
    return engine.create_account(name, 1000000, mode="trader")


def test_memory_write_permissions_and_evidence_gate():
    """三层记忆的写入权限与证据门：
    - AI 写的 lesson 只能是 candidate，且必须有证据
    - 证据不足不升 active（防止单次运气永久改变行为）
    - 证据够 + 置信度够 → 升 active，并聚合成 belief
    """
    from core.trader import memory as M

    tid = _new_trader("记忆规则测试")

    # 无证据的教训必须被拒
    try:
        M.write_lesson(tid, "没有证据的结论")
        assert False, "无证据的 lesson 应被拒收"
    except M.MemoryError_:
        pass

    f1 = M.write_fact(tid, "事实1：对 A 的预测错误", symbol="510300",
                      scope="symbol:510300", detail={"cls": "forecast_error"},
                      mirror=False)
    f2 = M.write_fact(tid, "事实2：对 A 的预测再次错误", symbol="510300",
                      scope="symbol:510300", detail={"cls": "forecast_error"},
                      mirror=False)

    lid = M.write_lesson(tid, "我对 A 的短线方向判断经常失准",
                         scope="symbol:510300", symbol="510300",
                         evidence_refs=[{"fact_id": f1}, {"fact_id": f2}],
                         proposed_confidence=0.5, mirror=False)
    assert M.recall(tid, kind=M.KIND_FACT, limit=10)  # fact 生来 active

    # 只有 2 条证据 < 门槛 3 → 不能升
    assert M.promote_lessons(tid, min_evidence=3, min_confidence=0.6,
                             mirror=False) == []
    rows = M.recall(tid, kind=M.KIND_LESSON, status=None, limit=10)
    assert rows and rows[0]["status"] == M.ST_CANDIDATE

    # 补第 3 条证据 → 达标升级
    f3 = M.write_fact(tid, "事实3：对 A 的预测第三次错误", symbol="510300",
                      scope="symbol:510300", mirror=False)
    M.add_evidence(tid, lid, {"fact_id": f3})
    changes = M.promote_lessons(tid, min_evidence=3, min_confidence=0.6,
                                mirror=False)
    assert changes and changes[0]["to"] == M.ST_ACTIVE, changes

    # 聚合出 belief
    b = M.rebuild_beliefs(tid, mirror=False)
    assert b["version"] >= 1 and b["beliefs"]
    assert M.recall(tid, kind=M.KIND_BELIEF, limit=5)

    # 反证超阈值 → 降级（记忆可自我纠正）
    for i in range(3):
        M.refute_lesson(tid, lid, {"reason": f"反证{i}"}, mirror=False)
    rows = M.recall(tid, kind=M.KIND_LESSON, status=None, limit=10)
    assert rows[0]["status"] == M.ST_REFUTED, rows[0]

    # Markdown 镜像可生成（人类可审计）
    p = M.render_markdown(tid)
    txt = Path(p).read_text(encoding="utf-8")
    assert "记忆镜像" in txt and "已被反驳" in txt


def test_memory_effectiveness_refutes_harmful_lessons():
    """记忆有效性：生效后判断质量反而变差 → 自动判定有害并记反证。"""
    from core.store.db import session_scope
    from core.store.models import Prediction, PredictionEval
    from core.trader import memory as M
    from core.trader import report as R

    tid = _new_trader("有效性测试")
    f = M.write_fact(tid, "事实：初始", symbol="600000",
                     scope="symbol:600000", mirror=False)
    lid = M.write_lesson(tid, "我应该在 600000 上更激进", scope="symbol:600000",
                         symbol="600000", evidence_refs=[{"fact_id": f}],
                         proposed_confidence=0.9, mirror=False)
    M.promote_lessons(tid, min_evidence=1, min_confidence=0.5, mirror=False)

    act_at = datetime.now() - timedelta(days=10)
    with session_scope() as s:
        m = s.get(M.TraderMemory, lid)
        m.activated_at = act_at
        # 生效前：4 次全对；生效后：4 次全错
        for i in range(4):
            p = Prediction(symbol="600000", asset_type="stock",
                           base_date=date.today() - timedelta(days=20 - i),
                           base_price=10.0, horizons={"next_day": {}})
            s.add(p)
            s.flush()
            s.add(PredictionEval(prediction_id=p.id, horizon="next_day",
                                 due_date=date.today(), hit=True,
                                 evaluated_at=act_at - timedelta(days=i + 1),
                                 predicted_direction="up", actual_direction="up"))
        for i in range(4):
            p = Prediction(symbol="600000", asset_type="stock",
                           base_date=date.today() - timedelta(days=5 - i),
                           base_price=10.0, horizons={"next_day": {}})
            s.add(p)
            s.flush()
            s.add(PredictionEval(prediction_id=p.id, horizon="next_day",
                                 due_date=date.today(), hit=False,
                                 evaluated_at=act_at + timedelta(days=i + 1),
                                 predicted_direction="up", actual_direction="down"))

    res = R.memory_effectiveness(tid, auto_refute=False, min_samples_per_side=3)
    row = next(x for x in res["lessons"] if x["id"] == lid)
    assert row["verdict"] == "harmful", row
    assert row["before_hit_rate"] == 1.0 and row["after_hit_rate"] == 0.0

    # 开启 auto_refute 后应写入反证
    R.memory_effectiveness(tid, auto_refute=True, min_samples_per_side=3)
    after = M.recall(tid, kind=M.KIND_LESSON, status=None, limit=50)
    target = next(x for x in after if x["id"] == lid)
    assert target["counter_evidence"], target


# ---------------------------------------------------------------------------
# 3. 风控门禁
# ---------------------------------------------------------------------------

def test_guard_blocks_and_hints():
    """风控必须拦得住，并且回灌可执行的 hint（拒单即教学）。"""
    from core.trader import guard as G

    # 常规组合：总资产 100 万，现金 50 万，两只各占 25%
    pf = {"cash": 500000.0, "market_value": 500000.0, "total_value": 1000000.0,
          "positions": [{"symbol": "510300", "volume": 62500,
                         "market_value": 250000.0, "available": 62500},
                        {"symbol": "512880", "volume": 250000,
                         "market_value": 250000.0, "available": 250000}]}

    # ① 单一标的上限：510300 已占 25%，想买 20 万 → 只能削到 5 万
    r = G.check_order({"side": "buy", "symbol": "510300", "price": 4.0,
                       "amount": 200000}, pf, {"trades_today": 0})
    assert r["allowed"] and r["rule"] == "clip_weight", r
    assert r["adjusted"] and r["adjusted"]["amount"] == 50000.0, r
    assert r["hint"], r

    # ② 现金不足 → 拒
    #    默认规则下"现金下限 5%"总会比"现金额度"先触顶，故用 rules_override
    #    把另外两个约束放松，隔离出现金这一项来单测。
    pf_cash = {"cash": 800.0, "market_value": 999200.0,
               "total_value": 1000000.0,
               "positions": [{"symbol": "512880", "volume": 1000,
                              "market_value": 999200.0, "available": 1000}]}
    r = G.check_order({"side": "buy", "symbol": "159915", "price": 4.0,
                       "amount": 50000}, pf_cash, {"trades_today": 0},
                      {"max_weight_per_symbol": 1.0,
                       "max_total_position": 1.0, "min_cash_weight": 0.0})
    assert not r["allowed"] and r["rule"] == "insufficient_cash", r
    assert r["hint"], r

    # ③ 单日亏损熔断 → 禁止买入
    r = G.check_order({"side": "buy", "symbol": "159915", "price": 4.0,
                       "amount": 10000}, pf,
                      {"trades_today": 0, "day_start_nav": 1000000.0,
                       "current_nav": 960000.0})
    assert not r["allowed"] and r["rule"] == "daily_loss_fuse", r

    # ④ 单日交易笔数上限
    r = G.check_order({"side": "buy", "symbol": "159915", "price": 4.0,
                       "amount": 10000}, pf, {"trades_today": 3})
    assert not r["allowed"] and r["rule"] == "max_trades_per_day", r

    # ⑤ 单笔最小金额
    r = G.check_order({"side": "buy", "symbol": "159915", "price": 4.0,
                       "amount": 100}, pf, {"trades_today": 0})
    assert not r["allowed"] and r["rule"] == "min_order_amount", r

    # ⑥ 卖出无持仓 → 拒
    r = G.check_order({"side": "sell", "symbol": "159915", "price": 4.0,
                       "volume": 100}, pf, {"trades_today": 0})
    assert not r["allowed"] and r["rule"] == "no_position", r

    # ⑦ T+1：可卖量为 0 → 拒
    pf2 = {"cash": 0.0, "market_value": 1000.0, "total_value": 1000.0,
           "positions": [{"symbol": "600000", "volume": 100,
                          "market_value": 1000.0, "available": 0}]}
    r = G.check_order({"side": "sell", "symbol": "600000", "price": 10.0,
                       "volume": 100}, pf2, {"trades_today": 0})
    assert not r["allowed"] and r["rule"] == "t_plus_1", r

    # ⑧ 黑名单
    r = G.check_order({"side": "buy", "symbol": "000001", "price": 10.0,
                       "amount": 5000}, pf,
                      {"trades_today": 0}, {"banned_symbols": ["000001"]})
    assert not r["allowed"] and r["rule"] == "banned", r

    # ⑨ 回归：削额必须取"所有约束的交集"，不得削出一个下不出来的假额度。
    #    此处单标的上限还剩 25 万、现金 10 万、但总仓位只剩 5 万——
    #    取交集只能削到 5 万。若只按权重削，会回灌 25 万的假额度。
    pf_tight = {"cash": 100000.0, "market_value": 900000.0,
                "total_value": 1000000.0,
                "positions": [{"symbol": "512880", "volume": 85000,
                               "market_value": 850000.0, "available": 85000},
                              {"symbol": "510300", "volume": 12500,
                               "market_value": 50000.0, "available": 12500}]}
    r = G.check_order({"side": "buy", "symbol": "510300", "price": 4.0,
                       "amount": 300000}, pf_tight, {"trades_today": 0})
    assert r["allowed"] and r["rule"] == "clip_position", r
    assert r["adjusted"]["amount"] == 50000.0, r
    assert r["adjusted"]["amount"] <= pf_tight["cash"], r


# ---------------------------------------------------------------------------
# 4. 完整每日循环
# ---------------------------------------------------------------------------

def test_run_day_end_to_end():
    """端到端：感知→对账→归因→反思→决策→风控→执行→留痕 全链路跑通。"""
    from unittest.mock import patch

    from core.paper import engine
    from core.store.db import session_scope
    from core.store.models import Prediction, PredictionEval, TraderMemory, TraderRun
    from core.trader import loop as L
    from core.trader import memory as M

    tid = engine.create_account("端到端测试", 1000000, mode="trader")
    as_of = date(2026, 9, 23)

    # 造一条已到期的错误预测 → 供对账与归因
    with session_scope() as s:
        p = Prediction(symbol="510300", name="沪深300ETF", asset_type="fund",
                       base_date=as_of - timedelta(days=2), base_price=4.0,
                       horizons={"next_day": {"up": 0.6, "flat": 0.2, "down": 0.2}},
                       confidence=60, engine="fused")
        s.add(p)
        s.flush()
        pid = p.id

    def _fake_eval(_as_of=None):
        """假对账：**自己开事务并提交**，忠实模仿真实 evaluate_due。

        这一点很关键：_reconcile_and_attribute 会在对账之后另开 session，
        按 evaluated_at >= t0 回查新增的评估结果。若把 eval 挂在调用方
        尚未提交的外层 session 上，新 session 根本看不见它，
        归因会静默地空手而归（表现为 new_facts=0，很难查）。
        """
        with session_scope() as s2:
            s2.add(PredictionEval(prediction_id=pid, horizon="next_day",
                                  due_date=as_of, actual_price=3.8,
                                  actual_change_pct=-5.0,
                                  actual_direction="down",
                                  predicted_direction="up", hit=False,
                                  brier_score=1.0))
        return 1

    deps = _make_deps(market_change=-1.0, engine=engine,
                      evaluate_due=_fake_eval)
    llm = FakeLLM(
        decision={"actions": [{"side": "buy", "symbol": "510300",
                               "amount": 200000, "reason": "测试买入"}],
                  "rationale": "测试", "watch_reason": ""},
        lessons=[{"statement": "我对沪深300ETF的次日方向判断偏乐观",
                  "scope": "symbol:510300", "symbol": "510300",
                  "evidence_fact_ids": "__FACT__", "_n": 1,
                  "confidence": 0.7}])

    with patch.object(engine, "current_price", return_value=4.0):
        res = L.run_day(tid, as_of, client=llm, deps=deps)

    assert res["status"] in ("ok", "watched"), res
    assert res["error"] == "", res
    assert res["orders"] == 1, res
    assert res["new_facts"] >= 1, res          # 归因后写了 fact
    assert res["new_lessons"] >= 1, res        # 反思提了教训
    assert res["turns"] >= 2, res              # 至少一次工具调用 + 一次决策

    with session_scope() as s:
        run = (s.query(TraderRun).filter_by(trader_id=tid, run_date=as_of).first())
        assert run is not None
        assert run.tool_calls, "工具轨迹必须留痕"
        assert run.perception.get("market_brief")
        cls = [a["cls"] for a in (run.attributions or [])]
        assert "forecast_error" in cls, cls    # 非 beta 的判断失误被识别出来
        assert run.guard_results, "每笔动作都要有风控结论"
        facts = s.query(TraderMemory).filter_by(
            trader_id=tid, kind=M.KIND_FACT).all()
        assert facts, "应写入经验事实"

    # 幂等：同日再跑不会重复交易
    with patch.object(engine, "current_price", return_value=4.0):
        res2 = L.run_day(tid, as_of, client=llm, deps=deps)
    assert res2.get("skipped") is True
    assert res2["orders"] == 1

    # 组合评估可出数
    from core.trader import report as R
    with patch.object(engine, "current_price", return_value=4.0):
        rev = R.trade_review(tid)
        assert rev["runs"] >= 1 and rev["orders"] >= 1
        ov = R.overview(tid)
        assert ov["run_stats"]["total"] >= 1
        assert ov["memory"]["stats"]["counts_all_status"][M.KIND_FACT] >= 1


def test_run_day_watch_is_not_failure():
    """观望是一等公民状态，不是错误。"""
    from unittest.mock import patch

    from core.paper import engine
    from core.store.models import TraderRun
    from core.trader import loop as L

    tid = engine.create_account("观望测试", 500000, mode="trader")
    as_of = date(2026, 9, 22)
    llm = FakeLLM(decision={"actions": [], "rationale": "看不清",
                            "watch_reason": "等待更清晰信号"},
                  lessons=[])
    deps = _make_deps(engine=engine)
    with patch.object(engine, "current_price", return_value=4.0):
        res = L.run_day(tid, as_of, client=llm, deps=deps)
    assert res["status"] == "watched", res
    assert res["orders"] == 0 and res["error"] == ""
    from core.store.db import session_scope
    with session_scope() as s:
        run = (s.query(TraderRun)
               .filter_by(trader_id=tid, run_date=as_of).first())
        assert run.status == "watched"
        assert run.decision.get("watch_reason")


def test_run_day_degrades_without_tool_support():
    """模型不支持 function calling 时走降级路径，仍能出决策。"""
    from unittest.mock import patch

    from core.paper import engine
    from core.trader import loop as L

    tid = engine.create_account("降级测试", 500000, mode="trader")
    as_of = date(2026, 9, 21)
    llm = FakeLLM(decision={"actions": [], "rationale": "降级决策",
                            "watch_reason": "降级观望"},
                  lessons=[], tool_unsupported=True)
    deps = _make_deps(engine=engine)
    with patch.object(engine, "current_price", return_value=4.0):
        res = L.run_day(tid, as_of, client=llm, deps=deps)
    assert res["status"] in ("watched", "ok")
    assert res["turns"] >= 1


def test_price_point_in_time_truncation():
    """时点取价必须严格截到 as_of：取不到未来价格。

    这是回放的命门——一旦 price_on 泄漏了 as_of 之后的价，
    回放就变成"看着答案做题"，所有结论作废。
    """
    from unittest.mock import patch

    import pandas as pd

    from core.data import price as P

    # 故意混入三种 dtype 的日期写法，验证归一化真的生效
    fake = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03",
                                "2026-09-04", "2026-09-08"]),
        "open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
        "close": [10.0, 11.0, 12.0, 13.0, 14.0],
        "volume": [100.0] * 5,
    })

    with patch.object(P, "_norm_history", return_value=fake):
        # as_of 恰为交易日 → 取当日
        assert P.price_on("X", "stock", date(2026, 9, 3)) == 12.0
        # as_of 为周末（9/5 六、9/6 日）→ 退到之前最近交易日 9/4
        assert P.price_on("X", "stock", date(2026, 9, 5)) == 13.0
        assert P.price_on("X", "stock", date(2026, 9, 6)) == 13.0
        # 早于全部数据 → None（不许倒推）
        assert P.price_on("X", "stock", date(2026, 8, 20)) is None

        # prev_close_on 取「严格早于」as_of 的最近一根
        assert P.prev_close_on("X", "stock", date(2026, 9, 3)) == 11.0

        # snapshot_on 的涨跌幅只能用 as_of 当天 vs 前一日
        snap = P.snapshot_on("X", "stock", date(2026, 9, 3))
        assert snap["ok"] is True and snap["price"] == 12.0
        assert abs(snap["change_pct"] - (12.0 / 11.0 - 1) * 100) < 1e-3
        assert "历史口径" in snap["note"]

        # closes_upto 截断 + limit 只从尾部取，且不许出现 as_of 之后的日期
        rows = P.closes_upto("X", "stock", date(2026, 9, 3))
        assert [r["date"] for r in rows] == ["2026-09-01", "2026-09-02",
                                             "2026-09-03"]
        tail = P.closes_upto("X", "stock", date(2026, 9, 3), limit=2)
        assert [r["date"] for r in tail] == ["2026-09-02", "2026-09-03"]
        assert all(r["date"] <= "2026-09-03" for r in rows)

    # 空数据不抛异常，只返回 None / 空
    with patch.object(P, "_norm_history", return_value=pd.DataFrame()):
        assert P.price_on("Y", "stock", date(2026, 9, 3)) is None
        assert P.closes_upto("Y", "stock", date(2026, 9, 3)) == []
        assert P.snapshot_on("Y", "stock", date(2026, 9, 3))["ok"] is False


def test_replay_hides_realtime_only_tools():
    """回放必须把「会泄漏未来」的工具从 AI 可见的工具表里摘掉。

    宁可少给工具，也不能给它一个能偷看未来的工具。
    """
    from core.trader import tools as T

    live = set(T.names(write_allowed=True, replay=False))
    rep = set(T.names(write_allowed=True, replay=True))
    assert T._REPLAY_UNAVAILABLE <= live, live
    assert not (T._REPLAY_UNAVAILABLE & rep), rep   # 已被摘掉
    # 只读且可回溯的工具仍应在场
    assert {"get_quote", "get_kline", "get_metrics", "get_portfolio"} <= rep

    # 即使硬调，也必须明确拒绝而不是静默返回实时数据
    ctx = {"trader_id": 1, "run_date": date(2026, 9, 3), "replay": True,
           "portfolio": None, "cache": {}, "deps": {}}
    res = T.call("get_market_snapshot", {}, ctx)
    assert res["ok"] is False
    assert res["rule"] == "replay_unavailable"


def test_replay_requires_fresh_trader():
    """回放必须跑在全新交易员上，否则持仓/净值会与真实账户混叠。"""
    from unittest.mock import patch

    from core.paper import engine
    from core.trader import loop as L

    tid = engine.create_account("回放非全新", 1000000, mode="trader")
    with patch.object(engine, "current_price", return_value=4.0):
        engine.buy(tid, "510300", amount=100000, asset_type="fund")

    out = L.replay(tid, date(2026, 9, 1), date(2026, 9, 3))
    assert out["ok"] == 0 and out["errors"] == 0
    assert "全新交易员" in out["error"], out

    # allow_existing 放行，并在 note 里明确标注结果不可信
    with patch.object(L, "run_day", return_value={
            "status": "watched", "orders": 0, "nav": 1000000.0,
            "new_facts": 0, "new_lessons": 0, "turns": 1, "cost": 0.0,
            "error": "", "skipped": False}), \
         patch("core.data.price.trading_days_ex",
               return_value=([date(2026, 9, 1)], "index")):
        out2 = L.replay(tid, date(2026, 9, 1), date(2026, 9, 3),
                        allow_existing=True)
    assert out2["days"] == 1 and out2["watched"] == 1, out2
    assert "allow_existing" in out2["note"], out2


def test_replay_drives_trading_days_in_order():
    """回放必须按交易日升序逐日推进，并汇总净值/成本/各类计数。"""
    from unittest.mock import patch

    from core.paper import engine
    from core.trader import loop as L

    tid = engine.create_account("回放驱动", 1000000, mode="trader")
    days = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]
    seen = []

    def _fake_run_day(trader_id, as_of, **kw):
        seen.append((as_of, kw.get("replay")))
        nav = 1000000.0 + len(seen) * 5000
        return {"status": "ok" if len(seen) != 2 else "watched",
                "orders": 1 if len(seen) != 2 else 0, "nav": nav,
                "new_facts": 1, "new_lessons": 2, "turns": 2, "cost": 0.25,
                "error": "", "skipped": False}

    with patch("core.data.price.trading_days_ex",
               return_value=(days, "index")), \
         patch.object(L, "run_day", side_effect=_fake_run_day):
        out = L.replay(tid, date(2026, 9, 1), date(2026, 9, 3))

    assert [d for d, _ in seen] == days, seen          # 严格升序
    assert all(rep is True for _, rep in seen), seen   # 每一日都走回放路径
    assert out["calendar_source"] == "index"
    assert (out["days"], out["ok"], out["watched"], out["errors"]) == (3, 2, 1, 0)
    assert [r["date"] for r in out["runs"]] == [str(d) for d in days]
    assert out["nav_start"] == 1005000.0 and out["nav_end"] == 1015000.0
    assert out["total_return_pct"] > 0
    assert abs(out["cost_total"] - 0.75) < 1e-9
    assert "隐藏" in out["note"]

    # 交易日历降级时必须显式告警（否则净值曲线会多出非交易日）
    with patch("core.data.price.trading_days_ex",
               return_value=(days, "weekday")), \
         patch.object(L, "run_day", side_effect=_fake_run_day):
        out3 = L.replay(tid, date(2026, 9, 1), date(2026, 9, 3))
    assert "降级" in out3["note"], out3["note"]


def test_replay_rejects_bad_range_and_long_span():
    """入参防御：日期非法 / 起晚于止 / 超出 max_days 都要明确拒绝，不静默跑。"""
    from unittest.mock import patch

    from core.paper import engine
    from core.trader import loop as L

    tid = engine.create_account("回放入参", 1000000, mode="trader")

    assert L.replay(tid, None, date(2026, 9, 3))["error"]
    assert "晚于" in L.replay(tid, date(2026, 9, 5), date(2026, 9, 1))["error"]

    many = [date(2026, 1, 1) + timedelta(days=i) for i in range(70)]
    with patch("core.data.price.trading_days_ex", return_value=(many, "index")), \
         patch.object(L, "run_day") as _spy:
        out = L.replay(tid, date(2026, 1, 1), date(2026, 3, 31))
    assert out["errors"] == 0
    assert out["days"] == 70          # 如实报出区间长度，便于判断该缩到多短
    assert "max_days" in out["error"], out
    assert _spy.call_count == 0, "超过 max_days 时不许开跑（每个交易日都是真金白银）"

    # allow_long=True 才放行超长区间
    with patch("core.data.price.trading_days_ex",
               return_value=(many, "index")), \
         patch.object(L, "run_day", return_value={
             "status": "watched", "orders": 0, "nav": 1000000.0,
             "new_facts": 0, "new_lessons": 0, "turns": 1, "cost": 0.0,
             "error": "", "skipped": False}):
        out_long = L.replay(tid, date(2026, 1, 1), date(2026, 3, 31),
                            allow_long=True)
    assert out_long["days"] == 70 and out_long["watched"] == 70, out_long

    with patch("core.data.price.trading_days_ex", return_value=([], "empty")):
        out2 = L.replay(tid, date(2026, 9, 1), date(2026, 9, 3))
    assert "无交易日" in out2["error"], out2


def test_replay_day_end_to_end_point_in_time_only():
    """真实走通回放全链路，并守住铁律：**每一次取价都必须带 as_of**。

    这条测试的价值在于它不 mock run_day —— 从 _perceive 到成交到净值快照
    走的是与实盘完全相同的代码路径，只是 replay=True。
    只要有任何一处忘了传 as_of，成交价就会变成"今天的价"，
    而回放结果看起来仍然"正常"（有净值、有曲线）——极难人工发现。
    """
    from unittest.mock import patch

    import pandas as pd

    from core.data import price as P
    from core.paper import engine
    from core.store.db import session_scope
    from core.store.models import NavHistory, PaperOrder
    from core.trader import loop as L

    tid = engine.create_account("回放端到端", 1000000, mode="trader")
    as_of = date(2026, 9, 3)

    # 9/3 收盘 12.0；若回放误用"实时价"，这个序列根本不会出现
    fake = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-28", "2026-09-01", "2026-09-02",
                                "2026-09-03"]),
        "open": [9.0, 10.0, 11.0, 12.0], "high": [9.0, 10.0, 11.0, 12.0],
        "low": [9.0, 10.0, 11.0, 12.0],
        "close": [9.0, 10.0, 11.0, 12.0], "volume": [1e4] * 4,
    })

    calls: list[tuple] = []
    real_cp = engine.current_price

    def _spy_cp(symbol, asset_type="stock", as_of=None):
        calls.append((symbol, asset_type, as_of))
        return real_cp(symbol, asset_type, as_of)

    llm = FakeLLM(decision={"actions": [{"side": "buy", "symbol": "510300",
                                         "amount": 200000, "reason": "回放买入"}],
                            "rationale": "回放", "watch_reason": ""},
                  lessons=[])

    with patch.object(P, "_norm_history", return_value=fake), \
         patch.object(engine, "current_price", side_effect=_spy_cp):
        res = L.run_day(tid, as_of, client=llm,
                        deps=_make_deps(engine=engine), replay=True)

    assert res["replay"] is True
    assert res["status"] == "ok", res
    assert res["orders"] == 1, res
    assert res["error"] == "", res

    # 铁律 1：回放中每一次取价都带 as_of
    assert calls, "回放必须取过价"
    assert all(a == as_of for _, _, a in calls), calls

    # 铁律 2：成交价是 9/3 的 12.0（历史口径），不是实时价
    with session_scope() as s:
        o = s.query(PaperOrder).filter_by(account_id=tid).first()
        assert o is not None and abs(o.price - 12.0) < 1e-9, (o.price if o else None)

    # 铁律 3：净值快照落在 as_of，而不是今天（否则多日回放会互相覆盖）
    with session_scope() as s:
        nav = s.query(NavHistory).filter_by(account_id=tid).all()
        assert [n.date for n in nav] == [as_of], [n.date for n in nav]

    # 铁律 4：回放不可用的工具不得出现在给 AI 的工具表里
    from core.trader import tools as T
    assert not (T._REPLAY_UNAVAILABLE & set(T.names(replay=True)))


if __name__ == "__main__":
    test_attribution_four_classes()
    test_memory_write_permissions_and_evidence_gate()
    test_memory_effectiveness_refutes_harmful_lessons()
    test_guard_blocks_and_hints()
    test_run_day_end_to_end()
    test_run_day_watch_is_not_failure()
    test_run_day_degrades_without_tool_support()
    test_price_point_in_time_truncation()
    test_replay_hides_realtime_only_tools()
    test_replay_requires_fresh_trader()
    test_replay_drives_trading_days_in_order()
    test_replay_rejects_bad_range_and_long_span()
    test_replay_day_end_to_end_point_in_time_only()
    print("[OK] all trader tests passed")
