"""定时任务调度：盘前简报、盘中异动、盘后预测、到期对账、净值快照、补打分。

用法：python -m jobs.scheduler  （常驻进程；docker 里独立容器）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from core.config import get


def _hhmm(key: str, default: str) -> tuple[int, int]:
    t = get("schedule", key, default)
    h, m = str(t).split(":")
    return int(h), int(m)


def job_morning_brief():
    """盘前简报：隔夜新闻 + 指数 → LLM 简报，存档可查。"""
    from core.intel.brief import morning_brief
    from core.predict.llm import LLMClient
    if not LLMClient().available():
        return
    try:
        text = morning_brief()
        print(f"[scheduler] 盘前简报已生成（{len(text)}字）")
    except Exception as e:
        print(f"[scheduler] 盘前简报失败: {e}")


def job_anomaly_scan():
    """盘中异动：自选标的涨跌超阈值 → 归因。"""
    from core.intel.brief import anomaly_scan
    try:
        found = anomaly_scan(threshold_pct=float(get("schedule", "anomaly_pct", 5.0)))
        for a in found:
            print(f"[scheduler] 异动 {a['name']}({a['symbol']}) {a['change_pct']:+.2f}% "
                  f"| {a.get('reason', '')[:60]}")
    except Exception as e:
        print(f"[scheduler] 异动扫描失败: {e}")


def job_eod_predict():
    """收盘后对自选标的逐个跑预测。"""
    from core.store.db import session_scope
    from core.store.models import Watchlist
    from core.predict.pipeline import predict
    from core.predict.llm import LLMClient, LLMError
    client = LLMClient()
    if not client.available():
        print("[scheduler] LLM 未配置，跳过盘后预测")
        return
    with session_scope() as s:
        symbols = [w.symbol for w in s.query(Watchlist).filter_by(auto_predict=True)]
    for sym in symbols:
        try:
            predict(sym, client=client)
            print(f"[scheduler] 预测完成 {sym}")
        except LLMError as e:
            print(f"[scheduler] 预测失败 {sym}: {e}")
        except Exception as e:
            print(f"[scheduler] 数据异常 {sym}: {e}")


def job_ai_trade():
    """尾盘执行 ai_signal 账户的自动调仓。"""
    from core.paper.auto_trade import run_all
    try:
        for aid, logs in run_all().items():
            print(f"[scheduler] AI账户{aid}: " + " | ".join(logs))
    except Exception as e:
        print(f"[scheduler] AI调仓失败: {e}")


def job_evaluate():
    """到期对账 → 顺手把新样本喂给自学习校准层。"""
    from core.verify.evaluator import evaluate_due, maybe_retrain
    n = evaluate_due()
    print(f"[scheduler] 到期对账完成，新增评估 {n} 条")
    r = maybe_retrain()
    if r and not r.get("skipped"):
        for key, v in r.items():
            if isinstance(v, dict) and v.get("trained"):
                print(f"[scheduler] 校准层重训 {key}: {v.get('method')} | {v.get('notes')}")


def job_trader_daily():
    """AI 交易员的每日自主循环。

    与 job_ai_trade 的区别（两者不可混淆）：
      job_ai_trade      跑的是 ai_signal 账户，按**固定规则**调仓，AI 不决策
      job_trader_daily  跑的是 mode='trader' 的账户，AI 自己选标的、自己下单、
                        自己从错误中总结教训写入记忆并持续修正
    调度顺序上排在 job_ai_trade 之前，避免两者同时打行情接口。
    """
    from core.predict.llm import LLMClient
    from core.trader.loop import run_all

    if not get("trader", "enabled", True):
        print("[scheduler] AI 交易员已禁用（config [trader] enabled=false）")
        return
    client = LLMClient()
    if not client.available():
        print("[scheduler] LLM 未配置，跳过 AI 交易员")
        return
    try:
        results = run_all(client=client)
    except Exception as e:
        print(f"[scheduler] AI 交易员运行失败: {e}")
        return
    if not results:
        print("[scheduler] 无 mode='trader' 的账户，跳过 AI 交易员")
        return
    for tid, r in results.items():
        if r.get("skipped"):
            print(f"[scheduler] 交易员{tid}: 该日已执行，跳过（幂等保护）")
            continue
        msg = (f"[scheduler] 交易员{tid}: {r.get('status')} "
               f"成交{r.get('orders', 0)}笔 净值{r.get('nav', 0):,.0f} "
               f"新事实{r.get('new_facts', 0)} 新教训{r.get('new_lessons', 0)} "
               f"升级{r.get('promotions', 0)} 轮次{r.get('turns', 0)}")
        if r.get("error"):
            msg += f" | 错误: {r['error']}"
        print(msg)


def job_retrain():
    """每日重训：量化模型滚动样本外 + 校准层元模型（形成学习曲线）。"""
    from core.config import get
    from core.predict import calibration
    from core.store.db import session_scope
    from core.store.models import Watchlist
    from core.predict import quant as quant_engine
    from core.data import stock as stock_data
    from core.data import fund as fund_data

    # K 线根数统一走配置，**不要硬编码 250**：
    # purged walk-forward 需要 min_train=250 + purge(horizon) + embargo，
    # 250 根时有效样本仅约 189 行，一条样本外样本都产不出来 →
    # 量化引擎照常"训练成功"，但样本外指标恒为空、融合权重被门控恒判 0，
    # 自学习池也就永远收不到 quant 样本（每日重训等于白跑，且日志看不出来）。
    hist_count = max(400, int(get("predict", "kline_count", 700)))

    # 1) 自选标的逐个重训量化模型（顺带产出 walk-forward 样本外样本）
    with session_scope() as s:
        symbols = [w.symbol for w in s.query(Watchlist)]
    for sym in symbols:
        try:
            info = stock_data.classify_symbol(sym)
            if sym in ("上证指数",) or info["asset_type"] == "ofund":
                df = fund_data.fetch_fund_nav_history(sym, count=hist_count)
            else:
                df = stock_data.fetch_kline(sym, count=hist_count)
            if df is not None and not df.empty:
                r = quant_engine.train(sym, df, max_age_days=0)
                if r.get("ok"):
                    # 只打 BSS：命中率会被类别不平衡骗（平盘占多数时"永远猜平盘"
                    # 也有体面命中率），BSS 才是"到底学到东西没有"的判据。
                    print(f"[scheduler] 量化重训 {sym}: n={r.get('n')} "
                          f"λ={r.get('lam')} 样本外n={r.get('oos_n')} "
                          f"BSS={r.get('oos_bss')} 技能={r.get('has_skill')}")
        except Exception as e:
            print(f"[scheduler] 量化重训失败 {sym}: {e}")

    # 2) 校准层重训（LLM 池 + 量化池 × 各周期）
    for key, v in calibration.train_all(record=True).items():
        print(f"[scheduler] 校准重训 {key}: {v.get('method')} | {v.get('notes')}")


def job_nav_snapshot():
    from core.store.db import session_scope
    from core.store.models import PaperAccount
    from core.paper import engine
    with session_scope() as s:
        ids = [a.id for a in s.query(PaperAccount)]
    for aid in ids:
        try:
            engine.settle_t1(aid)
            engine.snapshot_nav(aid)
        except Exception as e:
            print(f"[scheduler] 净值快照失败 acc={aid}: {e}")


def job_score_news():
    from core.intel.sentiment import score_db_backlog
    n = score_db_backlog()
    if n:
        print(f"[scheduler] 补打分 {n} 条新闻")


def main():
    if not get("schedule", "enabled", True):
        print("[scheduler] 调度已禁用")
        return
    sch = BlockingScheduler(timezone="Asia/Shanghai")

    # 盘前简报（工作日 8:50）
    h, m = _hhmm("morning_brief_time", "08:50")
    sch.add_job(job_morning_brief, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
                id="brief", name="盘前简报")

    # 盘中异动扫描（10:00 / 11:00 / 13:30 / 14:30）
    sch.add_job(job_anomaly_scan,
                CronTrigger(hour="10,11", minute=5, day_of_week="mon-fri"),
                id="anomaly_am", name="盘中异动(上午)")
    sch.add_job(job_anomaly_scan,
                CronTrigger(hour="13,14", minute=35, day_of_week="mon-fri"),
                id="anomaly_pm", name="盘中异动(下午)")

    # 尾盘 AI 自动调仓（14:50，留 10 分钟成交缓冲）
    h, m = _hhmm("ai_trade_time", "14:50")
    sch.add_job(job_ai_trade, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
                id="ai_trade", name="AI信号自动调仓")

    # 盘后预测（收盘后）
    h, m = _hhmm("eod_predict_time", "15:30")
    sch.add_job(job_eod_predict, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
                id="eod_predict", name="收盘后自选标的预测")

    # 到期对账
    h, m = _hhmm("evaluate_time", "18:00")
    sch.add_job(job_evaluate, CronTrigger(hour=h, minute=m), id="evaluate", name="预测到期对账")

    # 净值快照 + T+1 结算
    h, m = _hhmm("nav_snapshot_time", "16:00")
    sch.add_job(job_nav_snapshot, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
                id="nav", name="模拟盘净值快照+T+1结算")

    # AI 交易员自主循环（早于 AI 调仓，避开行情接口拥挤）
    h, m = _hhmm("trader_time", "14:40")
    sch.add_job(job_trader_daily, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
                id="trader_daily", name="AI交易员每日自主循环")

    # 情报补打分（每 30 分钟）
    sch.add_job(job_score_news, "interval", minutes=30, id="news_score", name="情报补打分")

    # 自学习重训（每日 18:20，对账之后）
    h, m = _hhmm("retrain_time", "18:20")
    sch.add_job(job_retrain, CronTrigger(hour=h, minute=m, day_of_week="mon-fri"),
                id="retrain", name="自学习重训(量化+校准)")

    print("[scheduler] 已启动：盘前简报 / 异动扫描 / AI调仓 / AI交易员 / 盘后预测 / "
          "对账 / 净值 / 打分 / 自学习重训")
    sch.start()


if __name__ == "__main__":
    main()
