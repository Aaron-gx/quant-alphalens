"""自学习校准层：把"引擎说出的概率"校准成"当真概率用的概率"。

为什么需要
----------
LLM 和量化模型都会系统性地说"过头话"：LLM 喜欢报 0.6/0.2/0.2，量化模型在小样本上
更是乱给概率。于是出现两个问题：概率不像概率（Brier 差），置信度不可解释（"模糊"）。

本模块用到期对账的真实结果做监督学习，形成三级递进的自学习闭环：

1. identity     —— 还没有样本：原样输出，并如实标注"未校准"（不假装有模型）
2. temperature  —— 温度缩放：用单个参数把概率收紧/放松，样本 ≥40 即可稳定估计
3. meta_logit   —— 元模型：纯 numpy 多分类逻辑回归，输入"引擎三方向概率 + 舆情分 +
                   置信度 + 引擎来源"，直接学习"该在多大程度上相信这个引擎"

每一步都做时序切分（前 75% 训练、后 25% 样本外评估），把校准前后的 Brier 与方向命中率
一起留痕到 model_training_runs 表 —— 这就是"预测率不断提高"的可核查证据链。
"""
from __future__ import annotations

import pickle
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np

from core.config import get, PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "data" / "models"
# 类别顺序**唯一来源**：core.predict.labels（0=跌 1=平 2=涨）。
# 原先这里自己写了一份 LABELS，而 feature_vector 又按 [涨,平,跌] 排列，
# 两者不一致 → 温度校准档位会把涨/跌概率对调后输出。详见 labels.py。
from core.predict.labels import (LABELS, LABEL_IDX, base_threshold,  # noqa: E402
                                 probs_to_vector, vector_to_probs)

MIN_SAMPLES_TEMP = 40        # 温度缩放的样本门槛
MIN_SAMPLES_META = 80        # 元模型的样本门槛
MIN_TEST = 12                # 样本外测试集最小条数
BUNDLE_TTL_DAYS = 3          # 校准包有效期：过期需重训（由调度器每天重训）


# ---------------- 基础工具 ----------------
# 实现已下沉到 core.ml（解除 quant → calibration 的循环依赖），此处保留旧名以兼容现有调用。
from core.ml import (SoftmaxRegression, TemperatureScaler,  # noqa: E402
                     align_proba as _align_proba,
                     brier_score as _brier,
                     hit_rate as _hit_rate,
                     majority_hit_rate as _majority_hit,
                     random_hit_rate as _random_hit,
                     softmax as _softmax)


def _bundle_path(horizon: str, scope: str = "global", stratum: str = "") -> Path:
    """校准包路径。`stratum` 为空表示全局池。

    分层（stratum）是 P1-5 的修法：原先 scope 只有引擎维度，
    5 个波动率/信噪比差异很大的标的混在一池训练，校准器互相污染——
    对高波动标的学到的"过冲"会被拿去修正低波动标的的概率。
    命名形如 `calib_llm_next_day_sym_510300.pkl` / `..._asset_fund.pkl`。
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{stratum}" if stratum else ""
    return MODEL_DIR / f"calib_{scope}_{horizon}{suffix}.pkl"


def stratum_for(symbol: str = "", asset_type: str = "") -> list[str]:
    """分层候选，**按优先级从细到粗**：单标的 → 资产类型 → 全局。

    apply() 会依次尝试，用第一个样本量达标的层——既享受同质样本的精度，
    又不会因为某标的新上市没有样本而彻底失去校准。
    """
    out: list[str] = []
    if symbol:
        out.append(f"sym_{symbol}")
    if asset_type:
        out.append(f"asset_{asset_type}")
    out.append("")
    return out


# ---------------- 样本构造 ----------------
def feature_vector(row) -> list[float]:
    """CalibSample -> 特征向量（与 meta_logit 输入维度严格对应）。

    前三维**必须**是 [跌, 平, 涨] —— 与标签索引 LABEL_IDX 同序。
    train() 直接拿 X[:, :3] 当 `raw` 去算 Brier / 拟合温度，
    一旦这里排成 [涨,平,跌]，温度档位就会把涨跌对调。
    """
    engine = (row.engine or "llm")
    return [
        float(row.p_down or 0), float(row.p_flat or 0), float(row.p_up or 0),
        float(row.senti_score or 0) / 100.0,
        float(row.confidence or 50) / 100.0,
        1.0 if engine == "fused" else 0.0,
        1.0 if engine == "quant" else 0.0,
        float(row.relevance_cov or 0.0),
    ]


FEATURE_NAMES = ["引擎-跌概率", "引擎-平概率", "引擎-涨概率", "标的舆情分",
                 "引擎置信度", "是否融合引擎", "是否量化引擎", "舆情相关度覆盖率"]


def _load_rows(horizon: str, source: str | None = None, since_days: int | None = 730,
               stratum: str | None = None):
    """取某周期 × 某引擎池（可选再按分层）的样本。

    stratum 形如 `sym_510300` / `asset_fund`；为空则取全局池。
    """
    from core.store.db import session_scope
    from core.store.models import CalibSample
    with session_scope() as s:
        q = s.query(CalibSample).filter(CalibSample.horizon == horizon,
                                        CalibSample.actual_direction.in_(("up", "flat", "down")))
        if source:
            q = q.filter(CalibSample.source == source)
        if since_days:
            q = q.filter(CalibSample.base_date >= date.today() - timedelta(days=since_days))
        if stratum:
            if stratum.startswith("sym_"):
                q = q.filter(CalibSample.symbol == stratum[4:])
            elif stratum.startswith("asset_"):
                q = q.filter(CalibSample.asset_type == stratum[6:])
        rows = q.order_by(CalibSample.base_date.asc(), CalibSample.id.asc()).all()
        for r in rows:
            s.expunge(r)
    return rows


def record_sample(*, engine: str, horizon: str, symbol: str, asset_type: str,
                  base_date: date, probs: dict, actual_direction: str,
                  actual_change_pct: float, flat_threshold: float,
                  source: str = "eval", confidence: int = 50,
                  senti_score: float = 0.0, universe_senti: float = 0.0,
                  relevance_cov: float = 0.0, prediction_id: int = 0,
                  payload: dict | None = None, dedup: bool = True) -> bool:
    """写入一条自学习样本（幂等：同源同引擎同标的同基准日同周期只留一条）。"""
    if actual_direction not in LABEL_IDX:
        return False
    from core.store.db import session_scope
    from core.store.models import CalibSample
    with session_scope() as s:
        if dedup:
            exists = (s.query(CalibSample)
                      .filter_by(source=source, engine=engine, horizon=horizon,
                                 symbol=symbol, base_date=base_date).first())
            if exists:
                return False
        s.add(CalibSample(
            source=source, engine=engine, horizon=horizon, symbol=symbol,
            asset_type=asset_type, base_date=base_date,
            p_up=float(probs.get("up", 0)), p_flat=float(probs.get("flat", 0)),
            p_down=float(probs.get("down", 0)), confidence=int(confidence or 50),
            senti_score=float(senti_score or 0), universe_senti=float(universe_senti or 0),
            relevance_cov=float(relevance_cov or 0),
            actual_direction=actual_direction,
            actual_change_pct=float(actual_change_pct or 0),
            flat_threshold=float(flat_threshold if flat_threshold
                                 else base_threshold()),
            prediction_id=int(prediction_id or 0), payload=payload or {}))
    return True


def record_samples_batch(items: list[dict], dedup: bool = True) -> int:
    """批量写入自学习样本：**一个 session 写完一批**，返回实际写入条数。

    为什么必须批量（性能 + 稳定性）
    -------------------------------
    walk-forward 留档一次产出 300~400 条样本，四个周期合计上千条。
    原实现对每条样本调一次 record_sample（各自开一个 session/连接周期），
    即一次收盘预测要开合上千次数据库连接。
    本机实测这种"长负载 + 高频连接开合"是**进程级崩溃**（无 traceback 猝死）
    的稳定共同项——注意：与 ML 后端无关（LightGBM / sklearn GBDT 崩溃率相当，
    见 core/predict/quant.py:_all_candidates 的实测记录）。
    批量后每个周期只开 1 个 session。

    items 的键与 record_sample 的参数同名；actual_direction 不合法或缺 base_date
    的条目直接跳过（与 record_sample 返回 False 的语义一致）。
    """
    if not items:
        return 0
    from core.store.db import session_scope
    from core.store.models import CalibSample

    valid = [it for it in items
             if it.get("actual_direction") in LABEL_IDX
             and it.get("probs") and it.get("base_date") is not None]
    if not valid:
        return 0

    def _key(it: dict):
        return (it.get("source", "eval"), it["engine"], it["horizon"],
                it["symbol"], it["base_date"])

    added = 0
    with session_scope() as s:
        existing: set = set()
        if dedup:
            syms = sorted({it["symbol"] for it in valid})
            for src, eng, hz, sym, bd in (
                    s.query(CalibSample.source, CalibSample.engine,
                            CalibSample.horizon, CalibSample.symbol,
                            CalibSample.base_date)
                     .filter(CalibSample.symbol.in_(syms)).all()):
                existing.add((src, eng, hz, sym, bd))
        seen: set = set()
        for it in valid:
            k = _key(it)
            if dedup and (k in existing or k in seen):
                continue
            seen.add(k)
            probs = it.get("probs") or {}
            s.add(CalibSample(
                source=it.get("source", "eval"), engine=it["engine"],
                horizon=it["horizon"], symbol=it["symbol"],
                asset_type=it.get("asset_type", "") or "",
                base_date=it["base_date"],
                p_up=float(probs.get("up", 0)), p_flat=float(probs.get("flat", 0)),
                p_down=float(probs.get("down", 0)),
                confidence=int(it.get("confidence", 50) or 50),
                senti_score=float(it.get("senti_score", 0) or 0),
                universe_senti=float(it.get("universe_senti", 0) or 0),
                relevance_cov=float(it.get("relevance_cov", 0) or 0),
                actual_direction=it["actual_direction"],
                actual_change_pct=float(it.get("actual_change_pct", 0) or 0),
                flat_threshold=float(it.get("flat_threshold") or base_threshold()),
                prediction_id=int(it.get("prediction_id", 0) or 0),
                payload=it.get("payload") or {}))
            added += 1
    return added


def sample_stats() -> dict:
    """样本盘点：按周期 × 引擎统计可用样本量（UI 与状态判断用）。"""
    from core.store.db import session_scope
    from core.store.models import CalibSample
    from sqlalchemy import func
    with session_scope() as s:
        rows = (s.query(CalibSample.horizon, CalibSample.engine, CalibSample.source,
                        func.count(CalibSample.id))
                .filter(CalibSample.actual_direction.in_(("up", "flat", "down")))
                .group_by(CalibSample.horizon, CalibSample.engine, CalibSample.source)
                .all())
        total = s.query(func.count(CalibSample.id)).scalar() or 0
    out: dict = {"total": int(total), "by_horizon": {}}
    for horizon, engine, source, n in rows:
        h = out["by_horizon"].setdefault(horizon, {"n": 0, "by_engine": {}, "by_source": {}})
        h["n"] += n
        h["by_engine"][engine] = h["by_engine"].get(engine, 0) + n
        h["by_source"][source] = h["by_source"].get(source, 0) + n
    return out


# ---------------- 校准器 ----------------
# `TemperatureScaler` / `SoftmaxRegression` 已下沉到 core.ml（见文件头说明），
# 由上方 import 引入。此处仅保留历史别名，避免破坏既有引用。
MetaLogit = SoftmaxRegression


def scope_of(engine: str) -> str:
    """按引擎分池训练：量化引擎的历史滚动样本与 LLM 的真实对账样本不能混池。

    量化样本是"用历史数据回溯"生成的，动辄几十条/标的；LLM 样本要等真实到期，
    条数少得多。混在一起训练会让校准器几乎只学会量化引擎的偏差，
    反而把 LLM 的概率校准坏。
    """
    return "quant" if str(engine).lower() == "quant" else "llm"


def _source_of(scope: str) -> str:
    return "walk_forward" if scope == "quant" else "eval"


def _meta_proba(meta, X) -> np.ndarray:
    """元模型概率 → 固定 [跌,平,涨] 三列（缺类补 0 并归一）。

    训练窗口里只有 2 个类别时，分类器的 predict_proba 只返回 2 列；
    而 Brier/命中率是按"列下标 = 类别索引"直接比较的，不对齐就崩或错位。
    """
    proba = meta.predict_proba(X)
    classes = getattr(meta, "classes_", None)
    return _align_proba(proba, classes, k=3)


def _evaluate(raw: np.ndarray, y: np.ndarray, scaler: TemperatureScaler | None,
              meta: MetaLogit | None, X: np.ndarray) -> dict:
    """样本外评估：**BSS 为主，命中率仅供人看**。

    三条基线全部显式给出，否则无法判断"模型是否真有信息量"：
      - brier_clim : 气候学基准（永远输出类别频率），BSS 的分母
      - base_hit   : 多数类基准命中率（最容易伪装成"有效"的假基准）
      - rand_hit   : 均匀随机基准（三分类 = 1/3）

    `*_bss = 1 - brier/brier_clim`，>0 才算有信息量，<=0 等于白干。
    """
    out = {"brier_before": round(_brier(raw, y), 4),
           "hit_before": round(_hit_rate(raw, y), 4)}
    temp = scaler.transform(raw) if scaler else raw
    out["brier_temp"] = round(_brier(temp, y), 4)
    out["hit_temp"] = round(_hit_rate(temp, y), 4)
    if meta is not None:
        # 必须对齐到 3 类列空间：训练窗口只有 2 类时 predict_proba 只给 2 列，
        # 直接送进 _brier 会 IndexError（原先日度重训就栽在这里）
        mp = _meta_proba(meta, X)
        out["brier_meta"] = round(_brier(mp, y), 4)
        out["hit_meta"] = round(_hit_rate(mp, y), 4)

    counts = np.bincount(y, minlength=3).astype(float)
    freq = counts / max(counts.sum(), 1.0)
    brier_clim = float(np.sum(freq * (1.0 - freq)))
    out["base_hit"] = round(float(_majority_hit(y)), 4)
    out["rand_hit"] = round(float(_random_hit(y)), 4)
    out["brier_clim"] = round(brier_clim, 4)
    out["dist"] = {k: int(counts[LABEL_IDX[k]]) for k in LABELS}

    def _bss(b: float) -> float:
        return round(1.0 - b / brier_clim, 4) if brier_clim > 1e-9 else 0.0

    out["bss_before"] = _bss(out["brier_before"])
    out["bss_temp"] = _bss(out["brier_temp"])
    if meta is not None:
        out["bss_meta"] = _bss(out["brier_meta"])
    return out


def train(horizon: str = "next_day", *, scope: str = "llm", stratum: str = "",
          record: bool = True) -> dict:
    """训练某周期 × 某引擎池 × 某分层 的校准器并落盘，同时写入 model_training_runs。

    stratum 为空 = 全局池；`sym_XXXXXX` = 单标的池；`asset_stock` = 资产类型池。
    样本不足时**如实退回 identity 并说明**，不硬用坏模型。
    """
    rows = _load_rows(horizon, source=_source_of(scope), stratum=stratum)
    n = len(rows)
    result = {"horizon": horizon, "scope": scope, "stratum": stratum,
              "n_samples": n, "method": "identity",
              "trained": False, "notes": ""}
    if n < MIN_SAMPLES_TEMP:
        tier = f"（分层 {stratum}）" if stratum else ""
        result["notes"] = (f"{'量化' if scope == 'quant' else 'LLM'}样本不足{tier}"
                           f"（{n}<{MIN_SAMPLES_TEMP}），保持原始概率不做校准")
        if record:
            _record_run(horizon, result, scope=scope, params={"stratum": stratum})
        return result

    X = np.array([feature_vector(r) for r in rows], dtype=float)
    y = np.array([LABEL_IDX[r.actual_direction] for r in rows], dtype=int)
    raw = np.clip(X[:, :3], 1e-6, 1.0)
    raw = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)

    # 时序切分：只用过去推未来，避免"用未来数据训练"的虚高
    n_test = max(MIN_TEST, int(n * 0.25))
    n_test = min(n_test, n - MIN_SAMPLES_TEMP // 2)
    if n_test <= 0 or n - n_test < 20:
        n_test = max(1, n // 5)
    cut = n - n_test
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    raw_tr, raw_te = raw[:cut], raw[cut:]

    scaler = TemperatureScaler().fit(raw_tr, ytr)
    meta = None
    if n >= MIN_SAMPLES_META and cut >= MIN_SAMPLES_META // 2 and len(set(ytr.tolist())) >= 2:
        meta = MetaLogit().fit(Xtr, ytr)
    ev = _evaluate(raw_te, yte, scaler, meta, Xte)

    # 择优：谁在样本外 Brier 最低就用谁；改善不明显则退回更保守的方案
    candidates = [("temperature", ev["brier_temp"]), ]
    if meta is not None:
        candidates.append(("meta_logit", ev["brier_meta"]))
    method, best = min(candidates, key=lambda x: x[1])
    if best >= ev["brier_before"] - 1e-4:      # 校准没带来改善
        method, best = "identity", ev["brier_before"]
    # 全量重训（部署用的模型用全部样本，评估结论仍来自上面的样本外）
    scaler_full = TemperatureScaler().fit(raw, y)
    meta_full = SoftmaxRegression().fit(X, y) if meta is not None else None

    bundle = {
        "horizon": horizon, "scope": scope, "stratum": stratum, "method": method,
        "n": n, "n_test": len(yte), "trained_at": time.time(),
        "temperature": scaler_full, "meta": meta_full if method == "meta_logit" else None,
        "features": FEATURE_NAMES,
        "evidence": ev, "base_hit": ev["base_hit"],
        "bss_before": ev.get("bss_before", 0.0),
        "bss_after": (ev.get("bss_meta") if method == "meta_logit"
                      else ev.get("bss_temp") if method == "temperature"
                      else ev.get("bss_before", 0.0)),
        "importance": (meta_full.importance() if (method == "meta_logit" and meta_full) else {}),
    }
    with open(_bundle_path(horizon, scope, stratum), "wb") as f:
        pickle.dump(bundle, f)
    hit = ev.get("hit_meta", ev.get("hit_before", 0.0)) if method == "meta_logit" \
        else ev.get("hit_temp", ev.get("hit_before", 0.0))
    # 择优判据用 BSS（等价于 Brier，但可跨周期/跨池比较）；命中率只做展示
    bss_after = float(bundle["bss_after"] or 0.0)
    tier = f"分层 {stratum} · " if stratum else ""
    result.update(trained=True, method=method, evidence=ev,
                  bss_after=bss_after,
                  importance=bundle["importance"],
                  notes=(f"{tier}样本外 {len(yte)} 条：Brier {ev['brier_before']} → "
                         f"{round(best, 4)}，BSS {ev.get('bss_before', 0):+.4f} → "
                         f"{bss_after:+.4f}，方向命中 {hit:.1%}"
                         f"（多数类基准 {ev['base_hit']:.1%}／随机 {ev['rand_hit']:.1%}）"))
    if record:
        _record_run(horizon, result, scope=scope,
                    params={"temperature": scaler_full.T, "stratum": stratum,
                            "bss_after": bss_after},
                    evidence=ev, n_test=len(yte), cut=cut)
    return result


def _record_run(horizon: str, result: dict, *, scope: str = "llm", params: dict,
                evidence: dict | None = None, n_test: int = 0, cut: int = 0) -> None:
    from core.store.db import session_scope
    from core.store.models import ModelTrainingRun
    ev = evidence or {}
    method = result.get("method", "identity")
    after = ev.get("brier_before", 0.0)
    if method == "meta_logit":
        after = ev.get("brier_meta", after)
    elif method == "temperature":
        after = ev.get("brier_temp", after)
    with session_scope() as s:
        s.add(ModelTrainingRun(
            scope=scope, horizon=horizon, method=method,
            n_samples=int(result.get("n_samples", 0)), n_train=int(cut), n_test=int(n_test),
            hit_rate=float(ev.get("hit_meta", ev.get("hit_before", 0.0)) or 0.0)
            if method == "meta_logit" else float(ev.get("hit_temp", ev.get("hit_before", 0.0)) or 0.0),
            base_hit_rate=float(ev.get("base_hit", 0.0) or 0.0),
            brier_before=float(ev.get("brier_before", 0.0) or 0.0),
            brier_after=float(after or 0.0),
            params=params, notes=str(result.get("notes", ""))[:250]))


def _available_strata(min_n: int = MIN_SAMPLES_TEMP, cap: int = 12) -> list[str]:
    """列出**值得单独建池**的分层：单标的优先，其次资产类型。

    门槛就用温度缩放的样本门槛——低于它建池也只会退回 identity，
    徒增训练次数。cap 防止标的池扩大后训练次数爆炸。
    """
    from core.store.db import session_scope
    from core.store.models import CalibSample
    from sqlalchemy import func
    with session_scope() as s:
        sym = (s.query(CalibSample.symbol, func.count(CalibSample.id))
               .group_by(CalibSample.symbol)
               .having(func.count(CalibSample.id) >= min_n)
               .order_by(func.count(CalibSample.id).desc()).all())
        ast = (s.query(CalibSample.asset_type, func.count(CalibSample.id))
               .group_by(CalibSample.asset_type)
               .having(func.count(CalibSample.id) >= min_n)
               .order_by(func.count(CalibSample.id).desc()).all())
    out = [f"sym_{a}" for a, _ in sym if a]
    out += [f"asset_{a}" for a, _ in ast if a]
    return out[:cap]


def train_all(*, record: bool = True, strata: bool = True) -> dict:
    """对所有周期 × 两个引擎池（可选再 × 各分层）跑一次训练（调度器每日调用）。

    分层池样本不足时 train() 自己会退回 identity 并留痕，所以这里可以放心广撒网：
    有样本的层自动生效，没样本的层如实记录"样本不足"。
    """
    from core.predict.labels import HORIZONS
    tiers = _available_strata() if strata else []
    out: dict = {}

    def _safe(key: str, **kw) -> None:
        """单个池训练失败不能拖垮整轮重训。

        原先 train_all 没有异常隔离：任一 (周期 × 引擎池 × 分层) 抛错，
        整轮日度重训全部作废，而且错误发生在调度器线程里，很容易被忽略——
        表现就是"校准层悄悄不再更新"。现在如实记下错误、继续跑其他池。
        """
        try:
            out[key] = train(record=record, **kw)
        except Exception as e:      # noqa: BLE001
            out[key] = {"trained": False, "method": "identity",
                        "error": f"{type(e).__name__}: {e}",
                        "notes": f"{key} 训练失败，已跳过（不影响其他池）"}

    for h in HORIZONS:
        for scope in ("llm", "quant"):
            _safe(f"{h}:{scope}", horizon=h, scope=scope)
            for st in tiers:
                _safe(f"{h}:{scope}:{st}", horizon=h, scope=scope, stratum=st)
    return out


def load(horizon: str, scope: str = "llm", stratum: str = "") -> dict | None:
    """读取校准包（超过有效期视为过期，不再使用，避免用陈旧模型骗自己）。"""
    p = _bundle_path(horizon, scope, stratum)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            b = pickle.load(f)
    except Exception:
        return None
    ttl = float(get("predict", "calib_ttl_days", BUNDLE_TTL_DAYS)) * 86400
    if time.time() - float(b.get("trained_at", 0)) > ttl:
        return None
    return b


def load_best(horizon: str, scope: str, symbol: str = "",
              asset_type: str = "") -> tuple[dict | None, str]:
    """按"单标的 → 资产类型 → 全局"顺序取第一个可用校准包，返回 (bundle, 实际分层)。

    为什么要分层回退：同标的样本最同质、校准最准，但新标的一开始没样本；
    此时退到资产类型池，再退到全局池，保证**既有精度又不至于没得用**。
    """
    for st in stratum_for(symbol, asset_type):
        b = load(horizon, scope, st)
        if b:
            return b, st
    return None, ""


def apply(probs: dict, *, horizon: str = "next_day", engine: str = "llm",
          senti_score: float = 0.0, confidence: int = 50,
          relevance_cov: float = 0.0, symbol: str = "",
          asset_type: str = "") -> dict:
    """对一组三方向概率做校准，返回 {probs, method, n, evidence, stratum}。

    没有可用校准包时原样返回并标注 "identity"（如实告知未校准，而不是假装）。
    传入 symbol/asset_type 时会启用**分层回退**（见 load_best）。

    列顺序统一走 `probs_to_vector` / `vector_to_probs`（[跌,平,涨]），
    不再手写 "p[0]→down" 这类映射——那正是涨跌被对调的原因。
    """
    raw = np.array([probs_to_vector(probs)], dtype=float)
    raw = np.clip(raw, 1e-6, 1.0)
    raw = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    scope = scope_of(engine)
    b, st = load_best(horizon, scope, symbol, asset_type)
    if not b:
        return {"probs": vector_to_probs(raw[0]),
                "method": "identity", "n": 0, "evidence": {}, "scope": scope,
                "stratum": "",
                "note": f"{scope} 池尚无校准样本（含所有分层）"}
    method = b.get("method", "identity")
    if method == "meta_logit" and b.get("meta") is not None:
        # 特征顺序（与 feature_vector 严格一致）：跌/平/涨概率 + 舆情分 + 置信度
        # + 融合引擎标记 + 量化引擎标记 + 相关度覆盖率
        X = np.array([[
            raw[0][0], raw[0][1], raw[0][2],
            float(senti_score or 0) / 100.0, float(confidence or 50) / 100.0,
            1.0 if engine == "fused" else 0.0,
            1.0 if engine == "quant" else 0.0,
            float(relevance_cov or 0.0),
        ]])
        p = _meta_proba(b["meta"], X)[0]
    elif method == "temperature" and b.get("temperature") is not None:
        p = b["temperature"].transform(raw)[0]
    else:
        p = raw[0]
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()
    return {"probs": vector_to_probs(p),
            "method": method, "n": int(b.get("n", 0)), "scope": scope,
            "stratum": st, "bss_after": b.get("bss_after"),
            "evidence": b.get("evidence", {}),
            "importance": b.get("importance", {}),
            "trained_at": datetime.fromtimestamp(float(b.get("trained_at", 0))).strftime("%Y-%m-%d %H:%M")}


def status() -> dict:
    """校准层现状（UI 用）：各周期 × 引擎池的校准包 + 样本量 + 学习曲线。"""
    from core.predict.labels import HORIZONS
    from core.store.db import session_scope
    from core.store.models import ModelTrainingRun
    stats = sample_stats()

    def _one(h: str, scope: str, stratum: str = "") -> dict:
        b = load(h, scope, stratum)
        by_source = stats["by_horizon"].get(h, {}).get("by_source", {})
        return {
            "available": bool(b), "method": (b or {}).get("method", "identity"),
            "n": (b or {}).get("n", 0), "stratum": stratum or "全局",
            "bss_after": (b or {}).get("bss_after"),
            "trained_at": datetime.fromtimestamp(float(b["trained_at"])).strftime("%Y-%m-%d %H:%M")
            if b else "",
            "evidence": (b or {}).get("evidence", {}),
            "importance": (b or {}).get("importance", {}),
            "samples": by_source.get(_source_of(scope), 0),
            "min_samples": MIN_SAMPLES_TEMP,
        }

    bundles = {h: {"llm": _one(h, "llm"), "quant": _one(h, "quant")} for h in HORIZONS}
    # 分层池单独列出，让"哪些标的已经能用自己的校准器"一眼可见
    tiered = {}
    for st in _available_strata():
        for h in HORIZONS:
            for scope in ("llm", "quant"):
                node = _one(h, scope, st)
                if node["available"]:
                    tiered.setdefault(st, {})[f"{h}:{scope}"] = node
    with session_scope() as s:
        runs = (s.query(ModelTrainingRun).order_by(ModelTrainingRun.id.desc())
                .limit(30).all())
        curve = [{"时间": str(r.trained_at)[:16], "周期": r.horizon,
                  "引擎池": "量化" if r.scope == "quant" else "LLM",
                  "分层": (r.params or {}).get("stratum") or "全局",
                  "方法": r.method, "样本数": r.n_samples, "样本外命中率": r.hit_rate,
                  "基准命中率": r.base_hit_rate, "校准前Brier": r.brier_before,
                  "校准后Brier": r.brier_after,
                  "校准后BSS": (r.params or {}).get("bss_after"),
                  "说明": r.notes} for r in runs]
    return {"samples": stats, "bundles": bundles, "tiered": tiered, "curve": curve,
            "thresholds": {"temperature": MIN_SAMPLES_TEMP, "meta": MIN_SAMPLES_META}}


def reliability_table(horizon: str = "next_day", buckets: int = 5) -> list[dict]:
    """置信度分桶 ↔ 实际命中率（让"置信度"这个数字可被检验，而不是玄学）。

    只统计真实归档预测的到期结果（source=eval），量化引擎的历史滚动样本不算，
    因为那些样本没有"引擎置信度"这个维度。

    ⚠️ 列顺序必须走 probs_to_vector（[跌,平,涨]）：这里原先按 [涨,平,跌] 建数组，
    却拿 `argmax == LABEL_IDX[实际方向]` 判命中，等于**把涨和跌当成同一个方向**，
    分桶命中率与 Brier 全是错的。
    """
    rows = _load_rows(horizon, source="eval")
    if not rows:
        return []
    out: dict[int, dict] = {}
    for r in rows:
        b = min(buckets - 1, int(float(r.confidence or 50) // (100 / buckets)))
        d = out.setdefault(b, {"n": 0, "hit": 0, "brier": 0.0})
        probs = np.array([probs_to_vector({"up": r.p_up, "flat": r.p_flat,
                                           "down": r.p_down})], dtype=float)
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / probs.sum()
        y = LABEL_IDX[r.actual_direction]
        d["n"] += 1
        d["hit"] += int(int(np.argmax(probs[0])) == y)
        d["brier"] += _brier(probs, np.array([y]))
    step = 100 / buckets
    return [{"置信度区间": f"{int(b * step)}-{int((b + 1) * step - 1)}",
             "样本数": v["n"], "实际命中率": round(v["hit"] / v["n"], 4),
             "Brier": round(v["brier"] / v["n"], 4)}
            for b, v in sorted(out.items())]


def brief(horizons=None) -> dict:
    """引擎自评证据块：把"引擎自己过往表现如何"整理成能进 prompt 的事实。

    这是 P1-4 的修法。原流水线给 `build_data_block` 留了 `calib` 形参，
    却**从来没有传过**——那段「## 引擎历史校准」是死代码，LLM 永远看不到
    自己被校准与对账的历史：自学习成果只写进了数据库，没有回流到决策输入，
    学习闭环在最后一厘米断掉了。

    每个周期给出四类事实：
      - 校准包：方法 / 样本数 / 校准后 BSS（>0 才算有信息量，≤0 等于瞎猜）
      - 命中率三基线：方向命中率 vs 多数类基准 vs 均匀随机基准
        （命中率必须跟这两个基准比才有意义，否则类别不平衡能骗出"体面"的数字）
      - 置信度分桶命中率：这个"置信度"数字到底值不值钱
      - 实际方向分布：让模型知道该周期的基准率，而不是凭空假设
    无样本的周期如实写明"未生效"，不留空让模型自由发挥。
    """
    from core.predict.labels import HORIZONS as _HORIZONS
    out: dict = {}
    for h in (horizons or _HORIZONS):
        rows = _load_rows(h, source="eval")
        bundle, stratum = load_best(h, "llm", "", "")
        bundle = bundle or {}
        item: dict = {
            "到期对账样本数": len(rows),
            "校准方法": bundle.get("method", "identity"),
            "校准样本数": int(bundle.get("n", 0) or 0),
            "校准分层": stratum or "全局",
            "校准后BSS": bundle.get("bss_after"),
        }
        if rows:
            y = np.array([LABEL_IDX[r.actual_direction] for r in rows], dtype=int)
            probs = np.array([probs_to_vector({"up": r.p_up, "flat": r.p_flat,
                                               "down": r.p_down}) for r in rows],
                             dtype=float)
            probs = np.clip(probs, 1e-9, 1.0)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            item["方向命中率"] = round(float(_hit_rate(probs, y)), 4)
            item["多数类基准"] = round(float(_majority_hit(y)), 4)
            item["随机基准"] = round(float(_random_hit(y)), 4)
            item["实际方向分布"] = {LABELS[i]: int((y == i).sum()) for i in range(3)}
        else:
            item["说明"] = "该周期尚无到期对账样本，校准层未生效（概率为原始输出）"
        rel = reliability_table(h)
        if rel:
            item["置信度分桶命中率"] = rel
        out[h] = item
    return out
