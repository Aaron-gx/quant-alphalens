"""量化基线引擎（引擎B）：技术指标 → 前向收益回归 → 残差经验分布 → 三方向概率。

七条设计要点（一次说清，避免后人重走弯路）
==========================================

1. **预测目标是回归，不是三分类。**
   原实现输出三分类（涨/平/跌）。问题是日收益里"平盘"占 57%，
   在弱信息特征集上三分类必然退化成"永远猜平盘"——
   命中率很体面（≈基准），BSS ≤ 0，等于白烧算力还占着 35% 融合权重。
   改为回归前向收益 μ，再由残差分布导出三方向概率：
   "平盘"的质量由残差在 0 附近的堆积自然给出，而不是让模型硬分。

2. **分布假设用经验分布（ECDF），不能用正态。**
   日收益是尖峰厚尾的（过量峰度中位数 ≈9.18）。同一批标的、同一个岭回归，
   只换概率导出方式（scripts/quant_bench.py，10 标的 × 1200 根日线）：
       正态 CDF → 命中率 0.2340，BSS −0.7928
       经验分布 → 命中率 0.4615，BSS −0.1555
   **仅分布假设一项就带来 +0.637 的 BSS 摆动**。原因：正态低估 P(|r|<1%)
   约 0.14（真实 0.5711 vs 正态 0.4309），把"平盘"永远压在 argmax 之外。
   诊断脚本：scripts/diag_kurtosis.py。

3. **μ 收缩系数 λ + 一标准差规则 —— 结构性的"无害保证"。**
   λ=0 ⇒ 完全不信模型的择时方向，概率退化为气候学基线（BSS ≡ 0）。
   λ 在训练内部留出的验证切片上按 Brier 择优，但**用 1-SE 规则取最小的 λ**：
   验证切片只有几十条样本，Brier 上 1% 的"优势"完全在噪声内，
   直接取 argmin 等于让 λ 拟合噪声（实测 h=20 会把 BSS 从 ~0 推到 −0.13）。
   ⇒ **没有统计显著证据时，模型自己承认无技能。**

4. **Purged + embargoed walk-forward（防标签泄漏）。**
   样本 t 的标签用到 close[t+h]。若训练集截至 tr_end，则 [tr_end−h+1, tr_end)
   的训练样本标签已经"看见"测试期价格——这是标签泄漏，会让样本外指标
   系统性偏乐观。必须先剔除（purge），再额外丢几个样本（embargo）
   以缓解序列自相关造成的残余泄漏。

5. **后端：默认 sklearn GBDT，LightGBM 显式开启（quant_backend）。**
   选后端有两道关：
   ① **真拟合探针**（_probe_fit）：判据不是 import 成功，而是"真的 fit 成功"。
      实测存在"import 成功、fit 抛 access violation"的后端，只看 import 会踩雷。
   ② **稳定性策略**（_all_candidates）：本机实测（2026-09-23）LightGBM 在多周期
      训练 + 与行情数据链(akshare/efinance)/sqlite 交替的真实负载下会**进程级硬崩溃**
      （单周期能过、第 2~3 个周期必崩）。硬崩溃无法被 try/except 捕获，所以
      进程内探针 + 降级这套机制对它**先天无效**——只有"不启用"才能规避。
      同负载下 sklearn GBDT 四周期全部稳定完成（各约 0.4s），样本外指标相当。
      ⇒ 默认 quant_backend="auto" 不含 LightGBM；想要的人设 "lgbm"，风险自担。
   降级链：sklearn GBDT → numpy 岭回归（LightGBM 开启时排在最前）。

6. **技能门控融合**：fuse() 的权重由样本外 BSS 反推，BSS ≤ 门槛即归零。
   给零信息量模型固定 35% 权重，等于把 LLM 的概率稀释成噪声。

7. **多周期**：标签步数 / 平盘阈值 / 样本记录 / 模型文件路径全部随 horizon 走。

诚实结论（勿删，勿美化）
========================
本特征集在**日线上没有可提取的方向 alpha**：实测所有方案在 h=5/20 上 BSS 皆 ≤0，
h=1 最好的也只是 +0.005 量级（且 λ→0）。本模块的价值**不是"更准"**，而是：
  ① 用可验证的 BSS 把这个事实暴露出来，而不是用命中率伪装成有效；
  ② 在无技能时**自动退化为无害**（λ→0 + 融合权重 0），
     而不是拿固定权重把噪声灌进 LLM 概率；
  ③ 一旦将来有真信号（更好的特征/频率/标的），同一套协议能立刻把它验出来。

模型存 data/models/{symbol}_{horizon}.pkl，按天复用；pkl 内含 kernel 版本号，
内核语义一变旧模型立即失效重训（否则"跑的还是旧内核"从 UI 上完全看不出）。
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import get, PROJECT_ROOT
from core.data.indicators import add_indicators
from core.ml import (NumpyRidge, brier_score, hit_rate, majority_hit_rate,
                     random_hit_rate)
# 标签口径（周期步数 / 平盘阈值 / 方向判定 / 概率列序）统一从 labels 取，
# 本模块不再自行定义——历史上三处各写一份导致过"训练用 ±0.5%、对账用 ±1.0%"的标签错位。
from core.predict.labels import (DEFAULT_HORIZON, HORIZON_STEPS, HORIZONS,
                                 LABEL_IDX, LABELS, PROB_KEYS, direction_of,
                                 flat_threshold_for)

MODEL_DIR = PROJECT_ROOT / "data" / "models"

# 扩展特征集：28 个，全部无量纲 / 平稳化。
# 必须与 scripts/quant_bench.py 的 NEW_FEATURES **完全一致**——
# 否则"基准脚本证明了有效"和"生产实际在跑什么"就是两件事，结论无法迁移。
FEATURES = ["ret1", "ret3", "ret5", "ret10", "ret20", "ret60",
            "ma5_r", "ma10_r", "ma20_r", "ma60_r", "ma5_slope", "ma20_slope",
            "macd_hist", "rsi6", "rsi14", "rsi_spread", "kdj_j",
            "boll_pos", "boll_width", "vol_ratio", "vol20", "vol60",
            "atr14", "amplitude", "gap", "turnover_chg",
            "dist_high20", "dist_low20"]

# 内核版本标记。写进 pkl，内核语义一变旧模型立即失效重训。
# 为什么必须有：换了内核但特征名恰好没变时，"特征集/阈值/后端等级"三道检查
# 都会放行旧模型，于是继续跑旧内核而 UI 上完全看不出来——本项目真实踩过。
KERNEL_TAG = "reg_emp_lam_v3"

# 后端能力等级：数字越大越强。用于"装了更强的库就立刻重训"。
_BACKEND_RANK = {"numpy_ridge": 1, "gbdt": 2, "lgbm": 3}

# μ 收缩网格与 walk-forward 协议参数（与 scripts/quant_bench.py 保持一致）
SHRINK_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
EMBARGO_STEPS = 2
MIN_TRAIN = 250
N_FOLDS = 5
MIN_SAMPLES = 60

__all__ = ["HORIZONS", "HORIZON_STEPS", "DEFAULT_HORIZON", "PROB_KEYS",
           "flat_threshold_for", "direction_of", "active_backend", "train",
           "predict", "predict_next", "predict_multi", "model_info",
           "model_info_all", "feature_importance", "fuse_weights", "fuse",
           "engine_report", "backend_diagnostics"]


# ---------------------------------------------------------------------------
# 后端（回归器）选择
# ---------------------------------------------------------------------------

_LGBM_READY = False

# 后端探针状态（进程内一次性）。
# _BACKEND_PROBE : 变体标签 → 失败原因（**只记失败**，用于诊断 + 剔除）
# _ALL_CANDIDATES: 惰性构造的全部候选（含 lightgbm 的两套线程参数）
# _BACKEND_CACHE : 探针选出的 (名称, 变体标签, 工厂)
_BACKEND_PROBE: dict[str, str] = {}
_ALL_CANDIDATES: list[tuple[str, str, object]] | None = None
_BACKEND_CACHE: tuple[str, str, object] | None = None


def _ensure_lgbm() -> None:
    """导入 LightGBM 之前**必须先 import sklearn**（Windows 上的硬性顺序要求）。

    为什么：lib_lightgbm.dll 的原生运行时依赖（OpenMP/BLAS）需要先由 sklearn
    的依赖链加载；否则 DLL 内的惰性函数指针为 NULL，一旦调用
    LGBM_DatasetSetField 就抛 `OSError: access violation reading 0x0`。

    实测（scripts/probe_lgb_sklearn.py，各 3 次重复）：
        无 sklearn 前置             → FAIL/FAIL/FAIL  必然崩溃
        import sklearn 前置         → OK/OK/OK        稳定可用
        import lightgbm.sklearn 前置 → FAIL（不能替代）

    ⚠️ 以下两点都要满足才真的可用，缺一不可：
    · 这是**必要不充分**条件：本机出现过"import 成功、fit 仍抛
      `access violation`"的情形（DLL 加载成功但原生运行时不可用）。
    · 即便 import + 单次 fit 都通过，本机 LightGBM 在多周期重负载下仍会
      **进程级硬崩溃**——这类问题探针拦不住，只能靠默认不启用（见 _all_candidates）。
    故本函数仅在 quant_backend="lgbm"（显式开启）时才会被调用。
    """
    global _LGBM_READY
    if _LGBM_READY:
        return
    import sklearn  # noqa: F401  必须排在 lightgbm 之前
    import lightgbm  # noqa: F401
    _LGBM_READY = True


def _lgbm_factory(variant: str = "default"):
    """LightGBM 回归器工厂。

    超参数刻意取"强正则 + 浅树 + 小学习率"：日线信噪比极低，默认参数的 GBDT
    会迅速把训练集噪声记住（训练残差很小、样本外 BSS 极负）。这里不做调参，
    只把正则调到"不该学的时候学不动"。

    variant="single_thread" 额外强制单线程 + row-wise 直方图构建，
    用于规避某些 Windows 环境下 OpenMP 线程池初始化失败导致的 access violation
    （本机的第二道缓解手段，第一道是 _ensure_lgbm 的导入顺序）。
    """
    import lightgbm as lgb
    params = dict(n_estimators=300, num_leaves=7, max_depth=3, learning_rate=0.02,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                  verbose=-1, random_state=42)
    if variant == "single_thread":
        params.update(force_row_wise=True, n_jobs=1)
    return lambda: lgb.LGBMRegressor(**params)


def _backend_preference() -> str:
    """读取 [predict].quant_backend：auto（默认）/ lgbm / gbdt / numpy。

    这是**手动降级开关**：若在本机遇到下面 docstring 描述的进程级崩溃，
    可设为 "gbdt" 或 "numpy" 换取更少的原生依赖（但实测崩溃与后端无关，
    见 _all_candidates 的说明——真正常见的诱因是样本留档路径）。
    """
    try:
        return str(get("predict", "quant_backend", "auto")).strip().lower()
    except Exception:
        return "auto"


def _all_candidates() -> list[tuple[str, str, object]]:
    """按配置给出候选后端：(名称, 变体标签, 工厂)，能力从高到低。

    默认 auto：LightGBM → LightGBM(单线程) → sklearn GBDT → numpy 岭回归，
    每个后端都必须先通过 _probe_fit 真拟合探针才会被采用。

    ⚠️ 关于本机"非确定性进程级崩溃"的实测记录（2026-09-23，勿再重复误判）
    -------------------------------------------------------------------
    现象：在**样本留档打开**（record_samples=True）的长负载下，进程会无 traceback
    猝死（access violation）。曾据此误判为"LightGBM 不稳定"，但复测否定了该结论：
        后端=lgbm : 正常 7 次 / 崩溃 4 次
        后端=gbdt : 正常 3 次 / 崩溃 1 次
    —— 两者崩溃率无显著差异；同一份代码同一模式也会一次崩一次过（非确定性）。
    ⇒ **后端选择不是崩溃的诱因**，不要用"换后端"当解决方案；真正的共同项是
      record_samples=True 的留档路径（已改为批量写入，见 _record_oos）。
    复现/验证工具：scripts/diag_quant_backend.py（子进程隔离，单模式可重跑）。
    """
    global _ALL_CANDIDATES
    if _ALL_CANDIDATES is not None:
        return _ALL_CANDIDATES
    pref = _backend_preference()
    cands: list[tuple[str, str, object]] = []

    if pref in ("auto", "lgbm"):
        try:
            _ensure_lgbm()
            cands.append(("lgbm", "lgbm", _lgbm_factory("default")))
            cands.append(("lgbm", "lgbm/单线程", _lgbm_factory("single_thread")))
        except Exception as e:
            _BACKEND_PROBE["lgbm"] = f"导入失败: {type(e).__name__}: {str(e)[:90]}"

    if pref != "numpy":
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            cands.append(("gbdt", "gbdt", lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.03,
                subsample=0.8, random_state=42)))
        except Exception as e:
            _BACKEND_PROBE["gbdt"] = f"导入失败: {type(e).__name__}: {str(e)[:90]}"
    # 纯 numpy 兜底：岭回归。线性假设 ⇒ 不保证准，只保证流水线不断。
    cands.append(("numpy_ridge", "numpy_ridge", lambda: NumpyRidge(alpha=1.0)))
    _ALL_CANDIDATES = cands
    return cands


def _live_candidates() -> list[tuple[str, str, object]]:
    """剔除本进程内**已确认失败**的后端，只留还值得一试的。"""
    return [c for c in _all_candidates() if c[1] not in _BACKEND_PROBE]


PROBE_ROUNDS = 3      # 探针连续拟合轮数


def _probe_fit(factory, rounds: int | None = None) -> None:
    """**真拟合多轮**（不是 import）；失败抛异常，由调用方记录。

    为什么不能只看 import：实测存在"import 成功、fit 抛 access violation"的后端，
    只看 import 会把它当成可用，然后每次训练都崩。

    为什么是**多轮**而不是单轮：可捕获的失败有时要热身几轮才出现；
    多轮能在选后端阶段就暴露它。代价是毫秒级。

    ⚠️ 边界（必须写清楚，否则后人会误以为探针是万能的）：
    探针**只能拦可捕获的异常**。若后端是"进程级硬崩溃"型不稳定，
    进程内探针要么在探针阶段把进程打死、要么拦不住——两种结果都没有意义。
    这类不稳定只能靠**修掉真实诱因**（本项目是样本留档的高频连接开合，
    已改为批量写入，见 _record_oos）或**子进程隔离**（scripts/diag_quant_backend.py）
    来定位；**换后端无效**（见 _all_candidates 的实测记录）。
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 12))
    y = rng.normal(size=120)
    if rounds is None:
        try:
            rounds = int(get("predict", "quant_probe_rounds", PROBE_ROUNDS))
        except Exception:
            rounds = PROBE_ROUNDS
    for _ in range(max(1, int(rounds))):
        model = factory().fit(X, y)
        pred = np.asarray(model.predict(X[:5]), dtype=float).reshape(-1)
        if not np.all(np.isfinite(pred)):
            raise RuntimeError("预测输出含非有限值")


def _backend() -> tuple[str, object]:
    """返回 (后端名, **回归器**工厂)。**先探针、后使用**，结论进程内缓存。

    降级链：LightGBM → LightGBM(单线程) → sklearn GBDT → numpy 岭回归。
    active_backend() 与 train() 共用这一个缓存，所以 UI 上显示的"当前后端"
    就是真正在训练的那个，不会出现"显示 lgbm、实际在崩 / 实际用线性"的错位
    （本项目真实踩过：装了 lightgbm 而 5 个 pkl 的 backend 全是 numpy_logit）。
    """
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE[0], _BACKEND_CACHE[2]
    for name, label, factory in _live_candidates():
        try:
            _probe_fit(factory)
            _BACKEND_CACHE = (name, label, factory)
            return name, factory
        except Exception as e:
            _BACKEND_PROBE[label] = f"探针失败: {type(e).__name__}: {str(e)[:90]}"
    # 理论上到不了这里（numpy_ridge 无外部依赖）：保底仍给出一个可用工厂
    factory = lambda: NumpyRidge(alpha=1.0)
    _BACKEND_CACHE = ("numpy_ridge", "numpy_ridge(兜底)", factory)
    return "numpy_ridge", factory


def active_backend() -> str:
    """当前环境**实际可训练**的后端名（UI/诊断用，不训练）。"""
    return _backend()[0]


def backend_diagnostics() -> dict:
    """后端探针结论：谁在干活、谁为什么不干活（给 engine_report / UI 用）。

    这是"装了 lightgbm 却在用线性模型"、或"lightgbm 一 fit 就崩所以自动降级"
    唯一能被看见的地方——否则用户只会看到"量化引擎没输出"，无从判断原因。
    """
    _backend()
    return {"active": _BACKEND_CACHE[1] if _BACKEND_CACHE else None,
            "active_name": _BACKEND_CACHE[0] if _BACKEND_CACHE else None,
            "probe": dict(_BACKEND_PROBE)}


# ---------------------------------------------------------------------------
# 特征与目标
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """28 个无量纲技术特征。

    与 scripts/quant_bench.py:build_features(enriched=True) 逐列对齐。
    全部做了无量纲化（比值 / 百分比 / 标准化位置），
    所以同一套权重能跨标的使用，也避免"价格绝对水平"泄漏成特征。
    """
    ind = add_indicators(df)
    c = ind["close"]
    f = pd.DataFrame(index=ind.index)
    f["ret1"] = c.pct_change(1) * 100
    f["ret3"] = c.pct_change(3) * 100
    f["ret5"] = c.pct_change(5) * 100
    f["ret10"] = c.pct_change(10) * 100
    f["ret20"] = c.pct_change(20) * 100
    f["ret60"] = c.pct_change(60) * 100
    for n in (5, 10, 20, 60):
        f[f"ma{n}_r"] = (c / ind[f"ma{n}"] - 1) * 100
    f["ma5_slope"] = ind["ma5"].pct_change(3) * 100
    f["ma20_slope"] = ind["ma20"].pct_change(5) * 100
    f["macd_hist"] = ind["macd_hist"] / c * 100
    f["rsi6"] = ind["rsi6"]
    f["rsi14"] = ind["rsi14"]
    f["rsi_spread"] = ind["rsi6"] - ind["rsi14"]
    f["kdj_j"] = ind["kdj_j"]
    rng = (ind["boll_up"] - ind["boll_dn"]).replace(0, np.nan)
    f["boll_pos"] = (c - ind["boll_dn"]) / rng
    f["boll_width"] = rng / ind["boll_mid"].replace(0, np.nan) * 100
    f["vol_ratio"] = ind.get("vol_ratio", pd.Series(1.0, index=ind.index))
    r1 = c.pct_change(1)
    f["vol20"] = r1.rolling(20).std() * 100
    f["vol60"] = r1.rolling(60).std() * 100
    tr = pd.concat([(ind["high"] - ind["low"]),
                    (ind["high"] - c.shift(1)).abs(),
                    (ind["low"] - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14).mean() / c * 100
    f["amplitude"] = (ind["high"] - ind["low"]) / c * 100
    f["gap"] = (ind["open"] / c.shift(1) - 1) * 100
    f["turnover_chg"] = (ind["volume"].pct_change(1) * 100
                         if "volume" in ind else 0.0)
    hi20 = ind["high"].rolling(20).max()
    lo20 = ind["low"].rolling(20).min()
    f["dist_high20"] = (hi20 - c) / c * 100
    f["dist_low20"] = (c - lo20) / c * 100
    return f


def _forward_return(df: pd.DataFrame, step: int = 1) -> pd.Series:
    """前向收益（**百分点**）：close[t+step]/close[t]-1。这是回归目标 μ 的真值。"""
    return (df["close"].shift(-step) / df["close"] - 1) * 100


def _label_of(ret_pct: float, tau: float) -> int:
    """收益(%) + 阈值(%) → 类别索引。**0=跌 1=平 2=涨**（列序 = LABEL_IDX）。"""
    if ret_pct > tau:
        return 2
    if ret_pct < -tau:
        return 0
    return 1


def _label(df: pd.DataFrame, flat_pct: float, step: int = 1) -> pd.Series:
    """三分类标签。**仅保留给诊断/对照用**——生产训练已改为回归目标。

    注意比较的是"百分点"：_forward_return 返回的就是百分数，
    所以直接和 flat_pct 比，不再需要 /100。
    """
    fwd = _forward_return(df, step)
    y = pd.Series(1, index=df.index)
    y[fwd > flat_pct] = 2
    y[fwd < -flat_pct] = 0
    return y


# ---------------------------------------------------------------------------
# 概率导出：残差经验分布
# ---------------------------------------------------------------------------

def _ecdf(resid) -> tuple[np.ndarray, np.ndarray]:
    """训练残差的**经验累积分布**（无分布假设）。

    返回 (sorted_resid, quantile_grid)，用 np.interp 线性插值求 F(x)。
    相比正态假设，它天然保留了残差在 0 附近的堆积（尖峰），
    因此"平盘"概率不会被压没——这正是修正命中率崩塌的关键所在。
    """
    r = np.sort(np.asarray(resid, dtype=float))
    n = len(r)
    if n == 0:
        return np.array([0.0]), np.array([0.5])
    return r, (np.arange(n) + 0.5) / n


def _probs_from_empirical(mu, resid, tau: float) -> np.ndarray:
    """由「预测收益 μ + 残差经验分布」导出三方向概率（向量化）。

    r = μ + ε,  ε ~ F_经验
      P(跌) = P(r < −τ) = F(−τ − μ)
      P(涨) = P(r > +τ) = 1 − F(+τ − μ)
      P(平) = 1 − P(涨) − P(跌)

    **列序 [跌, 平, 涨]，即 LABELS / LABEL_IDX 的顺序。**
    凡是要把概率排成 numpy 数组的地方列序都必须如此，
    否则 Brier / 命中率会拿错误的列下标去比（本项目真实踩过涨跌对调）。
    """
    r, q = _ecdf(resid)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    p_dn = np.interp(-tau - mu, r, q, left=0.0, right=1.0)
    p_up = 1.0 - np.interp(tau - mu, r, q, left=0.0, right=1.0)
    p_fl = np.maximum(0.0, 1.0 - p_up - p_dn)
    P = np.column_stack([p_dn, p_fl, p_up])
    P = np.clip(P, 1e-9, None)
    return P / P.sum(axis=1, keepdims=True)


def _row_to_probs(row) -> dict:
    """[跌, 平, 涨] → {up, flat, down}（归一）。"""
    arr = np.asarray(row, dtype=float).reshape(-1)
    d = {k: float(arr[LABEL_IDX[k]]) for k in PROB_KEYS}
    total = sum(d.values()) or 1.0
    return {k: d[k] / total for k in PROB_KEYS}


def _brier_rows(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐样本 Brier（用于配对的标准误估计，进而支撑 1-SE 规则）。"""
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return ((P - onehot) ** 2).sum(axis=1)


# ---------------------------------------------------------------------------
# 训练协议：purged walk-forward + λ 收缩
# ---------------------------------------------------------------------------

def _model_path(symbol: str, horizon: str = DEFAULT_HORIZON) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"{symbol}_{horizon}.pkl"


def _purged_splits(n: int, horizon: int, n_folds: int = N_FOLDS,
                   min_train: int = MIN_TRAIN, embargo: int = EMBARGO_STEPS):
    """生成 (train_slice, test_slice)，并**剔除标签与测试期重叠的训练样本**。

    为什么必须 purge：样本 t 的标签用到 close[t+h]。若训练集截至 tr_end，
    则 [tr_end−h+1, tr_end) 的训练标签已经"看见"测试期价格——标签泄漏会让
    样本外指标系统性偏乐观。embargo 再额外丢几个样本，缓解序列自相关。
    """
    if n < min_train + n_folds + horizon + embargo:
        return
    step = max(1, (n - min_train - horizon - embargo) // n_folds)
    for i in range(n_folds):
        tr_end = min_train + i * step
        te_start = tr_end + horizon + embargo
        te_end = min(n, te_start + step)
        if te_end <= te_start:
            break
        yield (slice(0, max(1, tr_end - horizon - embargo)),
               slice(te_start, te_end))


def _fit_regressor(factory, X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return factory().fit(X, y)


def _fit_with_lambda(factory, Xtr, rtr, tau: float):
    """回归 + **无泄漏的残差估计** + **μ 收缩系数 λ**（训练内验证集选）。

    为什么不能直接用训练集残差：树模型在训练集上残差天然偏小，
    直接拿来当分布尺度会过度自信（概率过尖 → Brier 反而变差）。
    这里先用前 70% 训练探针、在后 30% 取残差（真正样本外残差），
    并在这段上按 Brier 网格搜 λ。

    返回 (部署模型, 与 λ 一致的残差样本, λ)。
    """
    Xtr = np.asarray(Xtr, dtype=float)
    rtr = np.asarray(rtr, dtype=float)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(1, -1)
    cut = max(40, int(len(Xtr) * 0.7))
    if cut >= len(Xtr):                      # 样本太少：退化为无验证
        final = _fit_regressor(factory, Xtr, rtr)
        return final, rtr - float(np.mean(rtr)), 0.0

    probe = _fit_regressor(factory, Xtr[:cut], rtr[:cut])
    mu_val = np.asarray(probe.predict(Xtr[cut:]), dtype=float).reshape(-1)
    r_val = rtr[cut:]
    y_val = np.array([_label_of(v, tau) for v in r_val.tolist()], dtype=int)

    best_lam, best_resid = 0.0, r_val - float(np.mean(r_val))
    if len(np.unique(y_val)) >= 2:
        means: dict[float, float] = {}
        ses: dict[float, float] = {}
        cache: dict[float, np.ndarray] = {}
        for lam in SHRINK_GRID:
            resid = r_val - lam * mu_val      # 收缩后预测的残差
            if float(np.std(resid)) < 1e-9:
                continue
            rows = _brier_rows(_probs_from_empirical(lam * mu_val, resid, tau), y_val)
            cache[lam] = resid
            means[lam] = float(rows.mean())
            ses[lam] = (float(rows.std(ddof=1) / np.sqrt(len(rows)))
                        if len(rows) > 1 else 0.0)
        if means:
            # 一标准差规则（1-SE rule）：先取验证集 Brier 最优的 λ，
            # 再在"与最优值相差 <1 个标准误"的候选里取**最小**的 λ。
            # 理由：验证切片只有几十条样本，Brier 上 1% 的"优势"完全在噪声内；
            # 直接 argmin 等于让 λ 拟合验证集噪声（实测 h=20 把 BSS 推到 −0.13）。
            # 它的价值在于：**没有统计显著证据时，选最保守的 λ**。
            lam_best = min(means, key=lambda k: means[k])
            tol = means[lam_best] + ses[lam_best]
            for lam in sorted(means):
                if means[lam] <= tol:
                    best_lam, best_resid = float(lam), cache[lam]
                    break
    if float(np.std(best_resid)) < 1e-9:      # 退化保护：残差全同 → 用去均值残差
        best_resid = r_val - float(np.mean(r_val))

    final = _fit_regressor(factory, Xtr, rtr)  # 部署模型：全量训练
    return final, np.asarray(best_resid, dtype=float), float(best_lam)


def _walk_forward(factory, X: pd.DataFrame, r: pd.Series, tau: float,
                  horizon: int) -> tuple[list[dict], list[float]]:
    """Purged walk-forward：只用"过去"训练、预测"之后"的一段，逐折推进。

    这样得到的概率是真正样本外的，可以放心当校准样本用（不会自欺）。

    **气候学基准 = 该折"λ=0"时模型自己给出的概率**（不是训练集类别频率）。
    为什么必须同源：λ=0 的语义是"完全不信模型的择时方向"，此时模型退化为
    纯气候学预测。若基准改用 `bincount(训练标签)`，就变成拿"残差分布导出的
    连续气候学"去比"离散类别频率"——两套口径测的是不同东西，于是会出现
    **λ=0 却测出 BSS=+0.014** 的假技能（本项目实测过，并据此给模型发了权重）。
    用同一套概率构造做基准后，λ=0 ⇒ BSS 恒等于 0，BSS>0 才真正等价于
    "λ>0 的择时信息带来了增量"。

    返回 (样本列表, 各折 λ 列表)。样本里的 `pos` 是**过滤后**的下标，
    调用方需用 valid 掩码把它映射回原始 K 线位置（见 _record_oos）。
    """
    n = len(X)
    out: list[dict] = []
    lams: list[float] = []
    for tr, te in _purged_splits(n, horizon):
        Xtr, Xte = X.iloc[tr].values, X.iloc[te].values
        rtr, rte = r.iloc[tr].values, r.iloc[te].values
        ytr = np.array([_label_of(v, tau) for v in rtr.tolist()], dtype=int)
        if len(np.unique(ytr)) < 2:           # 单类训练集：无法学，跳过该折
            continue
        try:
            model, resid, lam = _fit_with_lambda(factory, Xtr, rtr, tau)
            mu_te = np.asarray(model.predict(Xte), dtype=float).reshape(-1)
            P = _probs_from_empirical(lam * mu_te, resid, tau)
            # 同源气候学基准：同一残差分布、但 μ 收缩到 0（即 λ=0）
            clim_row = _probs_from_empirical(np.zeros(1), resid, tau)[0]
        except Exception:
            continue
        lams.append(lam)
        clim = [float(clim_row[LABEL_IDX[k]]) for k in LABELS]   # [跌,平,涨]
        for j, idx in enumerate(range(te.start, te.stop)):
            yt = _label_of(float(rte[j]), tau)
            out.append({"pos": int(idx), "probs": _row_to_probs(P[j]),
                        "clim": clim, "actual": LABELS[yt],
                        "actual_pct": round(float(rte[j]), 3)})
    return out, lams


def _metrics(samples: list[dict]) -> dict:
    """样本外指标。**以 BSS 为主指标**——命中率会被类别不平衡骗，BSS 不会。

    - oos_brier      : 模型多分类 Brier（越小越好）
    - oos_brier_clim : 同源气候学基准 Brier（每折用该折 λ=0 的概率预测；
                       与模型同一套残差分布构造，故 λ=0 时 BSS 恰为 0）
    - oos_bss        : 1 − brier/brier_clim，**>0 才算有信息量**；≤0 等于白干
    - oos_hit        : 方向命中率（保留供人看，但不作主判据）
    - oos_base_hit   : 多数类基准命中率（最容易被伪装成"有效"的假基准）
    - oos_rand_hit   : 均匀随机基准（三分类 = 1/3）
    """
    if not samples:
        return {}
    n = len(samples)
    P = np.array([[float(s["probs"].get(k, 0.0)) for k in LABELS]
                  for s in samples], dtype=float)
    P = np.clip(P, 1e-9, None)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    y = np.array([LABEL_IDX.get(s["actual"], 1) for s in samples], dtype=int)

    brier = brier_score(P, y)
    if all(s.get("clim") for s in samples):
        C = np.array([s["clim"] for s in samples], dtype=float)
        C = np.clip(C, 1e-9, None)
        C = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-12)
        brier_clim = brier_score(C, y)
    else:  # 兜底：整段用样本类别频率
        counts = np.bincount(y, minlength=3).astype(float)
        freq = counts / max(counts.sum(), 1.0)
        brier_clim = float(np.sum(freq * (1.0 - freq)))
    bss = (1.0 - brier / brier_clim) if brier_clim > 1e-9 else 0.0

    dist = {k: int(sum(1 for s in samples if s["actual"] == k)) for k in LABELS}
    return {"oos_n": n,
            "oos_hit": round(hit_rate(P, y), 4),
            "oos_brier": round(brier, 4),
            "oos_brier_clim": round(brier_clim, 4),
            "oos_bss": round(bss, 4),
            "oos_base_hit": round(float(majority_hit_rate(y)), 4),
            "oos_rand_hit": round(float(random_hit_rate(y)), 4),
            "oos_dist": dist}


def _has_skill(metrics: dict, min_bss: float | None = None) -> bool:
    """量化引擎是否真有信息量。**这是融合权重门控的唯一依据。**"""
    if not metrics:
        return False
    if min_bss is None:
        min_bss = float(get("predict", "quant_min_bss", 0.0))
    return float(metrics.get("oos_bss", 0.0)) > float(min_bss)


# ---------------------------------------------------------------------------
# 训练 / 预测
# ---------------------------------------------------------------------------

def train(symbol: str, kline: pd.DataFrame, flat_pct: float | None = None,
          max_age_days: int = 5, record_samples: bool = True,
          horizon: str = DEFAULT_HORIZON) -> dict:
    """训练/复用单标的 × 单周期的量化模型。返回状态信息（含样本外指标）。

    缓存复用条件（任一条不满足就重训）：
      1. 未超过 max_age_days
      2. 特征集一致
      3. **内核版本一致**（换内核必须立刻重训，否则跑的还是旧内核）
      4. 后端等级**不低于**当前环境可选后端（装了 lightgbm 要立刻生效）
      5. 平盘阈值一致（阈值变了等于任务定义变了，旧模型作废）
    """
    step = HORIZON_STEPS.get(horizon, 1)
    if flat_pct is None:
        flat_pct = flat_threshold_for(horizon)
    flat_pct = float(flat_pct)
    name, factory = _backend()
    cur_rank = _BACKEND_RANK.get(name, 0)

    p = _model_path(symbol, horizon)
    if p.exists() and time.time() - p.stat().st_mtime < max_age_days * 86400:
        try:
            with open(p, "rb") as f_:
                b = pickle.load(f_)
            if (b.get("features") == FEATURES
                    and b.get("flat_pct") == flat_pct
                    and b.get("kernel") == KERNEL_TAG
                    and _BACKEND_RANK.get(b.get("backend"), 0) >= cur_rank
                    and b.get("resid") is not None):
                return {"ok": True, "reused": True, "backend": b.get("backend"),
                        "horizon": horizon, "flat_pct": flat_pct,
                        "kernel": b.get("kernel"), "lam": b.get("lam"),
                        "n": b.get("n"), **(b.get("oos") or {})}
        except Exception:
            pass

    X = _build_features(kline)
    r = _forward_return(kline, step)
    valid = (X[FEATURES].notna().all(axis=1) & r.notna()
             & kline["close"].shift(-step).notna())
    X = X[FEATURES][valid].reset_index(drop=True)
    r = r[valid].reset_index(drop=True)
    if len(X) < MIN_SAMPLES:
        return {"ok": False, "horizon": horizon,
                "error": f"样本不足({len(X)}<{MIN_SAMPLES})，该周期需要更长历史"}

    # 部署模型：全量训练 + 样本外残差 + λ（与基准脚本同一套协议）。
    # 逐级降级：探针选中的后端若**在真实数据上**仍失败（数值退化/全样本单类等），
    # 继续往下降一级再试，而不是把整个周期判死——量化引擎整体缺席才是最大损失。
    model = resid = lam = None
    backend_label: str | None = None
    tried: list[str] = []
    for name_i, label_i, factory_i in _live_candidates():
        try:
            model, resid, lam = _fit_with_lambda(factory_i, X.values, r.values, flat_pct)
            name, factory, backend_label = name_i, factory_i, label_i
            break
        except Exception as e:
            tried.append(f"{label_i}: {type(e).__name__}: {str(e)[:90]}")
    if model is None:
        return {"ok": False, "horizon": horizon,
                "error": "所有后端均训练失败 —— " + " | ".join(tried)}

    oos_samples, lams = _walk_forward(factory, X, r, flat_pct, step)
    oos = _metrics(oos_samples)
    if record_samples and oos_samples:
        _record_oos(symbol, kline, valid, oos_samples, flat_pct, horizon, step)

    with open(p, "wb") as f_:
        pickle.dump({"model": model, "backend": name, "features": FEATURES,
                     "backend_label": backend_label,
                     "kernel": KERNEL_TAG, "trained_at": time.time(), "n": len(X),
                     "horizon": horizon, "step": step, "oos": oos,
                     "flat_pct": flat_pct, "lam": lam,
                     "resid": np.asarray(resid, dtype=float)}, f_)
    return {"ok": True, "reused": False, "n": len(X), "backend": name,
            "backend_label": backend_label,
            "horizon": horizon, "flat_pct": flat_pct, "kernel": KERNEL_TAG,
            "lam": lam, "mean_fold_lam": (round(float(np.mean(lams)), 4) if lams else None),
            "has_skill": _has_skill(oos), **oos}


def _record_oos(symbol: str, kline: pd.DataFrame, valid, samples: list[dict],
                flat_pct: float, horizon: str = DEFAULT_HORIZON,
                step: int = 1) -> None:
    """把滚动样本外预测写成自学习样本（source=walk_forward, engine=quant）。

    `horizon` 与 `step` 随周期走——原先硬编码 next_day 正是"3/4 周期无量化样本"
    的根因：one_week / one_month / quarter 在架构上就无法接上自学习。
    写入的概率已经是**生产同款**（经验分布 + λ 收缩），所以校准层学到的
    分布与线上输出一致，不会出现"离线样本与线上概率不同源"的错配。

    **一条批次一个 session**（calibration.record_samples_batch）：
    原先每样本一次 session，一次留档要开合上千次连接，是本机进程级崩溃的
    稳定共同项（见 _all_candidates 的实测记录）。这里改成先算完再整批落库。
    """
    if not samples:
        return
    try:
        from core.data.stock import classify_symbol
        from core.predict import calibration
        asset_type = classify_symbol(symbol)["asset_type"]
        dates = pd.to_datetime(kline["date"], errors="coerce").to_numpy()
        closes = pd.to_numeric(kline["close"], errors="coerce").to_numpy()
        positions = np.flatnonzero(np.asarray(valid, dtype=bool))
        items: list[dict] = []
        for s in samples:
            j = int(s.get("pos", -1))
            if j < 0 or j >= len(positions):
                continue
            pos = int(positions[j])
            d = dates[pos] if pos < len(dates) else None
            if d is None or pd.isna(d):
                continue
            pct = float(s.get("actual_pct", 0.0) or 0.0)
            if pos + step < len(closes):
                c0, c1 = closes[pos], closes[pos + step]
                if not pd.isna(c0) and not pd.isna(c1) and float(c0):
                    pct = round((float(c1) / float(c0) - 1) * 100, 3)
            items.append({
                "engine": "quant", "horizon": horizon, "symbol": symbol,
                "asset_type": asset_type, "base_date": pd.Timestamp(d).date(),
                "probs": s["probs"], "actual_direction": s["actual"],
                "actual_change_pct": pct, "flat_threshold": flat_pct,
                "source": "walk_forward", "confidence": 50,
                "payload": {"backend": "walk_forward", "fold_pos": j,
                            "step": step, "kernel": KERNEL_TAG}})
        calibration.record_samples_batch(items)
    except Exception:
        pass  # 样本留档失败不影响预测主流程


def _needs_retrain(path: Path, horizon: str) -> tuple[bool, str]:
    """惰性体检：仅当"模型已过时"时返回 True，避免每次预测都重训。

    过时判定（任一命中即需重训）：
      - 内核版本变更（最高优先级：内核语义变了，别的检查都不可信）
      - 后端低于当前环境可选后端（例如刚装上 lightgbm，旧模型是线性的）
      - 平盘阈值与当前配置/周期不符（任务定义变了，旧模型作废）
      - 特征集不一致（代码升级过特征）
      - 缺少残差分布（旧内核产物，无法导概率）
    只读一次 pickle，代价可忽略。
    """
    if not path.exists():
        return True, "模型不存在"
    try:
        with open(path, "rb") as f_:
            b = pickle.load(f_)
    except Exception as e:
        return True, f"模型不可读({e})"
    if b.get("kernel") != KERNEL_TAG:
        return True, f"内核变更 {b.get('kernel')} → {KERNEL_TAG}"
    cur_rank = _BACKEND_RANK.get(active_backend(), 0)
    old = b.get("backend")
    if _BACKEND_RANK.get(old, 0) < cur_rank:
        return True, f"后端可升级 {old} → {active_backend()}"
    want = flat_threshold_for(horizon)
    if b.get("flat_pct") != want:
        return True, f"平盘阈值变更 {b.get('flat_pct')} → {want}"
    if b.get("features") != FEATURES:
        return True, "特征集变更"
    if b.get("resid") is None:
        return True, "缺少残差分布（旧内核产物）"
    return False, ""


def predict(symbol: str, kline: pd.DataFrame,
            horizon: str = DEFAULT_HORIZON) -> dict | None:
    """输出该周期 {up, flat, down} 概率；模型不可用返回 None。

    已有模型时**不无条件重训**（purged walk-forward 多折很贵），而是先做惰性体检：
    只有"内核/后端/阈值/特征"变更才重训。
    """
    p = _model_path(symbol, horizon)
    stale, why = _needs_retrain(p, horizon)
    if stale:
        r = train(symbol, kline, max_age_days=0, record_samples=(not p.exists()),
                  horizon=horizon)
        if not r.get("ok") and p.exists():
            pass  # 重训失败但旧模型在 → 降级用旧的（返回值带 stale_reason）
    try:
        with open(p, "rb") as f_:
            bundle = pickle.load(f_)
        model = bundle["model"]
        resid = np.asarray(bundle.get("resid"), dtype=float)
        lam = float(bundle.get("lam", 0.0) or 0.0)
        tau = float(bundle.get("flat_pct") or flat_threshold_for(horizon))
        X = _build_features(kline)
        X_last = X.iloc[[-1]][FEATURES]
        if X_last.isna().any(axis=1).iloc[0]:
            return None
        mu = float(np.asarray(model.predict(X_last.values),
                              dtype=float).reshape(-1)[0]) * lam
        out = _row_to_probs(_probs_from_empirical(np.array([mu]), resid, tau)[0])
        out.update(backend=bundle["backend"], n=bundle["n"], horizon=horizon,
                   kernel=bundle.get("kernel"), lam=bundle.get("lam"),
                   flat_pct=bundle.get("flat_pct"), step=bundle.get("step"),
                   **(bundle.get("oos") or {}))
        out["is_stale_backend"] = (
            _BACKEND_RANK.get(bundle.get("backend"), 0)
            < _BACKEND_RANK.get(active_backend(), 0))
        out["is_stale_kernel"] = bundle.get("kernel") != KERNEL_TAG
        out["has_skill"] = _has_skill(out)
        if stale and why:
            out["stale_reason"] = why
        return out
    except Exception:
        return None


def predict_next(symbol: str, kline: pd.DataFrame) -> dict | None:
    """次日概率（向后兼容的薄封装；新代码请用 predict(symbol, kline, horizon)）。"""
    return predict(symbol, kline, DEFAULT_HORIZON)


def predict_multi(symbol: str, kline: pd.DataFrame,
                  horizons=None) -> dict[str, dict]:
    """批量输出多个周期的量化概率：{horizon: probs}。

    某个周期样本不足时静默跳过该周期（返回里没有它），调用方按"缺失"处理
    ——而不是伪造一个均匀分布把它填上（那会让 UI 看不出"这个周期没量化"）。
    """
    out: dict[str, dict] = {}
    for h in (horizons or HORIZONS):
        if h not in HORIZON_STEPS:
            continue
        q = predict(symbol, kline, h)
        if q:
            out[h] = q
    return out


# ---------------------------------------------------------------------------
# 模型体检 / 可解释性
# ---------------------------------------------------------------------------

def model_info(symbol: str, horizon: str = DEFAULT_HORIZON) -> dict:
    """读取已训练模型的状态（UI 展示学习效果用；不触发训练）。"""
    p = _model_path(symbol, horizon)
    if not p.exists():
        return {"trained": False, "horizon": horizon}
    try:
        with open(p, "rb") as f_:
            b = pickle.load(f_)
        cur_rank = _BACKEND_RANK.get(active_backend(), 0)
        resid = b.get("resid")
        oos = b.get("oos") or {}
        return {"trained": True, "backend": b.get("backend"), "n": b.get("n"),
                "backend_label": b.get("backend_label"),
                "horizon": horizon, "flat_pct": b.get("flat_pct"),
                "kernel": b.get("kernel"), "lam": b.get("lam"),
                "resid_n": (len(resid) if resid is not None else 0),
                "stale_kernel": b.get("kernel") != KERNEL_TAG,
                "stale_backend": _BACKEND_RANK.get(b.get("backend"), 0) < cur_rank,
                "trained_at": time.strftime("%Y-%m-%d %H:%M",
                                            time.localtime(b.get("trained_at", 0))),
                "features": len(b.get("features") or []),
                "has_skill": bool(oos) and float(oos.get("oos_bss", 0) or 0) > 0.0,
                **oos}
    except Exception as e:
        return {"trained": False, "horizon": horizon, "error": str(e)}


def model_info_all(symbol: str) -> dict[str, dict]:
    """一个标的在所有周期上的模型状态（诊断"哪些周期根本没模型"）。"""
    return {h: model_info(symbol, h) for h in HORIZONS}


def feature_importance(symbol: str, top: int = 8,
                       horizon: str = DEFAULT_HORIZON) -> dict:
    """模型在看的特征（树模型用 feature_importances_，岭回归用 |权重|）。

    返回归一化后的占比，方便直接对比"哪个特征在起作用"。
    """
    p = _model_path(symbol, horizon)
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f_:
            clf = pickle.load(f_)["model"]
        imp = None
        if hasattr(clf, "feature_importances_"):
            imp = np.abs(np.asarray(clf.feature_importances_, dtype=float))
        elif hasattr(clf, "coef_"):
            imp = np.abs(np.asarray(clf.coef_, dtype=float).reshape(-1))
        elif getattr(clf, "w", None) is not None:
            imp = np.abs(np.asarray(clf.w, dtype=float).reshape(-1))
        if imp is not None and len(imp) == len(FEATURES):
            total = float(imp.sum()) or 1.0
            return {f: round(float(v / total), 4) for f, v in
                    sorted(zip(FEATURES, imp), key=lambda x: -x[1])[:top]}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------

def fuse_weights(quant: dict | None, horizon: str = DEFAULT_HORIZON) -> dict:
    """按量化引擎的样本外技能反推融合权重，并给出决策依据（可审计）。

    规则：
      - 无量化输出 / BSS <= quant_min_bss（默认 0）→ **权重 0**，LLM 概率不被稀释
      - 有技能 → 权重 = base_w × clip(BSS / quant_target_bss, 0, 1)
        即"技能越强给得越多，但不超过配置上限"

    返回 {weight, bss, reason, source}，便于在 UI/日志里解释"为什么这次没用量化"。
    """
    base_w = float(get("predict", "quant_weight", 0.35))
    base_w = max(0.0, min(1.0, base_w))
    min_bss = float(get("predict", "quant_min_bss", 0.0))
    target_bss = float(get("predict", "quant_target_bss", 0.05))

    if not quant:
        return {"weight": 0.0, "bss": None, "source": "none",
                "reason": "无量化输出"}

    bss = float(quant.get("oos_bss", 0.0) or 0.0)
    n = int(quant.get("oos_n", 0) or 0)
    if n <= 0:
        return {"weight": 0.0, "bss": None, "source": "no_oos", "n": 0,
                "reason": "无样本外指标，无法评估技能"}
    if bss <= min_bss:
        return {"weight": 0.0, "bss": round(bss, 4), "source": "gated", "n": n,
                "reason": (f"样本外 BSS {bss:+.4f} ≤ 门槛 {min_bss:+.4f}"
                           f"（无信息量，权重归零以免稀释 LLM）")}
    w = base_w * min(1.0, bss / max(target_bss, 1e-9))
    return {"weight": round(w, 4), "bss": round(bss, 4), "source": "skill", "n": n,
            "reason": (f"样本外 BSS {bss:+.4f} → 权重 {w:.3f}"
                       f"（上限 {base_w:.2f}，达到 {target_bss:+.3f} 即给满）")}


def fuse(llm_horizon: dict, quant: dict | None, quant_weight: float | None = None,
         horizon: str = DEFAULT_HORIZON) -> dict:
    """融合 LLM 与量化概率：加权平均后归一。

    权重来源（优先级）：
      1. 显式传入 quant_weight（测试/离线分析用）
      2. 按样本外 BSS 自适应（fuse_weights），无技能则权重 0 → 等于不融合

    注意：旧版固定 0.35 会把"零信息量模型"的噪声灌进 LLM 概率，
    这是本项目的真实缺陷之一，故改为自适应 + 门控。
    """
    if not quant:
        return llm_horizon
    detail = None
    if quant_weight is None:
        detail = fuse_weights(quant, horizon)
        w = float(detail["weight"])
    else:
        w = max(0.0, min(1.0, float(quant_weight)))
    if w <= 0:
        out = dict(llm_horizon)
        out["_fusion"] = {"weight": 0.0, "applied": False,
                          **({k: v for k, v in (detail or {}).items()} or {})}
        return out
    fused = dict(llm_horizon)
    for k in PROB_KEYS:
        fused[k] = llm_horizon.get(k, 0) * (1 - w) + quant.get(k, 0) * w
    total = sum(fused[k] for k in PROB_KEYS) or 1
    for k in PROB_KEYS:
        fused[k] = round(fused[k] / total, 4)
    fused["_fusion"] = {"weight": round(w, 4), "applied": True,
                        **({k: v for k, v in (detail or {}).items()} or {})}
    return fused


def engine_report(symbol: str, kline: pd.DataFrame | None = None) -> dict:
    """量化引擎自检报告：后端 / 各周期模型 / 技能门控 / 是否需要重训。

    这是"装了 lightgbm 却还在跑线性模型"、"换内核却还在跑旧内核"
    这类问题**唯一能被看见**的地方。
    """
    backend = active_backend()
    info = {h: model_info(symbol, h) for h in HORIZON_STEPS}
    need_retrain = [h for h, v in info.items()
                    if not v.get("trained") or v.get("stale_backend")
                    or v.get("stale_kernel")]
    return {
        "symbol": symbol,
        "active_backend": backend,
        "backend_probe": backend_diagnostics(),
        "kernel": KERNEL_TAG,
        "target": "forward_return_regression + empirical_residual",
        "quant_min_bss": float(get("predict", "quant_min_bss", 0.0)),
        "quant_target_bss": float(get("predict", "quant_target_bss", 0.05)),
        "base_weight": float(get("predict", "quant_weight", 0.35)),
        "models": info,
        "need_retrain": need_retrain,
        "note": ("stale_kernel/stale_backend=True 表示模型是用旧内核或更弱后端训练的，"
                 "predict() 会自动重训；也可手动调 train(max_age_days=0)。"
                 "BSS≤0 是正常结果——日线无方向 alpha 时不应当作故障处理。"
                 "backend_probe.probe 里会说明某个后端为何没被采用"
                 "（如 LightGBM 因进程级硬崩溃风险在 auto 下按策略跳过）。"),
    }
