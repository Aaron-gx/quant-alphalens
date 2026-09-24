"""量化引擎基准评测：purged walk-forward 下的多方案消融对照。

为什么需要这个脚本
------------------
"模型 v2 比 v1 更准"这类结论，如果没有统一的严格样本外协议，就是无法证伪的。
金融时序尤其容易自欺：标签重叠（label overlap）会让普通 K 折 CV 把未来信息
借给训练集，得到虚高的分数，从而**把噪声当成信号发布出去**。

本脚本用同一套 purged + embargoed walk-forward 协议做 2×2 消融，
把「用什么模型」和「用什么分布假设导概率」这两个因素彻底隔离：

  模型维度 : 线性逻辑回归 / 岭回归 / LightGBM 分类 / LightGBM 回归
  分布维度 : 正态 CDF（有参数假设） vs 训练残差经验分布（无分布假设）

变体清单
--------
  A_legacy        旧 16 特征 + 三分类逻辑回归          （改造前的生产实现，基线）
  B_clf_lgbm      新特征 + LightGBM 分类器
  C_reg_norm      新特征 + LightGBM 回归 + 正态 CDF    ← 想当然的"最优解"
  D_reg_emp       新特征 + LightGBM 回归 + 经验分布    ← 预期最优
  E_ridge_norm    新特征 + 岭回归 + 正态 CDF
  F_ridge_emp     新特征 + 岭回归 + 经验分布
  G_logit_new     新特征 + 逻辑回归（隔离"新特征"本身的贡献）

指标体系（三层，缺一不可）
--------------------------
1. 方向命中率  vs 多数类基准命中率   —— 方向层面有没有价值
2. Brier       vs 气候学基准 Brier   —— 概率层面有没有价值
3. Brier Skill Score = 1 - Brier/Brier_clim
   **这是唯一能回答"到底学到东西没有"的指标**：
   BSS <= 0 意味着这个模型不如"直接输出历史类别频率"，不该上生产。

重要结论（本脚本实测得到，见 README/诊断文档）
----------------------------------------------
日收益是**尖峰厚尾**分布（过量峰度中位数 ~9），真实 P(|r|<1%) ≈ 0.57，
而正态假设只给 0.43 —— 系统性低估平盘质量 +0.14。
后果：正态 CDF 导出的概率会把"平盘"永远压在 argmax 之外，命中率崩到 0.23。
=> **必须用训练残差的经验分布，不能用正态假设。**

用法
----
    .venv/Scripts/python.exe scripts/quant_bench.py
    .venv/Scripts/python.exe scripts/quant_bench.py --count 1500 --threshold 0.5
    .venv/Scripts/python.exe scripts/quant_bench.py --variants D_reg_emp,A_legacy
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data.indicators import add_indicators  # noqa: E402

DEFAULT_UNIVERSE = ["510300", "510500", "159915", "512880", "588000",
                    "600519", "000858", "601318", "300750", "600036"]

# (变体名, 是否用扩展特征, 模型类型, 概率导出方式)
VARIANTS: dict[str, tuple[bool, str, str]] = {
    "A_legacy":     (False, "logit",    "class"),
    "B_clf_lgbm":   (True,  "lgbm_clf", "class"),
    "C_reg_norm":   (True,  "lgbm_reg", "normal"),
    "D_reg_emp":    (True,  "lgbm_reg", "empirical"),
    "E_ridge_norm": (True,  "ridge",    "normal"),
    "F_ridge_emp":  (True,  "ridge",    "empirical"),
    "G_logit_new":  (True,  "logit",    "class"),
}
DEFAULT_ORDER = list(VARIANTS)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def probs_from_forecast(mu, sigma, tau: float):
    """由「预测收益 mu + 残差尺度 sigma」导出三方向概率（**正态假设**）。

    ⚠️ 仅作为对照保留。实测日收益峰度 ~9，正态会低估平盘质量约 0.14，
    用它导概率会把平盘永远压在 argmax 之外 —— 见 diag_kurtosis.py。
    """
    sigma = max(float(sigma), 1e-6)
    z_up = (float(mu) - tau) / sigma
    z_dn = (-tau - float(mu)) / sigma
    p_up = 1.0 - norm_cdf(z_up)
    p_dn = norm_cdf(z_dn)
    p_fl = max(0.0, 1.0 - p_up - p_dn)
    s = p_up + p_fl + p_dn
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_up / s, p_fl / s, p_dn / s


def _ecdf(resid: np.ndarray):
    """训练残差的**经验累积分布**（无分布假设）。

    返回 (sorted_resid, quantile_grid)，用 np.interp 线性插值求 F(x)。
    相比正态假设，它天然保留了残差在 0 附近的质量堆积（尖峰），
    因此"平盘"概率不会被压没 —— 这正是修正命中率崩塌的关键。
    """
    r = np.sort(np.asarray(resid, dtype=float))
    n = len(r)
    if n == 0:
        return np.array([0.0]), np.array([0.5])
    q = (np.arange(n) + 0.5) / n
    return r, q


def probs_from_empirical(mu, resid: np.ndarray, tau: float) -> np.ndarray:
    """由「预测收益 mu + 残差经验分布」导出三方向概率（向量化）。

    r = mu + eps,  eps ~ F_empirical
      P(跌) = P(r < -tau) = F(-tau - mu)
      P(涨) = P(r > +tau) = 1 - F(+tau - mu)
      P(平) = 1 - P(涨) - P(跌)
    """
    r, q = _ecdf(resid)
    mu = np.asarray(mu, dtype=float)
    p_dn = np.interp(-tau - mu, r, q, left=0.0, right=1.0)
    p_up = 1.0 - np.interp(tau - mu, r, q, left=0.0, right=1.0)
    p_fl = np.maximum(0.0, 1.0 - p_up - p_dn)
    P = np.column_stack([p_dn, p_fl, p_up])          # 列序 [down, flat, up]
    P = np.clip(P, 1e-9, None)
    return P / P.sum(axis=1, keepdims=True)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _brier_rows(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐样本 Brier（用于配对的标准误估计，进而支撑 1-SE 规则）。"""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return ((probs - onehot) ** 2).sum(axis=1)


def label_of(ret_pct, tau: float) -> int:
    """0=跌 1=平 2=涨（阈值 tau，单位百分点）。"""
    if ret_pct > tau:
        return 2
    if ret_pct < -tau:
        return 0
    return 1


# μ 收缩系数网格。λ=0 ⇒ 完全不信任模型的择时方向（退化为气候学基线，BSS≡0）；
# λ=1 ⇒ 完全信任。在训练集内部的验证切片上按 Brier 择优，
# 这就给了一个**结构性保证**：模型绝不会比"永远猜多数类"更差。
SHRINK_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)

# λ 的选择用「一标准差规则」(1-SE rule)，见 _fit_regression_with_residuals：
# 先取验证集 Brier 最优的 λ，再在「与最优值相差 <1 个标准误」的候选里取**最小**的 λ。
# 动机：验证切片只有几十条样本，Brier 上 1% 的"优势"完全在噪声内；
# 直接取 argmin 等于让 λ 拟合验证集噪声（实测 h=20 把 BSS 从 0 推到 −0.13）。


def _derive_probs(mu, resid, tau: float, dist_kind: str) -> np.ndarray:
    """按指定分布假设把「预测收益 mu + 残差分布」导成三方向概率。"""
    if dist_kind == "normal":
        sigma = float(np.std(resid))
        return np.array([probs_from_forecast(v, sigma, tau) for v in mu])
    if dist_kind == "empirical":
        return probs_from_empirical(mu, resid, tau)
    raise ValueError(dist_kind)


# ---------------------------------------------------------------------------
# 特征工程
# ---------------------------------------------------------------------------

LEGACY_FEATURES = ["ret1", "ret5", "ret10", "ret20", "ma5_r", "ma10_r", "ma20_r",
                   "ma60_r", "macd_hist", "rsi6", "rsi14", "kdj_j", "boll_pos",
                   "vol_ratio", "amplitude", "turnover_chg"]

NEW_FEATURES = ["ret1", "ret3", "ret5", "ret10", "ret20", "ret60",
                "ma5_r", "ma10_r", "ma20_r", "ma60_r", "ma5_slope", "ma20_slope",
                "macd_hist", "rsi6", "rsi14", "rsi_spread", "kdj_j",
                "boll_pos", "boll_width", "vol_ratio", "vol20", "vol60",
                "atr14", "amplitude", "gap", "turnover_chg",
                "dist_high20", "dist_low20"]


def build_features(df: pd.DataFrame, enriched: bool) -> pd.DataFrame:
    """构造特征矩阵。enriched=True 时用扩展特征集（全部无量纲/平稳化）。"""
    ind = add_indicators(df)
    c = ind["close"]
    f = pd.DataFrame(index=ind.index)
    f["ret1"] = c.pct_change(1) * 100
    f["ret5"] = c.pct_change(5) * 100
    f["ret10"] = c.pct_change(10) * 100
    f["ret20"] = c.pct_change(20) * 100
    for n in (5, 10, 20, 60):
        f[f"ma{n}_r"] = (c / ind[f"ma{n}"] - 1) * 100
    f["macd_hist"] = ind["macd_hist"] / c * 100
    f["rsi6"] = ind["rsi6"]
    f["rsi14"] = ind["rsi14"]
    f["kdj_j"] = ind["kdj_j"]
    rng = (ind["boll_up"] - ind["boll_dn"]).replace(0, np.nan)
    f["boll_pos"] = (c - ind["boll_dn"]) / rng
    f["vol_ratio"] = ind.get("vol_ratio", pd.Series(1.0, index=ind.index))
    f["amplitude"] = (ind["high"] - ind["low"]) / c * 100
    f["turnover_chg"] = ind["volume"].pct_change(1) * 100 if "volume" in ind else 0.0

    if enriched:
        f["ret3"] = c.pct_change(3) * 100
        f["ret60"] = c.pct_change(60) * 100
        f["ma5_slope"] = ind["ma5"].pct_change(3) * 100
        f["ma20_slope"] = ind["ma20"].pct_change(5) * 100
        f["rsi_spread"] = ind["rsi6"] - ind["rsi14"]
        f["boll_width"] = rng / ind["boll_mid"].replace(0, np.nan) * 100
        r1 = c.pct_change(1)
        f["vol20"] = r1.rolling(20).std() * 100
        f["vol60"] = r1.rolling(60).std() * 100
        tr = pd.concat([(ind["high"] - ind["low"]),
                        (ind["high"] - c.shift(1)).abs(),
                        (ind["low"] - c.shift(1)).abs()], axis=1).max(axis=1)
        f["atr14"] = tr.rolling(14).mean() / c * 100
        f["gap"] = (ind["open"] / c.shift(1) - 1) * 100
        hi20 = ind["high"].rolling(20).max()
        lo20 = ind["low"].rolling(20).min()
        f["dist_high20"] = (hi20 - c) / c * 100
        f["dist_low20"] = (c - lo20) / c * 100
    return f


def build_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """前瞻收益（百分点）：close[t+h]/close[t]-1。"""
    return (df["close"].shift(-horizon) / df["close"] - 1) * 100


# ---------------------------------------------------------------------------
# Purged + Embargoed Walk-Forward
# ---------------------------------------------------------------------------

def purged_splits(n: int, horizon: int, n_folds: int = 5, min_train: int = 250,
                  embargo: int = 2):
    """生成 (train_slice, test_slice)，并**剔除标签与测试集重叠的训练样本**。

    为什么必须 purge：样本 t 的标签用到 close[t+h]。若训练集截至 tr_end，
    则索引 [tr_end-h+1, tr_end) 的训练标签已经"看见"测试期价格 —— 这叫
    标签泄漏，会让样本外指标系统性偏乐观。embargo 再额外丢几个样本，
    缓解序列自相关带来的残余泄漏。
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
        tr_hi = max(1, tr_end - horizon - embargo)
        yield slice(0, tr_hi), slice(te_start, te_end)


# ---------------------------------------------------------------------------
# 候选模型
# ---------------------------------------------------------------------------

_LGBM_READY = False


def _ensure_lgbm():
    """导入 LightGBM 之前**必须先 import sklearn**（Windows 上的硬性顺序要求）。

    为什么：lib_lightgbm.dll 的原生运行时依赖（OpenMP/BLAS）需要先由 sklearn
    的依赖链加载；否则 DLL 内的惰性函数指针为 NULL，一旦调用
    LGBM_DatasetSetField 就抛 `OSError: access violation reading 0x0`。

    实测（scripts/probe_lgb_sklearn.py，各 3 次重复）：
        无 sklearn 前置      → FAIL/FAIL/FAIL  必然崩溃
        import sklearn 前置   → OK/OK/OK        稳定可用
        import lightgbm.sklearn 前置 → FAIL（无效，不能替代）
    """
    global _LGBM_READY
    if _LGBM_READY:
        return
    import sklearn  # noqa: F401  必须排在 lightgbm 之前
    import lightgbm  # noqa: F401
    _LGBM_READY = True


def make_lgbm_regressor():
    _ensure_lgbm()
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=300, num_leaves=7, max_depth=3, learning_rate=0.02,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        verbose=-1, random_state=42)


def make_lgbm_classifier():
    _ensure_lgbm()
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=300, num_leaves=7, max_depth=3, learning_rate=0.02,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        class_weight="balanced", verbose=-1, random_state=42)


class NumpyRidge:
    """纯 numpy 岭回归（闭式解，含截距 + 标准化）。无第三方依赖的兜底实现。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.mu = self.sigma = self.w = self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sigma
        n, k = Xs.shape
        A = np.hstack([np.ones((n, 1)), Xs])
        reg = np.eye(k + 1) * self.alpha
        reg[0, 0] = 0.0                      # 不惩罚截距
        coef = np.linalg.solve(A.T @ A + reg, A.T @ y)
        self.b, self.w = coef[0], coef[1:]
        return self

    def predict(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mu) / self.sigma
        return Xs @ self.w + self.b


class LegacyLogit:
    """三分类逻辑回归（复刻 core.predict.calibration.SoftmaxRegression 行为）。"""

    def __init__(self, l2: float = 0.1, lr: float = 0.3, iters: int = 500):
        self.l2, self.lr, self.iters = l2, lr, iters

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.mu, self.sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sigma
        n, k = Xs.shape
        m = len(self.classes_)
        idx = {c: i for i, c in enumerate(self.classes_.tolist())}
        self.W, self.b = np.zeros((k, m)), np.zeros(m)
        if m == 1:
            return self
        oh = np.zeros((n, m))
        for i, yi in enumerate(y.tolist()):
            oh[i, idx[yi]] = 1.0
        for _ in range(self.iters):
            z = Xs @ self.W + self.b
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
            self.W -= self.lr * (Xs.T @ (p - oh) / n + self.l2 * self.W)
            self.b -= self.lr * (p - oh).mean(axis=0)
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mu) / self.sigma
        z = Xs @ self.W + self.b
        if len(self.classes_) == 1:
            return np.ones((Xs.shape[0], 1))
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def probs_from_classes(proba, classes) -> np.ndarray:
    """把分类器输出重排成 [down, flat, up] 三列，缺失类别补 0。"""
    cls = list(classes)
    out = np.zeros((proba.shape[0], 3))
    for j, c in enumerate(cls):
        if int(c) in (0, 1, 2):
            out[:, int(c)] = proba[:, j]
    s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-12)


def _fit_regressor(kind: str, X: np.ndarray, y: np.ndarray):
    make = (make_lgbm_regressor if kind == "lgbm_reg"
            else (lambda: NumpyRidge(alpha=1.0)))
    return make().fit(X, y)


def _fit_regression_with_residuals(kind: str, Xtr: np.ndarray, rtr: np.ndarray,
                                   tau: float, dist_kind: str):
    """回归 + **无泄漏的残差估计** + **μ 收缩系数 λ**（训练内验证集选）。

    为什么不能直接用训练集残差：树模型在训练集上残差天然偏小，
    直接拿来当 sigma 会过度自信。这里先用前 70% 训练、在后 30% 上取残差
    （真正样本外残差），并在这段上网格搜 λ。

    返回 (最终模型, 与 λ 一致的残差样本, λ)。
    """
    cut = max(40, int(len(Xtr) * 0.7))
    probe = _fit_regressor(kind, Xtr[:cut], rtr[:cut])
    mu_val = probe.predict(Xtr[cut:])
    r_val = rtr[cut:]
    y_val = np.array([label_of(v, tau) for v in r_val.tolist()], dtype=int)

    best_lam, best_b, best_resid = 0.0, float("inf"), r_val - r_val.mean()
    if len(np.unique(y_val)) >= 2:
        means: dict[float, float] = {}
        ses: dict[float, float] = {}
        cache: dict[float, np.ndarray] = {}
        for lam in SHRINK_GRID:
            resid = r_val - lam * mu_val          # 收缩后预测的残差
            if np.std(resid) < 1e-9:
                continue
            rows = _brier_rows(_derive_probs(lam * mu_val, resid, tau, dist_kind),
                               y_val)
            cache[lam] = resid
            means[lam] = float(rows.mean())
            ses[lam] = float(rows.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0
        if means:
            # 一标准差规则（1-SE rule）：先找验证集 Brier 最优的 λ，
            # 再在所有"与最优值的差落在 1 个标准误内"的候选里取**最小**的 λ。
            #
            # 为什么必须这样：验证切片只有几十条样本，Brier 上 1% 的"优势"
            # 完全在噪声范围内。直接取 argmin 等于让 λ 去拟合验证集的噪声，
            # 实测在 h=20 上会把 BSS 从 0 附近推到 −0.13。
            # 1-SE 规则的价值在于：**只要没有统计显著的证据，就选最保守的 λ**。
            lam_best = min(means, key=lambda k: means[k])
            tol = means[lam_best] + ses[lam_best]
            for lam in sorted(means):
                if means[lam] <= tol:
                    best_lam, best_b, best_resid = lam, means[lam], cache[lam]
                    break
    if np.std(best_resid) < 1e-9:                 # 退化保护
        best_resid = r_val - r_val.mean()

    final = _fit_regressor(kind, Xtr, rtr)        # 部署模型：全量训练
    return final, np.asarray(best_resid, dtype=float), float(best_lam)


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

def evaluate_variant(symbol: str, kline: pd.DataFrame, *, variant: str,
                     horizon: int, tau: float, min_train: int, n_folds: int,
                     embargo: int) -> dict | None:
    enriched, model_kind, dist_kind = VARIANTS[variant]
    feats = build_features(kline, enriched=enriched)
    cols = NEW_FEATURES if enriched else LEGACY_FEATURES
    fwd = build_target(kline, horizon)
    valid = feats[cols].notna().all(axis=1) & fwd.notna()
    X = feats[cols][valid].reset_index(drop=True)
    r = fwd[valid].reset_index(drop=True)
    y_cls = np.array([label_of(v, tau) for v in r.tolist()], dtype=int)

    n = len(X)
    parts_probs: list[np.ndarray] = []
    parts_clim: list[np.ndarray] = []
    parts_true: list[np.ndarray] = []
    lams: list[float] = []

    for tr, te in purged_splits(n, horizon, n_folds=n_folds,
                                min_train=min_train, embargo=embargo):
        Xtr, Xte = X.iloc[tr].values, X.iloc[te].values
        rtr, rte = r.iloc[tr].values, r.iloc[te].values
        ytr = y_cls[tr]
        if len(np.unique(ytr)) < 2:
            continue

        if dist_kind == "class":
            if model_kind == "logit":
                m = LegacyLogit().fit(Xtr, ytr)
            elif model_kind == "lgbm_clf":
                m = make_lgbm_classifier().fit(Xtr, ytr)
            else:
                raise ValueError(model_kind)
            p = probs_from_classes(m.predict_proba(Xte), m.classes_)
        else:
            model, resid, lam = _fit_regression_with_residuals(
                model_kind, Xtr, rtr, tau, dist_kind)
            lams.append(lam)
            p = _derive_probs(model.predict(Xte) * lam, resid, tau, dist_kind)

        # 气候学基准：用**训练集**的历史类别频率当预测（真正无技能基准）
        freq = np.bincount(ytr, minlength=3).astype(float)
        freq = freq / max(freq.sum(), 1)
        parts_clim.append(np.tile(freq, (len(rte), 1)))

        parts_probs.append(p)
        parts_true.append(np.array([label_of(v, tau) for v in rte.tolist()], dtype=int))

    if not parts_probs:
        return None
    P = np.vstack(parts_probs)
    C = np.vstack(parts_clim)
    y = np.concatenate(parts_true)

    hit = float(np.mean(np.argmax(P, axis=1) == y))
    bri = brier(P, y)
    bri_clim = brier(C, y)
    cnt = np.bincount(y, minlength=3)
    base_hit = float(cnt.max() / max(cnt.sum(), 1))
    return {
        "symbol": symbol, "variant": variant, "n_oos": len(y),
        "hit": hit, "base_hit": base_hit, "hit_edge": hit - base_hit,
        "brier": bri, "brier_clim": bri_clim,
        "bss": 1.0 - bri / bri_clim if bri_clim > 0 else 0.0,
        "dist": cnt.tolist(),
        "lam": float(np.mean(lams)) if lams else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1200)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--min-train", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=2)
    ap.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    ap.add_argument("--variants", default=",".join(DEFAULT_ORDER))
    args = ap.parse_args()

    from core.data import stock as sd

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in want if v not in VARIANTS]
    if bad:
        raise SystemExit(f"未知变体：{bad}；可选 {DEFAULT_ORDER}")

    all_rows: list[dict] = []
    print(f"\n阈值 flat=±{args.threshold}%  周期 h={args.horizon}  "
          f"K线={args.count}  purged walk-forward: {args.folds} 折 / "
          f"min_train={args.min_train} / embargo={args.embargo}")
    print("=" * 100)
    print(f"{'标的':>8} {'方案':<13} {'样本外n':>7} {'命中率':>8} {'基准':>8} "
          f"{'增量':>8} {'Brier':>8} {'气候Brier':>9} {'BSS':>8}")
    print("-" * 100)

    for sym in syms:
        try:
            kline = sd.fetch_kline(sym, count=args.count)
        except Exception as e:
            print(f"{sym:>8} 取数失败 {type(e).__name__}")
            continue
        if kline is None or kline.empty:
            print(f"{sym:>8} 无数据")
            continue
        for variant in want:
            try:
                res = evaluate_variant(sym, kline, variant=variant,
                                       horizon=args.horizon, tau=args.threshold,
                                       min_train=args.min_train,
                                       n_folds=args.folds, embargo=args.embargo)
            except Exception as e:
                print(f"{sym:>8} {variant:<13} 失败 {type(e).__name__}: {str(e)[:50]}")
                continue
            if not res:
                print(f"{sym:>8} {variant:<13} 样本不足")
                continue
            all_rows.append(res)
            print(f"{sym:>8} {variant:<13} {res['n_oos']:>7} "
                  f"{res['hit']:>8.4f} {res['base_hit']:>8.4f} "
                  f"{res['hit_edge']:>+8.4f} {res['brier']:>8.4f} "
                  f"{res['brier_clim']:>9.4f} {res['bss']:>+8.4f}")
        print("-" * 100)

    if not all_rows:
        print("无可用结果")
        return

    df = pd.DataFrame(all_rows)
    print("\n按方案汇总（跨标的中位数，中位数比均值更抗单个标的的极端值）")
    print("=" * 104)
    print(f"{'方案':<13} {'模型':<9} {'分布':<10} {'标的数':>6} {'命中率':>8} "
          f"{'基准':>8} {'增量':>8} {'Brier':>8} {'BSS':>8} {'BSS>0':>7} {'λ均值':>7}")
    print("-" * 104)
    for v in want:
        g = df[df["variant"] == v]
        if g.empty:
            continue
        enriched, model_kind, dist_kind = VARIANTS[v]
        dname = {"class": "分类器", "normal": "正态", "empirical": "经验分布"}[dist_kind]
        lam = "  -  " if g["lam"].isna().all() else f"{g['lam'].mean():>6.2f}"
        print(f"{v:<13} {model_kind:<9} {dname:<10} {len(g):>6} "
              f"{g['hit'].median():>8.4f} {g['base_hit'].median():>8.4f} "
              f"{g['hit_edge'].median():>+8.4f} {g['brier'].median():>8.4f} "
              f"{g['bss'].median():>+8.4f} {int((g['bss'] > 0).sum()):>4}/{len(g)} {lam:>7}")
    print("=" * 104)
    print("判读：BSS <= 0 → 该方案不如『直接输出训练集类别频率』，不该上生产。")
    print("      命中率增量 <= 0 → 方向层面无价值（哪怕 Brier 看起来正常）。")
    print("      λ≈0 → 模型自己承认择时无技能，退化为气候学（BSS 被钉在 0 附近）。")
    print("      对照组：C_reg_norm vs D_reg_emp 只差分布假设 —— 差值就是正态偏差的代价。")


if __name__ == "__main__":
    main()
