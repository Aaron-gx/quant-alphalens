"""通用机器学习原语层。

为什么单独一层
--------------
原先 `SoftmaxRegression` / `TemperatureScaler` 住在 `core.predict.calibration`，
而 `core.predict.quant` 为了"没装 ML 依赖也能跑"又反向 import calibration 做兜底——
形成 `quant → calibration → (store)` 的层次倒置：**预测引擎依赖校准层的私有实现**。

把这两件纯数值工具下沉到这里之后：
- `quant`（预测引擎）依赖 `core.ml`（基础能力），方向正确
- `calibration`（校准层）也依赖 `core.ml`，两者是**兄弟关系**而非嵌套
- 依赖图变成 `ml ← predict.{quant,calibration}`，无环

本模块只做数值计算，**不碰任何业务状态、不读写数据库**，可独立单测。
"""
from __future__ import annotations

import numpy as np

# ---------------- 基础数值 ----------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """多分类 Brier：mean(sum((p-o)^2))，越小越好（0 最好，2 最差）。

    本项目把它当作**唯一可信的准确率主指标**：命中率会被类别不平衡骗，
    Brier 不会——它同时惩罚"方向错"和"自信地错"。
    """
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def hit_rate(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == y))


def majority_hit_rate(y: np.ndarray, k: int = 3) -> float:
    """多数类基准命中率：**永远猜最常见方向**这个笨蛋的表现。

    这是最容易被误认成"模型有效"的假基准——只要 flat 占多数，
    一个无脑模型就能拿到体面分数。必须显式对照。
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(np.asarray(y, dtype=int), minlength=k)
    return float(counts.max() / max(counts.sum(), 1))


def random_hit_rate(y: np.ndarray, k: int = 3) -> float:
    """均匀随机基准：三分类即 1/3。低于它说明模型在**主动帮倒忙**。"""
    return 1.0 / max(k, 1)


def align_proba(proba: np.ndarray, classes, k: int = 3) -> np.ndarray:
    """把 predict_proba 的输出对齐到固定 k 类列空间（缺失类补 0 后归一）。

    为什么必须对齐：训练窗口里只出现 2 个类别时（小样本下极常见），
    分类器的 predict_proba 只返回 2 列，而下游的 Brier / 命中率是拿
    "列下标 == 类别索引"直接比较的（见 calibration._evaluate）——
    列数一少就 IndexError 或整体错位。本项目真实踩过：
    日度重训在任一池只有涨/平两类样本时直接崩。
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    out = np.zeros((proba.shape[0], k))
    cls = np.asarray(classes).astype(int).tolist() if classes is not None else list(range(proba.shape[1]))
    for j, c in enumerate(cls):
        if j < proba.shape[1] and 0 <= c < k:
            out[:, c] = proba[:, j]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


# ---------------- 校准器 ----------------
class TemperatureScaler:
    """温度缩放：p ∝ p^(1/T)。T>1 让分布变平（模型过度自信），T<1 让其更锐利。

    只估 **1 个参数**，所以几十条样本就能稳定拟合——这是冷启动期的默认校准器。
    """

    def __init__(self, T: float = 1.0):
        self.T = float(T)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        grid = np.arange(0.5, 3.01, 0.05)
        best_t, best_b = 1.0, float("inf")
        for t in grid:
            b = brier_score(self._apply(probs, t), y)
            if b < best_b:
                best_t, best_b = float(t), b
        self.T = best_t
        return self

    @staticmethod
    def _apply(probs: np.ndarray, t: float) -> np.ndarray:
        p = np.clip(probs, 1e-9, 1.0) ** (1.0 / t)
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if abs(self.T - 1.0) < 1e-6:
            return probs
        return self._apply(probs, self.T)


class SoftmaxRegression:
    """多分类逻辑回归（纯 numpy，L2 正则 + 特征标准化）。

    两处复用：
    - 校准层的元模型：学习"引擎概率 → 真实方向"的映射
    - 量化引擎的兜底分类器：没装 lightgbm/sklearn 时也能训练出可校准的概率
      （**注意：兜底是线性模型，日频三分类上几乎学不到非线性，
        它的存在意义是"保证流水线不断"，不是"保证准"。**）
    """

    def __init__(self, feature_names: list[str] | None = None, l2: float = 0.05,
                 lr: float = 0.5, iters: int = 700):
        self.feature_names = feature_names or []
        self.l2, self.lr, self.iters = l2, lr, iters
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None:
            self.mu, self.sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        index = {c: i for i, c in enumerate(self.classes_.tolist())}
        Xs = self._standardize(X)
        n, f = Xs.shape
        k = len(self.classes_)
        self.W, self.b = np.zeros((f, k)), np.zeros(k)
        if k == 1:                     # 只有一个类别：常数概率
            return self
        onehot = np.zeros((n, k))
        for i, yi in enumerate(y.tolist()):
            onehot[i, index[yi]] = 1.0
        for _ in range(self.iters):
            p = softmax(Xs @ self.W + self.b)
            self.W -= self.lr * (Xs.T @ (p - onehot) / n + self.l2 * self.W)
            self.b -= self.lr * (p - onehot).mean(axis=0)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("SoftmaxRegression 未训练")
        X = np.asarray(X, dtype=float)
        if len(self.classes_) == 1:
            return np.ones((X.shape[0], 1))
        return softmax(self._standardize(X) @ self.W + self.b)

    def importance(self) -> dict:
        """特征重要度（各特征权重的绝对值均值），用于解释模型在看什么。"""
        if self.W is None:
            return {}
        imp = np.abs(self.W).mean(axis=1)
        total = float(imp.sum()) or 1.0
        names = self.feature_names or [f"f{i}" for i in range(len(imp))]
        return {name: round(float(v / total), 3)
                for name, v in sorted(zip(names, imp), key=lambda x: -x[1])}


class NumpyRidge:
    """纯 numpy 岭回归（闭式解，含截距 + 特征标准化）。无第三方依赖的回归兜底。

    为什么需要"回归兜底"而不是"分类兜底"
    ------------------------------------
    量化引擎的预测目标是**前向收益 μ**，再由残差分布导出三方向概率。
    所以兜底实现必须给出 μ，而不是类别概率——分类器无法表达
    "看多但不确定"这类连续强度，只会把概率压在类别频率上，
    在平盘占多数的日线上直接退化成"永远猜平盘"。

    它在依赖清单里的定位：没装 lightgbm/sklearn 时保证流水线不断，
    其线性假设意味着**不保证准**——这一点必须显式写在文档里，
    否则下一个接手的人会以为"有兜底 = 有保障"。
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X, y) -> "NumpyRidge":
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
        self.b, self.w = float(coef[0]), coef[1:]
        return self

    def predict(self, X) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("NumpyRidge 未训练")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X - self.mu) / self.sigma @ self.w + self.b


__all__ = ["softmax", "brier_score", "hit_rate", "majority_hit_rate",
           "random_hit_rate", "align_proba", "TemperatureScaler",
           "SoftmaxRegression", "NumpyRidge"]
