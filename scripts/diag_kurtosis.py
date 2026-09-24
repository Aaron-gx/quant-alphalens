"""诊断：金融收益的"平盘概率"，正态假设到底错多少？

背景
----
quant_bench 里 D_reg_ridge（岭回归 + 正态CDF导概率）样本外命中率只有 0.23，
而"永远猜平盘"的基准是 0.56 —— 也就是说：**它在系统性地把平盘判成涨/跌**。

根因假设：日收益分布是**尖峰厚尾**（leptokurtic），
真正落在 |r|<1% 里的天数远多于 Normal(0, σ) 给出的概率。
正态假设会低估"平盘"质量 → 概率平盘被压到接近 0 → argmax 永远不是平盘。

本脚本用真实数据把这个偏差量化出来，作为"必须改用经验分布"的证据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_UNIVERSE = ["510300", "510500", "159915", "512880", "588000",
                    "600519", "000858", "601318", "300750", "600036"]


def norm_cdf(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def main() -> None:
    from core.data import stock as sd

    taus = [0.5, 1.0, 1.5, 2.0]
    print("\n平盘质量对比：经验 P(|r|<tau)  vs  正态假设 P(|r|<tau)")
    print("（正态用该标的的实际 std；两者差得越多，说明正态假设越不可用）")
    print("=" * 108)
    hdr = f"{'标的':>8} {'n':>6} {'std%':>7} {'峰度':>7}"
    for t in taus:
        hdr += f" {'经tau'+str(t):>9} {'正tau'+str(t):>9} {'偏差':>8}"
    print(hdr)
    print("-" * 108)

    rows = []
    for sym in DEFAULT_UNIVERSE:
        try:
            k = sd.fetch_kline(sym, count=1200)
        except Exception as e:
            print(f"{sym:>8} 取数失败 {type(e).__name__}")
            continue
        if k is None or k.empty:
            print(f"{sym:>8} 无数据")
            continue
        r = (k["close"].shift(-1) / k["close"] - 1).dropna().values * 100
        n = len(r)
        sd_ = float(np.std(r))
        # 过量峰度（正态=0）
        kurt = float(np.mean(((r - r.mean()) / sd_) ** 4) - 3.0)
        line = f"{sym:>8} {n:>6} {sd_:>7.3f} {kurt:>7.2f}"
        rec = {"sym": sym, "n": n, "std": sd_, "kurt": kurt}
        for t in taus:
            emp = float(np.mean(np.abs(r) < t))
            norm = norm_cdf(t / sd_) - norm_cdf(-t / sd_)
            rec[f"emp{t}"] = emp
            rec[f"gap{t}"] = emp - norm
            line += f" {emp:>9.4f} {norm:>9.4f} {emp - norm:>+8.4f}"
        print(line)
        rows.append(rec)
        if sym == DEFAULT_UNIVERSE[0]:
            print("-" * 108)

    print("=" * 108)
    med = f"{'中位数':>8} {'':>6} {'':>7} {np.median([r['kurt'] for r in rows]):>7.2f}"
    for t in taus:
        med += (f" {np.median([r['emp'+str(t)] for r in rows]):>9.4f} "
                f"{np.median([r['gap'+str(t)] for r in rows]):>+8.4f}")
    print(med)
    print("=" * 108)
    print("判读：偏差为正且量级 >0.05 → 正态假设显著低估平盘质量，")
    print("      用它导出的概率会把『平盘』永远压在 argmax 之外。")
    print("      修正：用**训练残差的经验分布**代替正态分布（无分布假设）。")


if __name__ == "__main__":
    main()
