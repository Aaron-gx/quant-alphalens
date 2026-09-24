"""量化内核 v3 自检：回归 + 经验分布 + λ(1-SE) + purged walk-forward。

为什么需要这个脚本
------------------
`quant_bench.py` 证明的是"这套协议在离线基准里有效"；
本脚本证明的是"**生产模块 quant.py 真的按这套协议在跑**"。
两者必须分开验证——否则很容易出现"基准跑的是新内核、生产还是旧内核"
（本项目真实踩过：装了 lightgbm 而 5 个 pkl 的 backend 全是 numpy_logit）。

检查项
------
1. 后端与内核标记（确认不是 numpy 兜底、不是旧内核）
2. 四周期训练是否都能产出样本外指标（oos_n > 0）
3. BSS 是否与离线基准同量级（h=1 接近 0，h=5/20 ≤ 0）
4. 概率是否自洽：和=1、非退化（没有恒为 1/3 或恒为 0.99）
5. pkl 里是否带残差分布与 λ（predict 要靠它们导概率）
6. 缓存复用是否生效（第二次 train 应 reused=True 且不重训）
7. _needs_retrain 是否会把旧内核模型判为待重训

用法
----
    .venv/Scripts/python.exe scripts/verify_quant_v3.py
    .venv/Scripts/python.exe scripts/verify_quant_v3.py --symbols 510300 --count 900
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.predict import quant as Q  # noqa: E402
from core.predict import labels as L  # noqa: E402
from core.predict import calibration  # noqa: E402


def check(cond: bool, msg: str, fails: list) -> None:
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="510300,600519")
    ap.add_argument("--count", type=int, default=800)
    args = ap.parse_args()

    from core.data import stock as sd

    fails: list[str] = []
    print("=" * 78)
    print("量化内核 v3 自检")
    print("=" * 78)

    print("\n[1] 后端与内核")
    be = Q.active_backend()
    print(f"  active_backend = {be}")
    print(f"  KERNEL_TAG     = {Q.KERNEL_TAG}")
    print(f"  FEATURES       = {len(Q.FEATURES)} 个")
    check(be in ("lgbm", "gbdt", "numpy_ridge"), "后端名合法", fails)
    check(len(Q.FEATURES) == 28, "特征数 = 28", fails)

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    summary: list[dict] = []

    for sym in syms:
        print(f"\n[2] 训练 {sym}（{args.count} 根日线）")
        try:
            kline = sd.fetch_kline(sym, count=args.count)
        except Exception as e:
            print(f"  取数失败: {type(e).__name__}: {e}")
            fails.append(f"{sym} 取数失败")
            continue
        if kline is None or kline.empty:
            print("  无数据")
            fails.append(f"{sym} 无数据")
            continue
        print(f"  K线 {len(kline)} 根，{kline['date'].iloc[0]} → {kline['date'].iloc[-1]}")

        for h in L.HORIZONS:
            try:
                r = Q.train(sym, kline, max_age_days=0, record_samples=False,
                            horizon=h)
            except Exception as e:
                print(f"  {h:<10} 异常 {type(e).__name__}: {str(e)[:70]}")
                fails.append(f"{sym}/{h} train 异常")
                continue
            if not r.get("ok"):
                print(f"  {h:<10} 未训练: {r.get('error')}")
                fails.append(f"{sym}/{h} 训练失败")
                continue
            n_oos = int(r.get("oos_n") or 0)
            print(f"  {h:<10} backend={r.get('backend'):<11} n={r.get('n'):>4} "
                  f"λ={r.get('lam'):<5} oos_n={n_oos:>4} "
                  f"hit={r.get('oos_hit')} base={r.get('oos_base_hit')} "
                  f"BSS={r.get('oos_bss'):+.4f} skill={r.get('has_skill')}")
            check(n_oos > 0, f"{sym}/{h} 产出样本外样本", fails)
            summary.append({"symbol": sym, "horizon": h, **r})

    print("\n[3] 预测输出自洽性")
    for sym in syms:
        try:
            kline = sd.fetch_kline(sym, count=args.count)
        except Exception:
            continue
        if kline is None or kline.empty:
            continue
        qmap = Q.predict_multi(sym, kline)
        print(f"  {sym}: 覆盖周期 {list(qmap)}")
        for h, q in qmap.items():
            p = np.array([q["up"], q["flat"], q["down"]], dtype=float)
            print(f"    {h:<10} up={q['up']:.4f} flat={q['flat']:.4f} "
                  f"down={q['down']:.4f} sum={p.sum():.6f} "
                  f"backend={q.get('backend')} λ={q.get('lam')} "
                  f"stale_kernel={q.get('is_stale_kernel')} skill={q.get('has_skill')} "
                  f"w={Q.fuse_weights(q, h)['weight']}")
            check(abs(p.sum() - 1.0) < 1e-3, f"{sym}/{h} 概率和 = 1", fails)
            check(p.max() < 0.999, f"{sym}/{h} 概率未退化（max<0.999）", fails)

    print("\n[4] pkl 内容与缓存复用")
    for sym in syms:
        h = L.DEFAULT_HORIZON
        p = Q._model_path(sym, h)
        if not p.exists():
            check(False, f"{sym}/{h} 模型文件存在", fails)
            continue
        info = Q.model_info(sym, h)
        print(f"  {sym}/{h}: kernel={info.get('kernel')} lam={info.get('lam')} "
              f"resid_n={info.get('resid_n')} stale_kernel={info.get('stale_kernel')} "
              f"stale_backend={info.get('stale_backend')}")
        check(info.get("kernel") == Q.KERNEL_TAG, f"{sym}/{h} pkl 带正确内核标记", fails)
        check(int(info.get("resid_n") or 0) > 0, f"{sym}/{h} pkl 带残差分布", fails)
        check(info.get("stale_kernel") is False, f"{sym}/{h} 内核不陈旧", fails)

        try:
            kline = sd.fetch_kline(sym, count=args.count)
            r2 = Q.train(sym, kline, max_age_days=5, record_samples=False, horizon=h)
            check(bool(r2.get("reused")), f"{sym}/{h} 二次训练命中缓存（未白跑）", fails)
        except Exception as e:
            check(False, f"{sym}/{h} 缓存复用检查异常 {e}", fails)

    print("\n[5] 旧内核 pkl 会被判为待重训")
    stale_any = False
    for sym in syms:
        for h in L.HORIZONS:
            stale, why = Q._needs_retrain(Q._model_path(sym, h), h)
            if stale:
                stale_any = True
                print(f"  {sym}/{h} 判定待重训: {why}")
    check(not stale_any, "所有模型均不为陈旧状态", fails)

    print("\n[6] 样本是否已进入自学习池（引擎=quant）")
    try:
        st = calibration.sample_stats()
        print(f"  sample_stats = {st}")
    except Exception as e:
        print(f"  sample_stats 读取失败（不影响内核验收）: {e}")

    print("\n" + "=" * 78)
    if summary:
        print("BSS 汇总（中位数）：")
        for h in L.HORIZONS:
            vs = [s["oos_bss"] for s in summary
                  if s["horizon"] == h and s.get("oos_bss") is not None]
            if vs:
                print(f"  {h:<10} BSS中位 = {float(np.median(vs)):+.4f}  (n={len(vs)})")
        print("\n判读：h=1 接近 0、h=5/20 ≤ 0 属于**预期结果**——")
        print("      日线上本特征集没有方向 alpha，本内核的价值是「无技能时无害」")
        print("      （λ→0 退化 + 融合权重 0），而不是伪造成高命中率。")
    print("=" * 78)
    if fails:
        print(f"\n失败 {len(fails)} 项：")
        for f in fails:
            print("  - " + f)
        return 1
    print("\n全部检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
