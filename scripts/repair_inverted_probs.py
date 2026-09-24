"""把被「标签顺序缺陷」对调过的历史预测概率修回真值。

为什么可以确定 `probs_before` 才是真值、且可以安全回写
----------------------------------------------------
`calibration.apply()` 在 `method == "identity"` 时的代码路径是：

    raw = probs_to_vector(probs)      # {up,flat,down} -> [跌,平,涨]
    ...
    return {"probs": vector_to_probs(raw[0])}   # [跌,平,涨] -> {up,flat,down}

两步互为逆运算且列序同源 ⇒ **恒等映射必然原样返回**，`probs_before` 必须等于
`probs_after`。任何 `before ≠ after` 都只可能来自那段著名的缺陷：特征向量按
[涨,平,跌] 排、标签索引按 [跌,平,涨] 读，于是 up/down 被对调（见 labels.py 顶部）。

所以：
  - 缺陷发生在**后处理**（calibration），不在模型；
  - `probs_before` 是模型（含融合）的原始输出，是模型真正想表达的概率；
  - `probs_after` 是被错误后处理污染后的值，也就是落库、前端读取、对账依赖的那个。

回写范围（只动被证明对调的那一项，其余字段一律不碰）：
  1. `predictions.horizons[key].up/down`  ← 用 before 覆盖
  2. `input_snapshot.calibration[key].probs_after` ← 用 before 覆盖（恢复自洽）
  3. 追加 `repaired` 审计字段，保留"曾经被对调成什么"的痕迹

用法：
    .venv/Scripts/python.exe scripts/repair_inverted_probs.py           # 预演（不改库）
    .venv/Scripts/python.exe scripts/repair_inverted_probs.py --apply   # 真正写入
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("data/_repair_inverted_probs.json")
PROB_KEYS = ("up", "flat", "down")
EPS = 1e-6


def _near(a, b) -> bool:
    try:
        return abs(float(a or 0) - float(b or 0)) < EPS
    except Exception:
        return False


def _is_swap(before: dict, after: dict) -> bool:
    return (_near(before.get("up"), after.get("down"))
            and _near(before.get("down"), after.get("up"))
            and _near(before.get("flat"), after.get("flat"))
            and not _near(before.get("up"), after.get("up")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写库（默认只预演）")
    args = ap.parse_args()

    from core.store.db import session_scope
    from core.store.models import Prediction, PredictionEval

    report: dict = {"applied": bool(args.apply), "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "records": [], "skipped": [], "eval_warnings": []}

    with session_scope() as s:
        preds = s.query(Prediction).order_by(Prediction.id).all()
        for p in preds:
            snap = copy.deepcopy(p.input_snapshot or {})
            calib = snap.get("calibration") or {}
            if not isinstance(calib, dict):
                continue
            # 收集该记录的待修项
            pending: dict[str, dict] = {}
            for key, info in calib.items():
                if not isinstance(info, dict) or info.get("method") != "identity":
                    continue
                if info.get("probs_before") is None or info.get("probs_after") is None:
                    continue
                if _is_swap(info["probs_before"], info["probs_after"]):
                    pending[key] = info
            if not pending:
                continue

            hz = copy.deepcopy(p.horizons or {})
            changed = []
            for key, info in pending.items():
                before = info["probs_before"]
                after = info["probs_after"]
                h = hz.get(key)
                if not isinstance(h, dict):
                    report["skipped"].append({"id": p.id, "key": key, "why": "horizons 无该周期"})
                    continue
                changed.append({
                    "key": key,
                    "horizons_before": {"up": h.get("up"), "flat": h.get("flat"), "down": h.get("down")},
                    "horizons_after": {"up": before["up"], "flat": before["flat"], "down": before["down"]},
                    "probs_after_before": dict(after),
                    "probs_after_fixed": dict(before),
                })
                # ① 把方向概率修回模型真值（只动 up/down，flat 与 range 等字段不碰）
                h["up"] = before["up"]
                h["down"] = before["down"]
                # ② 恢复 calibration 自洽：identity 下 after 就该等于 before
                calib[key]["probs_after"] = {"up": before["up"], "flat": before["flat"],
                                             "down": before["down"]}
                # ③ 留审计痕迹，便于日后回答"这条被改过、改的是什么"
                calib[key]["repaired"] = {
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "label_order_inversion",
                    "swapped_back_from": {"up": after.get("up"), "flat": after.get("flat"),
                                          "down": after.get("down")},
                }

            if not changed:
                continue

            report["records"].append({
                "id": p.id, "symbol": p.symbol, "base_date": str(p.base_date),
                "created_at": str(p.created_at), "horizons_fixed": changed,
            })

            if args.apply:
                # JSON 列必须整体重新赋值，原地改嵌套 dict 不会被 SQLAlchemy 标记为脏
                snap["calibration"] = calib
                p.horizons = hz
                p.input_snapshot = snap

        if args.apply:
            s.flush()
            # 对账记录里由"被对调的概率"推出的 predicted_direction 也一并体检（只报告不擅改）
            evs = s.query(PredictionEval).all()
            for e in evs:
                if e.horizon not in ("next_day", "one_week", "one_month", "quarter"):
                    continue
                pred = s.get(Prediction, e.prediction_id)
                if not pred or not isinstance(pred.horizons, dict):
                    continue
                h = pred.horizons.get(e.horizon)
                if not isinstance(h, dict):
                    continue
                top = max(("up", "flat", "down"), key=lambda k: float(h.get(k) or 0))
                if e.predicted_direction and e.predicted_direction != top:
                    report["eval_warnings"].append({
                        "eval_id": e.id, "prediction_id": e.prediction_id,
                        "horizon": e.horizon,
                        "predicted_direction": e.predicted_direction,
                        "argmax_after_fix": top,
                        "hit": e.hit, "brier": e.brier_score,
                    })

    n_h = sum(len(r["horizons_fixed"]) for r in report["records"])
    report["summary"] = {
        "records_to_fix": len(report["records"]),
        "horizon_cells_fixed": n_h,
        "ids": [r["id"] for r in report["records"]],
        "mode": "APPLIED" if args.apply else "DRY-RUN",
        "eval_warnings": len(report["eval_warnings"]),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for r in report["records"]:
        print(f"  id={r['id']:>3} {r['symbol']} {r['base_date']}  修正 {len(r['horizons_fixed'])} 个周期")
        for c in r["horizons_fixed"]:
            print(f"      {c['key']:<10} up/down {c['horizons_before']['up']}/"
                  f"{c['horizons_before']['down']} -> {c['horizons_after']['up']}/"
                  f"{c['horizons_after']['down']}")
    if report["eval_warnings"]:
        print("\n  ⚠ 对账记录里 predicted_direction 与修正后 argmax 不一致：")
        for w in report["eval_warnings"]:
            print(f"      eval#{w['eval_id']} pred#{w['prediction_id']} {w['horizon']} "
                  f"{w['predicted_direction']} -> {w['argmax_after_fix']}")


if __name__ == "__main__":
    main()
