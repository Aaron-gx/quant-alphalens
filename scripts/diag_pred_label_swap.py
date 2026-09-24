"""检查库里的预测记录是否被「旧标签顺序」污染（涨/跌概率对调）。

判据（自洽，不需要任何外部对照）
--------------------------------
`calibration.apply()` 在无可用样本时返回 `method="identity"`，语义是**恒等映射：
概率原样返回**。因此对每一条 identity 的记录，必然有

    probs_before == probs_after   （逐字段相等）

旧代码把特征向量按 [涨,平,跌] 排、标签索引按 [跌,平,涨] 读（见 core/predict/labels.py
顶部注释记录的那次事故），于是 `apply()` 会把 up/down **对调后**再输出。
对调后 `method` 仍然是 "identity"、命中率/Brier 这些指标也全都"看起来正常"，
唯一能露馅的就是 before/after 不再相等——本脚本就查这个。

用法：
    .venv/Scripts/python.exe scripts/diag_pred_label_swap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("data/_diag_label_swap.json")
PROB_KEYS = ("up", "flat", "down")
# 允许的浮点误差：pipeline 写 before 时会 round(...,3)
EPS = 1e-6


def _eq(a: dict, b: dict) -> bool:
    try:
        return all(abs(float(a.get(k, 0) or 0) - float(b.get(k, 0) or 0)) < EPS
                   for k in PROB_KEYS)
    except Exception:
        return False


def _swapped(a: dict, b: dict) -> bool:
    """b 是否是 a 把 up/down 对调的结果（且确实发生了变化）。

    ⚠️ 必须排除 `up == down` 的情形：此时"对调"与"恒等"是同一个变换，
    对称记录会被误报成污染（本项目真实踩过：某周期 up=down=0.36，
    被误判为 swapped 并触发了不该出现的告警）。判污染的前提是
    **改动了**，所以先要求 up 确实变了，再看是否恰好换成 down。
    """
    try:
        if _eq(a, b):
            return False
        return (abs(float(a.get("up", 0) or 0) - float(b.get("down", 0) or 0)) < EPS
                and abs(float(a.get("down", 0) or 0) - float(b.get("up", 0) or 0)) < EPS
                and abs(float(a.get("flat", 0) or 0) - float(b.get("flat", 0) or 0)) < EPS)
    except Exception:
        return False


def main() -> None:
    from core.predict.labels import argmax_direction
    from core.store.db import session_scope
    from core.store.models import Prediction

    rep: dict = {"rows": [], "summary": {}}
    with session_scope() as s:
        preds = (s.query(Prediction).order_by(Prediction.id.desc()).limit(10).all())
        for p in preds:
            snap = p.input_snapshot or {}
            calib = snap.get("calibration") or {}
            row = {
                "id": p.id, "symbol": p.symbol, "base_date": str(p.base_date),
                "created_at": str(p.created_at), "engine": p.engine,
                "confidence": p.confidence,
                "horizons": [],
            }
            swapped_n = identity_n = 0
            for key, info in (calib.items() if isinstance(calib, dict) else []):
                if not isinstance(info, dict):
                    continue
                method = info.get("method")
                before = info.get("probs_before") or {}
                after = info.get("probs_after") or {}
                same = _eq(before, after)
                swap = _swapped(before, after)
                if method == "identity":
                    identity_n += 1
                    if swap:
                        swapped_n += 1
                row["horizons"].append({
                    "key": key, "method": method, "n": info.get("n"),
                    "before": before, "after": after,
                    "before_eq_after": same, "up_down_swapped": swap,
                })
            # 落库的最终概率（前端直接读这个）与报告文字的倾向是否一致
            stored = p.horizons or {}
            row["stored_argmax"] = {k: argmax_direction(v) for k, v in stored.items()
                                    if isinstance(v, dict)}
            row["report_head"] = (p.report_text or "")[:90]
            row["identity_horizons"] = identity_n
            row["swapped_horizons"] = swapped_n
            row["verdict"] = (
                "TAINTED" if swapped_n else
                ("identity-but-clean" if identity_n else "no-identity")
            )
            rep["rows"].append(row)

    tainted = [r for r in rep["rows"] if r["verdict"] == "TAINTED"]
    rep["summary"] = {
        "checked": len(rep["rows"]),
        "tainted_ids": [r["id"] for r in tainted],
        "latest_id": rep["rows"][0]["id"] if rep["rows"] else None,
        "latest_tainted": bool(rep["rows"]) and rep["rows"][0]["verdict"] == "TAINTED",
        "conclusion": (
            "最新一条预测即含涨跌对调：产出它的后端进程仍在跑旧标签顺序，"
            "必须重启后端并重新生成研判。" if (rep["rows"] and rep["rows"][0]["verdict"] == "TAINTED")
            else "最新一条预测未检出对调。" if rep["rows"] else "库里没有预测记录。"
        ),
    }
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=2))
    for r in rep["rows"]:
        print(f"  id={r['id']:>3} {r['symbol']} {r['base_date']} {r['created_at'][:19]} "
              f"identity={r['identity_horizons']} swapped={r['swapped_horizons']} -> {r['verdict']}")


if __name__ == "__main__":
    main()
