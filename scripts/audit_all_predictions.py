"""全量盘点：每条预测的方向数据、校准状态、以及界面会渲染出的判决文案。

用途：一次看清"用户看到的这句话到底是哪条记录算出来的、数字与结论是否自相矛盾"。
不做任何修改（只读）。

用法： .venv/Scripts/python.exe scripts/audit_all_predictions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("data/_audit_all.json")
PROB_KEYS = ("up", "flat", "down")
EPS = 1e-6

# 与前端 forecast.ts 的门槛/标签保持一致（此处只为便于人读，判定以代码为准）
NO_EDGE_TOP = 0.40
LBL = {"next_day": "次日", "one_week": "一周", "one_month": "一月", "quarter": "一季"}


def _eq(a, b) -> bool:
    try:
        return all(abs(float(a.get(k, 0) or 0) - float(b.get(k, 0) or 0)) < EPS for k in PROB_KEYS)
    except Exception:
        return False


def _swap(a, b) -> bool:
    try:
        if _eq(a, b):
            return False
        return (abs(float(a.get("up", 0) or 0) - float(b.get("down", 0) or 0)) < EPS
                and abs(float(a.get("down", 0) or 0) - float(b.get("up", 0) or 0)) < EPS
                and abs(float(a.get("flat", 0) or 0) - float(b.get("flat", 0) or 0)) < EPS)
    except Exception:
        return False


def verdict(h: dict) -> tuple[str, str]:
    """与前端 verdictOf 同规则，返回 (kind, text)。"""
    up = float(h.get("up") or 0)
    fl = float(h.get("flat") or 0)
    dn = float(h.get("down") or 0)
    tot = up + fl + dn
    if tot <= 0:
        return "", ""
    nu, nf, nd = up / tot, fl / tot, dn / tot
    top, bot = max(nu, nf, nd), min(nu, nf, nd)
    if top < NO_EDGE_TOP:
        k = "unclear"
    elif nf >= top - 1e-9:
        k = "flat"
    elif abs(nu - nd) < 1e-9:
        k = "unclear"
    else:
        k = "up" if nu > nd else "down"
    txt = {"up": "看涨", "down": "看跌", "flat": "震荡", "unclear": "方向不明"}[k]
    return k, f"{txt}[涨{up:.0%}/平{fl:.0%}/跌{dn:.0%} 区分度{top - bot:.0%}]"


def main() -> None:
    from core.store.db import session_scope
    from core.store.models import Prediction

    rows = []
    with session_scope() as s:
        for p in s.query(Prediction).order_by(Prediction.id).all():
            snap = p.input_snapshot or {}
            calib = snap.get("calibration") or {}
            hz = p.horizons or {}
            per = {}
            sw = 0
            for k, v in hz.items():
                if not isinstance(v, dict):
                    continue
                kind, txt = verdict(v)
                info = calib.get(k) if isinstance(calib, dict) else None
                flagged = bool(info and info.get("method") == "identity"
                               and info.get("probs_before") and info.get("probs_after")
                               and _swap(info["probs_before"], info["probs_after"]))
                sw += 1 if flagged else 0
                per[k] = {"verdict": kind, "text": txt, "inverted": flagged}
            rows.append({
                "id": p.id, "symbol": p.symbol, "base_date": str(p.base_date),
                "created_at": str(p.created_at), "engine": p.engine,
                "llm_model": p.llm_model,
                "asset_type": p.asset_type,
                "confidence": p.confidence,
                "has_snapshot": bool(snap),
                "calibration_keys": list(calib.keys()) if isinstance(calib, dict) else None,
                "inverted_cells": sw,
                "per_horizon": per,
                "report_head": (p.report_text or "")[:120],
                "advice": (p.action or {}).get("advice", "")[:120],
            })

    OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'id':>3} {'symbol':>7} {'base_date':<11} {'created':<20} {'eng':<6} "
          f"{'snap':<5} {'inv':<4} 结论")
    for r in rows:
        v = r["per_horizon"]
        summary = " ".join(f"{LBL.get(k, k)}:{v[k]['text']}" for k in
                           ("next_day", "one_week", "one_month", "quarter") if k in v)
        print(f"{r['id']:>3} {r['symbol']:>7} {r['base_date']:<11} {r['created_at'][:19]:<20} "
              f"{r['engine']:<6} {str(r['has_snapshot']):<5} {r['inverted_cells']:<4} {summary}")


if __name__ == "__main__":
    main()
