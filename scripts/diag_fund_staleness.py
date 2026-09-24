"""016665 场外基金诊断：净值新鲜度 + 预测方向可读性量化。

回答两个用户问题：
  ① 为什么 09-23 数据没有（今日 09-24）
  ② 为什么图上"预测"看不出涨跌

用法：.venv/Scripts/python.exe scripts/diag_fund_staleness.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYMBOL = "016665"
OUT = Path("data/_diag_fund.json")


def main() -> None:
    rep: dict = {"symbol": SYMBOL, "now": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ── ① 缓存文件状态 ────────────────────────────────────────────────
    try:
        from core.data.cache import CACHE_DIR, _key_path
        rows = []
        d = CACHE_DIR / "kline"
        if d.exists():
            for p in sorted(d.glob("*.parquet")):
                st = p.stat()
                rows.append({
                    "file": p.name,
                    "size": st.st_size,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                    "age_hours": round((time.time() - st.st_mtime) / 3600, 2),
                })
        rep["cache_files_total"] = len(rows)
        # 定位 016665:120 对应的那个 key
        target = _key_path("kline", f"ofund:{SYMBOL}:120")
        rep["cache_target"] = {
            "path": str(target),
            "exists": target.exists(),
            "age_hours": round((time.time() - target.stat().st_mtime) / 3600, 2) if target.exists() else None,
        }
    except Exception as e:
        rep["cache_error"] = repr(e)

    # ── ② 缓存里的最后几个日期（走缓存，等价于前端拿到的东西）──────────
    try:
        from core.data.fund import fetch_fund_nav_history
        df_c = fetch_fund_nav_history(SYMBOL, count=120, use_cache=True)
        rep["cached_tail"] = [
            {"date": str(r["date"])[:10], "close": float(r["close"])}
            for r in df_c.tail(6).to_dict(orient="records")
        ]
        rep["cached_rows"] = int(len(df_c))
    except Exception as e:
        rep["cached_error"] = repr(e)

    # ── ③ 绕开缓存直连数据源（判断"上游到底有没有 09-23"）────────────
    try:
        from core.data.fund import fetch_fund_nav_history
        t0 = time.time()
        df_f = fetch_fund_nav_history(SYMBOL, count=120, use_cache=False)
        rep["fresh_fetch_seconds"] = round(time.time() - t0, 2)
        rep["fresh_tail"] = [
            {"date": str(r["date"])[:10], "close": float(r["close"])}
            for r in df_f.tail(8).to_dict(orient="records")
        ]
        rep["fresh_rows"] = int(len(df_f))
        rep["fresh_last_date"] = str(df_f["date"].iloc[-1])[:10] if len(df_f) else None
    except Exception as e:
        rep["fresh_error"] = repr(e)

    # ── ④ 实时估值接口（含"净值日期"字段，可交叉验证）────────────────
    try:
        from core.data.fund import fetch_fund_realtime_estimate
        rep["realtime"] = fetch_fund_realtime_estimate(SYMBOL)
    except Exception as e:
        rep["realtime_error"] = repr(e)

    # ── ⑤ 预测概率 → 前端会渲染成什么方向 ────────────────────────────
    try:
        from core.store.models import Prediction
        from core.store.db import session_scope
        with session_scope() as s:
            ps = (s.query(Prediction)
                    .filter(Prediction.symbol == SYMBOL)
                    .order_by(Prediction.id.desc()).limit(3).all())
            items = []
            for p in ps:
                items.append({
                    "id": p.id,
                    "base_date": str(p.base_date)[:10] if p.base_date else None,
                    "created_at": str(p.created_at)[:19] if getattr(p, "created_at", None) else None,
                    "horizons": p.horizons,
                })
            rep["predictions"] = items
    except Exception as e:
        rep["pred_error"] = repr(e)

    # ── ⑥ 复算前端的 midPct（把"看不懂"量化成数字）──────────────────
    try:
        import re as _re

        def parse_range_pct(s):
            if not s:
                return 0.0
            m = _re.search(r"([\d.]+)\s*%", str(s))
            return float(m.group(1)) / 100 if m else 0.0

        calc = []
        preds = rep.get("predictions") or []
        if preds:
            hz = preds[0].get("horizons") or {}
            base = 2.892
            for key, tday in (("next_day", 1), ("one_week", 5), ("one_month", 20), ("quarter", 60)):
                h = hz.get(key)
                if not h or not isinstance(h.get("up"), (int, float)):
                    calc.append({"horizon": key, "status": "missing"})
                    continue
                band = parse_range_pct(h.get("range_pct") or h.get("range")) / 2 or 0.02
                diff = h["up"] - (h.get("down") or 0)
                mid_pct = diff * band
                # 三分类概率和
                tot = sum(float(h.get(k) or 0) for k in ("up", "flat", "down"))
                calc.append({
                    "horizon": key,
                    "trading_days": tday,
                    "up": h.get("up"), "flat": h.get("flat"), "down": h.get("down"),
                    "prob_sum": round(tot, 4),
                    "range_pct": h.get("range_pct") or h.get("range"),
                    "band": round(band, 5),
                    "prob_diff(up-down)": round(diff, 4),
                    "midPct": round(mid_pct, 6),
                    "midPct_display": f"{mid_pct * 100:+.3f}%",
                    "mid_price": round(base * (1 + mid_pct), 4),
                    # 在 y 轴跨度 ~2.4 的图上，占多少像素（图高约 380px）
                    "pixels_on_chart": round(abs(base * mid_pct) / 2.4 * 380, 2),
                })
            rep["dir_amplitude_check"] = calc
            rep["dir_amplitude_note"] = (
                "pixels_on_chart 是把该中枢偏移折算到图上高度："
                "<2px 即肉眼不可分辨，必然被用户读成『平的/看不懂』"
            )
    except Exception as e:
        rep["calc_error"] = repr(e)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
