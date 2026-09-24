"""聊天机器人适配层示例：一个极简 webhook，把聊天指令映射到核心引擎。

对接方式（任选）：
- 微信/企业微信/Telegram/QQ 机器人框架把用户消息 POST 到 /chat
- 或直接在你们自己的机器人里 import core 引擎（同机部署不必走 HTTP）

指令示例（发送给机器人）：
  预测 600519            -> 跑一次 AI 预测（fast 模式）
  深析 上证指数           -> deep 分层预测
  行情                   -> 大盘概览
  准确率                 -> 预测验证报表
  记录 微信群都在聊券商合并  -> 人工情报写入情报库

运行：uvicorn bots.bot_server:app --port 8001
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel

from core.predict.pipeline import predict
from core.predict.llm import LLMClient
from core.verify.evaluator import accuracy_report
from core.data import market as market_data
from core.intel import news as news_data
from core.predict.schema import HORIZON_LABELS

app = FastAPI(title="chat-bot-adapter")


class ChatMsg(BaseModel):
    text: str
    user: str = ""


def _fmt_prediction(p) -> str:
    lines = [f"🔮 {p.name}({p.symbol}) 预测  [置信度 {p.confidence}]"]
    for k in ("next_day", "one_week", "one_month", "quarter"):
        h = (p.horizons or {}).get(k)
        if h:
            lines.append(f"{HORIZON_LABELS.get(k, k)}: 涨{h['up']:.0%} 平{h['flat']:.0%} 跌{h['down']:.0%} {h.get('range_pct','')}")
    if p.report_text:
        lines.append(f"摘要：{p.report_text}")
    if p.risks:
        lines.append("风险：" + "；".join(p.risks[:3]))
    return "\n".join(lines)


@app.post("/chat")
def chat(msg: ChatMsg):
    text = msg.text.strip()
    if not text:
        return {"reply": "？"}

    if text.startswith(("预测 ", "深析 ")):
        mode = "deep" if text.startswith("深析") else "fast"
        symbol = text.split(None, 1)[1].strip()
        if not LLMClient().available():
            return {"reply": "LLM 未配置，请先在 config.toml 填 api_key"}
        try:
            p = predict(symbol, mode=mode)
            return {"reply": _fmt_prediction(p)}
        except Exception as e:
            return {"reply": f"预测失败：{e}"}

    if text in ("行情", "大盘"):
        ov = market_data.market_overview()
        idx = ov.get("indices", {}).get("indices", {})
        senti = ov.get("sentiment", {})
        lines = [f"{n}: {d['price']:.2f} ({d['change_pct']:+.2f}%)" for n, d in sorted(idx.items())]
        lines.append(f"市场情绪 {senti.get('score', 0):+.0f} ({senti.get('level', '')})")
        return {"reply": "\n".join(lines) or "行情暂不可用"}

    if text in ("准确率", "对账"):
        rep = accuracy_report()
        if not rep.get("total"):
            return {"reply": "暂无到期评估"}
        lines = [f"已评估 {rep['total']} 条，方向命中率 {rep['hit_rate']:.1%}，Brier {rep['avg_brier']:.3f}"]
        for h, s_ in rep.get("by_horizon", {}).items():
            lines.append(f"{HORIZON_LABELS.get(h, h)}: {s_['hit_rate']:.1%} (n={s_['n']})")
        return {"reply": "\n".join(lines)}

    if text.startswith("记录 "):
        content = text[3:].strip()
        news_data.save_news([{"symbol": "", "source": f"user:{msg.user}",
                              "category": "manual", "title": content[:250],
                              "content": content, "url": "", "publish_time": None}])
        return {"reply": "已写入情报库，后续分析会参考 ✅"}

    return {"reply": "支持的指令：预测 <代码> / 深析 <代码> / 行情 / 准确率 / 记录 <情报>"}
