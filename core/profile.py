"""用户画像：持久化偏好，自动注入每次预测。"""
from __future__ import annotations

from datetime import datetime

from core.store.db import session_scope
from core.store.models import UserProfile


def get_profile() -> dict:
    """取画像；没有则返回默认。"""
    with session_scope() as s:
        p = s.query(UserProfile).first()
        if not p:
            return {"risk_preference": "neutral", "custom_principles": "",
                    "position_text": "", "trade_style": "", "common_mistakes": ""}
        return {"risk_preference": p.risk_preference,
                "custom_principles": p.custom_principles,
                "position_text": p.position_text,
                "trade_style": p.trade_style,
                "common_mistakes": p.common_mistakes}


def save_profile(**kw) -> None:
    """保存画像（单例）。"""
    with session_scope() as s:
        p = s.query(UserProfile).first()
        if not p:
            p = UserProfile()
            s.add(p)
        for k in ("risk_preference", "custom_principles", "position_text",
                  "trade_style", "common_mistakes"):
            if k in kw and kw[k] is not None:
                setattr(p, k, kw[k])
        p.updated_at = datetime.now()


def profile_prompt_block(profile: dict, user_opinion: str = "") -> str:
    """把画像拼成 prompt 段。空字段自动跳过。"""
    parts = []
    if user_opinion.strip():
        parts.append(f"## 用户本次观点\n{user_opinion.strip()}")
    if profile.get("position_text"):
        parts.append(f"## 用户当前持仓\n{profile['position_text']}")
    if profile.get("trade_style"):
        parts.append(f"## 用户交易风格\n{profile['trade_style']}")
    if profile.get("common_mistakes"):
        parts.append(f"## 用户常犯错误（请在建议中针对性提醒）\n{profile['common_mistakes']}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n请在分析中针对性回应用户情况。"
