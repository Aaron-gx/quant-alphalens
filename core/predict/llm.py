"""OpenAI 兼容协议 LLM 客户端（DeepSeek/Qwen/GPT/Ollama 均可）。"""
from __future__ import annotations

import json
import time
from typing import Optional

from core.config import get


class LLMError(RuntimeError):
    pass


class LLMClient:
    """薄封装：chat / ask / ask_json，带用量统计。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, reasoner_model: str | None = None,
                 temperature: float | None = None, timeout: int | None = None):
        self.base_url = base_url or get("llm", "base_url", "")
        self.api_key = api_key or get("llm", "api_key", "")
        self.model = model or get("llm", "model", "deepseek-chat")
        self.reasoner_model = reasoner_model or get("llm", "reasoner_model", self.model)
        self.temperature = temperature if temperature is not None else get("llm", "temperature", 0.3)
        self.timeout = timeout or get("llm", "timeout", 120)
        self.last_usage: dict = {}

    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def _client(self):
        from openai import OpenAI  # 延迟导入，未装依赖时其他模块仍可跑
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def chat(self, messages: list[dict], *, model_type: str = "chat",
             temperature: float | None = None, json_mode: bool = False,
             max_retries: int = 2, caller: str = "") -> str:
        """model_type: 'chat' 普通模型 / 'reasoner' 推理模型（综合裁决用）。"""
        if not self.available():
            raise LLMError("LLM 未配置：请在 config.toml 的 [llm] 段填 api_key/base_url/model")
        model = self.reasoner_model if model_type == "reasoner" else self.model
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client().chat.completions.create(**kwargs)
                self.last_usage = getattr(resp, "usage", None)
                self.last_usage = self.last_usage.model_dump() if self.last_usage else {}
                _log_usage(model, caller, self.last_usage)
                return resp.choices[0].message.content or ""
            except Exception as e:  # 网络抖动/限流重试
                last_err = e
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        raise LLMError(f"LLM 调用失败: {last_err}")

    def ask(self, prompt: str, *, system: str = "", json_mode: bool = False,
            model_type: str = "chat", caller: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model_type=model_type, json_mode=json_mode,
                         caller=caller)

    def ask_json(self, prompt: str, *, system: str = "", model_type: str = "chat",
                 caller: str = "") -> dict:
        """JSON 模式调用并解析；失败时做一次花括号截取兜底。"""
        raw = self.ask(prompt, system=system, json_mode=True,
                       model_type=model_type, caller=caller)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start:end + 1])
            raise

    def supports_tools(self) -> bool:
        """模型是否声明支持 function calling（粗略判断，失败时会降级）。"""
        m = (self.model or "").lower()
        return not any(k in m for k in ("reasoner", "r1", "o1", "o3"))
        # 推理类模型对 tools 的支持不稳定 → 视为不支持，由上层走降级路径

    def chat_with_tools(
        self, messages: list[dict], tools: list[dict], *,
        model_type: str = "chat", tool_choice: str = "auto",
        temperature: float | None = None, max_retries: int = 1,
        caller: str = "",
    ) -> dict:
        """带工具的一轮调用（AI 交易员决策循环用）。

        返回：
          {"content": str,
           "tool_calls": [{"id": str, "name": str, "arguments": str}],  # arguments 为原始 JSON 串
           "usage": dict, "raw_finish_reason": str}

        失败：抛 LLMError。若模型不支持 tools（400/401 类参数错误），
        消息里会带 "tools_unsupported" 标记，由调用方降级为
        「工具结果预置上下文 + JSON 输出」模式。
        """
        if not self.available():
            raise LLMError("LLM 未配置：请在 config.toml 的 [llm] 段填 api_key/base_url/model")
        model = self.reasoner_model if model_type == "reasoner" else self.model
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client().chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                usage = usage.model_dump() if usage else {}
                _log_usage(model, caller or "trader", usage)
                msg = resp.choices[0].message
                calls = []
                for c in (getattr(msg, "tool_calls", None) or []):
                    fn = getattr(c, "function", None)
                    calls.append({
                        "id": getattr(c, "id", "") or "",
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "") if fn else "{}",
                    })
                return {
                    "content": msg.content or "",
                    "tool_calls": calls,
                    "usage": usage,
                    "raw_finish_reason": getattr(resp.choices[0], "finish_reason", ""),
                }
            except Exception as e:
                last_err = e
                txt = str(e).lower()
                if "tool" in txt or "function" in txt or "400" in txt:
                    raise LLMError(f"tools_unsupported: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        raise LLMError(f"LLM 工具调用失败: {last_err}")


def _log_usage(model: str, caller: str, usage: dict):
    """token 用量落库；失败静默，不影响主流程。"""
    if not usage:
        return
    try:
        from core.store.db import session_scope
        from core.store.models import TokenUsage
        with session_scope() as s:
            s.add(TokenUsage(
                model=model, caller=caller,
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0)))
    except Exception:
        pass


def usage_summary(days: int = 30) -> dict:
    """近 N 天 token 用量统计（按 caller/model 聚合）。"""
    from datetime import datetime, timedelta
    from core.store.db import session_scope
    from core.store.models import TokenUsage
    cutoff = datetime.now() - timedelta(days=days)
    with session_scope() as s:
        rows = s.query(TokenUsage).filter(TokenUsage.called_at >= cutoff).all()
    total = sum(r.total_tokens for r in rows)
    by_caller: dict = {}
    by_model: dict = {}
    for r in rows:
        by_caller[r.caller or "unknown"] = by_caller.get(r.caller or "unknown", 0) + r.total_tokens
        by_model[r.model] = by_model.get(r.model, 0) + r.total_tokens
    return {"days": days, "calls": len(rows), "total_tokens": total,
            "by_caller": by_caller, "by_model": by_model}


def test_connection(client: LLMClient | None = None, timeout: int = 15) -> dict:
    """测试模型 API 连通性与配置可用性。

    返回 dict 包含：
    - ok: bool
    - chat_model: str
    - chat_ok: bool
    - chat_latency_ms: int
    - chat_reply: str
    - reasoner_model: str
    - reasoner_ok: bool
    - reasoner_latency_ms: int
    - reasoner_reply: str
    - error: str | None
    """
    c = client or LLMClient()
    if not c.available():
        return {
            "ok": False,
            "error": "配置不完整：请先配置有效的 API Key、Base URL 以及模型名称",
            "chat_ok": False,
            "reasoner_ok": False,
        }

    # 建立测试用客户端，避免因网络无响应卡死 120 秒
    test_c = LLMClient(
        base_url=c.base_url,
        api_key=c.api_key,
        model=c.model,
        reasoner_model=c.reasoner_model,
        temperature=c.temperature,
        timeout=timeout,
    )

    res = {
        "ok": False,
        "chat_model": test_c.model,
        "chat_ok": False,
        "chat_latency_ms": 0,
        "chat_reply": "",
        "reasoner_model": test_c.reasoner_model,
        "reasoner_ok": False,
        "reasoner_latency_ms": 0,
        "reasoner_reply": "",
        "error": None,
    }

    # 1. 测试通识分析模型
    t0 = time.time()
    try:
        reply = test_c.ask("请回复'pong'四个字母确认连接正常", model_type="chat", caller="test_conn")
        latency = int((time.time() - t0) * 1000)
        res["chat_ok"] = True
        res["chat_latency_ms"] = latency
        res["chat_reply"] = (reply or "").strip()[:60]
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Incorrect API key" in err_msg or "AuthenticationError" in err_msg:
            res["error"] = "API Key 鉴权失败 (401)：密钥无效或未激活，请检查 API Key 是否完整正确复制。"
        elif "404" in err_msg or "NotFoundError" in err_msg or "model_not_found" in err_msg:
            res["error"] = f"模型不存在 (404)：当前端点未找到模型「{test_c.model}」，请检查模型名称是否拼写正确。"
        elif "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
            res["error"] = f"连接超时 ({timeout}s)：无法在规定时间内访问 {test_c.base_url}，请检查网络或代理设置。"
        elif "connection" in err_msg.lower():
            res["error"] = f"网络连接失败：无法连通 {test_c.base_url}，请检查 Base URL 是否可访问。"
        else:
            res["error"] = f"通识模型 ({test_c.model}) 测试失败: {err_msg}"
        return res

    # 2. 如果推理模型不同，测试推理模型
    if test_c.reasoner_model and test_c.reasoner_model != test_c.model:
        t0 = time.time()
        try:
            r_reply = test_c.ask("请回复'pong'四个字母确认连接正常", model_type="reasoner", caller="test_conn")
            r_latency = int((time.time() - t0) * 1000)
            res["reasoner_ok"] = True
            res["reasoner_latency_ms"] = r_latency
            res["reasoner_reply"] = (r_reply or "").strip()[:60]
        except Exception as e:
            res["error"] = f"通识模型连接正常，但推理模型 ({test_c.reasoner_model}) 调用失败: {e}"
            return res
    else:
        res["reasoner_ok"] = True
        res["reasoner_latency_ms"] = res["chat_latency_ms"]

    res["ok"] = True
    return res

