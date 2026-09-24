"""配置管理：读取 config.toml，环境变量可覆盖敏感项。"""
from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
EXAMPLE_CONFIG = PROJECT_ROOT / "config.example.toml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """加载 config.toml；不存在则回退到 example（方便开箱跑通界面）。"""
    path = Path(os.environ.get("JIJIN_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        path = EXAMPLE_CONFIG
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    # 环境变量覆盖（服务器部署时用）
    cfg.setdefault("llm", {})
    cfg["llm"]["api_key"] = os.environ.get("JIJIN_LLM_KEY", cfg["llm"].get("api_key", ""))
    cfg["llm"]["base_url"] = os.environ.get("JIJIN_LLM_BASE_URL", cfg["llm"].get("base_url", ""))
    cfg["llm"]["model"] = os.environ.get("JIJIN_LLM_MODEL", cfg["llm"].get("model", ""))
    cfg.setdefault("api", {})
    cfg["api"]["api_key"] = os.environ.get("JIJIN_API_KEY", cfg["api"].get("api_key", ""))
    cfg.setdefault("db", {})
    cfg["db"]["url"] = os.environ.get("JIJIN_DB_URL", cfg["db"].get("url", ""))
    cfg.setdefault("http", {})
    cfg["http"]["proxy"] = os.environ.get("JIJIN_HTTP_PROXY", cfg["http"].get("proxy", ""))
    return cfg


def get(section: str, key: str, default=None):
    return load_config().get(section, {}).get(key, default)


def http_proxy() -> str:
    """HTTP/HTTPS 代理地址（如本地 Clash 混合端口 http://127.0.0.1:7890），空串=直连。"""
    return load_config().get("http", {}).get("proxy", "").strip()


def reload():
    """测试或改配置后调用，清缓存重读。"""
    load_config.cache_clear()


def _esc_toml(s) -> str:
    """TOML 基础字符串转义（反斜杠与双引号）。"""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def save_llm_config(*, base_url: str, api_key: str, model: str,
                    reasoner_model: str, temperature: float) -> None:
    """把 LLM 配置写回 config.toml 的 [llm] 段并清缓存即时生效。

    只替换 [llm] 段，保留文件其余部分（含注释与其它配置段）。
    """
    path = Path(os.environ.get("JIJIN_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        path = EXAMPLE_CONFIG
    timeout = int(get("llm", "timeout", 120) or 120)
    block = (
        "# 任意兼容 OpenAI 协议的端点\n"
        f'base_url = "{_esc_toml(base_url)}"\n'
        f'api_key = "{_esc_toml(api_key)}"\n'
        "# 普通分析模型（便宜、快）\n"
        f'model = "{_esc_toml(model)}"\n'
        "# 推理模型（用于综合裁决层，可与普通模型相同）\n"
        f'reasoner_model = "{_esc_toml(reasoner_model)}"\n'
        f"temperature = {float(temperature):g}\n"
        f"timeout = {timeout}\n"
    )
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    i, replaced = 0, False
    while i < len(lines):
        if lines[i].strip() == "[llm]":
            out.append(lines[i])
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("["):
                i += 1
            out.append(block)
            replaced = True
        else:
            out.append(lines[i])
            i += 1
    if not replaced:
        out.append("\n[llm]\n" + block)
    path.write_text("".join(out), encoding="utf-8")
    reload()
