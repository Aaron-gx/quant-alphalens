"""三层记忆：fact（经验）/ lesson（教训）/ belief（信念）。

设计要点（见 docs/AI交易员Harness-架构设计.md §4.4 / §6.1）：

1. **写入权限分离**（这是防污染的核心）
   - fact   : 系统自动写（对账 + 归因后），只追加、永不修改
   - lesson : AI 在反思阶段写，但**只能写成 candidate**
   - belief : 系统从 active lesson 聚合生成，AI 不可直接写

2. **证据门**：lesson 从 candidate 升 active 需要
   evidence_count >= min_evidence 且 confidence >= min_confidence
   且反证数不超过 evidence_count 的 max_counter_ratio。
   这样防止 AI 凭单次运气就永久改变行为（自我强化错误）。

3. **不可硬删除**：只改 status（refuted/archived 保留反证痕迹），
   保证"AI 曾经以为 X，后来被 Y 推翻"这条链路可审计。

4. **双真相面**：SQLite 是真相源（可查询/可关联/可统计）；
   Markdown 是只读派生镜像（人类可审计、可 diff）。
   任何写入以 DB 为准，MD 由 DB 重新渲染，**不允许反向修改**。
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from core.config import PROJECT_ROOT, load_config
from core.store.db import session_scope
from core.store.models import TraderMemory

# --- 记忆种类 ---
KIND_FACT, KIND_LESSON, KIND_BELIEF = "fact", "lesson", "belief"

# --- 记忆状态机 ---
ST_CANDIDATE, ST_ACTIVE, ST_REFUTED, ST_ARCHIVED = (
    "candidate", "active", "refuted", "archived")

# --- 作用域前缀 ---
SCOPE_GLOBAL = "global"
SCOPE_ASSET = "asset_type"     # scope = "asset_type:fund"
SCOPE_SYMBOL = "symbol"        # scope = "symbol:510300"

_MAX_STATEMENT_LEN = 800


class MemoryError_(RuntimeError):
    """记忆层规则违例（如 AI 试图写无证据的教训）。"""


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _mem_cfg() -> dict:
    """读取 [trader.memory] 段。"""
    return (load_config().get("trader") or {}).get("memory") or {}


def _min_evidence() -> int:
    return int(_mem_cfg().get("min_evidence_to_activate", 3))


def _min_confidence() -> float:
    return float(_mem_cfg().get("min_confidence_to_activate", 0.6))


def _max_counter_ratio() -> float:
    return float(_mem_cfg().get("max_counter_ratio", 0.333))


def _mirror_dir() -> Path:
    d = (load_config().get("trader") or {}).get(
        "memory_mirror_dir", "data/trader_memory")
    p = Path(d)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_scope(symbol: str = "", asset_type: str = "") -> str:
    """生成作用域字符串。优先级：symbol > asset_type > global。"""
    if symbol:
        return f"{SCOPE_SYMBOL}:{symbol}"
    if asset_type:
        return f"{SCOPE_ASSET}:{asset_type}"
    return SCOPE_GLOBAL


def _norm(statement: str) -> str:
    s = " ".join(str(statement or "").split())
    return s[:_MAX_STATEMENT_LEN]


def _calc_confidence(base: float, evidence_count: int, counter_count: int) -> float:
    """置信度 = 基础分随证据数增长，再按反证比例打折。

    base=0.5、证据 3 条 → 0.5 + 0.08*2 = 0.66（刚好过 0.6 门槛）
    有反证时按反证占比线性折扣，防止"证据多就能无视反例"。
    """
    conf = float(base) + 0.08 * max(0, evidence_count - 1)
    if evidence_count > 0:
        conf *= max(0.0, 1.0 - counter_count / max(evidence_count, 1))
    return round(min(0.95, max(0.0, conf)), 3)


def _to_dict(m: TraderMemory) -> dict:
    return {
        "id": m.id, "trader_id": m.trader_id, "kind": m.kind, "scope": m.scope,
        "symbol": m.symbol, "statement": m.statement, "status": m.status,
        "confidence": round(float(m.confidence or 0), 3),
        "evidence_count": int(m.evidence_count or 0),
        "evidence_refs": list(m.evidence_refs or []),
        "counter_evidence": list(m.counter_evidence or []),
        "source": m.source, "policy_patch": dict(m.policy_patch or {}),
        "version": int(m.version or 1),
        "activated_at": str(m.activated_at) if m.activated_at else "",
        "created_at": str(m.created_at), "updated_at": str(m.updated_at),
    }


# ---------------------------------------------------------------------------
# 写入：fact（系统）
# ---------------------------------------------------------------------------

def write_fact(
    trader_id: int,
    statement: str,
    *,
    scope: str = SCOPE_GLOBAL,
    symbol: str = "",
    detail: dict | None = None,
    run_date: date | None = None,
    mirror: bool = True,
) -> int:
    """系统写经验事实（对账 + 归因后调用）。只追加、永不修改。

    fact 是 lesson 的证据来源——每条 lesson 的 evidence_refs 指向 fact id。
    返回 fact_id。
    """
    stmt = _norm(statement)
    if not stmt:
        raise MemoryError_("fact 内容不能为空")
    ref = {"run_date": str(run_date or date.today()),
           "symbol": symbol, "detail": dict(detail or {})}
    with session_scope() as s:
        m = TraderMemory(
            trader_id=trader_id, kind=KIND_FACT, scope=scope, symbol=symbol,
            statement=stmt, status=ST_ACTIVE,          # fact 生来即"事实"
            confidence=1.0, evidence_count=1, evidence_refs=[ref],
            source="system",
        )
        s.add(m)
        s.flush()
        fid = m.id
    if mirror:
        render_markdown(trader_id)
    return fid


# ---------------------------------------------------------------------------
# 写入：lesson（AI 反思）
# ---------------------------------------------------------------------------

def write_lesson(
    trader_id: int,
    statement: str,
    *,
    scope: str = SCOPE_GLOBAL,
    symbol: str = "",
    evidence_refs: list[dict] | None = None,
    proposed_confidence: float = 0.5,
    mirror: bool = True,
) -> int:
    """AI 在反思阶段提教训。**只能写成 candidate**（升 active 由 promote_lessons 决定）。

    硬规则（pre_reflect 钩子的落地实现）：
    - evidence_refs 必须非空 —— 教训不能凭空产生，必须引用 fact
    - 同 (scope, symbol, statement) 的既有 candidate/active lesson → 累加证据，不新建
    返回 lesson_id。
    """
    stmt = _norm(statement)
    if not stmt:
        raise MemoryError_("lesson 内容不能为空")
    refs = [dict(r) for r in (evidence_refs or []) if isinstance(r, dict)]
    if not refs:
        raise MemoryError_("lesson 必须携带证据引用（evidence_refs 不能为空）")

    with session_scope() as s:
        existing = (s.query(TraderMemory)
                    .filter(TraderMemory.trader_id == trader_id,
                            TraderMemory.kind == KIND_LESSON,
                            TraderMemory.scope == scope,
                            TraderMemory.symbol == symbol,
                            TraderMemory.statement == stmt)
                    .filter(TraderMemory.status.in_([ST_CANDIDATE, ST_ACTIVE]))
                    .first())
        if existing:
            # 同一教训再次被观察到 → 累加证据并重算置信度
            ev = list(existing.evidence_refs or [])
            seen = {str(e.get("fact_id") or e) for e in ev}
            for r in refs:
                key = str(r.get("fact_id") or r)
                if key not in seen:
                    ev.append(r)
                    seen.add(key)
            existing.evidence_refs = ev
            existing.evidence_count = len(ev)
            cnt = len(existing.counter_evidence or [])
            existing.confidence = _calc_confidence(
                proposed_confidence, existing.evidence_count, cnt)
            existing.updated_at = datetime.now()
            lid = existing.id
        else:
            m = TraderMemory(
                trader_id=trader_id, kind=KIND_LESSON, scope=scope, symbol=symbol,
                statement=stmt, status=ST_CANDIDATE,
                confidence=_calc_confidence(proposed_confidence, len(refs), 0),
                evidence_count=len(refs), evidence_refs=refs,
                counter_evidence=[], source="ai_reflection",
            )
            s.add(m)
            s.flush()
            lid = m.id
    if mirror:
        render_markdown(trader_id)
    return lid


def add_evidence(
    trader_id: int, lesson_id: int, ref: dict, *,
    mirror: bool = False,
) -> dict:
    """给既有 lesson 追加一条证据（反思阶段发现同一规律再次应验时用）。

    返回：{"lesson_id", "evidence_count", "confidence", "status"}
    """
    with session_scope() as s:
        m = s.get(TraderMemory, lesson_id)
        if not m or m.trader_id != trader_id or m.kind != KIND_LESSON:
            raise MemoryError_(f"lesson {lesson_id} 不存在或不属于该交易员")
        ev = list(m.evidence_refs or [])
        ev.append(dict(ref))
        m.evidence_refs = ev
        m.evidence_count = len(ev)
        base = max(0.3, float(m.confidence or 0.5) - 0.08 * max(0, len(ev) - 2))
        m.confidence = _calc_confidence(base, len(ev), len(m.counter_evidence or []))
        m.updated_at = datetime.now()
        out = {"lesson_id": lesson_id, "evidence_count": m.evidence_count,
               "confidence": m.confidence, "status": m.status}
    if mirror:
        render_markdown(trader_id)
    return out


# ---------------------------------------------------------------------------
# 状态机：candidate → active / refuted / archived
# ---------------------------------------------------------------------------

def promote_lessons(
    trader_id: int, *,
    min_evidence: int | None = None,
    min_confidence: float | None = None,
    mirror: bool = True,
) -> list[dict]:
    """评估所有 candidate → 达标者升 active。

    达标条件（三者同时满足）：
      1. evidence_count >= min_evidence
      2. confidence >= min_confidence
      3. 反证数 <= evidence_count * max_counter_ratio

    返回：[{"id","from","to","reason"}]（本条本轮发生的状态变更）
    """
    me = _min_evidence() if min_evidence is None else int(min_evidence)
    mc = _min_confidence() if min_confidence is None else float(min_confidence)
    mr = _max_counter_ratio()
    changes: list[dict] = []
    now = datetime.now()

    with session_scope() as s:
        rows = (s.query(TraderMemory)
                .filter(TraderMemory.trader_id == trader_id,
                        TraderMemory.kind == KIND_LESSON,
                        TraderMemory.status == ST_CANDIDATE)
                .all())
        for m in rows:
            n_ev = int(m.evidence_count or 0)
            n_ct = len(m.counter_evidence or [])
            if n_ev < me:
                continue
            if float(m.confidence or 0) < mc:
                continue
            if n_ev > 0 and n_ct > n_ev * mr:
                continue
            m.status = ST_ACTIVE
            m.activated_at = now
            m.updated_at = now
            changes.append({
                "id": m.id, "from": ST_CANDIDATE, "to": ST_ACTIVE,
                "reason": f"证据 {n_ev} 条(≥{me})、置信度 {m.confidence}(≥{mc})、"
                          f"反证 {n_ct} 条未超限",
            })
    if changes and mirror:
        render_markdown(trader_id)
    return changes


def refute_lesson(
    trader_id: int, lesson_id: int, counter: dict, *,
    auto: bool = False, mirror: bool = True,
) -> dict:
    """给 lesson 记一条反证；反证占比超阈值 → 降级 refuted。

    auto=True 表示由 report.memory_effectiveness() 自动触发（记忆自我纠正）。
    返回：{"lesson_id","counter_count","new_status","reason"}
    """
    mr = _max_counter_ratio()
    with session_scope() as s:
        m = s.get(TraderMemory, lesson_id)
        if not m or m.trader_id != trader_id:
            raise MemoryError_(f"lesson {lesson_id} 不存在或不属于该交易员")
        ce = list(m.counter_evidence or [])
        payload = dict(counter or {})
        if auto:
            payload["auto"] = True
        ce.append(payload)
        m.counter_evidence = ce
        m.updated_at = datetime.now()
        n_ev = int(m.evidence_count or 0)
        n_ct = len(ce)
        # 反证占比超阈值 → 降级
        if n_ev > 0 and n_ct > n_ev * mr:
            m.status = ST_REFUTED
            reason = (f"反证 {n_ct} 条 / 证据 {n_ev} 条，超过 {mr:.0%} 阈值 → 降级 refuted")
        else:
            # 未超阈值：置信度按反证比例下调
            base = max(0.3, float(m.confidence or 0.5))
            m.confidence = round(max(0.1, base * (1 - n_ct / max(n_ev, 1))), 3)
            reason = f"记入反证 {n_ct} 条，置信度下调至 {m.confidence}"
        out = {"lesson_id": lesson_id, "counter_count": n_ct,
               "new_status": m.status, "reason": reason}
    if mirror:
        render_markdown(trader_id)
    return out


def archive_lesson(trader_id: int, lesson_id: int, reason: str = "") -> dict:
    """归档某条 lesson（人工或规则触发）。保留痕迹，不删除。"""
    with session_scope() as s:
        m = s.get(TraderMemory, lesson_id)
        if not m or m.trader_id != trader_id:
            raise MemoryError_(f"lesson {lesson_id} 不存在")
        m.status = ST_ARCHIVED
        m.updated_at = datetime.now()
    render_markdown(trader_id)
    return {"lesson_id": lesson_id, "new_status": ST_ARCHIVED, "reason": reason}


# ---------------------------------------------------------------------------
# 写入：belief（系统聚合）
# ---------------------------------------------------------------------------

def rebuild_beliefs(trader_id: int, *, mirror: bool = True) -> dict:
    """从 active lessons 聚合生成 belief（version+1）。系统行为，AI 不可直接调用。

    聚合规则（可编码的确定性映射）：
    - 旧 belief 全部先置 archived（保留历史版本，可回滚）
    - 新 belief 逐条由 active lesson 生成：statement 即 lesson 的陈述，
      policy_patch 是该 lesson 影响到的策略键（默认空，由 lesson 的 detail 决定）
    - version = 历史最大 version + 1

    返回：{"version": int, "beliefs": [...]}
    """
    with session_scope() as s:
        active = (s.query(TraderMemory)
                  .filter(TraderMemory.trader_id == trader_id,
                          TraderMemory.kind == KIND_LESSON,
                          TraderMemory.status == ST_ACTIVE)
                  .order_by(TraderMemory.confidence.desc())
                  .all())
        old = (s.query(TraderMemory)
               .filter(TraderMemory.trader_id == trader_id,
                       TraderMemory.kind == KIND_BELIEF)
               .all())

        version = 1
        for b in old:
            version = max(version, int(b.version or 1) + 1)
            if b.status == ST_ACTIVE:
                b.status = ST_ARCHIVED

        created: list[dict] = []
        now = datetime.now()
        for les in active:
            patch = dict(les.policy_patch or {})
            b = TraderMemory(
                trader_id=trader_id, kind=KIND_BELIEF,
                scope=les.scope, symbol=les.symbol,
                statement=les.statement, status=ST_ACTIVE,
                confidence=float(les.confidence or 0.5),
                evidence_count=int(les.evidence_count or 0),
                evidence_refs=[{"lesson_id": les.id}],
                counter_evidence=[], source="system",
                policy_patch=patch, version=version,
                activated_at=now,
            )
            s.add(b)
            s.flush()
            created.append({"id": b.id, "statement": b.statement,
                            "version": version, "policy_patch": patch,
                            "from_lesson": les.id})
        out = {"version": version, "beliefs": created}
    if mirror:
        render_markdown(trader_id)
    return out


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def recall(
    trader_id: int, *,
    kind: str | None = None,
    scope: str | None = None,
    symbol: str | None = None,
    status: str | None = ST_ACTIVE,
    limit: int = 10,
) -> list[dict]:
    """检索记忆（供 AI 的 recall_memory 工具 + 上下文注入）。

    **默认只返回 active**——candidate 不进决策上下文，避免未证实的教训影响决策。
    status=None 时返回全部状态（给审计/报告用）。
    """
    with session_scope() as s:
        q = s.query(TraderMemory).filter(TraderMemory.trader_id == trader_id)
        if kind:
            q = q.filter(TraderMemory.kind == kind)
        if status:
            q = q.filter(TraderMemory.status == status)
        if scope:
            q = q.filter(TraderMemory.scope == scope)
        if symbol:
            q = q.filter(TraderMemory.symbol == symbol)
        rows = (q.order_by(TraderMemory.confidence.desc(),
                           TraderMemory.id.desc())
                .limit(max(1, int(limit))).all())
        return [_to_dict(r) for r in rows]


def view(trader_id: int, *, limit_lessons: int = 10,
         limit_mistakes: int = 5) -> dict:
    """组装注入决策上下文的记忆视图（只含 active）。

    返回：{"facts": [...], "lessons": [...], "beliefs": [...],
           "stats": {...}}
    """
    facts = recall(trader_id, kind=KIND_FACT, limit=limit_mistakes)
    lessons = recall(trader_id, kind=KIND_LESSON, limit=limit_lessons)
    beliefs = recall(trader_id, kind=KIND_BELIEF, status=ST_ACTIVE, limit=10)
    counts = {k: len(recall(trader_id, kind=k, status=None, limit=9999))
              for k in (KIND_FACT, KIND_LESSON, KIND_BELIEF)}
    return {"facts": facts, "lessons": lessons, "beliefs": beliefs,
            "stats": {"counts_all_status": counts,
                      "active_lessons": len(lessons),
                      "active_beliefs": len(beliefs)}}


def history(trader_id: int, *, kind: str | None = None,
            limit: int = 200) -> list[dict]:
    """全部状态的历史记忆（供 report/审计）。"""
    return recall(trader_id, kind=kind, status=None, limit=limit)


# ---------------------------------------------------------------------------
# Markdown 镜像（只读派生视图）
# ---------------------------------------------------------------------------

_STATUS_LABEL = {
    ST_CANDIDATE: "候选（证据不足，未生效）",
    ST_ACTIVE: "生效",
    ST_REFUTED: "已被反驳",
    ST_ARCHIVED: "已归档",
}


def render_markdown(trader_id: int, out_dir: Path | None = None) -> str:
    """把 DB 中的记忆渲染成 Markdown 镜像到磁盘。

    DB 是真相源，本文件是**只读派生视图**：不要手工修改后再期望回灌。
    返回写入的文件路径。
    """
    d = out_dir or _mirror_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"trader_{trader_id}.md"
    items = history(trader_id, limit=5000)

    lines: list[str] = [
        f"# 交易员 #{trader_id} 记忆镜像", "",
        f"> 本文由系统从数据库自动渲染（{datetime.now():%Y-%m-%d %H:%M:%S}）。",
        "> 数据库是唯一真相源；请勿手工修改本文后期望回灌。", "",
    ]
    for kind, title in ((KIND_BELIEF, "信念（当前生效的策略认知）"),
                        (KIND_LESSON, "教训（AI 总结的规律）"),
                        (KIND_FACT, "经验（客观对账事实）")):
        group = [m for m in items if m["kind"] == kind]
        lines.append(f"## {title}（{len(group)} 条）")
        lines.append("")
        if not group:
            lines.append("_暂无_")
            lines.append("")
            continue
        for m in group:
            st = _STATUS_LABEL.get(m["status"], m["status"])
            tag = f"`{st}`"
            if kind == KIND_LESSON:
                tag += (f" 置信度 {m['confidence']} · 证据 {m['evidence_count']} 条"
                        f" · 反证 {len(m['counter_evidence'])} 条")
            if kind == KIND_BELIEF:
                tag += f"  v{m['version']}"
            lines.append(f"- **[{m['id']}]** {m['statement']}")
            lines.append(f"  - {tag} · 作用域 `{m['scope']}`")
            if kind == KIND_FACT and m["evidence_refs"]:
                d0 = (m["evidence_refs"][0] or {}).get("detail") or {}
                if d0:
                    brief = ", ".join(f"{k}={v}" for k, v in list(d0.items())[:6])
                    lines.append(f"  - 依据：{brief}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
