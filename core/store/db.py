"""数据库连接与会话管理。"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from core.config import get, PROJECT_ROOT
from core.store.models import Base

_engine = None
_SessionLocal: sessionmaker | None = None


def _resolve_sqlite_path(url: str) -> str:
    """sqlite:///data/jijin.db 相对路径锚定到项目根目录。"""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url[len("sqlite:///"):]
        if not Path(rel).is_absolute() and not rel.startswith(":memory:"):
            abs_path = PROJECT_ROOT / rel
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{abs_path.as_posix()}"
    return url


def _default_sql(col) -> str:
    """把模型里的标量默认值翻译成 SQLite 能接受的 DEFAULT 片段。"""
    if col.default is None or not getattr(col.default, "is_scalar", False):
        return ""
    v = col.default.arg
    if isinstance(v, bool):
        return f" DEFAULT {1 if v else 0}"
    if isinstance(v, (int, float)):
        return f" DEFAULT {v}"
    if isinstance(v, str):
        return f" DEFAULT '{v}'"
    return ""


def _light_migrations(engine) -> None:
    """轻量加列迁移：SQLite 的 create_all 不会给已存在的表补列。

    对模型里新增的列做幂等 ALTER TABLE ADD COLUMN（只加列，不改类型、不删数据），
    保证老库升级后不会因为缺列而报错。
    """
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    existing = set(insp.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing:
                continue
            try:
                cols = {c["name"] for c in insp.get_columns(table.name)}
            except Exception:
                continue
            for col in table.columns:
                if col.name in cols:
                    continue
                col_type = col.type.compile(engine.dialect)
                ddl = (f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" '
                       f"{col_type}{_default_sql(col)}")
                try:
                    conn.execute(text(ddl))
                except Exception:
                    pass  # 加列失败（如列名冲突）不阻塞启动


def get_engine():
    global _engine
    if _engine is None:
        url = _resolve_sqlite_path(get("db", "url", "sqlite:///data/jijin.db"))
        kwargs: dict = {"pool_pre_ping": True, "future": True}
        if ":memory:" in url:
            # 内存库：单连接池共享，否则每个连接都是全新空库（测试用）
            from sqlalchemy.pool import StaticPool
            kwargs.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        Base.metadata.create_all(_engine)
        if url.startswith("sqlite"):
            _light_migrations(_engine)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal()


@contextmanager
def session_scope():
    """事务作用域：正常提交，异常回滚。"""
    s = get_session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def reset_engine():
    """测试用：重置连接。"""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
