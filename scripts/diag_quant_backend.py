"""量化后端可用性诊断（子进程隔离）。

用途
----
回答一个问题：**本机到底哪个量化后端能真的跑完全部四个周期？**

为什么必须用子进程隔离：可疑后端（LightGBM）的失败形态是**进程级硬崩溃**
（access violation），不是可捕获的异常。崩溃在子进程里只是一个非零退出码，
在父进程里就是"预测流水线整条被杀"。所以本脚本让每个组合各自跑在独立
子进程里，父进程只收集退出码与输出。

背景（2026-09-23 实测，本机 Windows + 预编译 wheel）
----------------------------------------------------
现象：在**样本留档打开**（record_samples=True）的长负载下，进程会无 traceback
猝死（access violation）。曾据此误判为"LightGBM 不稳定"，**复测否定了该结论**：

    后端=lgbm : 正常 7 次 / 崩溃 4 次
    后端=gbdt : 正常 3 次 / 崩溃 1 次

两者崩溃率无显著差异，且同一模式同一代码也会一次崩一次过（非确定性）。
⇒ **后端选择不是崩溃的诱因**；真正的共同项是 record_samples=True 的留档路径
（原先每样本开一次 session，已改为批量写入，见 calibration.record_samples_batch）。
所以 [predict].quant_backend 默认仍是 "auto"（LightGBM → GBDT → numpy）。

用法
----
    python scripts/diag_quant_backend.py

输出落在 data/diag_quant_backend.txt，包含每个组合的退出码与逐步日志。
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUT_PATH = os.path.join(ROOT, "data", "diag_quant_backend.txt")

# 子进程脚本：mode 里的字母是开关（见下方 MODES 注释）
CHILD = r'''
import io, os, sys, traceback
ROOT = r"{root}"
sys.path.insert(0, ROOT); os.chdir(ROOT)
# faulthandler：原生硬崩溃（access violation）时把 **Python 调用栈** 打到 stderr。
# 这是定位"无 traceback 的进程猝死"唯一可靠的手段。
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass
# 抑制硬崩溃时的 Windows 错误上报弹窗（否则可能连带把父进程/控制台一起干掉）。
try:
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
except Exception:
    pass
MODE = sys.argv[1]
_FH = io.open(os.path.join(ROOT, "data", "diag_child_%s.txt" % MODE), "w",
              encoding="utf-8")
def out(line=""):
    _FH.write(str(line) + "\n"); _FH.flush()
out("mode=%s" % MODE)
try:
    import numpy as np, pandas as pd
    if "N" in MODE:
        out("step: import core.data.stock（行情数据链）")
        import core.data.stock
    if "P" in MODE:
        out("step: import core.predict.pipeline")
        import core.predict.pipeline
    out("step: import core.predict.quant")
    from core.predict import quant, labels
    if "G" in MODE:
        quant._ALL_CANDIDATES = [c for c in quant._all_candidates() if c[0] == "gbdt"]
        out("forced candidates -> %s" % [c[1] for c in quant._ALL_CANDIDATES])
    out("probe -> %s" % (quant.backend_diagnostics(),))
    rng = np.random.default_rng(7); n = 700
    r = rng.normal(0, 1.2, n); close = 10 * np.exp(np.cumsum(r) / 100.0)
    k = pd.DataFrame({{"date": pd.date_range("2023-01-02", periods=n, freq="B"),
                       "open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": rng.integers(1e6, 5e6, n)}})
    rec = "R" in MODE
    for h in labels.HORIZONS:
        out("step: train horizon=%s record_samples=%s" % (h, rec))
        res = quant.train("_DIAG_" + h, k, max_age_days=0,
                          record_samples=rec, horizon=h)
        out("   -> ok=%s oos_n=%s bss=%s backend=%s"
            % (res.get("ok"), res.get("oos_n"), res.get("oos_bss"),
               res.get("backend")))
        if not res.get("ok"):
            out("   !! %s" % res.get("error"))
    out("DONE（四周期全部完成）")
except Exception as e:
    out("EXC %s: %s" % (type(e).__name__, e))
    out(traceback.format_exc())
finally:
    # 清理本次产生的临时模型。注意 os.listdir 本身也要包住：data/models
    # 不存在时它会抛 FileNotFoundError，从 finally 逃出去会把退出码污染成 1
    # （实测假象：训练四周期全 DONE 却 rc=1）。退出码必须干净地只反映"训练是否崩"。
    try:
        for f in os.listdir(os.path.join(ROOT, "data", "models")):
            if f.startswith("_DIAG_"):
                try:
                    os.remove(os.path.join(ROOT, "data", "models", f))
                except OSError:
                    pass
    except OSError:
        pass
    # 清理本诊断写进校准库的样本：record_samples=True 是本诊断的**核心负载**，
    # 但不能因此把 _DIAG_* 垃圾样本长期留在生产库里（真实标的不可能以 "_" 开头）。
    try:
        from core.store.db import session_scope
        from core.store.models import CalibSample
        with session_scope() as s:
            n = (s.query(CalibSample)
                 .filter(CalibSample.symbol.like("_DIAG_%")).delete(
                     synchronize_session=False))
        out("已清理校准库 _DIAG_* 样本: %s 行" % n)
    except Exception as e:
        out("(校准库清理跳过: %s)" % type(e).__name__)
'''

# 组合说明：
#   A = 四周期训练   R = 记样本（触发数据链 + sqlite 写入，生产同款）
#   N = 预导入行情数据链   P = 预导入 pipeline   G = 只留 GBDT 后端
MODES = [
    ("A",    "四周期 · 不记样本"),
    ("AR",   "四周期 · 记样本（生产同款，触发崩溃的关键负载）"),
    ("NAR",  "预导入数据链 + 四周期 · 记样本"),
    ("PAR",  "预导入 pipeline + 四周期 · 记样本"),
    ("AGR",  "强制 GBDT + 四周期 · 记样本（验证兜底是否稳）"),
]

_FH = io.open(OUT_PATH, "a", encoding="utf-8")


def out(line=""):
    _FH.write(str(line) + "\n")
    _FH.flush()


def run_mode(mode: str, desc: str) -> None:
    code = CHILD.format(root=ROOT.replace("\\", "\\\\"))
    child_log = os.path.join(ROOT, "data", "diag_child_%s.txt" % mode)
    outfn = os.path.join(ROOT, "data", "diag_child_%s.out" % mode)
    for stale in (child_log, outfn):          # 清掉上一轮残留，避免误读
        if os.path.exists(stale):
            os.remove(stale)
    flags = 0
    if os.name == "nt":
        flags = 0x00000200 | 0x00000008        # NEW_PROCESS_GROUP | DETACHED_PROCESS
    try:
        with open(outfn, "wb") as fo:
            p = subprocess.run([PY, "-c", code, mode], stdout=fo, stderr=fo,
                               timeout=1800, creationflags=flags)
        out("[%-4s] rc=%-12s %s" % (mode, p.returncode, desc))
        if os.path.exists(child_log):
            with io.open(child_log, encoding="utf-8") as fh:
                for ln in fh.read().splitlines():
                    out("        " + ln)
        else:
            out("        (子进程未写出日志：启动即崩)")
    except subprocess.TimeoutExpired:
        out("[%-4s] 超时            %s" % (mode, desc))
    out("")


if len(sys.argv) > 1:
    # 单模式运行：结果可信，且一个模式崩溃不会影响其他模式。
    want = sys.argv[1].upper()
    hit = [(m, d) for m, d in MODES if m == want]
    if not hit:
        print("未知模式 %s；可用：%s" % (want, ", ".join(m for m, _ in MODES)))
        raise SystemExit(2)
    out("== 量化后端可用性诊断（单模式 %s）==" % want)
    out("")
    run_mode(*hit[0])
else:
    out("== 量化后端可用性诊断（全模式）==")
    out("")
    out("提示：某个模式若触发硬崩溃，可能连带终止本进程，剩余模式不会执行。")
    out("      出现这种情况时请用单模式重跑：python scripts/diag_quant_backend.py <MODE>")
    out("")
    for m, d in MODES:
        run_mode(m, d)

out("== 判读 ==")
out("  rc != 0 且日志停在某个 step → 该负载下崩了（硬崩溃无 traceback）。")
out("  关键对照：AR（记样本）若在 one_week/one_month 崩，而 A（不记样本）四周期全 DONE，")
out("  则说明崩溃来自样本留档路径，与后端无关（历史误判见本文件顶部背景）。")
out("  批量写入（record_samples_batch）落地后，AR/AGR 应四周期全部 DONE。")

out("")
out("== 环境 ==")
for mod in ("numpy", "pandas", "scipy", "sklearn", "lightgbm"):
    try:
        m = __import__(mod)
        out("  %-9s %s" % (mod, getattr(m, "__version__", "?")))
    except Exception as e:
        out("  %-9s 不可用 (%s)" % (mod, type(e).__name__))
out("  Python    %s" % sys.version.split()[0])
out("  CPU 核数  %s" % os.cpu_count())

print("written -> data/diag_quant_backend.txt")
