"""环境体检 + LightGBM 可用性探测（写文件读结果，避免 shell 缺工具）。"""
import sys
import traceback

for name in ("numpy", "pandas", "scipy", "sklearn", "lightgbm"):
    try:
        m = __import__(name)
        print(f"OK   {name:<10} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"FAIL {name:<10} {type(e).__name__}: {e}")

print("-" * 60)
try:
    import numpy as np
    import lightgbm as lgb
    X = np.random.randn(300, 10)
    y = np.random.randn(300)
    m = lgb.LGBMRegressor(n_estimators=50, num_leaves=7, verbose=-1).fit(X, y)
    print("LGBMRegressor OK  ->", np.round(m.predict(X[:2]), 4).tolist())
except Exception:
    print("LGBMRegressor FAIL")
    traceback.print_exc(limit=3)

print("-" * 60)
try:
    import numpy as np
    import lightgbm as lgb
    X = np.random.randn(300, 10)
    y = np.random.randint(0, 3, 300)
    m = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, verbose=-1).fit(X, y)
    print("LGBMClassifier OK ->", np.round(m.predict_proba(X[:2]), 4).tolist())
except Exception:
    print("LGBMClassifier FAIL")
    traceback.print_exc(limit=3)

print("python", sys.version.split()[0])
