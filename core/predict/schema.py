"""结构化预测的 Schema 定义与校验。

预测的每个周期输出三方向概率 + 区间，机器可读 —— 这是预测验证闭环的前提。
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field, model_validator

# 周期口径唯一来源：core.predict.labels（这里只做再导出，不再自行定义）
from core.predict.labels import HORIZON_DAYS, HORIZONS, argmax_direction  # noqa: F401

logger = logging.getLogger(__name__)

HORIZON_LABELS = {
    "next_day": "次日",
    "one_week": "一周",
    "one_month": "一个月",
    "quarter": "三个月",
}


class HorizonForecast(BaseModel):
    # 不设上界：LLM 偶尔返回 60/0.6 两种口径，统一由下面的口径钉死器归一化
    up: float = Field(ge=0, description="上涨概率")
    flat: float = Field(ge=0, description="平盘概率")
    down: float = Field(ge=0, description="下跌概率")
    range_pct: str = Field(default="", description="预期波动区间，如 ±2%")

    @model_validator(mode="after")
    def _pin_scale(self) -> "HorizonForecast":
        """把概率口径钉死：混合百分数 → [0,1]，和恒为 1，全零退化为均匀。

        原实现 `_sum_check` 是**空壳**（`if not (...): pass`）——校验不出任何东西。
        于是 LLM 返回 40/30/30 这类百分数时，会一路带进落库、Brier 打分、
        校准训练：概率不再是概率，所有下游指标都是噪声，且完全静默。

        三条规则：
          1. 混合口径：逐个判，>1.5 的按百分数 /100（容忍 0.4/30/30 混用）
          2. 和 ≤ 0（全零/负值）→ 均匀分布 1/3，不抛异常打断流水线
          3. 归一化到和 = 1
        另附低区分度告警（max < 0.4 ≈ 三路都接近 1/3，等于没表态）。
        """
        vals = (self.up, self.flat, self.down)
        if any(v > 1.5 for v in vals):
            self.up, self.flat, self.down = [v / 100.0 if v > 1.5 else v
                                             for v in vals]
        total = self.up + self.flat + self.down
        if total <= 0:
            self.up = self.flat = self.down = 1.0 / 3
        elif abs(total - 1.0) > 1e-6:
            self.up, self.flat, self.down = (self.up / total, self.flat / total,
                                             self.down / total)
        if max(self.up, self.flat, self.down) < 0.40:
            logger.warning("方向概率区分度过低(max=%.3f)，近似无判断: up=%.3f flat=%.3f down=%.3f",
                           max(self.up, self.flat, self.down),
                           self.up, self.flat, self.down)
        return self

    def normalize(self) -> "HorizonForecast":
        """兼容旧调用：口径已在验证期钉死，这里只做幂等的再归一。"""
        total = self.up + self.flat + self.down
        if total > 0 and abs(total - 1.0) > 1e-6:
            self.up, self.flat, self.down = self.up/total, self.flat/total, self.down/total
        return self

    @property
    def spread(self) -> float:
        """最大概率 − 最小概率：这个预测有没有"态度"。"""
        return round(max(self.up, self.flat, self.down)
                     - min(self.up, self.flat, self.down), 4)

    @property
    def direction(self) -> str:
        # 走统一实现：平手取 flat（无观点），避免"三路均等"被记成看涨
        return argmax_direction({"up": self.up, "flat": self.flat, "down": self.down})


class Scenario(BaseModel):
    probability: float = Field(ge=0, default=0.0)
    target_range: str = ""
    triggers: list[str] = Field(default_factory=list)


class ActionAdvice(BaseModel):
    advice: str = ""                 # 一句话操作建议
    entry: str = ""                  # 买入参考位/条件
    stop_loss: str = ""              # 止损位/条件
    target: str = ""                 # 目标位
    position_hint: str = ""          # 仓位建议


class PredictionResult(BaseModel):
    """综合裁决层的结构化输出。"""
    horizons: dict[str, HorizonForecast]
    scenarios: dict[str, Scenario] = Field(default_factory=dict)  # optimistic/base/pessimistic
    confidence: int = Field(ge=0, le=100, default=50)
    key_factors: list[str] = Field(default_factory=list)
    action: ActionAdvice = Field(default_factory=ActionAdvice)
    risks: list[str] = Field(default_factory=list)
    summary: str = ""

    @model_validator(mode="after")
    def _check_horizons(self) -> "PredictionResult":
        """缺周期告警。

        缺一个周期不只是"少一格 UI"——该周期的到期对账、Brier、校准样本**全部断档**，
        而流水线不会报错。所以这里至少留下一条可追踪的告警。
        """
        missing = [h for h in HORIZONS if h not in self.horizons]
        if missing:
            logger.warning("综合裁决缺少周期 %s：这些周期将无法对账、不进校准样本", missing)
        return self

    def normalized(self) -> "PredictionResult":
        for h in self.horizons.values():
            h.normalize()
        scen_total = sum(s.probability for s in self.scenarios.values())
        if scen_total > 1.0:  # 兼容百分数口径
            for s in self.scenarios.values():
                s.probability = s.probability / scen_total
        return self


PREDICTION_JSON_INSTRUCTION = """严格按以下 JSON Schema 输出（只输出 JSON，不要任何多余文字）：
{
  "horizons": {
    "next_day":  {"up": 0~1, "flat": 0~1, "down": 0~1, "range_pct": "±x%"},
    "one_week":  {"up": 0~1, "flat": 0~1, "down": 0~1, "range_pct": "±x%"},
    "one_month": {"up": 0~1, "flat": 0~1, "down": 0~1, "range_pct": "±x%"},
    "quarter":   {"up": 0~1, "flat": 0~1, "down": 0~1, "range_pct": "±x%"}
  },
  "scenarios": {
    "optimistic": {"probability": 0~1, "target_range": "...", "triggers": ["触发条件..."]},
    "base":       {"probability": 0~1, "target_range": "...", "triggers": ["..."]},
    "pessimistic":{"probability": 0~1, "target_range": "...", "triggers": ["..."]}
  },
  "confidence": 0-100,
  "key_factors": ["影响预测的最关键因素，最多5条"],
  "action": {"advice": "一句话操作建议", "entry": "买入参考位/条件", "stop_loss": "止损位/条件", "target": "目标位", "position_hint": "仓位建议"},
  "risks": ["最需警惕的风险，最多3条"],
  "summary": "150字内的结论摘要"
}
要求：
1. 概率一律用 **0~1 的小数**（写 0.42，不要写 42 当 42%）；每个 horizon 的 up+flat+down 之和必须为 1。
2. 概率要有区分度：三个方向不许都挤在 0.33 附近（系统会把区分度不足的预测判为"没有判断"）。
3. 期限越长不确定性越大：flat 概率应随周期拉长而整体抬升；各周期方向不应自相矛盾
   （例如次日看涨、一周看跌，就必须有 key_factor 能解释这个转折）。
4. "平盘"的口径按周期不同，由系统按 ±0.5%×√周期 判定
   （次日 ±0.5%、一周 ±1.1%、一月 ±2.2%、一季 ±3.9%），请按此口径给 flat。
5. range_pct 用"±x%"格式，且要与该周期的 flat 概率相称：flat 越高，区间应越窄。"""
