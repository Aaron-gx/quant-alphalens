"""AI 交易员（harness 形态）：自主循环 + 交易台工具 + 三层记忆。

设计文档：docs/AI交易员Harness-架构设计.md

与本项目其它模块的关系：
- 复用 core.paper.engine 作为交易内核（撮合/持仓/T+1/佣金）
- 复用 core.predict.pipeline.predict 作为 AI 可调用的预测工具
- 复用 core.verify.evaluator 的到期对账能力作为反思输入
- 新增本包：memory（三层记忆）/ attributor（纠错归因）/ tools（交易台）
  / guard（风控门禁）/ loop（每日自主循环）/ report（三层评估）

权限铁律（见设计文档 §4.2 / §4.5）：
- 只有 2 个写工具（place_buy_order / place_sell_order），且必须过 guard
- AI 没有"写记忆"的工具：记忆写入只能在反思阶段由系统接口完成
"""
