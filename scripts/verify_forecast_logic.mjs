#!/usr/bin/env node
/**
 * 方向裁决口径的可执行自检（不依赖后端，直接对 src/lib/forecast.ts 断言）。
 *
 * ⚠️ 夹具必须**冻结**，不能读实时数据
 * ------------------------------------
 * 最初这版直接从 `data/_diag_label_swap.json`（诊断脚本的实时输出）取 id=12 当输入。
 * 于是数据一旦被修好，这个自检自己就红了 —— 断言依赖了**会被修复掉的生产数据**，
 * 修完 bug 反而"测试失败"，属于典型的自毁式测试。
 * 现在改为把修复前的原始数值**硬编码成快照夹具**：它就该永远保持"被污染"的样子，
 * 用来钉死"这套逻辑必须能识别出污染、且能算出被反转的结论"。
 *
 * 用法： node scripts/verify_forecast_logic.mjs
 */
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
// Windows 上动态 import 绝对路径必须是 file:// URL，直接给 "E:\..." 会报
// ERR_UNSUPPORTED_ESM_URL_SCHEME；.ts 由 Node 22 的类型剥离直接加载。
const { detectInvertedProbs, allVerdicts, overallVerdict, resolveForecastAnchor,
        buildForecastPlan, forecastMarks } =
  await import(pathToFileURL(join(ROOT, 'frontend/src/lib/forecast.ts')).href);

let pass = 0;
let fail = 0;
const check = (name, actual, expected) => {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  ok ? pass++ : fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `\n      expected=${JSON.stringify(expected)}\n      actual  =${JSON.stringify(actual)}`}`);
};

/* ── 冻结夹具：id=12（016665，base_date=2026-09-24）**修复前**的原始快照 ──────
 * before = 模型（含融合）的原始输出，是模型真正想表达的概率
 * after  = 被"标签顺序缺陷"对调后的值，即当时落库/前端显示的那个
 * 证据：该记录 report_text 写「短线超买不宜追高…MACD金叉提供中期支撑」（短空长多），
 *      与 before 完全吻合；而 after 把次日说成看涨，与它自己的报告文字矛盾。 */
const FIXTURE = [
  { key: 'next_day',  range: '±1.2%', before: { up: 0.32, flat: 0.29, down: 0.39 }, after: { up: 0.39, flat: 0.29, down: 0.32 } },
  { key: 'one_week',  range: '±3%',   before: { up: 0.36, flat: 0.26, down: 0.38 }, after: { up: 0.38, flat: 0.26, down: 0.36 } },
  { key: 'one_month', range: '±6%',   before: { up: 0.42, flat: 0.22, down: 0.36 }, after: { up: 0.36, flat: 0.22, down: 0.42 } },
  { key: 'quarter',   range: '±12%',  before: { up: 0.45, flat: 0.20, down: 0.35 }, after: { up: 0.35, flat: 0.20, down: 0.45 } },
];

const snap = { calibration: {} };
for (const f of FIXTURE) {
  snap.calibration[f.key] = { method: 'identity', n: 0, probs_before: f.before, probs_after: f.after };
}
const toHorizons = (side) => Object.fromEntries(
  FIXTURE.map((f) => [f.key, { ...f[side], range_pct: f.range }]),
);

// ── ① 对调指纹检出 ──────────────────────────────────────────────────────
check('污染夹具被检出', detectInvertedProbs(snap), true);
check('干净的 identity（before==after）不误报', detectInvertedProbs({
  calibration: { next_day: { method: 'identity', probs_before: { up: 0.4, flat: 0.3, down: 0.3 },
                             probs_after: { up: 0.4, flat: 0.3, down: 0.3 } } },
}), false);
// 陷阱：涨=跌时，"对调"与"恒等"是同一个变换。若不先要求"确实发生了变化"，
// 对称记录会被误判为污染 → 健康数据上弹出不该出现的告警。
// （真实踩过：某周期 up=down=0.36，诊断脚本因此误报。）
check('涨=跌的对称记录不误报（对调≡恒等）', detectInvertedProbs({
  calibration: { one_week: { method: 'identity', probs_before: { up: 0.36, flat: 0.28, down: 0.36 },
                             probs_after: { up: 0.36, flat: 0.28, down: 0.36 } } },
}), false);
check('涨=跌但 flat 也参与对调的对称记录不误报', detectInvertedProbs({
  calibration: { q: { method: 'identity', probs_before: { up: 0.5, flat: 0.0, down: 0.5 },
                      probs_after: { up: 0.5, flat: 0.0, down: 0.5 } } },
}), false);
check('真校准（method!=identity）不误报', detectInvertedProbs({
  calibration: { next_day: { method: 'temperature', probs_before: { up: 0.4, flat: 0.3, down: 0.3 },
                             probs_after: { up: 0.3, flat: 0.3, down: 0.4 } } },
}), false);
check('无快照不误报', detectInvertedProbs(null), false);
check('空快照不误报', detectInvertedProbs({}), false);
check('缺 calibration 不误报', detectInvertedProbs({ quote: { price: 1 } }), false);

// ── ② before 才是真值：修正后的结论应为「偏多」 ─────────────────────────
const vb = allVerdicts(toHorizons('before'));
const ob = overallVerdict(vb);
check('before → 综合结论为偏多', ob.kind, 'up');
check('before → 一月看涨', vb.find((v) => v.key === 'one_month').kind, 'up');
check('before → 一季看涨', vb.find((v) => v.key === 'quarter').kind, 'up');
check('before → 次日方向不明（0.39 < 0.40 门槛）', vb.find((v) => v.key === 'next_day').kind, 'unclear');

// ── ③ after 是当时页面显示的（错的）值：结论被反转成「偏空」 ────────────
const va = allVerdicts(toHorizons('after'));
const oa = overallVerdict(va);
check('after → 综合结论为偏空（与真值相反，故必须告警）', oa.kind, 'down');
// 次日两边的"方向"都没过 0.40 门槛（同判「方向不明」），所以不能拿 kind 做断言；
// 真正暴露污染的是**涨跌相对大小被反转**，这才是与报告文字矛盾的那个点。
const ndB = vb.find((v) => v.key === 'next_day');
const ndA = va.find((v) => v.key === 'next_day');
check('before → 次日 跌>涨（与报告「短线超买不宜追高」一致）', ndB.down > ndB.up, true);
check('after  → 次日 涨>跌（与报告文字直接矛盾）', ndA.up > ndA.down, true);
check('次日两侧同判「方向不明」（0.40 门槛下都无优势）', [ndB.kind, ndA.kind], ['unclear', 'unclear']);

// ── ④ 预期偏移算法：概率趋近均衡必然趋近 0，这就是紫线看起来是平的的原因 ──
const midMax = Math.max(...va.map((v) => Math.abs(v.midPct)));
check('四周期预期偏移全部 < 1%', midMax < 0.01, true);
console.log(`\n  参考：夹具侧最大预期偏移 = ${(midMax * 100).toFixed(3)}%`
  + `（折算到 y 轴跨度约 2.4、图高 380px 的图上 ≈ ${(midMax / 2.4 * 380).toFixed(2)} px）`);
console.log('  ⇒ 在价格刻度的图上，这点位移人眼不可分辨：平线本身是"没有方向"的正确表达。');

/* ── ⑤ 结论措辞不得退回"模板填空"形态（本次返工的核心不变量）──────────────
 * 用户原文指出：结论读起来不像模型生成，而像一张填好的表 ——
 *   「依据：次日、一周」＋「次日、一周 给出看跌；其余周期方向不明或震荡。
 *     注意这是概率上的相对优势，不是幅度承诺。」
 * 特征有三：① 同一批周期名在一句里出现两次；② 每个分支末尾挂同一句免责套话；
 * ③ 结论里嵌着本应由明细表展示的周期名。这三条都必须钉住，否则一改就复发。
 * 断言的写法刻意不检查具体句子（那是"断言实现细节"），只检查**结构性质**。 */
const NAMES = ['次日', '一周', '一月', '一季'];
const hasDupName = (s) => NAMES.some((n) => (s.match(new RegExp(n, 'g')) || []).length > 1);
for (const [tag, o, vs] of [['before(偏多)', ob, vb], ['after(偏空)', oa, va]]) {
  check(`⑤ ${tag} 结论不复述周期名（周期名只在明细表出现一次）`, hasDupName(o.detail), false);
  // 免责话术只允许出现在组件底部固定的规则区；结论句里每档都挂一遍就是模板签名
  check(`⑤ ${tag} 结论不含固定免责套话`, /不是幅度承诺|不构成|仅供|请注意/.test(o.detail), false);
  // 计数事实必须等于逐周期裁决的真实分布，供界面显示"看涨 N · 看跌 M"
  check(`⑤ ${tag} counts 合计等于周期数`, o.counts.up + o.counts.down + o.counts.flat + o.counts.unclear, o.counts.total);
  const real = (k) => vs.filter((v) => v.kind === k).length;
  check(`⑤ ${tag} counts 看涨/看跌与实际裁决一致`,
    [o.counts.up, o.counts.down], [real('up'), real('down')]);
  check(`⑤ ${tag} counts 震荡/不明与实际裁决一致`,
    [o.counts.flat, o.counts.unclear], [real('flat'), real('unclear')]);
}
// 结论句必须非空且不含未替换的占位符
for (const [tag, o] of [['before', ob], ['after', oa]]) {
  check(`⑤ ${tag} detail 非空且无占位符`, !!o.detail && !/\{\{|\}\}|undefined|NaN/.test(o.detail), true);
}
check('⑤ 已移除 drivers 字段（周期名不再由结论携带）', 'drivers' in ob, false);

/* ── ⑥ 预测起算基准：不得再用"行情最后一日"当起算日 ─────────────────────────
 * 缺陷原貌：一条 base_date = 09-24 的研判，图上「次日 T+1」的标签被钉在 **09-23**,
 * 一个**已经过去**的日期。用户读到的是"模型在预测昨天"——后面所有解读都无从谈起。
 * 根因：起算日取的是"行情最后一日净值"(09-22)，而不是研判自己的 base_date。
 *
 * ⚠️ 这张到期日表是**冻结夹具**，同时也是后端 labels.due_dates_for 在同一组输入下的
 * 输出（交易日口径、跳过周末）。两侧各断言一遍同一张表，任何一侧改口径都会立刻变红——
 * 这正是"两处实现各自漂移"那类缺陷的防线。
 * 夹具：base_date = 2026-09-24（周四），行情最后一日 = 2026-09-22（周二）。 */
const LAST_NAV = '2026-09-22';
const BASE_DATE = '2026-09-24';
const EXPECT_DUE = {
  next_day: '2026-09-25',
  one_week: '2026-10-01',
  one_month: '2026-10-22',
  quarter: '2026-12-17',
};
const HORIZON_FIXTURE = {
  next_day: { up: 0.32, flat: 0.29, down: 0.39, range_pct: '±1.2%' },
  one_week: { up: 0.36, flat: 0.26, down: 0.38, range_pct: '±3%' },
  one_month: { up: 0.42, flat: 0.22, down: 0.36, range_pct: '±6%' },
  quarter: { up: 0.45, flat: 0.20, down: 0.35, range_pct: '±12%' },
};

check('⑥ 起算日优先取 base_date（不再用行情最后一日）',
  resolveForecastAnchor(BASE_DATE, LAST_NAV), { date: BASE_DATE, source: 'base_date' });
check('⑥ base_date 缺失时才退回行情最后一日',
  resolveForecastAnchor(null, LAST_NAV), { date: LAST_NAV, source: 'last_price' });
check('⑥ 两者都缺 → null（宁可不画，也不编一个日期）',
  resolveForecastAnchor('', ''), null);
check('⑥ 非法日期串不被当成锚点', resolveForecastAnchor('2026-9-24', ''), null);

const plan = buildForecastPlan(HORIZON_FIXTURE, BASE_DATE, LAST_NAV, null);
check('⑥ 起算日 = base_date', plan.anchor.date, BASE_DATE);
check('⑥ 起算日比行情最后一日晚 2 天（界面披露用）', plan.gapDays, 2);
check('⑥ 未接后端时 dueOrigin=weekend（谁算的）', plan.dueOrigin, 'weekend');
check('⑥ 未接后端时 dueCalendar=weekday（什么质量）', plan.dueCalendar, 'weekday');
check('⑥ 各周期到期日（交易日口径，冻结表）', plan.dueDates, EXPECT_DUE);
check('⑥ 所有到期日都严格晚于起算日',
  Object.values(plan.dueDates).every((d) => d > BASE_DATE), true);
check('⑥ 「次日」不再落在已过去的 09-23',
  plan.dueDates.next_day !== '2026-09-23', true);
check('⑥ 到期日全部存在于预测区日期网格里',
  Object.values(plan.dueDates).every((d) => plan.dates.includes(d)), true);
check('⑥ steps 指向的日期与到期日集合一致',
  Object.entries(plan.steps).map(([, i]) => plan.dates[i]).sort(),
  Object.values(EXPECT_DUE).sort());

// 标注日期只要不在预测曲线里，KlineChart 会直接 return null —— **静默消失**，无任何报错。
const markList = forecastMarks(HORIZON_FIXTURE, plan);
check('⑥ 四个周期都有标注（不会静默消失）', markList.length, 4);
check('⑥ 标注日期全部落在预测曲线上',
  markList.every((m) => plan.dates.includes(m.date)), true);
check('⑥ 标注与到期日一一对应',
  markList.map((m) => `${m.tradingDays}:${m.date}`).sort(),
  ['1:2026-09-25', '20:2026-10-22', '5:2026-10-01', '60:2026-12-17']);

/* 后端给了到期日就必须**覆盖**前端推算：后端按交易日历算，前端算不出节假日。
 * 这里让 quarter 的到期日比"排周末"的结果再晚 2 个交易日（模拟长假顺延），
 * 断言它被并进日期网格 —— 否则那条标注会正好在长假那次静默消失。 */
const withBackend = buildForecastPlan(HORIZON_FIXTURE, BASE_DATE, LAST_NAV,
  { ...EXPECT_DUE, quarter: '2026-12-21' }, 'index');
check('⑥ 有后端日期时 dueOrigin=backend', withBackend.dueOrigin, 'backend');
check('⑥ 后端给 index 就该照实说 index（已跳过节假日）', withBackend.dueCalendar, 'index');
check('⑥ 后端到期日覆盖前端推算', withBackend.dueDates.quarter, '2026-12-21');
check('⑥ 后端到期日被并进日期网格（否则标注静默消失）',
  withBackend.dates.includes('2026-12-21'), true);
check('⑥ steps 指向的是后端到期日',
  withBackend.dates[withBackend.steps.quarter], '2026-12-21');
check('⑥ 后端日期下标注依然落得上去',
  forecastMarks(HORIZON_FIXTURE, withBackend).some((m) => m.date === '2026-12-21'), true);

/* ⚠️ 口径防伪：后端**实际**返回的是 due_source='weekday'（base_date 在近未来时
 * 那一刻的交易日历还不存在，谁也拿不到）。绝不能因为"日期是后端给的"就写成
 * "已按真实交易日历跳过节假日" —— 那是给数据贴一个比它本身更强的保证，
 * 正是本次排查要消灭的那类缺陷。 */
const backendWeekday = buildForecastPlan(HORIZON_FIXTURE, BASE_DATE, LAST_NAV,
  { ...EXPECT_DUE, quarter: '2026-12-21' }, 'weekday');
check('⑥ 后端只排周末时 dueOrigin 仍为 backend', backendWeekday.dueOrigin, 'backend');
check('⑥ 但质量必须照实标成 weekday（不得升级成 index）',
  backendWeekday.dueCalendar, 'weekday');
check('⑥ 后端没标口径时按最保守的 weekday 处理',
  buildForecastPlan(HORIZON_FIXTURE, BASE_DATE, LAST_NAV, EXPECT_DUE, null).dueCalendar, 'weekday');

console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
