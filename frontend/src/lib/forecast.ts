/**
 * 预测曲线与方向裁决的**唯一口径来源**。
 *
 * 为什么必须单独成模块
 * --------------------
 * 「周期 → 交易日步数」「振幅字符串解析」「预期中枢的算法」原先只在
 * `Analysis.tsx` 的 `buildForecast()` 里各写一份。而方向裁决条需要用到
 * **同一套**判定，如果各写一份，必然出现"图上画一个方向、文字说另一个方向"
 * 的漂移——本项目已经吃过同源的亏（config 里同一个阈值三处各读一遍）。
 *
 * 方向判定阈值与后端 `core/predict/schema.py` **同源**：
 * 后端在 `max(up,flat,down) < 0.40` 时会打日志提示"方向概率区分度过低，
 * 近似无判断"。前端沿用同一判据，保证"前端说无方向"与"后端认为无判断"
 * 指的是同一件事，而不是两个各自拍脑袋的数字。
 */

/** 周期 → 前向交易日步数（与后端 labels.HORIZON_STEPS 一致）。 */
export const HORIZON_TDAYS: Array<[string, number]> = [
  ['next_day', 1],
  ['one_week', 5],
  ['one_month', 20],
  ['quarter', 60],
];

export const HORIZON_LABELS: Record<string, string> = {
  next_day: '次日 T+1',
  one_week: '一周 5日',
  one_month: '一月 20日',
  quarter: '一季 60日',
};

/** 与后端 schema 同一门槛：最大类别概率低于此值 ⇒ 近似无判断。 */
export const NO_EDGE_TOP = 0.4;

/** 无样本时的默认半幅振幅（与 buildForecast 原实现保持一致）。 */
export const DEFAULT_BAND = 0.02;

export interface HorizonLike {
  up?: number | null;
  flat?: number | null;
  down?: number | null;
  range_pct?: string;
  range?: string;
}

/** 解析 "±4.5%" / "4.5%" 这类字符串为 0.045。 */
export function parseRangePct(s?: string): number {
  if (!s) return 0;
  const m = String(s).match(/([\d.]+)\s*%/);
  return m ? parseFloat(m[1]) / 100 : 0;
}

/**
 * 从最后一天往后推 n 个**交易日**（跳过周末）的日期串。
 *
 * 原先是 `Analysis.tsx` 里的私有函数。标注要落在与预测锚点**完全同一天**上，
 * 如果图表标注自己再算一遍日期，只要有一处差一天，标签就会钉错位置——
 * 与"图上画一个方向、文字说另一个方向"是同一类错误。所以收进这里，只留一份。
 */
export function futureTradeDates(lastDate: string, n: number): string[] {
  const out: string[] = [];
  // lastDate 可能是 "2026-09-22" 或 "2026-09-22T00:00:00"，先取日期部分再按本地时间构造
  const m = String(lastDate).slice(0, 10).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return out;
  const d = new Date(+m[1], +m[2] - 1, +m[3]);
  if (isNaN(d.getTime())) return out;
  const fmt = (x: Date) =>
    `${x.getFullYear()}-${String(x.getMonth() + 1).padStart(2, '0')}-${String(x.getDate()).padStart(2, '0')}`;
  while (out.length < n) {
    d.setDate(d.getDate() + 1);
    const w = d.getDay();
    if (w === 0 || w === 6) continue;
    out.push(fmt(d));
  }
  return out;
}

/** 方向短词（图上标注用，比 directionText 更短以免互相压字）。 */
export const DIRECTION_SHORT: Record<DirectionKind, string> = {
  up: '看涨', down: '看跌', flat: '震荡', unclear: '方向不明',
};

/** 预测起算基准：模型自报的 base_date，或（仅当 base_date 缺失时）行情最后一日。 */
export interface ForecastAnchor {
  date: string;
  source: 'base_date' | 'last_price';
}

/**
 * 解析预测的起算基准日。
 *
 * ⚠️ 这里曾经错得不显眼：起算日取的是**行情最后一日**，而不是研判自己的
 * `base_date`。两者在多数日子只差 0~2 天，看着"差不多"，语义却完全不同：
 *   - `base_date` 是模型的基准日 —— 后端的到期日就是从它往后数交易日；
 *   - 行情最后一日是数据的末端 —— 场外基金净值 T+1~T+2 公布，天然滞后。
 * 结果：一条 `base_date = 09-24` 的研判，图上「次日 T+1」的标签钉在 **09-23**，
 * 一个**已经过去**的日期。用户读到的是"模型在预测昨天"，后面所有解读都无从谈起。
 */
export function resolveForecastAnchor(
  baseDate?: string | null,
  lastPriceDate?: string | null,
): ForecastAnchor | null {
  const norm = (v?: string | null) => {
    const m = String(v ?? '').slice(0, 10).match(/^(\d{4})-(\d{2})-(\d{2})$/);
    return m ? `${m[1]}-${m[2]}-${m[3]}` : '';
  };
  const bd = norm(baseDate);
  if (bd) return { date: bd, source: 'base_date' };
  const lp = norm(lastPriceDate);
  if (lp) return { date: lp, source: 'last_price' };
  return null;
}

/** 预测区的日期网格与各周期落点 —— 曲线、标注、明细表共用同一份，杜绝各算一遍。 */
export interface ForecastPlan {
  anchor: ForecastAnchor;
  /** 预测区 x 轴（锚点之后逐交易日，升序） */
  dates: string[];
  /** horizon key → 该到期日在 dates 中的下标（0 基） */
  steps: Record<string, number>;
  /** horizon key → 到期日（恒等于 dates[steps[key]]） */
  dueDates: Record<string, string>;
  /**
   * 到期日的**计算方**：'backend' = 预测接口给的；'weekend' = 前端兜底算的。
   *
   * ⚠️ 这**不是**日期质量。它只回答"谁算的"。质量看下面的 dueCalendar。
   */
  dueOrigin: 'backend' | 'weekend';
  /**
   * 日历**质量**（后端 due_source 原样透传）：
   *   'index'   —— 真实交易日历（沪深300 日 K），已跳过法定节假日，严格口径；
   *   'weekday' —— 只排除了周末。bp 在近未来时**必然**如此：那时的交易日历
   *                还不存在，谁也拿不到，只能排周末 —— 所以这不是"降级得可耻"，
   *                而是必须照实说出来；
   *   'none'    —— 后端没给这个字段（旧版本后端），口径未知。
   *
   * 为什么必须把"谁算的"和"什么质量"分成两个字段：
   * 合成一个 `dueSource: 'backend' | 'weekend'` 时，代码会写出
   * "后端给的 ⇒ 已跳过节假日"这种推断 —— 而实测后端返回的正是 weekday。
   * 那就会在界面上贴一个**比数据本身更强的保证**，属于同类缺陷里最坏的一种。
   */
  dueCalendar: 'index' | 'weekday' | 'none';
  /** 起算日比行情最后一日晚几个自然日（披露数据缺口用，0 = 不晚） */
  gapDays: number;
}

/**
 * 把「起算基准 + 各周期到期日 + 预测区日期网格」一次算清。
 *
 * 为什么必须合成一个 plan
 * ----------------------
 * KlineChart 用 `fcByDate` 按**日期串**取预测值来定位 markPoint，
 * 取不到就 `return null` —— 也就是说：**标注日期只要不在预测曲线里，标签就静默消失**。
 * 所以"曲线用的日期"和"标注用的日期"绝不能各算一遍，必须是同一份数据。
 */
export function buildForecastPlan(
  horizons: Record<string, HorizonLike> | undefined | null,
  baseDate?: string | null,
  lastPriceDate?: string | null,
  dueDates?: Record<string, string> | null,
  dueSource?: string | null,
): ForecastPlan | null {
  const anchor = resolveForecastAnchor(baseDate, lastPriceDate);
  if (!anchor) return null;
  const vs = allVerdicts(horizons);
  if (!vs.length) return null;

  const iso = (v?: string | null) => {
    const s = String(v ?? '').slice(0, 10);
    return /^\d{4}-\d{2}-\d{2}$/.test(s) ? s : '';
  };

  // 后端给的到期日优先：它按交易日历算，前端算不出节假日。
  const backend: Record<string, string> = {};
  for (const [k, v] of Object.entries(dueDates || {})) {
    const s = iso(v);
    if (s) backend[k] = s;
  }
  const useBackend = vs.some((v) => backend[v.key]);

  // 前端兜底：从锚点起跳周末取第 N 个交易日。**这是降级档**，不排节假日。
  const maxStep = Math.max(...vs.map((v) => v.tradingDays));
  const weekendGrid = futureTradeDates(anchor.date, maxStep);

  const dueDatesOut: Record<string, string> = {};
  for (const v of vs) {
    dueDatesOut[v.key] = (useBackend && backend[v.key]) || weekendGrid[v.tradingDays - 1] || '';
  }

  // 网格 = 兜底网格 ∪ 后端到期日。
  // 节假日会把后端到期日推到兜底网格之外（也可能提前），不并进来，
  // 那条标注就会因为"日期不在 x 轴上"而静默消失。
  const grid = new Set<string>(weekendGrid);
  for (const d of Object.values(dueDatesOut)) if (d) grid.add(d);
  const dates = Array.from(grid).sort();

  const steps: Record<string, number> = {};
  for (const v of vs) {
    const idx = dates.indexOf(dueDatesOut[v.key]);
    if (idx >= 0) steps[v.key] = idx;
  }
  if (!Object.keys(steps).length) return null;

  const lp = iso(lastPriceDate);
  const gapDays =
    lp && anchor.date > lp
      ? Math.max(0, Math.round((Date.parse(anchor.date) - Date.parse(lp)) / 86400000))
      : 0;

  // 日历质量：后端明确说了就照它说，没说或没接后端就按实际发生的算。
  const rawCal = String(dueSource ?? '');
  const dueCalendar: ForecastPlan['dueCalendar'] =
    useBackend && (rawCal === 'index' || rawCal === 'weekday')
      ? rawCal
      : useBackend ? 'weekday'   // 后端给了日期但没标口径 → 只能按最保守的算
      : 'weekday';               // 前端兜底：确实只排了周末

  return {
    anchor,
    dates,
    steps,
    dueDates: dueDatesOut,
    dueOrigin: useBackend ? 'backend' : 'weekend',
    dueCalendar,
    gapDays,
  };
}

/** 图上标注点：把裁决结果钉到 K 线预测区的对应日期上。 */
export interface ForecastMark {
  date: string;
  text: string;
  tone: 'up' | 'down' | 'flat';
  /** 该周期对应的交易日步数，供图表决定标签上下轮换、避免相邻标签重叠 */
  tradingDays: number;
}

/**
 * 标在 K 线上的文字 = 「周期 + 方向」，**不带预期偏移数字**。
 *
 * 为什么不带数字：预测区只占整图右侧约 1/3 宽（120 根历史 + 60 个未来交易日），
 * 四个标签都带上 "+0.60%" 后总宽 ~444px，而预测区的横跨度只有 ~246px ——
 * **两行也放不下**，实测「一月 看涨 +0.18%」直接压住了「次日」的尾字。
 * 而那个数字在价格刻度上本体就不足 1%（见下方说明），印在图上既挤又不增加信息量；
 * 它的准确值在「方向裁决」表的「预期偏移」列（小数点后 3 位）和情景卡里都有。
 * 图上只保留**方向**这个用户真正读不出来的信息。
 */
export function forecastMarks(
  horizons: Record<string, HorizonLike> | undefined | null,
  plan: ForecastPlan | null,
): ForecastMark[] {
  const vs = allVerdicts(horizons);
  if (!vs.length || !plan) return [];
  const out: ForecastMark[] = [];
  for (const v of vs) {
    const idx = plan.steps[v.key];
    if (idx == null) continue;
    const date = plan.dates[idx];
    if (!date) continue;
    const tone: ForecastMark['tone'] =
      v.kind === 'up' ? 'up' : v.kind === 'down' ? 'down' : 'flat';
    const short = v.label.replace(/\s.*/, '');
    out.push({
      date, tone, tradingDays: v.tradingDays,
      text: `${short} ${DIRECTION_SHORT[v.kind]}`,
    });
  }
  return out;
}

export type DirectionKind = 'up' | 'down' | 'flat' | 'unclear';

export interface HorizonVerdict {
  key: string;
  label: string;
  tradingDays: number;
  /** 归一化后的三方向概率 */
  up: number;
  flat: number;
  down: number;
  kind: DirectionKind;
  /** 展示用方向名 */
  directionText: string;
  /** 涨跌概率差 up - down，∈ [-1,1]，只代表方向信心 */
  edge: number;
  /** 最大与最小概率之差，代表整体区分度 */
  separation: number;
  /** 半幅振幅（= range_pct / 2） */
  band: number;
  /** 预期偏移：与 K 线图上"AI 预测中枢"完全同一公式 */
  midPct: number;
}

/**
 * 判定单个周期的方向。
 *
 * 规则（刻意保守，宁说"不知道"也不错报方向）：
 *   1. `max(up,flat,down) < 0.40` → **方向不明**（与后端 schema 同判据）
 *   2. 平盘为最大类别        → **震荡**（平盘为主）
 *   3. 涨跌概率几乎相等      → **方向不明**
 *   4. 否则按 up/down 大小   → 看涨 / 看跌
 */
export function verdictOf(key: string, h: HorizonLike | undefined): HorizonVerdict | null {
  if (!h || typeof h.up !== 'number') return null;
  const up = Number(h.up) || 0;
  const flat = Number(h.flat) || 0;
  const down = Number(h.down) || 0;
  const total = up + flat + down;
  if (total <= 0) return null;

  const nu = up / total;
  const nf = flat / total;
  const nd = down / total;
  const top = Math.max(nu, nf, nd);
  const bottom = Math.min(nu, nf, nd);

  let kind: DirectionKind;
  if (top < NO_EDGE_TOP) kind = 'unclear';
  else if (nf >= top - 1e-9) kind = 'flat';
  else if (Math.abs(nu - nd) < 1e-9) kind = 'unclear';
  else kind = nu > nd ? 'up' : 'down';

  const directionText =
    kind === 'up' ? '看涨'
      : kind === 'down' ? '看跌'
        : kind === 'flat' ? '震荡（平盘为主）'
          : '方向不明';

  const band = parseRangePct(h.range_pct || h.range) / 2 || DEFAULT_BAND;

  return {
    key,
    label: HORIZON_LABELS[key] || key,
    tradingDays: HORIZON_TDAYS.find(([k]) => k === key)?.[1] ?? 1,
    up: nu,
    flat: nf,
    down: nd,
    kind,
    directionText,
    edge: nu - nd,
    separation: top - bottom,
    band,
    // 与 K 线「AI 预测中枢」同一公式：概率差 × 半幅振幅。
    // 概率趋近均衡时该值必然趋近 0 —— 这正是图上紫线看起来"永远是平的"的原因。
    midPct: (nu - nd) * band,
  };
}

export function allVerdicts(horizons: Record<string, HorizonLike> | undefined | null): HorizonVerdict[] {
  if (!horizons) return [];
  return HORIZON_TDAYS
    .map(([k]) => verdictOf(k, horizons[k]))
    .filter((v): v is HorizonVerdict => v !== null);
}

/** 各方向的周期计数，界面据此自行组织措辞（不再套模板句）。 */
export interface VerdictCounts {
  up: number;
  down: number;
  flat: number;
  unclear: number;
  total: number;
}

export interface OverallVerdict {
  kind: DirectionKind;
  headline: string;
  tone: 'up' | 'down' | 'flat';
  counts: VerdictCounts;
  /** 一句结论；只陈述计数事实，不复述周期名、不塞固定免责套话 */
  detail: string;
}

/**
 * 综合四个周期给一句人话结论。
 *
 * 刻意**不做加权平均**：四个周期的概率差都是弱信号，平均只会把"没有方向"
 * 洗成"微弱方向"，反而误导。这里只做计数，并在分歧时直说分歧。
 *
 * 关于措辞（2026-09-24 返工）
 * ------------------------
 * 旧实现返回 `drivers`（周期名数组）并在 `detail` 里把同一批周期名**再打印一遍**，
 * 于是界面渲染出成这样：
 *     「依据：次日、一周」+「次日、一周 给出看跌；其余周期方向不明或震荡。
 *       注意这是概率上的相对优势，不是幅度承诺。」
 * 同一句话里同一个列表出现两次、外加一句每档都一字不差的免责尾句，
 * 一眼就能看出是"模板填空"，用户直接指出这是露馅文字。
 *
 * 现在只返回 `counts` 事实 + 一句自然结论；周期名由界面在明细表里展示**一次**，
 * 免责说明放进组件底部固定的规则区（只有一条，不再每个分支各抄一遍）。
 */
export function overallVerdict(verdicts: HorizonVerdict[]): OverallVerdict | null {
  if (!verdicts.length) return null;

  const counts: VerdictCounts = {
    up: verdicts.filter((v) => v.kind === 'up').length,
    down: verdicts.filter((v) => v.kind === 'down').length,
    flat: verdicts.filter((v) => v.kind === 'flat').length,
    unclear: verdicts.filter((v) => v.kind === 'unclear').length,
    total: verdicts.length,
  };
  const { up: nUp, down: nDown, flat: nFlat, unclear: nUnclear, total } = counts;
  const threshold = `${Math.round(NO_EDGE_TOP * 100)}%`;

  if (!nUp && !nDown) {
    return {
      kind: 'unclear',
      headline: nFlat ? '震荡为主' : '多空均衡',
      tone: 'flat',
      counts,
      detail: nFlat
        ? `${total} 个周期里 ${nFlat} 个以平盘为最大类别，其余区分度不足（最大类别概率低于 ${threshold}），`
          + '没有形成可使用方向。'
        : `${total} 个周期的最大类别概率都不足 ${threshold}，模型这次没有给出方向。`,
    };
  }

  if (nUp > nDown) {
    return {
      kind: 'up',
      headline: '偏多',
      tone: 'up',
      counts,
      detail: `${total} 个周期中 ${nUp} 个看涨、${nDown} 个看跌`
        + (nUnclear || nFlat ? `，另有 ${nFlat + nUnclear} 个方向不明` : '')
        + '。多方占优的幅度很有限。',
    };
  }
  if (nDown > nUp) {
    return {
      kind: 'down',
      headline: '偏空',
      tone: 'down',
      counts,
      detail: `${total} 个周期中 ${nDown} 个看跌、${nUp} 个看涨`
        + (nUnclear || nFlat ? `，另有 ${nFlat + nUnclear} 个方向不明` : '')
        + '。空方占优的幅度很有限。',
    };
  }
  return {
    kind: 'unclear',
    headline: '多空分歧',
    tone: 'flat',
    counts,
    detail: `看涨与看跌各 ${nUp} 个周期，短周期与长周期结论相反，属于信号冲突。`,
  };
}

/**
 * 识别「涨跌概率被整体对调」的历史记录 —— 旧引擎标签顺序缺陷的指纹。
 *
 * 判据是自洽的，不需要任何外部对照：
 * `calibration.method === 'identity'` 的语义就是**恒等映射：概率原样返回**，
 * 所以对每一条 identity 记录必然有 `probs_before === probs_after`。
 * 若两者出现 up/down 对调，说明产出该记录的后端进程把特征向量按 [涨,平,跌] 排、
 * 标签索引却按 [跌,平,涨] 读（详见后端 core/predict/labels.py 顶部记录的那次事故）。
 * 这种记录 `method` 仍是 identity、Brier/命中率也全都"看着正常"，
 * 只有前端把具体数字摆到一起时才会暴露：**报告文字说跌，数字却显示涨**。
 *
 * 检出后绝不能静默展示——方向被对调的结论比"没有结论"更危险。
 */
export function detectInvertedProbs(inputSnapshot: unknown): boolean {
  const calib = (inputSnapshot as any)?.calibration;
  if (!calib || typeof calib !== 'object') return false;
  const infos: any[] = Array.isArray(calib) ? calib : Object.values(calib);
  const near = (a: unknown, b: unknown) => Math.abs((Number(a) || 0) - (Number(b) || 0)) < 1e-6;
  return infos.some((info) => {
    if (!info || info.method !== 'identity') return false;
    const b = info.probs_before;
    const a = info.probs_after;
    if (!b || !a) return false;
    return !near(b.up, a.up) && near(b.up, a.down) && near(b.down, a.up);
  });
}
