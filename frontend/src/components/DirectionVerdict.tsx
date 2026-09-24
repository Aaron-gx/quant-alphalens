import React from 'react';
import { ArrowDownRight, ArrowUpRight, Minus, Scale, TriangleAlert } from 'lucide-react';
import {
  NO_EDGE_TOP,
  allVerdicts,
  overallVerdict,
  type DirectionKind,
  type HorizonLike,
  type HorizonVerdict,
} from '../lib/forecast';

/**
 * AI 方向裁决条。
 *
 * 存在理由：原先"预测方向"只由 K 线图上那条紫色虚线表达，而该线的纵坐标是
 * `(P涨 − P跌) × 半幅振幅`。在概率接近均衡时这个值必然接近 0（实测四个周期
 * 全部落在现价的 ±0.6% 内），折算到 y 轴跨度 2.4 的图上不足 3 个像素——
 * 用户看到的就是一条完全水平的线，于是必然问"那到底是涨还是跌"。
 *
 * 本组件把同一批概率翻译成**明确的文字结论 + 每周期方向 + 预期偏移百分比**，
 * 并在模型确实没有方向时直说"方向不明"，而不是画一条平线让用户自己猜。
 */

const UP = '#E03E3E';      // 中国习惯：涨为红
const DOWN = '#34C759';    // 跌为绿
const FLAT = '#86868B';
const ACCENT = '#0071E3';

const TONE: Record<'up' | 'down' | 'flat', { color: string; bg: string; border: string }> = {
  up: { color: UP, bg: '#FDF0F0', border: 'rgba(224,62,62,0.18)' },
  down: { color: DOWN, bg: '#F0FAF2', border: 'rgba(52,199,89,0.18)' },
  flat: { color: FLAT, bg: '#F5F5F7', border: 'rgba(0,0,0,0.06)' },
};

const KindIcon: React.FC<{ kind: DirectionKind; className?: string; style?: React.CSSProperties }> = ({
  kind,
  className,
  style,
}) => {
  if (kind === 'up') return <ArrowUpRight className={className} style={style} />;
  if (kind === 'down') return <ArrowDownRight className={className} style={style} />;
  if (kind === 'flat') return <Minus className={className} style={style} />;
  return <Scale className={className} style={style} />;
};

const pct = (x: number) => `${Math.round(x * 100)}%`;
/** 预期偏移保留 3 位小数：本体常只有 ±0.05% 量级，少了就全被舍成 0.000 */
const signed = (x: number) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(3)}%`;

export interface DirectionVerdictProps {
  horizons?: Record<string, HorizonLike> | null;
  /** 预测基准日，用于说明"这套判断是哪天做的" */
  baseDate?: string | null;
  /** 净值数据截止日；与基准日不同时提示数据滞后 */
  dataAsOf?: string | null;
  /**
   * 各周期到期日（后端按**交易日**口径算好）。
   *
   * 为什么要在结论卡上标到期日：用户看完"偏空"必然下一个问题是"到什么时候"。
   * 不给日期，这个结论就没法执行，也没法跟对账记录核对。
   */
  dueDates?: Record<string, string> | null;
  /**
   * 到期日是谁算的：'backend' = 预测接口给的，'weekend' = 前端兜底。
   *
   * ⚠️ 它**不等于**日期质量。前端兜底当然只排了周末；但**后端给的也可能是只排周末的**
   * —— base_date 在近未来时，那时的交易日历根本还不存在。所以不要写
   * "后端给的 ⇒ 已跳过节假日"这种推断，那是在给数据贴一个比它本身更强的保证。
   * 质量看 dueCalendar。
   */
  dueOrigin?: string | null;
  /** 日历质量：'index' = 真实交易日历（含节假日）；'weekday' = 只排了周末 */
  dueCalendar?: string | null;
  /** 该记录已被检出「涨/跌概率整体对调」（旧引擎缺陷），结论不可信 */
  inverted?: boolean;
  className?: string;
}

export const DirectionVerdict: React.FC<DirectionVerdictProps> = ({
  horizons,
  baseDate,
  dataAsOf,
  dueDates,
  dueOrigin,
  dueCalendar,
  inverted = false,
  className = '',
}) => {
  const verdicts = allVerdicts(horizons);
  const overall = overallVerdict(verdicts);
  if (!verdicts.length || !overall) return null;

  const tone = TONE[overall.tone];
  const maxAbsMid = Math.max(...verdicts.map((v) => Math.abs(v.midPct)), 0.0005);

  // 基准日与净值截止日不一致 ⇒ 用户会以为"数据丢了"，如实说明
  const base = baseDate ? String(baseDate).slice(0, 10) : '';
  const asOf = dataAsOf ? String(dataAsOf).slice(0, 10) : '';
  const lagged = !!(base && asOf && base > asOf);

  /**
   * 数据异常时**不展示结论**，只给一句产品语言 + 下一步动作。
   *
   * 这里刻意不写"涨跌概率被对调""标签顺序缺陷""重启后端"这类内部细节：
   * 对用户而言那是实现细节，说清楚"这条不可用、请重新生成"就够了；
   * 更关键的是——**已知方向是错的数，就不能继续摆在界面上**，
   * 否则用户会拿一个反转的结论去下单，比"什么都不显示"危险得多。
   */
  if (inverted) {
    return (
      <div className={`apple-card p-5 ${className}`} data-testid="direction-verdict">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-[#86868B] uppercase tracking-wider flex items-center gap-1.5">
            <Scale className="w-3.5 h-3.5" style={{ color: ACCENT }} />
            方向裁决
          </span>
          {base && (
            <span className="text-[11px] text-[#A1A1A6] font-mono">
              预测基准日 {base}
              {lagged ? ` · 数据截至 ${asOf}` : ''}
            </span>
          )}
        </div>
        <div
          className="rounded-xl p-4 flex items-start gap-3"
          style={{ background: '#FFF8E6', border: '1px solid rgba(178,80,0,0.20)' }}
        >
          <TriangleAlert className="w-4 h-4 mt-0.5 shrink-0" style={{ color: '#B25000' }} />
          <div className="min-w-0">
            <div className="text-sm font-semibold" style={{ color: '#8A4B00' }}>
              这次研判的数据异常，方向结论已隐藏
            </div>
            <p className="text-xs text-[#6E6E73] leading-relaxed mt-1.5">
              该条研判的方向与概率不可靠，为避免误导不在此展示。
              请点击右上角「生成时序研判」重新生成一次即可。
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`apple-card p-5 ${className}`} data-testid="direction-verdict">
      {/* 标题 */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-[#86868B] uppercase tracking-wider flex items-center gap-1.5">
          <Scale className="w-3.5 h-3.5" style={{ color: ACCENT }} />
          方向裁决
        </span>
        <div className="flex items-center gap-2">
          <span
            className="text-[10px] font-medium px-2 py-0.5 rounded-full border border-black/[0.05]"
            style={{ background: '#F5F5F7', color: '#86868B' }}
            title="本结论由固定规则从模型输出的三分类概率推导，措辞固定，不是大模型生成的文字"
          >
            规则判定
          </span>
          {base && (
            <span className="text-[11px] text-[#A1A1A6] font-mono">
              预测基准日 {base}
              {lagged ? ` · 数据截至 ${asOf}` : ''}
            </span>
          )}
        </div>
      </div>

      {/* 结论横幅 */}
      <div
        className="rounded-xl p-4 flex items-start gap-3"
        style={{ background: tone.bg, border: `1px solid ${tone.border}` }}
      >
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: 'rgba(255,255,255,0.85)' }}
        >
          <KindIcon kind={overall.kind} className="w-5 h-5" style={{ color: tone.color }} />
        </div>
        <div className="min-w-0">
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className="text-xl font-bold tracking-tight" style={{ color: tone.color }}>
              {overall.headline}
            </span>
            {/* 事实计数：周期名不在这里出现，避免与下方明细表重复 */}
            <span className="text-[11px] text-[#86868B] tabular-nums">
              {overall.counts.total} 个周期中 看涨 {overall.counts.up} · 看跌 {overall.counts.down}
              {' · '}震荡 {overall.counts.flat} · 不明 {overall.counts.unclear}
            </span>
          </div>
          <p className="text-xs text-[#6E6E73] leading-relaxed mt-1.5">{overall.detail}</p>
        </div>
      </div>

      {/* 分周期明细 */}
      <div className="mt-4 space-y-1.5">
        <div className="grid grid-cols-12 gap-2 px-2 text-[10px] font-medium text-[#A1A1A6] uppercase tracking-wider">
          <div className="col-span-3">周期</div>
          <div className="col-span-2">方向</div>
          <div className="col-span-4 text-center">涨 / 平 / 跌</div>
          <div className="col-span-3 text-right">预期偏移</div>
        </div>

        {verdicts.map((v) => (
          <Row key={v.key} v={v} maxAbsMid={maxAbsMid} due={dueDates?.[v.key]} />
        ))}
      </div>

      {/* 判词依据 —— 每次都把规则写出来，避免"这是怎么算的"变成黑箱 */}
      <div className="mt-3.5 pt-3 border-t border-black/[0.05] space-y-1.5">
        <p className="text-[11px] text-[#86868B] leading-relaxed">
          <span className="font-medium text-[#6E6E73]">判定规则：</span>
          最大类别概率低于 <span className="font-mono">{Math.round(NO_EDGE_TOP * 100)}%</span> 即判「方向不明」；
          平盘为最大类别判「震荡」；否则按涨跌概率大小判看涨/看跌。该门槛与后端预测校验器一致。
        </p>
        <p className="text-[11px] text-[#86868B] leading-relaxed">
          <span className="font-medium text-[#6E6E73]">这段字是谁写的：</span>
          上方结论由上述规则从模型输出的三分类概率推导，句式固定、逐条可复算，
          不是大模型生成的文字；本页「推理链与深度研报」里的内容才是模型生成。
        </p>
        <p className="text-[11px] text-[#86868B] leading-relaxed">
          <span className="font-medium text-[#6E6E73]">为什么图上那条紫线几乎是平的：</span>
          「AI 预测中枢」= (涨概率 − 跌概率) × 半幅振幅。概率接近均衡时它必然趋近 0，
          平线本身就是"没有方向"的正确表达，而不是预测失灵。
        </p>
        <p className="text-[11px] text-[#A1A1A6] leading-relaxed">
          概率上的相对优势不等于价格幅度，以上均不构成收益承诺。
        </p>
        {!(dueOrigin === 'backend' && dueCalendar === 'index') && (
          <p className="text-[11px] text-[#B25000] leading-relaxed flex items-start gap-1.5">
            <TriangleAlert className="w-3.5 h-3.5 mt-px shrink-0" />
            <span>
              到期日按「起算日之后第 N 个交易日、仅跳过周末」推算
              {dueOrigin === 'weekend' ? '（未接后端交易日历）' : ''}。
              未来的法定节假日尚未可知，实际交割日可能比上表早 1~几天；
              这是数据源的限制，不是本页故障。
            </span>
          </p>
        )}
        {lagged && (
          <p className="text-[11px] text-[#B25000] leading-relaxed flex items-start gap-1.5">
            <TriangleAlert className="w-3.5 h-3.5 mt-px shrink-0" />
            <span>
              净值数据最新为 <span className="font-mono">{asOf}</span>，落后预测基准日 <span className="font-mono">{base}</span>。
              场外基金（尤其 QDII）净值由基金公司延后公布，属数据源延迟而非本页故障。
            </span>
          </p>
        )}
      </div>
    </div>
  );
};

const Row: React.FC<{ v: HorizonVerdict; maxAbsMid: number; due?: string }> = ({ v, maxAbsMid, due }) => {
  const t = TONE[v.kind === 'up' ? 'up' : v.kind === 'down' ? 'down' : 'flat'];
  // 偏移值的颜色跟着「方向判定」走，而不是跟着符号走。
  // 判「方向不明」的周期里 ±0.04% 全在噪声内，染红/染绿会和旁边灰色的「方向不明」徽标自相矛盾；
  // 那正是用户抱怨的"看不懂"，不能在新组件里再造一次。
  const midTone = v.kind === 'up' ? UP : v.kind === 'down' ? DOWN : FLAT;
  const barW = Math.min(100, (Math.abs(v.midPct) / maxAbsMid) * 100);

  return (
    <div className="grid grid-cols-12 gap-2 items-center px-2 py-2 rounded-lg hover:bg-[#F5F5F7] transition-colors">
      <div className="col-span-3">
        <div className="text-xs font-semibold text-[#1D1D1F] flex items-baseline gap-1.5">
          <span>{v.label}</span>
          {/* 到期日：结论必须能落到一个具体日期上才可执行、可核对。
              取不到就不显示，不拿交易日步数去"估"一个日期出来。 */}
          {due && (
            <span className="text-[10px] font-mono text-[#86868B] font-normal">
              到期 {due.slice(5)}
            </span>
          )}
        </div>
        <div className="text-[10px] text-[#A1A1A6] font-mono">
          区分度 {(v.separation * 100).toFixed(1)}pp
        </div>
      </div>

      <div className="col-span-2">
        <span
          className="inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-md"
          style={{ color: t.color, background: t.bg }}
        >
          <KindIcon kind={v.kind} className="w-3 h-3" />
          {v.directionText}
        </span>
      </div>

      <div className="col-span-4">
        <div className="flex items-center gap-1.5 text-[11px] font-mono font-semibold tabular-nums justify-center">
          <span style={{ color: UP }}>涨 {pct(v.up)}</span>
          <span className="text-[#E5E5EA]">·</span>
          <span className="text-[#86868B]">平 {pct(v.flat)}</span>
          <span className="text-[#E5E5EA]">·</span>
          <span style={{ color: DOWN }}>跌 {pct(v.down)}</span>
        </div>
      </div>

      <div className="col-span-3">
        <div className="flex items-center justify-end gap-2">
          <div className="flex-1 h-1.5 bg-[#EFEFF4] rounded-full overflow-hidden max-w-[70px]">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${barW}%`, background: midTone }}
            />
          </div>
          <span
            className="text-[11px] font-mono font-bold tabular-nums w-[64px] text-right"
            style={{ color: midTone }}
          >
            {signed(v.midPct)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default DirectionVerdict;
