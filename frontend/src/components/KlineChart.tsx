import React, { useMemo } from 'react';
import { EChart } from './EChart';
import type { ForecastMark } from '../lib/forecast';

interface KlineItem {
  date: string;
  open?: number;
  close?: number;
  high?: number;
  low?: number;
  volume?: number;
  nav?: number;
}

export interface ForecastPoint {
  date: string;
  mid: number | null;
  low: number | null;
  high: number | null;
}

interface KlineChartProps {
  data: KlineItem[];
  title?: string;
  isNav?: boolean;
  /**
   * 已移除的 `basePrice?: number`（默认值 0.533）。
   *
   * 该参数在组件内部**从未被引用**（只有声明和默认值两处），是个死参数；
   * 但那个 `= 0.533` 的默认值很危险 —— 任何未来真的用到它的改动，都会在
   * 调用方忘记传值时，静默拿一个编造的价格（0.533）当作基准价画图。
   * 与其留个带假数据的坑，不如删掉；组件内部需要基准价的地方已全部
   * 改为从传入的 data 序列里取真实末值。
   */
  /** AI 预测点序列（未来日期），传入后在图表右端叠加中枢虚线 + 区间阴影 */
  forecast?: ForecastPoint[] | null;
  /**
   * 方向标注（钉在预测区各周期对应日期上）。
   *
   * 为什么必须有：中枢线的纵坐标是 (涨概率−跌概率)×半幅振幅，概率接近均衡时
   * 折算到图上不足 3 像素 —— 光靠那条线，用户永远读不出涨跌。
   * 方向只能以文字标注的形式落到图上，这是唯一诚实的表达。
   */
  forecastMarks?: ForecastMark[] | null;
  /** 图右上角角标，如 "含 AI 预测" */
  badge?: string;
  /** 图表高度 px，默认 340 */
  height?: number;
}

// ─── 已删除的 generateFallbackCandles (2026-09-24) ────────────────────────────
// 这里原先有一个用 `Math.random()` 生成 60 根"平滑逼真的兜底 K 线"的函数：
// 用正弦波 + 随机扰动造出开高低收，用随机数造成交量，日期按今天往回推。
// 触发条件写在 displayData 里 —— `data 为空` 就用它顶上。
//
// 后果：接口失败 / 新标的还没拉到数据时，页面上会出现一张**形态完整、有涨有跌、
// 带成交量的 K 线图**，唯一的问题是所有数字都是随机数，而且用户无从分辨。
// 拿它去对照预测曲线、甚至做决策，后果比"图表空白"严重得多。
// 现在改为：没有数据就渲染明确的无数据占位，绝不生成任何形态。

// 与 MA20 的 #5856D6 区分开：预测中枢用 Apple 品紫
const FORECAST_COLOR = '#AF52DE';
const FORECAST_BAND = 'rgba(175, 82, 222, 0.10)';
// 方向标注配色：涨红跌绿（A 股习惯），方向不明用中性灰——
// 绝不给"方向不明"染红或染绿，否则等于用颜色暗示一个并不存在的方向。
const MARK_TONE: Record<'up' | 'down' | 'flat', string> = {
  up: '#E03E3E', down: '#34C759', flat: '#86868B',
};

export const KlineChart: React.FC<KlineChartProps> = ({
  data = [],
  title = '日K线走势与量能',
  isNav = false,
  forecast = null,
  forecastMarks = null,
  badge,
  height = 340,
}) => {
  const displayData = useMemo(() => data || [], [data]);
  const hasData = displayData.length > 0;

  const chartOption = useMemo(() => {
    // 后端可能返回 "2026-09-22T00:00:00"，统一截断成日期，避免 x 轴标签带时间
    const histDates = displayData.map((d) => String(d.date).slice(0, 10));
    const hasForecast = !!(forecast && forecast.length > 0);
    const fcDates = hasForecast ? forecast!.map((f) => f.date) : [];
    const dates = histDates.concat(fcDates);
    const histLen = histDates.length;
    const histNulls = fcDates.map(() => null);
    // 预测序列在历史区为 null，与最后一日收盘价相接形成连续曲线
    const fcByDate = new Map((forecast || []).map((f) => [f.date, f]));
    const lastClose = displayData[histLen - 1]?.close ?? displayData[histLen - 1]?.nav ?? 0;
    // junction 为衔接点（最后历史日）：mid/low 取收盘价连成线，range 必须取 0，
    // 否则堆叠区在衔接点会算成 close+close 把 Y 轴顶高一倍
    const alignFc = (pick: (f: ForecastPoint) => number | null, junction: number) =>
      dates.map((dt, i) => {
        if (i < histLen - 1) return null;
        if (i === histLen - 1) return junction;
        const f = fcByDate.get(dt);
        return f ? pick(f) : null;
      });
    const fcMid = hasForecast ? alignFc((f) => f.mid, lastClose) : null;
    const fcLow = hasForecast ? alignFc((f) => f.low, lastClose) : null;
    const fcRange = hasForecast
      ? alignFc((f) => (f.high != null && f.low != null ? +(f.high - f.low).toFixed(4) : null), 0)
      : null;

    // 方向标注点：y 取该日期预测中枢的值。
    // 用 markPoint 而不是 markLine/graphic —— markPoint 的 coord 会跟着
    // 数据缩放/缩放条一起走，缩放图时标签不会漂到错误位置。
    //
    // ⚠️ 排布为什么不能只按 index 奇偶交替
    // ------------------------------------
    // T+1 与 T+5 只隔 4 个交易日，奇偶交替碰巧能错开；但 T+20 是偶数位、又回到同一行，
    // 而它到 T+5 的横向距离放不下两个标签，于是「一月」压住了「次日」的尾字
    // （实测像素：次日右缘 591.4 > 一月左缘 559.8，重叠 32px）。
    // 所以改成真正的**贪心避让**：按 x 从左到右，逐个找第一个放得下的行。
    // 行用「上下 + 更远的上一行」三档，保证窄屏下也有解。
    //
    // 宽度按真实字宽估：CJK 10px、ASCII 6px（fontSize=10），再加 padding。
    // 坐标轴是等距 category，所以换算到**下标空间**做比较，不依赖 DOM 测量。
    // 参考绘图宽度取 800px：本项目主图在此宽度下正好排成两行，窄于此时退化成三行。
    const PLOT_REF_PX = 800;
    const LANES: Array<{ position: 'top' | 'bottom'; distance: number }> = [
      { position: 'top', distance: 8 },
      { position: 'bottom', distance: 8 },
      { position: 'top', distance: 28 },
    ];
    const textW = (s: string) =>
      [...String(s)].reduce((w, ch) => w + (/[\u2e80-\u9fff\uff00-\uffef]/.test(ch) ? 10 : 6), 0) + 10;

    const markPlacement = (() => {
      if (!hasForecast || !forecastMarks || !forecastMarks.length) return null;
      const n = Math.max(dates.length, 1);
      const laneRight = LANES.map(() => Number.NEGATIVE_INFINITY);
      const picks: number[] = [];
      const ordered = forecastMarks
        .map((m, i) => ({ i, idx: fcDates.indexOf(m.date), w: textW(m.text) }))
        .filter((o) => o.idx >= 0)
        .sort((a, b) => a.idx - b.idx);
      for (const o of ordered) {
        const half = (o.w / 2 / PLOT_REF_PX) * n;
        let lane = LANES.findIndex((_, li) => o.idx - half > laneRight[li]);
        if (lane < 0) lane = 0;   // 三档都放不下：至少不越出画布
        laneRight[lane] = o.idx + half + (4 / PLOT_REF_PX) * n;
        picks[o.i] = lane;
      }
      return picks;
    })();

    const forecastMarkPoint =
      hasForecast && forecastMarks && forecastMarks.length > 0
        ? {
            silent: true,
            data: forecastMarks
              .map((m, i) => {
                const mid = fcByDate.get(m.date)?.mid;
                if (mid == null) return null;
                const color = MARK_TONE[m.tone];
                const lane = LANES[markPlacement?.[i] ?? i % LANES.length];
                return {
                  coord: [m.date, mid],
                  value: m.text,
                  symbol: 'circle',
                  symbolSize: 5,
                  itemStyle: { color },
                  label: {
                    formatter: m.text,
                    position: lane.position,
                    distance: lane.distance,
                    color,
                    fontSize: 10,
                    fontWeight: 600 as const,
                    backgroundColor: 'rgba(255,255,255,0.92)',
                    borderColor: color,
                    borderWidth: 1,
                    borderRadius: 4,
                    padding: [2, 5] as [number, number],
                  },
                };
              })
              .filter(Boolean),
            // 三个行档都不一样高，避免第一行就在顶部的标签被裁掉
            z: 5,
          }
        : undefined;

    // 最右端的标签会被画布右边缘截断（实测「一季 看涨 +0.60%」只显示成 "+0.6…"）。
    // markPoint 的标签默认以锚点居中，而最远的锚点正好落在绘图区右边缘，
    // 于是半个标签跑到画布外。给绘图区留出右侧余量即可 —— 代价是绘图区窄一点，
    // 比"标签读不全"小得多。
    const markRightPad = forecastMarkPoint
      ? Math.max(0, ...forecastMarks!.map((m) => textW(m.text) / 2 + 8 - 20))
      : 0;

    const forecastSeries = hasForecast
      ? [
          {
            name: 'forecast_low',
            type: 'line',
            data: fcLow,
            stack: 'fc-band',
            symbol: 'none',
            lineStyle: { opacity: 0 },
            tooltip: { show: false },
            silent: true,
          },
          {
            name: 'AI预测区间',
            type: 'line',
            data: fcRange,
            stack: 'fc-band',
            symbol: 'none',
            lineStyle: { opacity: 0 },
            areaStyle: { color: FORECAST_BAND },
            tooltip: { show: false },
            silent: true,
          },
          {
            name: 'AI预测中枢',
            type: 'line',
            data: fcMid,
            smooth: true,
            showSymbol: false,
            itemStyle: { color: FORECAST_COLOR },
            lineStyle: { width: 2, type: 'dashed', color: FORECAST_COLOR },
            // 方向标注：把裁决结果钉在对应周期的日期上。
            // 相邻锚点（T+1 与 T+5 日）日期很近，标签上下轮换以避免压字。
            markPoint: forecastMarkPoint,
          },
        ]
      : [];

    const startMarkLine = hasForecast
      ? {
          symbol: 'none',
          silent: true,
          data: [{ xAxis: histDates[histLen - 1] }],
          lineStyle: { color: '#A1A1A6', type: 'dashed', width: 1 },
          label: { formatter: '预测起点', color: '#86868B', fontSize: 10, position: 'insideEndTop' },
        }
      : undefined;

    // 预测延伸区淡紫底标识：放在 invisible 的辅助 line 系列上，
    // 避免 candlestick 上 markArea 与 category 轴末端坐标解析异常导致整图渲染失败
    const forecastMarkArea = hasForecast
      ? {
          silent: true,
          itemStyle: { color: 'rgba(175, 82, 222, 0.05)' },
          data: [[{ xAxis: histDates[histLen - 1] }, { xAxis: fcDates[fcDates.length - 1] }]],
          label: {
            show: true,
            position: 'insideTop',
            color: '#AF52DE',
            fontSize: 10,
            formatter: 'AI 预测区',
          },
        }
      : undefined;

    const markAreaCarrier = hasForecast
      ? {
          name: 'fc_zone',
          type: 'line',
          data: dates.map(() => null),
          symbol: 'none',
          lineStyle: { opacity: 0 },
          tooltip: { show: false },
          silent: true,
          markArea: forecastMarkArea,
          markLine: startMarkLine,
          z: 0,
        }
      : null;

    const calculateMA = (dayCount: number, prices: number[]) => {
      const result: any[] = [];
      for (let i = 0, len = prices.length; i < len; i++) {
        if (i < dayCount - 1) {
          result.push(null);
          continue;
        }
        let sum = 0;
        for (let j = 0; j < dayCount; j++) {
          sum += prices[i - j];
        }
        result.push(Number((sum / dayCount).toFixed(3)));
      }
      return result;
    };

    if (isNav) {
      // 场外基金：画净值走势折线图 (Apple Stocks Area) + AI 预测延伸
      const navsHist = displayData.map((d) => d.close || d.nav || 0);
      const navs = (navsHist as any[]).concat(histNulls);
      const ma20 = calculateMA(20, navsHist).concat(histNulls);

      return {
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'line', lineStyle: { color: '#86868B', type: 'dashed' } },
          backgroundColor: 'rgba(255, 255, 255, 0.96)',
          borderColor: 'rgba(0, 0, 0, 0.08)',
          shadowBlur: 10,
          shadowColor: 'rgba(0,0,0,0.06)',
          textStyle: { color: '#1D1D1F', fontSize: 12 },
        },
        legend: {
          data: hasForecast ? ['单位净值', 'MA20', 'AI预测中枢'] : ['单位净值', 'MA20'],
          top: 0,
          right: 8,
          textStyle: { color: '#86868B', fontSize: 11 },
          itemWidth: 14,
        },
        grid: { left: 45, right: 20 + markRightPad, top: 25, bottom: 30 },
        xAxis: {
          type: 'category',
          data: dates,
          axisLine: { lineStyle: { color: '#E5E5EA' } },
          axisLabel: { color: '#86868B', fontSize: 11 },
        },
        yAxis: {
          type: 'value',
          scale: true,
          splitLine: { lineStyle: { color: '#F2F2F7', type: 'dashed' } },
          axisLabel: { color: '#86868B', fontSize: 11 },
        },
        series: [
          {
            name: '单位净值',
            type: 'line',
            data: navs,
            smooth: true,
            showSymbol: false,
            itemStyle: { color: '#0071E3' },
            lineStyle: { width: 2 },
            markLine: startMarkLine,
            markArea: forecastMarkArea,
            areaStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  { offset: 0, color: 'rgba(0, 113, 227, 0.20)' },
                  { offset: 1, color: 'rgba(0, 113, 227, 0.00)' },
                ],
              },
            },
          },
          {
            name: 'MA20',
            type: 'line',
            data: ma20,
            smooth: true,
            showSymbol: false,
            itemStyle: { color: '#FF9500' },
            lineStyle: { width: 1.5, type: 'dashed' },
          },
          ...forecastSeries,
        ],
      };
    }

    // 场内标的：K线 + MA5/10/20 + 成交量 + AI 预测延伸
    const ohlc = displayData.map((d) => [d.open ?? 0, d.close ?? 0, d.low ?? 0, d.high ?? 0]);
    const closePrices = displayData.map((d) => d.close ?? 0);
    const volumes = displayData.map((d) => d.volume ?? 0);

    const ma5 = calculateMA(5, closePrices).concat(histNulls);
    const ma10 = calculateMA(10, closePrices).concat(histNulls);
    const ma20 = calculateMA(20, closePrices).concat(histNulls);
    // 蜡烛图不接受 null 项（WhiskerBoxCommonMixin 会抛错），预测延伸区用 '-' 占位
    const ohlcExt = (ohlc as any[]).concat(fcDates.map(() => '-'));
    const volExt = (volumes as any[]).concat(histNulls);

    return {
      animation: false,
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross', lineStyle: { color: '#86868B', type: 'dashed' } },
        backgroundColor: 'rgba(255, 255, 255, 0.96)',
        borderColor: 'rgba(0, 0, 0, 0.08)',
        shadowBlur: 12,
        shadowColor: 'rgba(0, 0, 0, 0.06)',
        textStyle: { color: '#1D1D1F', fontSize: 11 },
        position: (pos: any, _params: any, _el: any, _rect: any, size: any) => {
          const obj: any = { top: 10 };
          obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
          return obj;
        },
      },
      legend: {
        data: hasForecast ? ['日K', 'MA5', 'MA10', 'MA20', 'AI预测中枢'] : ['日K', 'MA5', 'MA10', 'MA20'],
        top: 2,
        right: 10,
        textStyle: { color: '#86868B', fontSize: 11 },
      },
      grid: [
        { left: 45, right: 20 + markRightPad, top: 30, height: '62%' },
        { left: 45, right: 20 + markRightPad, top: '78%', height: '16%' },
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          axisLine: { lineStyle: { color: '#E5E5EA' } },
          axisLabel: { color: '#86868B', fontSize: 10 },
          min: 'dataMin',
          max: 'dataMax',
        },
        {
          type: 'category',
          gridIndex: 1,
          data: dates,
          axisLabel: { show: false },
          axisLine: { lineStyle: { color: '#E5E5EA' } },
          min: 'dataMin',
          max: 'dataMax',
        },
      ],
      yAxis: [
        {
          scale: true,
          splitLine: { lineStyle: { color: '#F2F2F7', type: 'dashed' } },
          axisLabel: { color: '#86868B', fontSize: 10 },
        },
        {
          scale: true,
          gridIndex: 1,
          splitNumber: 2,
          axisLabel: { show: false },
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { show: false },
        },
      ],
      dataZoom: [
        // 有预测延伸区时往前多展示一些历史，避免右半屏显得空
        { type: 'inside', xAxisIndex: [0, 1], start: hasForecast ? 30 : 45, end: 100 },
        {
          show: true,
          xAxisIndex: [0, 1],
          type: 'slider',
          bottom: 2,
          height: 12,
          borderColor: '#EFEFF4',
          fillerColor: 'rgba(0, 113, 227, 0.12)',
          textStyle: { color: 'transparent' },
          start: hasForecast ? 30 : 45,
          end: 100,
        },
      ],
      series: [
        {
          name: '日K',
          type: 'candlestick',
          data: ohlcExt,
          itemStyle: {
            color: '#E03E3E', // Apple Coral Red
            color0: '#34C759', // Apple Mint Green
            borderColor: '#E03E3E',
            borderColor0: '#34C759',
          },
          markLine: startMarkLine,
          markArea: forecastMarkArea,
        },
        {
          name: 'MA5',
          type: 'line',
          data: ma5,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1.5, color: '#FF9500' },
        },
        {
          name: 'MA10',
          type: 'line',
          data: ma10,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1.5, color: '#0071E3' },
        },
        {
          name: 'MA20',
          type: 'line',
          data: ma20,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1.5, color: '#5856D6' },
        },
        {
          name: '成交量',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          itemStyle: {
            color: (params: any) => {
              const i = params.dataIndex;
              const cur = displayData[i];
              if (!cur) return '#E5E5EA';
              return (cur.close ?? 0) >= (cur.open ?? 0) ? '#E03E3E' : '#34C759';
            },
          },
          data: volExt,
        },
        ...forecastSeries,
      ],
    };
  // forecastMarks 必须在依赖里：它同时决定 markPoint 数据和绘图区右侧余量
  // （markRightPad 按最长标签估的），漏了会出现"标签没变但余量还是旧的"→ 截断复发。
  }, [displayData, isNav, forecast, forecastMarks]);

  return (
    <div className="apple-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#0071E3]" />
          <h4 className="text-sm font-bold text-[#1D1D1F] tracking-tight">{title}</h4>
          {badge && (
            <span className="text-[10px] font-semibold text-[#5856D6] bg-[#5856D6]/10 px-2 py-0.5 rounded-full border border-[#5856D6]/20">
              {badge}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2.5">
          <span className="text-[11px] text-[#86868B]">滚轮可无级缩放</span>
          <span className="text-[11px] text-[#1D1D1F] bg-[#F5F5F7] px-2.5 py-0.5 rounded-full font-semibold border border-black/[0.04]">
            日线
          </span>
        </div>
      </div>
      {hasData ? (
        <div className="w-full" style={{ height }}>
          <EChart
            option={chartOption}
            style={{ width: '100%', height }}
            notMerge={true}
          />
        </div>
      ) : (
        <div
          className="w-full flex flex-col items-center justify-center gap-1.5 rounded-xl border border-dashed border-black/[0.08] bg-[#F5F5F7]/60"
          style={{ height }}
        >
          <span className="text-xs font-medium text-[#6E6E73]">暂无行情数据</span>
          <span className="text-[11px] text-[#A1A1A6] text-center leading-relaxed px-6">
            行情接口未返回该标的的日线序列。
            <br />
            此处不会绘制任何模拟走势，以免被误读为真实行情。
          </span>
        </div>
      )}
    </div>
  );
};
