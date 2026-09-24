import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import * as echarts from 'echarts';

/**
 * 轻量 ECharts 容器（替代 echarts-for-react）。
 *
 * 为什么不再用 echarts-for-react@3
 * --------------------------------
 * 该库的实例初始化是**异步两段式**（node_modules/echarts-for-react/esm/core.js
 * `initEchartsInstance`）：
 *
 *   1. echarts.init(ele)            → 建一个"临时实例"
 *   2. 等实例抛出 'finished' 事件   → echarts.dispose(ele) 销毁临时实例
 *   3.                            → 再用真实宽高 echarts.init(ele) 重建
 *
 * 而 componentDidUpdate 里会直接 `getEchartsInstance().setOption()` 并
 * `hideLoading()`。于是只要"props 更新"落在第 2、3 步之间（实例已 dispose、
 * 重建尚未完成），就会在**已销毁的实例**上调用方法，报出：
 *
 *   [ECharts] Instance ec_xxxx has been disposed
 *
 * 本项目开启了 <StrictMode>（main.tsx），开发模式下 React 会刻意
 * mount → unmount → remount 一次；再加上 Analysis 页用
 * `<ErrorBoundary key={symbol + activeTab}>` 和 `<KlineChart key={symbol}>`
 * 在切标的/切页签时频繁卸载重建 —— 这个竞态因此几乎必现。
 *
 * 这里的做法是把生命周期收敛成**同步且严格配对**的 useEffect：
 * 挂载即 init、卸载即 dispose，option 变化只 setOption、绝不触碰实例生命周期。
 * 配合 ResizeObserver 处理尺寸变化，行为上比原库的 size-sensor 更可靠
 * （页签从隐藏变为可见时容器尺寸会变化，能自动触发 resize）。
 * echarts-for-react 用的也是 `import * as echarts from 'echarts'`，
 * 所以换成这里渲染结果完全一致。
 */
/**
 * 图表配置对象。
 *
 * 这里刻意允许"松散对象"，不是偷懒，而是为了让替换 echarts-for-react 保持**无回归**：
 * · 原库的 prop 类型就是 `any`（见 node_modules/echarts-for-react/esm/types.d.ts
 *   的 `export type EChartsOption = any`，注释写着"Solve the type conflict caused by
 *   multiple type files"），所以本项目的 option 字面量从未被类型检查过。
 * · ECharts 官方的 EChartsOption 对字面量极严：`type: 'category'` 会被推断成
 *   `string` 而报 TS2322，6 个调用点会同时失败；真正做严格化还要连带调整
 *   markArea 结构、回调式 itemStyle、dataZoom slider 等，属于另一件事。
 *
 * 想启用强类型时：把 `EChartsOption` 之外的那一项删掉，再把各调用点的
 * `useMemo` 返回值标注成 `echarts.EChartsOption`（让字面量获得上下文类型），
 * 逐个修完即可。
 */
export type EChartOption = echarts.EChartsOption | Record<string, unknown>;

export interface EChartProps {
  option: EChartOption;
  style?: CSSProperties;
  className?: string;
  /**
   * true → 每次 setOption 都丢弃旧配置重建（叠加预测区这类"系列数量会变"的场景必须开）。
   * 默认 false，与原库保持一致。
   */
  notMerge?: boolean;
  theme?: string | object;
  /** 容器尺寸变化时自动 resize，默认 true */
  autoResize?: boolean;
  /** 实例就绪回调（需要手动挂事件/取实例时用） */
  onChartReady?: (chart: echarts.ECharts) => void;
  /** 声明式事件绑定：{ 事件名: (params, chart) => void } */
  onEvents?: Record<string, (params: unknown, chart: echarts.ECharts) => void>;
}

export const EChart: React.FC<EChartProps> = ({
  option,
  style,
  className,
  notMerge = false,
  theme,
  autoResize = true,
  onChartReady,
  onEvents,
}) => {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  // 回调放进 ref：避免它们变化时重建实例（重建会导致图表闪烁、丢失缩放状态）
  const readyRef = useRef(onChartReady);
  readyRef.current = onChartReady;

  // ① 生命周期：只在 theme / autoResize 变化时重建实例
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const chart = echarts.init(host, theme, { renderer: 'canvas' });
    chartRef.current = chart;
    readyRef.current?.(chart);

    let observer: ResizeObserver | undefined;
    if (autoResize && typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => {
        if (!chart.isDisposed()) chart.resize();
      });
      observer.observe(host);
    }

    return () => {
      observer?.disconnect();
      // 先 dispose 再清引用：保证任何后续 effect 都能通过 isDisposed() 判断出失效
      chart.dispose();
      chartRef.current = null;
    };
  }, [theme, autoResize]);

  // ② 配置：只改 option，不碰实例生命周期。
  //    依赖里带上 theme 是必需的 —— theme 变化会触发 ① 重建实例，
  //    而 effect 按声明顺序执行，① 先建好、② 紧跟着铺配置，顺序天然正确。
  //    isDisposed() 守卫是最后一道保险：万一某个版本里 ② 晚于 cleanup 执行，
  //    也不会再复现 "has been disposed" 的告警。
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || chart.isDisposed()) return;
    chart.setOption(option as echarts.EChartsOption, { notMerge, lazyUpdate: false });
  }, [option, notMerge, theme]);

  // ③ 事件：声明式绑定，变化前先解绑自己挂上的 handler
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || chart.isDisposed() || !onEvents) return;
    const bound: Array<[string, (params: unknown) => void]> = [];
    for (const [name, fn] of Object.entries(onEvents)) {
      if (typeof fn !== 'function') continue;
      const handler = (params: unknown) => fn(params, chart);
      chart.on(name, handler);
      bound.push([name, handler]);
    }
    return () => {
      if (chart.isDisposed()) return;
      for (const [name, handler] of bound) chart.off(name, handler);
    };
  }, [onEvents, theme]);

  return (
    <div
      ref={hostRef}
      className={className}
      style={{ width: '100%', height: 300, ...style }}
    />
  );
};

export default EChart;
