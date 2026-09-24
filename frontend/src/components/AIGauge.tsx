import React from 'react';
import { EChart } from './EChart';

interface AIGaugeProps {
  /** 0 - 100；传 null 表示该标的还没有研判（或后端未给置信度），此时不打分 */
  score: number | null;
  title?: string;
  sub?: string;
}

export const AIGauge: React.FC<AIGaugeProps> = ({
  score,
  title = 'AI 预测置信度',
  sub = '',
}) => {
  // 旧实现在这里写了 `score = 65` 的默认值：调用方少传一个参数，仪表盘就会
  // 稳定显示一个凭空捏造的 65%，且和真值长得完全一样。缺就是缺，显示「暂无」。
  const hasScore = typeof score === 'number' && Number.isFinite(score);
  const normalized = hasScore ? Math.min(Math.max(score as number, 0), 100) : 0;

  const getThemeColor = (val: number) => {
    if (val >= 75) return '#0071E3'; // Apple Blue
    if (val >= 50) return '#0077ED';
    return '#86868B'; // Apple Gray
  };

  const ringColor = getThemeColor(normalized);

  const option = {
    series: [
      {
        type: 'gauge',
        startAngle: 190,
        endAngle: -10,
        min: 0,
        max: 100,
        radius: '86%',
        center: ['50%', '62%'],
        itemStyle: {
          color: ringColor,
        },
        progress: {
          show: true,
          roundCap: true,
          width: 9,
        },
        pointer: {
          show: false,
        },
        axisLine: {
          roundCap: true,
          lineStyle: {
            width: 9,
            color: [[1, '#EFEFF4']],
          },
        },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        title: {
          show: true,
          offsetCenter: [0, '26%'],
          fontSize: 11,
          color: '#86868B',
          fontWeight: 500,
        },
        detail: {
          valueAnimation: true,
          offsetCenter: [0, '-8%'],
          fontSize: 32,
          fontWeight: 700,
          color: '#1D1D1F',
          formatter: hasScore ? '{value}%' : '暂无',
        },
        data: [
          {
            value: normalized,
            name: '综合置信水平',
          },
        ],
      },
    ],
  };

  return (
    <div className="apple-card p-5 flex flex-col items-center justify-between">
      <div className="w-full flex items-center justify-between mb-1">
        <span className="text-xs font-semibold text-[#86868B] uppercase tracking-wider flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#0071E3]" />
          {title}
        </span>
        <span className="text-[11px] font-semibold text-[#1D1D1F] bg-[#F5F5F7] px-2.5 py-0.5 rounded-full border border-black/[0.04]">
          {!hasScore ? '无研判' : normalized >= 75 ? '高确信' : normalized >= 50 ? '中度确信' : '低信噪比'}
        </span>
      </div>
      <div className="w-full h-36 flex items-center justify-center">
        <EChart option={option} style={{ width: '100%', height: '145px' }} />
      </div>
      {sub && (
        <p className="text-[11px] text-[#86868B] text-center font-normal mt-0.5">
          {sub}
        </p>
      )}
    </div>
  );
};
