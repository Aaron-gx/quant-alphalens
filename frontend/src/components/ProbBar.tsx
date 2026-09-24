import React from 'react';

interface ProbBarProps {
  label: string;
  up: number; // 0 - 1
  flat: number;
  down: number;
  range?: string;
  calibrated?: boolean;
}

export const ProbBar: React.FC<ProbBarProps> = ({
  label,
  up,
  flat,
  down,
  range = '',
  // 默认必须是 false：这是「自学习已校准」徽标的开关，默认 true 等于
  // 调用方一旦漏传就凭空宣称校准已生效。调用方有真实信息才打开它。
  calibrated = false,
}) => {
  const total = Math.max(up + flat + down, 0.0001);
  const upPct = Math.round((up / total) * 100);
  const flatPct = Math.round((flat / total) * 100);
  const downPct = Math.max(0, 100 - upPct - flatPct);

  return (
    <div className="flex flex-col gap-1.5 py-2.5 border-b border-black/[0.04] last:border-none">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-[#1D1D1F]">{label}</span>
          {range && (
            <span className="text-[10px] text-[#86868B] bg-[#F5F5F7] px-2 py-0.5 rounded font-mono border border-black/[0.04]">
              预期振幅 {range}
            </span>
          )}
          {calibrated && (
            <span className="text-[10px] text-[#0071E3] bg-[#EBF5FF] px-2 py-0.5 rounded-full font-medium">
              自学习已校准
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 font-mono font-semibold text-xs tabular-nums">
          <span className="text-[#E03E3E]">涨 {upPct}%</span>
          <span className="text-[#E5E5EA]">·</span>
          <span className="text-[#86868B]">平 {flatPct}%</span>
          <span className="text-[#E5E5EA]">·</span>
          <span className="text-[#34C759]">跌 {downPct}%</span>
        </div>
      </div>

      {/* Apple 极简三色分段胶囊条 */}
      <div className="w-full h-2 bg-[#EFEFF4] rounded-full flex overflow-hidden">
        <div
          style={{ width: `${upPct}%` }}
          className="h-full bg-[#E03E3E] transition-all duration-500 rounded-l-full"
          title={`上涨概率: ${upPct}%`}
        />
        <div
          style={{ width: `${flatPct}%` }}
          className="h-full bg-[#86868B] transition-all duration-500"
          title={`平盘概率: ${flatPct}%`}
        />
        <div
          style={{ width: `${downPct}%` }}
          className="h-full bg-[#34C759] transition-all duration-500 rounded-r-full"
          title={`下跌概率: ${downPct}%`}
        />
      </div>
    </div>
  );
};
