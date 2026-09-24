import React from 'react';
import { Star, ShieldAlert } from 'lucide-react';

interface VerdictCardProps {
  advice?: string;
  entry?: number | string;
  target?: number | string;
  stopLoss?: number | string;
  stars?: number;
  risks?: string[];
}

/**
 * 投资策略与量化裁决卡。
 *
 * 2026-09-24 返工：原先 props 带 `advice = '逢低关注 · 分批布局'`、`stars = 4` 两个
 * 默认值。也就是说只要调用方漏传（或后端没返回 action），卡片会稳定显示一条
 * **"逢低关注 · 分批布局 + 四星"的建仓建议**，与真实建议无法区分 —— 用户会直接
 * 照做。同时星级是五档主观评价，后端从未提供过这个字段，四个亮星纯属装饰。
 * 现在：没有 advice 就渲染明文空状态，stars 未提供就不渲染星级。
 */
export const VerdictCard: React.FC<VerdictCardProps> = ({
  advice,
  entry = '—',
  target = '—',
  stopLoss = '—',
  stars,
  risks = [],
}) => {
  const getBadgeStyle = (adv: string) => {
    if (adv.includes('买入') || adv.includes('多') || adv.includes('加仓') || adv.includes('积极')) {
      return {
        bg: 'bg-[#FFECEB] text-[#E03E3E] border-[#FFD3D0]',
        actionLabel: '进攻配置',
      };
    }
    if (adv.includes('逢低') || adv.includes('关注') || adv.includes('波段') || adv.includes('试探')) {
      return {
        bg: 'bg-[#EBF5FF] text-[#0071E3] border-[#CCE4FF]',
        actionLabel: '分批定投',
      };
    }
    if (adv.includes('减仓') || adv.includes('防守') || adv.includes('止损') || adv.includes('空')) {
      return {
        bg: 'bg-[#E8F8EE] text-[#107C41] border-[#C8F0D5]',
        actionLabel: '防御回避',
      };
    }
    return {
      bg: 'bg-[#F5F5F7] text-[#1D1D1F] border-black/[0.06]',
      actionLabel: '中性观望',
    };
  };

  const hasAdvice = typeof advice === 'string' && advice.trim() !== '';
  const cleanAdvice = (advice || '').replace(/^建议[：:]\s*/, '');
  const hasStars = typeof stars === 'number' && Number.isFinite(stars);
  const style = getBadgeStyle(advice || '');

  return (
    <div className="apple-card p-5 flex flex-col justify-between">
      <div>
        {/* 卡片顶栏 */}
        <div className="flex items-center justify-between mb-3">
          <div className="text-xs font-semibold text-[#86868B] uppercase tracking-wider flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#0071E3]" />
            投资策略与量化裁决
          </div>
          <span className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${style.bg}`}>
            {style.actionLabel}
          </span>
        </div>

        {/* 核心建议大字与星级评定 */}
        <div className="flex items-center justify-between gap-3 my-2">
          <h3 className="text-[17px] font-bold text-[#1D1D1F] tracking-tight leading-snug flex-1">
            {hasAdvice ? cleanAdvice : '尚未给出操作建议'}
          </h3>
          {/* 星级是主观评价档位，只在调用方确实给出时才画；缺省不画空星 */}
          {hasStars && (
            <div className="flex items-center gap-0.5 shrink-0">
              {[1, 2, 3, 4, 5].map((i) => (
                <Star
                  key={i}
                  className={`w-3.5 h-3.5 ${
                    i <= (stars as number) ? 'text-[#FF9500] fill-[#FF9500]' : 'text-[#E5E5EA] fill-[#E5E5EA]'
                  }`}
                />
              ))}
            </div>
          )}
        </div>

        {/* 关键交易点位三栏网格 (Apple HIG Table/Card) */}
        <div className="grid grid-cols-3 gap-2 my-3 p-3 bg-[#F5F5F7] rounded-xl border border-black/[0.03] text-center">
          <div>
            <div className="text-[11px] text-[#86868B] font-medium mb-0.5">参考建仓</div>
            <div className="text-sm font-bold text-[#1D1D1F] font-mono tracking-tight">
              {entry || '现价附近'}
            </div>
          </div>
          <div>
            <div className="text-[11px] text-[#86868B] font-medium mb-0.5">目标止盈</div>
            <div className="text-sm font-bold text-[#E03E3E] font-mono tracking-tight">
              {target || '看上行通道'}
            </div>
          </div>
          <div>
            <div className="text-[11px] text-[#86868B] font-medium mb-0.5">防守止损</div>
            <div className="text-sm font-bold text-[#34C759] font-mono tracking-tight">
              {stopLoss || '破位离场'}
            </div>
          </div>
        </div>
      </div>

      {/* 风险边界提示 */}
      {risks && risks.length > 0 && (
        <div className="pt-2.5 mt-1 border-t border-black/[0.04] flex items-start gap-2 text-xs">
          <ShieldAlert className="w-3.5 h-3.5 text-[#FF9500] shrink-0 mt-0.5" />
          <p className="line-clamp-2 text-[11px] text-[#86868B] leading-relaxed font-normal">
            {risks[0]}
          </p>
        </div>
      )}
    </div>
  );
};
