import React from 'react';
import { TrendingUp, Scale, TrendingDown, ArrowUpRight, ArrowDownRight } from 'lucide-react';

interface ScenarioItem {
  prob?: number;
  range?: string;
  target?: number;
  condition?: string;
}

interface ScenarioCardsProps {
  scenarios?: {
    optimistic?: ScenarioItem;
    base?: ScenarioItem;
    pessimistic?: ScenarioItem;
  };
}

/**
 * 三情景路径推演卡。
 *
 * 2026-09-24 返工：原先每个情景都挂了一整套 `|| { prob: 0.35, range: '+5% ~ +12%',
 * condition: '量能持续放大突破前高阻力位，北向资金持续净流入' }` 兜底。也就是说
 * 只要后端没返回 scenarios（或某个情景缺失），界面就会**凭空生成**三条带概率、
 * 带涨幅区间、带触发条件的推演，其中「北向资金持续净流入」「主力资金大幅离场」
 * 这类句子读起来像是有数据支撑的判断，实际是硬编码的常量。
 *
 * 现在：情景缺了就直说缺了，概率缺了显示 '—'，触发条件缺了也明写缺失，
 * 不再用默认值把空位填满。
 */
export const ScenarioCards: React.FC<ScenarioCardsProps> = ({ scenarios }) => {
  const norm = (x?: ScenarioItem) => {
    const prob = typeof x?.prob === 'number' && Number.isFinite(x.prob) ? x.prob : null;
    const range = (x?.range || '').trim();
    const condition = (x?.condition || '').trim();
    return { prob, range, condition, empty: prob === null && !range && !condition };
  };

  const slots = [
    {
      key: 'optimistic' as const,
      title: '乐观推演',
      tag: '进攻',
      icon: <TrendingUp className="w-4 h-4 text-[#E03E3E]" />,
      badgeColor: 'bg-[#FFECEB] text-[#E03E3E] border-[#FFD3D0]',
      probColor: 'text-[#E03E3E]',
      dirIcon: <ArrowUpRight className="w-3.5 h-3.5 text-[#E03E3E]" />,
      dirHint: '上行',
    },
    {
      key: 'base' as const,
      title: '基准中枢',
      tag: '中性',
      icon: <Scale className="w-4 h-4 text-[#0071E3]" />,
      badgeColor: 'bg-[#EBF5FF] text-[#0071E3] border-[#CCE4FF]',
      probColor: 'text-[#0071E3]',
      dirIcon: <span className="text-xs text-[#0071E3] font-bold">~</span>,
      dirHint: '震荡',
    },
    {
      key: 'pessimistic' as const,
      title: '悲观防守',
      tag: '防守',
      icon: <TrendingDown className="w-4 h-4 text-[#34C759]" />,
      badgeColor: 'bg-[#E8F8EE] text-[#107C41] border-[#C8F0D5]',
      probColor: 'text-[#34C759]',
      dirIcon: <ArrowDownRight className="w-3.5 h-3.5 text-[#34C759]" />,
      dirHint: '下行',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-3">
      {slots.map((s) => {
        const d = norm(scenarios?.[s.key]);
        return (
          <div key={s.key} className="apple-card apple-card-hover p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 font-bold text-[#1D1D1F] text-xs">
                  {s.icon}
                  <span>{s.title}</span>
                </div>
                <span className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${s.badgeColor}`}>
                  {s.tag}
                </span>
              </div>

              {d.empty ? (
                <div className="py-9 text-center text-[11px] text-[#A1A1A6] leading-relaxed">
                  该条研判未给出「{s.title}」情景
                </div>
              ) : (
                <>
                  <div className="flex items-baseline justify-between my-2">
                    <span className={`text-3xl font-extrabold tabular-nums tracking-tight font-mono ${s.probColor}`}>
                      {d.prob === null ? '—' : `${Math.round(d.prob * 100)}%`}
                    </span>
                    {d.range && (
                      <div className="flex items-center gap-1 font-mono font-semibold text-xs text-[#1D1D1F] bg-[#F5F5F7] px-2.5 py-1 rounded-lg border border-black/[0.04]">
                        {s.dirIcon}
                        <span>{d.range}</span>
                      </div>
                    )}
                  </div>

                  <p className="text-[11px] text-[#86868B] leading-relaxed mt-3 pt-3 border-t border-black/[0.04] line-clamp-3">
                    <span className="font-semibold text-[#1D1D1F]">触发逻辑：</span>
                    {d.condition || `该情景未给出触发条件（${s.dirHint}方向）`}
                  </p>
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
