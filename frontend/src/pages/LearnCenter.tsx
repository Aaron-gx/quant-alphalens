import React, { useEffect, useState } from 'react';
import { Button, Table, Tag, message } from 'antd';
import { EChart } from '../components/EChart';
import {
  BrainCircuit,
  RefreshCw,
  Layers,
  BookOpen,
  TrendingUp,
  ShieldCheck,
  Zap,
  Activity,
} from 'lucide-react';
import { api } from '../api/client';

export const LearnCenterPage: React.FC = () => {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const res: any = await api.get('/model/status');
      setStatus(res);
    } catch {
      message.error('获取自学习模型状态失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleRetrain = async () => {
    setTraining(true);
    try {
      await api.post('/model/train');
      message.success('全量自学习模型校准重训完成！');
      fetchStatus();
    } catch {
      message.error('重训失败');
    } finally {
      setTraining(false);
    }
  };

  const samples = status?.samples || {};
  const bundles = status?.bundles || {};
  // 样本库总量缺失时**不兜底成 600**：600 是早期手工造的演示数字，
  // 接口挂掉时界面会稳定显示"累计学习样本总库 600 条"，
  // 与 `/model/status` 真实返回（当前 1852 条）毫无关系。缺就是缺 → null → '—'。
  const totalSamples = typeof samples?.total === 'number' ? samples.total : null;
  const curve = status?.curve || [];

  const horizonLabels: Record<string, { label: string; icon: string }> = {
    next_day: { label: '次日超短期', icon: '⚡' },
    one_week: { label: '一周短期', icon: '📅' },
    one_month: { label: '一月中期', icon: '📊' },
    quarter: { label: '季度趋势', icon: '🎯' },
  };

  const tableRows: any[] = [];
  if (bundles) {
    Object.entries(bundles).forEach(([h, pools]: [string, any]) => {
      Object.entries(pools).forEach(([poolName, b]: [string, any]) => {
        const ev = b?.evidence || {};
        const method = b?.method || 'identity';
        const methodLabel =
          method === 'meta_logit'
            ? '元模型 (逻辑回归)'
            : method === 'temperature'
            ? '温度缩放 (Temp Scaling)'
            : '未校准 (原始概率)';

        const meta = horizonLabels[h] || { label: h, icon: '📌' };

        // 三个数字字段一律从后端取，取不到就是 null，由列渲染成 '—'。
        // 这一行原先有三处硬编码：Brier 缺失时兜底 '0.564'、命中率缺失时兜底
        // '56.7%'、available 缺失时兜底 `quant && next_day === true`。
        // 三者叠加的效果是：一个从没训练过的池子，会被显示成
        // "已通过样本外检验 · 命中率 56.7% · Brier 0.564" —— 全是编的。
        const hitRaw = ev?.hit_temp ?? ev?.hit_before;
        tableRows.push({
          key: `${h}-${poolName}`,
          horizon: meta.label,
          icon: meta.icon,
          rawHorizon: h,
          pool: poolName === 'quant' ? '量化引擎池' : 'LLM / 融合推理池',
          method: methodLabel,
          // b.samples 是真实字段；缺失记 0 条（不是编一个数）
          samples: typeof b?.samples === 'number' ? b.samples : 0,
          brierBefore: typeof ev?.brier_before === 'number' ? Number(ev.brier_before).toFixed(3) : null,
          brierAfter:
            typeof ev?.brier_meta === 'number' ? Number(ev.brier_meta).toFixed(3)
              : typeof ev?.brier_temp === 'number' ? Number(ev.brier_temp).toFixed(3)
                : typeof ev?.brier_before === 'number' ? Number(ev.brier_before).toFixed(3)
                  : null,
          hitRate: typeof hitRaw === 'number' ? `${(hitRaw * 100).toFixed(1)}%` : null,
          // 后端 bundles[*][*].available 是真实的布尔判定（n >= min_samples 且样本外检验通过）。
          // 原先的 `?? (quant && next_day)` 兜底会让这一行在接口异常时假装"已就绪"。
          available: b?.available === true,
        });
      });
    });
  }

  // ── 图表数据源（2026-09-24 返工）────────────────────────────────────
  // 改用 `bundles` 而不是 `curve`。
  //
  // 原因：两者覆盖面不一致。`curve` 只含"分层校准"（asset_fund / asset_stock /
  // sym_* 等）的记录，实测里 next_day 与 one_week 的全局分层样本数为 0，
  // 于是被过滤后图上只剩 2 组柱，而下方表格（读 bundles）明明有 4 个已激活池 ——
  // 同一页两个组件对"模型状况"给出不同规模的画面。
  //
  // `bundles[horizon][pool]` 是权威的池子级数据，表格用的就是它，图表也用它，
  // 两边自然对齐。取 evidence.hit_temp ?? hit_before 作为该池的真实样本外命中率，
  // base_hit 作为多数类基准；样本数为 0 的池子不参与（没跑过就不该有柱子）。
  const chartRows = (() => {
    const out: Array<{ label: string; hit: number; base: number }> = [];
    for (const [h, poolsRaw] of Object.entries(bundles as Record<string, any>)) {
      const pools = poolsRaw as Record<string, any>;
      for (const [poolName, b] of Object.entries(pools)) {
        const ev = (b as any)?.evidence || {};
        const hit = ev?.hit_temp ?? ev?.hit_before;
        const base = ev?.base_hit;
        if (!(Number((b as any)?.samples) > 0)) continue;
        if (typeof hit !== 'number' || !Number.isFinite(hit)) continue;
        out.push({
          label: `${horizonLabels[h]?.label ?? h} · ${poolName === 'quant' ? '量化' : 'LLM'}`,
          hit: hit * 100,
          base: typeof base === 'number' && Number.isFinite(base) ? base * 100 : 0,
        });
      }
    }
    return out;
  })();

  // 曲线 y 轴范围：由真实曲线的命中率与基准线共同决定（单位 %）。
  // 上下各留 5 个百分点边距，取整到 5 的倍数；无数据时给 0~100 的空坐标系。
  const curvePcts: number[] = [];
  for (const p of chartRows) {
    curvePcts.push(p.hit, p.base);
  }
  const yRange = (() => {
    if (curvePcts.length === 0) return { min: 0, max: 100, interval: 25 };
    const lo = Math.max(0, Math.floor((Math.min(...curvePcts) - 5) / 5) * 5);
    const hi = Math.min(100, Math.ceil((Math.max(...curvePcts) + 5) / 5) * 5);
    const span = Math.max(hi - lo, 10);       // 至少留 10 个点的跨度，避免压成一条线
    const interval = Math.max(5, Math.ceil(span / 4 / 5) * 5);
    return { min: lo, max: lo + interval * 4 >= hi ? lo + interval * 4 : hi, interval };
  })();

  // Apple HIG ECharts 学习曲线配置
  const curveOption = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: 'rgba(0, 0, 0, 0.08)',
      borderWidth: 1,
      padding: [10, 14],
      textStyle: { color: '#1D1D1F', fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text"' },
      extraCssText: 'box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08); border-radius: 12px; backdrop-filter: blur(12px);',
      formatter: (params: any) => {
        const arr = Array.isArray(params) ? params : [params];
        const hit = arr.find((p: any) => p.seriesName === '样本外实测命中率');
        const base = arr.find((p: any) => p.seriesName === '多数类基准');
        return `<div style="font-size:12px; line-height: 1.6;">
          <div style="color:#86868B; margin-bottom: 2px;">引擎池：<span style="color:#1D1D1F; font-weight:600;">${arr[0]?.name ?? ''}</span></div>
          <div>样本外命中率：<span style="color:#0071E3; font-weight:600; font-family: SF Mono, monospace;">${hit?.value ?? '—'}%</span></div>
          <div>多数类基准线：<span style="color:#86868B; font-weight:600; font-family: SF Mono, monospace;">${base?.value ?? '—'}%</span></div>
        </div>`;
      },
    },
    legend: {
      data: ['样本外实测命中率', '多数类基准'],
      top: 0,
      right: 16,
      icon: 'circle',
      itemWidth: 8,
      itemHeight: 8,
      textStyle: { color: '#86868B', fontSize: 12, fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text"' },
    },
    grid: { top: 36, right: 16, bottom: 58, left: 45 },
    xAxis: {
      type: 'category',
      // x 轴用「周期 · 引擎池」标签（与下方表格同一数据源，逐个池子对齐）。
      // 原先用 c.时间：30 条记录时间戳全相同，坐标轴被 30 个重复标签塞满。
      // 无可用点时留空数组，不再兜底 ['Run 1 (264样本)', ...] 这类编造的轮次。
      data: chartRows.map((p) => p.label),
      axisLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.08)' } },
      axisTick: { show: false },
      axisLabel: {
        color: '#86868B', fontSize: 11, fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text"',
        rotate: 30, interval: 0,
      },
    },
    yAxis: {
      type: 'value',
      // 原先写死 min:45 / max:75。但真实命中率会落在这个窗口之外 ——
      // 例如 one_week/quant 池的 hit_before = 30.11%，直接掉到轴下方看不见，
      // 图上只剩一条"看起来还行"的线。改为按真实数据算范围并留边距，
      // 数据为空时才退回 0~100 的空坐标系。
      min: yRange.min,
      max: yRange.max,
      interval: yRange.interval,
      axisLabel: { formatter: '{value}%', color: '#86868B', fontSize: 11, fontFamily: 'SF Mono, monospace' },
      splitLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.04)', type: 'dashed' } },
    },
    series: [
      {
        name: '样本外实测命中率',
        type: 'bar',
        // 用柱状而不是折线：x 轴是互不相干的池子（离散类别），不是连续时间轴。
        // 折线会在两个池子之间连出一条"中间的命中率"——那条曲线段并不存在，
        // 而且平滑插值会让它看起来像性能演进。柱状天然表达"各自独立取值"。
        barMaxWidth: 26,
        itemStyle: { color: '#0071E3', borderRadius: [4, 4, 0, 0] },
        // 无回测样本时留空数组（图上不画柱），不再用 [63.6, 60.2, 56.7] 这类
        // 写死的"漂亮的样本外命中率"顶上 —— 那会让没跑过回测的库看起来已验证过。
        data: chartRows.map((p) => p.hit.toFixed(1)),
      },
      {
        name: '多数类基准',
        type: 'bar',
        barMaxWidth: 26,
        itemStyle: { color: 'rgba(134, 134, 139, 0.45)', borderRadius: [4, 4, 0, 0] },
        data: chartRows.map((p) => p.base.toFixed(1)),
      },
    ],
  };

  return (
    <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* ── 头部卡片 (Apple HIG) ── */}
      <div className="apple-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-2xl bg-black/[0.04] border border-black/[0.04] flex items-center justify-center text-[#1D1D1F]">
            <BrainCircuit className="w-5 h-5 text-[#0071E3]" />
          </div>
          <div>
            <div className="flex items-center gap-2.5 flex-wrap">
              <h2 className="text-lg font-semibold text-[#1D1D1F] tracking-tight">
                模型自学习中心
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-black/[0.04] text-[#1D1D1F] border border-black/[0.06]">
                样本外切分
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#34C759]/10 text-[#248A3D] border border-[#34C759]/20">
                防过拟合
              </span>
            </div>
          </div>
        </div>

        <Button
          type="primary"
          icon={<RefreshCw className={`w-3.5 h-3.5 ${training ? 'animate-spin' : ''}`} />}
          loading={training}
          onClick={handleRetrain}
          className="apple-press !bg-[#0071E3] hover:!bg-[#0077ED] !text-white !rounded-full !font-medium !h-9 !px-5 shadow-sm border-none"
        >
          全量重训与校准
        </Button>
      </div>

      {/* ── 核心指标四卡片 ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="apple-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">累计学习样本总库</span>
            <div className="w-8 h-8 rounded-xl bg-black/[0.03] flex items-center justify-center text-[#1D1D1F]">
              <Layers className="w-4 h-4 text-[#0071E3]" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-semibold text-[#1D1D1F] font-mono tracking-tight">
              {totalSamples === null ? '—' : totalSamples} <span className="text-xs font-normal text-[#86868B]">条</span>
            </div>
            <div className="mt-2">
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-black/[0.04] text-[#1D1D1F]">
                真实履约 + 滚动前向
              </span>
            </div>
          </div>
        </div>

        <div className="apple-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">已激活校准器单元</span>
            <div className="w-8 h-8 rounded-xl bg-[#34C759]/10 flex items-center justify-center text-[#34C759]">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            {/* 原先写 `...length || 1`：一个校准器都没通过样本外检验时，
                这张卡会显示"1 个池 · 通过样本外检验"，把 0 说成 1。 */}
            <div className="text-2xl font-semibold text-[#248A3D] font-mono tracking-tight">
              {tableRows.filter((r) => r.available).length} <span className="text-xs font-normal text-[#86868B]">个池</span>
            </div>
            <div className="mt-2">
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-[#34C759]/10 text-[#248A3D]">
                通过样本外检验
              </span>
            </div>
          </div>
        </div>

        <div className="apple-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">校准层分级机制</span>
            <div className="w-8 h-8 rounded-xl bg-black/[0.03] flex items-center justify-center text-[#1D1D1F]">
              <Zap className="w-4 h-4 text-[#0071E3]" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-semibold text-[#1D1D1F] font-mono tracking-tight">
              3 <span className="text-xs font-normal text-[#86868B]">阶梯</span>
            </div>
            <div className="mt-2">
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-black/[0.04] text-[#86868B]">
                未校准 → 温度 → 元模型
              </span>
            </div>
          </div>
        </div>

        <div className="apple-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">防过拟合自退火</span>
            <div className="w-8 h-8 rounded-xl bg-black/[0.03] flex items-center justify-center text-[#1D1D1F]">
              <Activity className="w-4 h-4 text-[#FF9500]" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-semibold text-[#1D1D1F] tracking-tight">
              零未来函数
            </div>
            <div className="mt-2">
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-black/[0.04] text-[#86868B]">
                跑输基准自动熔断
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── 学习曲线图表 ── */}
      <div className="apple-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-[#0071E3]" />
            <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight">
              各引擎池样本外命中率 vs 基准
            </h3>
            {/* 原来叫「重训迭代学习曲线」。但后端 /model/status 的 curve 里所有记录
                时间戳相同（是同一个训练时刻的截面快照而非时间序列），
                称它为"迭代学习曲线"会让人以为看到了性能随训练的演进。
                这里按实际内容命名，并说明只画已跑满样本的池子。 */}
            <span className="text-[11px] text-[#86868B]">
              仅含样本数 &gt; 0 的池子，与下方表格同一数据源
            </span>
          </div>
        </div>
        <EChart option={curveOption} style={{ height: 260 }} />
      </div>

      {/* ── 校准器状态详情表格 ── */}
      <div className="apple-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#0071E3]" />
            <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight">
              各周期与引擎池校准包状态详情
            </h3>
          </div>
          <span className="text-xs text-[#86868B]">共 8 个细分校准池</span>
        </div>

        <div className="overflow-x-auto">
          <Table
            dataSource={tableRows}
            pagination={false}
            rowClassName="hover:bg-black/[0.015] transition-colors"
            columns={[
              {
                title: '周期尺度',
                dataIndex: 'horizon',
                key: 'horizon',
                render: (_, r: any) => (
                  <div className="flex items-center gap-2 font-medium text-[#1D1D1F] text-xs">
                    <span>{r.icon}</span>
                    <span>{r.horizon}</span>
                  </div>
                ),
              },
              {
                title: '引擎池',
                dataIndex: 'pool',
                key: 'pool',
                render: (t) => <span className="font-medium text-[#1D1D1F] text-xs">{t}</span>,
              },
              {
                title: '当前校准策略',
                dataIndex: 'method',
                key: 'method',
                render: (m: string) => (
                  <span className="inline-block px-2.5 py-0.5 rounded-full text-xs font-medium bg-black/[0.04] text-[#1D1D1F]">
                    {m}
                  </span>
                ),
              },
              {
                title: '训练样本量',
                dataIndex: 'samples',
                key: 'samples',
                render: (v: number) => <span className="font-mono text-xs text-[#1D1D1F]">{v} 条</span>,
              },
              {
                title: '样本外 Brier 变化 (校准前 → 校准后)',
                key: 'brier',
                // null（后端未给出）渲染成 '—'，不要出现 "null → null" 这种字样
                render: (_, r: any) => (
                  <span className="font-mono text-xs text-[#86868B]">
                    {r.brierBefore ?? '—'} → <span className="text-[#1D1D1F] font-semibold">{r.brierAfter ?? '—'}</span>
                  </span>
                ),
              },
              {
                title: '样本外验证命中率',
                dataIndex: 'hitRate',
                key: 'hitRate',
                // 命中率不是涨跌幅，不套用红涨绿跌；这里统一用中性深色，
                // 未计算时如实显示 '—'。
                render: (v: string | null) => (
                  <span className={`font-mono font-semibold text-xs ${v === null ? 'text-[#A1A1A6]' : 'text-[#1D1D1F]'}`}>
                    {v ?? '—'}
                  </span>
                ),
              },
              {
                title: '生效状态',
                dataIndex: 'available',
                key: 'available',
                render: (a: boolean) => (
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      a
                        ? 'bg-[#34C759]/10 text-[#248A3D] border border-[#34C759]/20'
                        : 'bg-black/[0.04] text-[#86868B]'
                    }`}
                  >
                    {a ? '已激活生效' : '待累积样本'}
                  </span>
                ),
              },
            ]}
          />
        </div>
      </div>

      {/* ── 自学习工作原理三阶段卡片 ── */}
      <div className="apple-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="w-4 h-4 text-[#0071E3]" />
          <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight">
            AlphaLens 概率自学习工作原理与防欺骗机制
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-[#F5F5F7] border border-black/[0.04] space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-[#1D1D1F] text-white text-[11px] font-semibold flex items-center justify-center">
                1
              </span>
              <span className="text-xs font-semibold text-[#1D1D1F]">分池累积无偏样本</span>
            </div>
            <p className="text-xs text-[#86868B] leading-relaxed">
              真实到期对账与量化模型历史滚动前向检验（Walk-Forward）样本物理隔离存储，防止不同数量级带偏校准器。
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#F5F5F7] border border-black/[0.04] space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-[#1D1D1F] text-white text-[11px] font-semibold flex items-center justify-center">
                2
              </span>
              <span className="text-xs font-semibold text-[#1D1D1F]">三级校准防过拟合阶梯</span>
            </div>
            <p className="text-xs text-[#86868B] leading-relaxed">
              样本 &lt; 40 条时如实标注“未校准”；≥ 40 条启动温度缩放；≥ 80 条启动元模型逻辑回归，科学递进。
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#F5F5F7] border border-black/[0.04] space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-[#1D1D1F] text-white text-[11px] font-semibold flex items-center justify-center">
                3
              </span>
              <span className="text-xs font-semibold text-[#1D1D1F]">时序切分防未来函数</span>
            </div>
            <p className="text-xs text-[#86868B] leading-relaxed">
              严格只用过去样本拟合并在未来样本验证。若校准后指标不如基准多数类，系统自动拒绝生效并退回原始概率。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
