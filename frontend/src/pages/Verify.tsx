import React, { useEffect, useState } from 'react';
import { Button, Table, Progress, message } from 'antd';
import { EChart } from '../components/EChart';
import {
  CheckCircle2,
  Play,
  TrendingUp,
  BarChart3,
  AlertCircle,
  Target,
  ShieldCheck,
  Zap,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Clock,
  History,
  FileCheck2,
} from 'lucide-react';
import { api } from '../api/client';

// ─── 已删除的 MOCK_AUDIT_LOGS (2026-09-24) ─────────────────────────────────────
// 这里原先是一个 6 条的硬编码对账流水常量：编造的核销单号（EVAL-20260922-01）、
// 编造的标的、编造的实际涨跌幅与命中结果、编造的 Brier 分数，而表格标题写着
// 「已对账前 6 笔记录」—— 用户会把它当成系统真实跑出来的对账台账。
// 根因是后端 accuracy_report() 只返回聚合统计、不含明细，前端没有真数据可读，
// 就用常量顶上了。现在补了 /verify/logs 明细接口，这里改读真实记录。

/** 方向码 → 中文标签（与后端 core/predict/labels.py 的 up/flat/down 同源）。 */
const dirLabel = (d?: string) => (d === 'up' ? '看涨' : d === 'down' ? '看跌' : d === 'flat' ? '震荡' : '—');
/** 中国习惯：涨红跌绿。 */
const dirColor = (d?: string) => (d === 'up' ? 'text-[#E03E3E]' : d === 'down' ? 'text-[#34C759]' : 'text-[#86868B]');


export const VerifyPage: React.FC = () => {
  const [report, setReport] = useState<any>(null);
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  const fetchReport = async () => {
    setLoading(true);
    try {
      const [rep, lg] = await Promise.all([
        api.get('/verify/report').catch(() => null),
        api.get('/verify/logs', { params: { limit: 20 } }).catch(() => null),
      ]);
      // 后端在没有样本时只返回 {"total": 0}，此时报告置空、走空态；
      // 不再用 `res.total > 0` 静默保留上一次的旧值（切换标的时会出现数据串台）。
      setReport(rep && (rep as any).total > 0 ? rep : null);
      setLogs(Array.isArray((lg as any)?.items) ? (lg as any).items : []);
    } catch {
      setReport(null);
      setLogs([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReport();
  }, []);

  const handleRunVerify = async () => {
    setRunning(true);
    try {
      const res: any = await api.post('/verify/run');
      message.success(`自动对账完成！本次新核销对账 ${res?.evaluated ?? 0} 条记录`);
      fetchReport();
    } catch {
      message.error('自动对账异常');
    } finally {
      setRunning(false);
    }
  };

  // 2026-09-24 返工：
  // ① 原先四个核心指标各带一个写死的兜底值（命中率 0.642、样本 30 笔、
  //    Brier 0.218），从未跑过对账的库也会显示一份完整的"历史绩效"。
  // ② **字段名也是错的**：后端 core/verify/evaluator.accuracy_report() 返回的是
  //    `hit_rate` / `avg_brier` / `total` / `by_horizon[h]={n,hit_rate,avg_brier}`，
  //    而这里读的是 `overall_hit_rate` / `overall_brier` / `total_evaluations` /
  //    `v.total` / `v.hits` / `v.brier` —— 全部为 undefined，于是**真实数据永远
  //    进不来**，页面长期稳定显示那份假报告。两处一起修。
  const hitRate = typeof report?.hit_rate === 'number' ? report.hit_rate : null;
  const totalEvals = typeof report?.total === 'number' ? report.total : null;
  const brierScore = typeof report?.avg_brier === 'number' ? report.avg_brier : null;
  const excessHit = hitRate === null ? null : hitRate - 0.5;

  const horizonBreakdown: Record<string, any> = report?.by_horizon || {};

  const horizonLabels: Record<string, { label: string; desc: string }> = {
    next_day: { label: '次日超短期', desc: '次日T+1收盘价验证' },
    one_week: { label: '一周短期', desc: '5个交易日累计收益' },
    one_month: { label: '一月中期', desc: '20个交易日区间收益' },
    quarter: { label: '季度趋势', desc: '60个交易日趋势' },
  };

  const horizonChartOption = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.96)',
      borderColor: 'rgba(0, 0, 0, 0.08)',
      shadowBlur: 10,
      shadowColor: 'rgba(0, 0, 0, 0.06)',
      textStyle: { color: '#1D1D1F', fontSize: 11 },
    },
    grid: { left: 40, right: 20, top: 25, bottom: 25 },
    xAxis: {
      type: 'category',
      data: ['次日(T+1)', '一周(5D)', '一月(20D)', '季度(60D)'],
      axisLine: { lineStyle: { color: '#E5E5EA' } },
      axisLabel: { color: '#86868B', fontSize: 11 },
    },
    yAxis: {
      type: 'value',
      min: 0.4,
      max: 1.0,
      splitLine: { lineStyle: { color: '#F2F2F7', type: 'dashed' } },
      axisLabel: { color: '#86868B', fontSize: 11, formatter: (v: number) => `${Math.round(v * 100)}%` },
    },
    series: [
      {
        name: '模型胜率',
        type: 'bar',
        barWidth: 28,
        itemStyle: {
          color: '#0071E3',
          borderRadius: [6, 6, 0, 0],
        },
        data: (['next_day', 'one_week', 'one_month', 'quarter'] as const).map((k) => {
          // 缺哪个周期就留空（ECharts 对 null 不画柱），不再用 0.58/0.62/0.67/0.75 顶上
          const hr = horizonBreakdown?.[k]?.hit_rate;
          return typeof hr === 'number' ? Number(hr.toFixed(2)) : null;
        }),
        markLine: {
          silent: true,
          lineStyle: { color: '#86868B', type: 'dashed', width: 1.5 },
          data: [{ yAxis: 0.5, name: '无偏基准 50%' }],
          label: { formatter: '50% 随机基准', position: 'end', color: '#86868B', fontSize: 10 },
        },
      },
    ],
  };

  // 置信度分桶命中率：真实来源是 report.by_confidence（桶按置信度 20 分档，
  // 形如 {"40-59": {n, hit_rate}, "60-79": {...}}）。
  // 原先这条"校准可靠性曲线"的 y 值是写死的 [0.48, 0.56, 0.65, 0.76, 0.84]，
  // x 轴标签也是自己编的 0.4~0.5 分段 —— 和真实分桶完全对不上，
  // 等于画了一条永久"看起来很校准"的曲线。现在按真实分桶渲染。
  const confBuckets = Object.entries((report?.by_confidence || {}) as Record<string, any>)
    .map(([k, v]) => {
      const lo = parseInt(k, 10);
      return {
        label: Number.isFinite(lo) ? `${lo}~${lo + 19}` : k,
        lo: Number.isFinite(lo) ? lo : 0,
        rate: typeof v?.hit_rate === 'number' ? v.hit_rate : null,
        n: typeof v?.n === 'number' ? v.n : 0,
      };
    })
    .sort((a, b) => a.lo - b.lo);

  const calibrationChartOption = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.96)',
      borderColor: 'rgba(0, 0, 0, 0.08)',
      textStyle: { color: '#1D1D1F', fontSize: 11 },
    },
    grid: { left: 40, right: 20, top: 25, bottom: 25 },
    xAxis: {
      type: 'category',
      data: confBuckets.map((c) => c.label),
      axisLine: { lineStyle: { color: '#E5E5EA' } },
      axisLabel: { color: '#86868B', fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1.0,
      splitLine: { lineStyle: { color: '#F2F2F7', type: 'dashed' } },
      axisLabel: { color: '#86868B', fontSize: 10, formatter: (v: number) => `${Math.round(v * 100)}%` },
    },
    series: [
      {
        name: '实际命中比例',
        type: 'line',
        smooth: true,
        itemStyle: { color: '#34C759' },
        lineStyle: { width: 2.5, color: '#34C759' },
        data: confBuckets.map((c) => c.rate),
      },
      {
        // 完全校准时的理想值：落在该置信度区间的中点
        name: '完全校准理想对角线',
        type: 'line',
        lineStyle: { color: '#86868B', type: 'dashed', width: 1.5 },
        symbol: 'none',
        data: confBuckets.map((c) => (c.lo + 9.5) / 100),
      },
    ],
  };

  return (
    <div className="max-w-[1600px] mx-auto px-6 sm:px-8 py-6 space-y-6">
      {/* ── 头部卡片 ── */}
      <div className="apple-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#F5F5F7] flex items-center justify-center text-[#34C759] border border-black/[0.04]">
            <FileCheck2 className="w-5 h-5 text-[#34C759]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-[#1D1D1F] tracking-tight">
                预测验证看板
              </h2>
              {/* 原先徽标固定写「已核销」，哪怕一条都没核销过也这么显示。
                  现在按实际核销笔数切换。
                  计数取 totalEvals（后端汇报的真实总数），而不是 logs.length ——
                  `/verify/logs` 有 limit=20 的上限，核销超过 20 笔时用 logs.length
                  会在页头少报一个确切的数字，而右边样本卡显示的是真实总数，两处对不上。 */}
              {(totalEvals ?? logs.length) > 0 ? (
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#34C759]/10 text-[#248A3D]">
                  已核销 {totalEvals ?? logs.length} 笔
                </span>
              ) : (
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-black/[0.04] text-[#86868B]">
                  待首次核销
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 self-start md:self-auto shrink-0">
          <Button
            type="primary"
            icon={<Play className="w-3.5 h-3.5 fill-current" />}
            loading={running}
            onClick={handleRunVerify}
            className="apple-press !bg-[#0071E3] hover:!bg-[#0077ED] !rounded-full !font-medium !h-9 !px-5 !text-xs shadow-sm text-white border-none"
          >
            执行核销
          </Button>
        </div>
      </div>

      {/* ── 核心指标四卡片 ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="apple-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[#86868B] uppercase tracking-wider">全局方向命中率</span>
            {/* 图标配色必须随数值走。原先这里是写死的绿色（#34C759）——
                0.0% 命中率、比随机基准低 50 个百分点的最差结果，配一个绿色对勾图标，
                视觉上读起来像"达标"。绿色在这里既不表示涨也不表示好，纯属误导，
                所以改成与下方徽标同判据：跑赢基准=品牌蓝，未跑赢/无数据=中性灰。 */}
            <div
              className={`w-8 h-8 rounded-lg bg-[#F5F5F7] flex items-center justify-center ${
                excessHit !== null && excessHit >= 0 ? 'text-[#0071E3]' : 'text-[#A1A1A6]'
              }`}
            >
              <Target className="w-4 h-4" />
            </div>
          </div>
          <div className={`text-3xl font-extrabold font-mono tracking-tight mt-2 ${hitRate === null ? 'text-[#A1A1A6]' : 'text-[#0071E3]'}`}>
            {hitRate === null ? '—' : `${(hitRate * 100).toFixed(1)}%`}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            {excessHit === null ? (
              <span className="text-[11px] text-[#86868B]">尚未执行对账，无绩效数据</span>
            ) : (
              <>
                {/* 超额为正才给"超额胜率"的蓝色徽标；为负如实标成"低于基准" */}
                <span
                  className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${
                    excessHit >= 0 ? 'bg-[#EBF5FF] text-[#0071E3]' : 'bg-[#F5F5F7] text-[#86868B]'
                  }`}
                  title="相对 50% 随机基准的超额命中率"
                >
                  {excessHit >= 0
                    ? `+${(excessHit * 100).toFixed(1)}% 超额胜率`
                    : `${(excessHit * 100).toFixed(1)}% 低于基准`}
                </span>
                <span className="text-[11px] text-[#86868B]">vs 随机基准 50%</span>
              </>
            )}
          </div>
        </div>

        <div className="apple-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[#86868B] uppercase tracking-wider">Brier 概率校准均方差</span>
            <div className="w-8 h-8 rounded-lg bg-[#F5F5F7] flex items-center justify-center text-[#0071E3]">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <div className={`text-3xl font-extrabold font-mono tracking-tight mt-2 ${brierScore === null ? 'text-[#A1A1A6]' : 'text-[#0071E3]'}`}>
            {brierScore === null ? '—' : Number(brierScore).toFixed(3)}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            {brierScore === null ? (
              <span className="text-[11px] text-[#86868B]">尚未执行对账，无校准数据</span>
            ) : (
              <>
                {/* 分档由实际数值派生：越低越好，0.25 以下视为优良 */}
                <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${brierScore < 0.25 ? 'bg-[#EBF5FF] text-[#0071E3]' : 'bg-[#F5F5F7] text-[#86868B]'}`}>
                  {brierScore < 0.15 ? '优秀 (<0.15)' : brierScore < 0.25 ? '优良区间 (<0.25)' : '待改善 (≥0.25)'}
                </span>
                <span className="text-[11px] text-[#86868B]">概率诚实度</span>
              </>
            )}
          </div>
        </div>

        <div className="apple-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[#86868B] uppercase tracking-wider">已完整对账履约样本</span>
            <div className="w-8 h-8 rounded-lg bg-[#F5F5F7] flex items-center justify-center text-[#1D1D1F]">
              <CheckCircle2 className="w-4 h-4 text-[#0071E3]" />
            </div>
          </div>
          <div className={`text-3xl font-extrabold font-mono tracking-tight mt-2 ${totalEvals === null ? 'text-[#A1A1A6]' : 'text-[#1D1D1F]'}`}>
            {totalEvals === null ? '—' : <>{totalEvals} <span className="text-sm font-normal text-[#86868B]">笔</span></>}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-[#F5F5F7] text-[#86868B] border border-black/[0.04]">
              按到期日收盘价核销
            </span>
          </div>
        </div>

        {/* 原先这里还有第四张卡「信号履约超额 Alpha +5.8% / 盈亏比 1.92:1」，
            两个数字都是硬编码常量，后端也没有任何字段能支撑它们（对账只统计
            方向命中与 Brier，不统计盈亏比）。没有数据源就不该有这张卡，直接移除，
            指标行改为三列。 */}
      </div>

      {/* ── 可视化图表行 ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="apple-card p-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[#0071E3]" />
              <h3 className="text-sm font-bold text-[#1D1D1F] tracking-tight">
                各预测尺度命中率 vs 50% 随机基准
              </h3>
            </div>
            <span className="text-[11px] text-[#86868B]">基准超额</span>
          </div>
          <p className="text-xs text-[#86868B] mb-3">
            蓝色柱体为模型胜率，灰色虚线为无信息量基准（50%）。
          </p>
          <EChart option={horizonChartOption} style={{ height: 250 }} />
        </div>

        <div className="apple-card p-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-[#34C759]" />
              <h3 className="text-sm font-bold text-[#1D1D1F] tracking-tight">
                模型置信度与实际命中拟合曲线
              </h3>
            </div>
            <span className="text-[11px] text-[#86868B]">校准可靠性</span>
          </div>
          <p className="text-xs text-[#86868B] mb-3">
            绿色实线越贴合灰色对角线，表明预测置信度越客观无偏。
          </p>
          <EChart option={calibrationChartOption} style={{ height: 250 }} />
        </div>
      </div>

      {/* ── 周期细分表现表 ── */}
      <div className="apple-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-[#1D1D1F] tracking-tight">
            多时间尺度预测表现细分
          </h3>
          <span className="text-xs text-[#86868B]">分层对账统计</span>
        </div>
        <Table
          dataSource={Object.entries(horizonBreakdown).map(([k, v]: [string, any]) => {
            const meta = horizonLabels[k] || { label: k, desc: '' };
            // 命中率 / Brier 缺失时保留 null，由列渲染成 '—'；
            // 不再用 `?? 0.5` / `?? 0.22` 把空位填成一个"刚好等于基准"的漂亮数字。
            const rate = typeof v?.hit_rate === 'number' ? v.hit_rate : null;
            return {
              key: k,
              horizon: meta.label,
              desc: meta.desc,
              // 真实字段名是 n / hit（不是 total / hits）
              total: typeof v?.n === 'number' ? v.n : null,
              hits: typeof v?.hit === 'number' ? v.hit : null,
              rate,
              ratePct: rate === null ? null : Math.round(rate * 100),
              diffPct: rate === null ? null : Math.round(rate * 100) - 50,
              brier: typeof v?.avg_brier === 'number' ? v.avg_brier : null,
            };
          })}
          pagination={false}
          size="middle"
          locale={{ emptyText: '暂无分层对账数据 —— 执行「核销」并等到期后生成' }}
          columns={[
            {
              title: '预测尺度',
              key: 'horizon',
              render: (_, r) => (
                <div>
                  <div className="font-bold text-xs text-[#1D1D1F]">{r.horizon}</div>
                  <div className="text-[11px] text-[#86868B]">{r.desc}</div>
                </div>
              ),
            },
            {
              title: '核销样本 / 命中数',
              key: 'sample',
              render: (_, r: any) => (
                <div className="font-mono text-xs text-[#1D1D1F]">
                  <span className="font-bold">{r.hits ?? '—'} 命中</span>
                  <span className="text-[#86868B]"> / 共 {r.total ?? '—'} 笔</span>
                </div>
              ),
            },
            {
              title: '周期命中率',
              dataIndex: 'ratePct',
              key: 'ratePct',
              width: 220,
              render: (val: number | null, r: any) => (
                <div className="space-y-1">
                  {val === null ? (
                    <span className="text-xs text-[#A1A1A6]">—</span>
                  ) : (
                    <>
                      <div className="flex items-center justify-between text-xs font-mono">
                        {/* 命中率不是涨跌幅，不能套用"红涨绿跌"的价格语义。
                            这里用品牌蓝=跑赢基准、灰色=未跑赢，避免把 0% 命中
                            渲染成一片"优良绿"。 */}
                        <span className={`font-bold ${r.diffPct >= 0 ? 'text-[#0071E3]' : 'text-[#86868B]'}`}>
                          {val}%
                        </span>
                        <span
                          className={`font-semibold px-1.5 py-0.5 rounded text-[10px] ${
                            r.diffPct >= 0
                              ? 'text-[#0071E3] bg-[#EBF5FF]'
                              : 'text-[#86868B] bg-[#F5F5F7]'
                          }`}
                        >
                          {r.diffPct >= 0 ? '+' : ''}{r.diffPct}% vs 基准
                        </span>
                      </div>
                      <Progress
                        percent={val}
                        showInfo={false}
                        strokeColor={r.diffPct >= 0 ? '#0071E3' : '#86868B'}
                        railColor="#EFEFF4"
                        size={['100%', 5]}
                      />
                    </>
                  )}
                </div>
              ),
            },
            {
              title: 'Brier 校准分',
              dataIndex: 'brier',
              key: 'brier',
              render: (v: number | null) => (
                <span className="font-mono font-bold text-xs text-[#1D1D1F]">
                  {typeof v === 'number' ? v.toFixed(3) : '—'}
                </span>
              ),
            },
          ]}
        />
      </div>

      {/* ── 实时履约对账流水表 (Apple Audit Ledger) ── */}
      <div className="apple-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <History className="w-4 h-4 text-[#0071E3]" />
            <h3 className="text-sm font-bold text-[#1D1D1F] tracking-tight">
              近期到期预测履约核销对账流水 (Audit Trail)
            </h3>
          </div>
          <span className="text-xs text-[#86868B]">
            {logs.length > 0 ? `最近 ${logs.length} 笔核销记录` : '暂无核销记录'}
          </span>
        </div>

        {/* 2026-09-24 返工：原先 dataSource 是硬编码的 MOCK_AUDIT_LOGS —— 6 条编造的
            核销单号、编造的实际涨跌幅与命中结果，标题写着「已对账前 6 笔记录」。
            现在改读后端 /verify/logs（真实 PredictionEval 明细），无记录就空态。 */}
        <Table
          dataSource={logs}
          rowKey="id"
          pagination={false}
          size="middle"
          loading={loading}
          locale={{ emptyText: '暂无核销记录 —— 点击右上角「执行核销」后，到期预测的核销结果会出现在这里' }}
          columns={[
            {
              title: '对账序号 / 核销时间',
              key: 'id',
              render: (_, r: any) => (
                <div>
                  <div className="font-mono text-xs font-semibold text-[#1D1D1F]">
                    #{r.prediction_id}-{r.id}
                  </div>
                  <div className="text-[11px] text-[#86868B] flex items-center gap-1 font-mono">
                    <Clock className="w-3 h-3" />
                    {r.evaluated_at || '—'}
                  </div>
                </div>
              ),
            },
            {
              title: '标的名称',
              key: 'target',
              render: (_, r: any) => (
                <div>
                  <div className="font-bold text-xs text-[#1D1D1F]">{r.name || '—'}</div>
                  <span className="text-[11px] font-mono text-[#86868B]">{r.symbol || '—'}</span>
                </div>
              ),
            },
            {
              title: '周期',
              dataIndex: 'horizon',
              key: 'horizon',
              render: (t: string) => (
                <span className="text-xs text-[#86868B]">{horizonLabels[t]?.label || t || '—'}</span>
              ),
            },
            {
              title: '预测 / 实际方向',
              key: 'dir',
              render: (_, r: any) => (
                <div className="text-xs">
                  <span className={`font-semibold ${dirColor(r.predicted_direction)}`}>
                    {dirLabel(r.predicted_direction)}
                  </span>
                  <span className="text-[#86868B]"> → </span>
                  <span className={`font-semibold ${dirColor(r.actual_direction)}`}>
                    {dirLabel(r.actual_direction)}
                  </span>
                </div>
              ),
            },
            {
              title: '到期真实涨跌',
              dataIndex: 'actual_change_pct',
              key: 'actual_change_pct',
              render: (v: number) => (
                <span className={`font-mono font-bold text-xs ${v > 0 ? 'text-[#E03E3E]' : v < 0 ? 'text-[#34C759]' : 'text-[#86868B]'}`}>
                  {v > 0 ? `+${Number(v).toFixed(2)}%` : `${Number(v).toFixed(2)}%`}
                </span>
              ),
            },
            {
              title: 'Brier',
              dataIndex: 'brier_score',
              key: 'brier_score',
              render: (v: number) => (
                <span className="font-mono text-xs text-[#1D1D1F]">
                  {typeof v === 'number' ? v.toFixed(3) : '—'}
                </span>
              ),
            },
            {
              title: '核销结论',
              dataIndex: 'hit',
              key: 'hit',
              render: (hit: boolean) => (
                <span className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${hit ? 'bg-[#E8F8EE] text-[#107C41] border-[#C8F0D5]' : 'bg-[#FFECEB] text-[#E03E3E] border-[#FFD3D0]'}`}>
                  {hit ? '✓ 命中预期' : '✕ 未命中'}
                </span>
              ),
            },
          ]}
        />
      </div>

      {/* ── 闭环机制 ── */}
      <div className="apple-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <AlertCircle className="w-4 h-4 text-[#86868B]" />
          <h3 className="text-sm font-bold text-[#1D1D1F] tracking-tight">
            对账与模型校准自闭环机制
          </h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-[#F5F5F7] border border-black/[0.03] space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-[#1D1D1F] text-white text-[11px] font-bold flex items-center justify-center">
                1
              </span>
              <span className="text-xs font-bold text-[#1D1D1F]">不可篡改存证</span>
            </div>
            <p className="text-xs text-[#86868B] leading-relaxed">
              每次发起的研判推演，均以只读 JSON 即时持久化到数据库，附带时间戳凭证。
            </p>
          </div>
          <div className="p-4 rounded-xl bg-[#F5F5F7] border border-black/[0.03] space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-[#1D1D1F] text-white text-[11px] font-bold flex items-center justify-center">
                2
              </span>
              <span className="text-xs font-bold text-[#1D1D1F]">客观收盘核销</span>
            </div>
            <p className="text-xs text-[#86868B] leading-relaxed">
              到期后每日 18:00 自动调取真实收盘价比对核销，打上命中标记并计算 Brier 分数。
            </p>
          </div>
          <div className="p-4 rounded-xl bg-[#F5F5F7] border border-black/[0.03] space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-[#1D1D1F] text-white text-[11px] font-bold flex items-center justify-center">
                3
              </span>
              <span className="text-xs font-bold text-[#1D1D1F]">自学习模型进化</span>
            </div>
            <p className="text-xs text-[#86868B] leading-relaxed">
              评估对账流水自动回填至校准器再训练集，约束极端置信度，持续消除模型幻觉。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
