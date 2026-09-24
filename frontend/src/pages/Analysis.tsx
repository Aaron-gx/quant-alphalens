import React, { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Button, Table, message } from 'antd';
import { AppleSegmented } from '../components/AppleSegmented';
import {
  BrainCircuit,
  Play,
  Star,
  TrendingUp,
  User,
  PieChart,
  History,
  Shield,
  Layers,
  FileText,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Compass,
  Building2,
  Coins,
  CalendarDays,
  Flag,
  Gauge,
  Wallet,
  Cpu,
} from 'lucide-react';
import { api, QuoteData, PredictionData } from '../api/client';
import { KlineChart, ForecastPoint } from '../components/KlineChart';
import { AIGauge } from '../components/AIGauge';
import { VerdictCard } from '../components/VerdictCard';
import { ScenarioCards } from '../components/ScenarioCards';
import { ProbBar } from '../components/ProbBar';
import { MarkdownView } from '../components/MarkdownView';
import { ErrorBoundary } from '../components/ErrorBoundary';
import { DirectionVerdict } from '../components/DirectionVerdict';
// 周期步数 / 振幅解析 / 方向裁决口径统一来自 lib/forecast，本页不再自行实现，
// 否则"图上画的方向"与"文字说的方向"会各算各的。
import { HORIZON_TDAYS, HORIZON_LABELS, parseRangePct, detectInvertedProbs, forecastMarks, buildForecastPlan, type ForecastPlan } from '../lib/forecast';

// ─── 热门标的快捷切换列表 ───────────────────────────────────────────────────────
// 刻意**不带 defaultPrice**：原先每条都挂了一个写死的价格，并在行情/净值/K线
// 全部缺失时兜底显示，等价于给用户看一个假报价。价格只能来自接口。
const QUICK_TARGETS = [
  { symbol: '161725', name: '招商中证白酒', type: '基金/LOF', asset_type: 'fund' },
  { symbol: '005827', name: '易方达蓝筹精选', type: '混合型', asset_type: 'ofund' },
  { symbol: '000997', name: '新大陆', type: 'A股', asset_type: 'stock' },
  { symbol: '510300', name: '沪深300ETF', type: '场内ETF', asset_type: 'fund' },
  { symbol: '300750', name: '宁德时代', type: '创业板', asset_type: 'stock' },
  { symbol: '016665', name: '天弘全球高端制造', type: 'QDII', asset_type: 'ofund' },
];

/**
 * 「研判来源」明细行。
 * 值缺失时统一显示 '—'：这类卡片存在的意义就是让用户能核对结论的来路，
 * 任何一个字段靠猜或靠默认值补齐，整张卡的信任价值就归零了。
 */
const ProvRow: React.FC<{ label: string; value?: string | number | null }> = ({ label, value }) => (
  <div className="flex items-baseline justify-between gap-3 text-xs">
    <dt className="text-[#86868B] shrink-0">{label}</dt>
    <dd className="font-mono text-[#1D1D1F] text-right truncate" title={value == null ? '' : String(value)}>
      {value === undefined || value === null || value === '' ? '—' : value}
    </dd>
  </div>
);

// ─── 已删除的「演示兜底数据」(2026-09-24) ──────────────────────────────────────
// 这里原先有一套 DEFAULT_AI_REPORT / DEFAULT_AUDIT_HISTORY 常量，对 161725 这个
// **真实标的**兜底渲染：一段编造的「综合研判深度推演报告」（含"内置量化分类器
// (GBDT + 均线动量因子)给出胜率 62.4%"这类根本不存在的模型输出）、四条编造的
// 对账流水（命中率、Brier 分数都是手写的）、以及 `现价 × 0.98 / 1.08 / 0.94`
// 冒充的建仓-止盈-止损点位。
//
// 后果有两层：一是用户对着假研报做决策；二是任何一条真数据缺失时，界面上立刻
// 冒出一段"看起来很专业"的假文字，这正是用户指出的「露馅文字」。
// 现在一律改为如实的空状态，缺数据就说缺数据。

// ─── 由 horizons 概率推演未来价格中枢与区间（交易日）─────────────────────────
// 日期口径全部由 ../lib/forecast 的 ForecastPlan 提供：起算基准、各周期到期日、
// 预测区日期网格。本页只负责把 anchors 插值成一条连续的带状曲线。
// 为什么不再自己算日期：KlineChart 按**日期串**取预测值定位标注，取不到就静默丢弃，
// 所以"曲线日期"和"标注日期"必须是同一份数据。

function buildForecast(
  horizons: PredictionData['horizons'] | undefined,
  basePrice: number,
  plan: ForecastPlan | null,
): ForecastPoint[] | null {
  if (!horizons || !plan || !(basePrice > 0)) return null;
  const anchors: Array<{ idx: number; mid: number; lo: number; hi: number }> = [];
  for (const [key] of HORIZON_TDAYS) {
    const h = (horizons as any)?.[key];
    if (!h || typeof h.up !== 'number') continue;
    // 该周期在预测区日期网格里的下标。**取不到就跳过这个周期**，
    // 而不是硬塞一个"按交易日数估的位置"——那会画出一条位置错误的曲线。
    const idx = plan.steps[key];
    if (idx == null) continue;
    // 预期中枢 = 涨跌概率差 × 半幅振幅：概率差只代表方向信心，
    // 幅度由模型的预期振幅决定，避免把 0.2 的概率差误读成 20% 涨幅
    //
    // 振幅缺失时 halfBand = 0。旧实现写成 `... / 2 || 0.02`，会在模型压根没给
    // 振幅时凭空画出一条 ±2% 的区间阴影 —— 和"模型真的预测了 2% 波动"长得一模一样。
    // 没有振幅就不画区间，幅度一律不编；方向由 forecastMarks 的文字标注承担。
    const halfBand = parseRangePct(h.range_pct || h.range) / 2;
    const band = Number.isFinite(halfBand) && halfBand > 0 ? halfBand : 0;
    // 概率差必须涨、跌两侧都有值才成立；缺一侧就无从比较，中枢落在基准价上。
    const midPct = typeof h.down === 'number' ? (h.up - h.down) * band : 0;
    anchors.push({
      idx,
      mid: basePrice * (1 + midPct),
      lo: basePrice * (1 + midPct - band),
      hi: basePrice * (1 + midPct + band),
    });
  }
  if (!anchors.length) return null;
  anchors.sort((a, b) => a.idx - b.idx);

  // 起算点 = 基准价，位于网格下标 -1（即锚点日当天，图上表现为"衔接最后一日收盘"）
  const lerp = (i: number, pick: 'mid' | 'lo' | 'hi') => {
    let a = { idx: -1, mid: basePrice, lo: basePrice, hi: basePrice };
    let b = anchors[0];
    for (const an of anchors) {
      if (an.idx >= i) { b = an; break; }
      a = an; b = an;
    }
    const t = b.idx === a.idx ? 1 : (i - a.idx) / (b.idx - a.idx);
    return a[pick] + (b[pick] - a[pick]) * Math.max(0, Math.min(1, t));
  };
  return plan.dates.map((date, i) => ({
    date,
    mid: +lerp(i, 'mid').toFixed(4),
    low: +lerp(i, 'lo').toFixed(4),
    high: +lerp(i, 'hi').toFixed(4),
  }));
}

// ─── 指标小块 ────────────────────────────────────────────────────────────────
// tone 刻意不设默认值：不传 = 该指标没有可用数值，不做任何着色。
// 若默认成 'flat'，一个 value 为 '—' 的空卡片会被渲染成「中性」，
// 等于替一个从未计算过的指标下结论。
const IndTile: React.FC<{ label: string; value: React.ReactNode; sub?: string; tone?: 'up' | 'down' | 'flat' }> = ({
  label, value, sub, tone,
}) => (
  <div className="p-3.5 bg-white rounded-xl border border-black/[0.05]">
    <div className="text-[10px] font-medium text-[#86868B] mb-1">{label}</div>
    <div
      className={`text-sm font-bold font-mono ${
        tone === 'up' ? 'text-[#E03E3E]' : tone === 'down' ? 'text-[#34C759]' : 'text-[#1D1D1F]'
      }`}
    >
      {value}
    </div>
    {sub && <div className="text-[10px] text-[#A1A1A6] mt-0.5">{sub}</div>}
  </div>
);

export const Analysis: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const symbol = searchParams.get('symbol') || '161725';
  const typeParam = searchParams.get('type') || '';

  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [quote, setQuote] = useState<QuoteData | null>(null);
  const [fundProfile, setFundProfile] = useState<any>(null);
  const [klineData, setKlineData] = useState<any[]>([]);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [newsData, setNewsData] = useState<any>(null);
  const [accuracyHistory, setAccuracyHistory] = useState<any[]>([]);
  const [indicators, setIndicators] = useState<any>(null);
  const [isWatchlisted, setIsWatchlisted] = useState(false);
  const [watchlistType, setWatchlistType] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  const matchedQuick = QUICK_TARGETS.find((q) => q.symbol === symbol);
  // 资产类型提示：URL type 参数 > 快捷标的表 > 自选池记录 > 行情返回字段
  const assetTypeHint =
    typeParam || matchedQuick?.asset_type || watchlistType || (quote as any)?.asset_type || '';

  const isFund =
    ['fund', 'ofund'].includes(assetTypeHint) ||
    quote?.estimate_nav !== undefined ||
    fundProfile?.is_ofund ||
    fundProfile?.is_etf ||
    /^(16|15|51|50|56|58|11|01|05|06|07|08|09)/.test(symbol);
  const isOfund = assetTypeHint === 'ofund' || fundProfile?.is_ofund;

  const loadData = async (sym: string, hint: string) => {
    setLoading(true);
    // 清空上一个标的的残留数据，避免切标的瞬间旧 K 线/旧报价串到新标的
    setQuote(null);
    setKlineData([]);
    setFundProfile(null);
    setNewsData(null);
    setIndicators(null);
    setPrediction(null);
    setAccuracyHistory([]);
    const p = hint ? { asset_type: hint } : {};
    let quoteRes: any = null;
    try {
      // 1. 核心行情与预测记录优先获取，快速解除主卡片 loading。
      // /watchlist 是慢接口（后端逐个标的串行拉行情），移出关键路径放第二阶段。
      const [q, klineRes, predListRes] = await Promise.all([
        api.getDedup(`/quote/${sym}`, { params: p }).catch(() => null),
        api.getDedup(`/kline/${sym}`, { params: { count: 120, ...p } }).catch(() => []),
        api.getDedup('/predictions', { params: { symbol: sym, limit: 1 } }).catch(() => []),
      ]);
      quoteRes = q;

      if (quoteRes && (quoteRes as any).ok !== false) setQuote(quoteRes as QuoteData);
      if (Array.isArray(klineRes) && klineRes.length > 0) setKlineData(klineRes);
      if (Array.isArray(predListRes) && predListRes.length > 0) {
        const pid = predListRes[0].id;
        api.getDedup(`/predictions/${pid}`).then((pDetail: any) => {
          if (pDetail) setPrediction(pDetail);
        }).catch(() => null);
      } else {
        setPrediction(null);
      }
    } catch {
      message.error('加载标的数据异常');
    } finally {
      setLoading(false);
    }

    // 2. 自选池校验 + 档案/舆情/指标/对账，异步并发静默加载，不阻塞主屏
    api.getDedup('/watchlist')
      .then(async (watchRes: any) => {
        if (!Array.isArray(watchRes)) return;
        const w = watchRes.find((x: any) => x.symbol === sym);
        setIsWatchlisted(!!w);
        if (w?.asset_type) setWatchlistType(w.asset_type);
        // 自选池说它是场外基金但行情走了股票路径且没数据 → 带正确类型重取
        if (w?.asset_type === 'ofund' && !hint && !(quoteRes as any)?.estimate_nav && !(quoteRes as any)?.price) {
          const q2: any = await api.getDedup(`/quote/${sym}`, { params: { asset_type: 'ofund' } }).catch(() => null);
          if (q2 && q2.ok !== false) setQuote(q2);
          const k2: any = await api.getDedup(`/kline/${sym}`, { params: { count: 120, asset_type: 'ofund' } }).catch(() => null);
          if (Array.isArray(k2) && k2.length > 0) setKlineData(k2);
        }
      })
      .catch(() => null);

    const hint2 = hint || watchlistType;
    const p2 = hint2 ? { asset_type: hint2 } : {};
    // 已知是股票时跳过基金档案请求，避免无谓的 400 噪声
    if (hint2 !== 'stock') {
      api.getDedup('/fund/profile', { params: { symbol: sym, ...p2 } })
        .then((fp: any) => {
          if (fp && fp.ok !== false) setFundProfile(fp);
        })
        .catch(() => null);
    }

    api.getDedup(`/news/${sym}`, { params: p2 })
      .then((nRes: any) => {
        if (nRes) setNewsData(nRes);
      })
      .catch(() => null);

    api.getDedup(`/indicators/${sym}`, { params: p2 })
      .then((ind: any) => {
        if (ind && Object.keys(ind).length > 1) setIndicators(ind);
      })
      .catch(() => null);

    api.getDedup(`/analysis/${sym}/accuracy-history`)
      .then((accRes: any) => {
        if (Array.isArray(accRes) && accRes.length > 0) setAccuracyHistory(accRes);
        else setAccuracyHistory([]);
      })
      .catch(() => {
        setAccuracyHistory([]);
      });
  };

  useEffect(() => {
    loadData(symbol, typeParam || matchedQuick?.asset_type || '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  // 发起 AI 预测推演
  const handlePredict = async () => {
    setPredicting(true);
    message.loading({ content: 'AI 引擎正在多源采样与分层推演中…', key: 'pred_msg' });
    try {
      const res: any = await api.post('/predict', {
        symbol: symbol,
        risk_preference: 'neutral',
        mode: 'fast',
        asset_type: assetTypeHint || undefined,
      });
      message.success({ content: 'AI 研判推演完成！', key: 'pred_msg' });
      if (res?.id) {
        const pDetail: any = await api.get(`/predictions/${res.id}`);
        setPrediction(pDetail);
      }
    } catch (e: any) {
      message.error({ content: `预测失败: ${e.response?.data?.detail || e.message}`, key: 'pred_msg' });
    } finally {
      setPredicting(false);
    }
  };

  // 添加/移出自选
  const toggleWatchlist = async () => {
    try {
      if (isWatchlisted) {
        await api.delete(`/watchlist/${symbol}`);
        setIsWatchlisted(false);
        message.success('已移出重点监控池');
      } else {
        await api.post('/watchlist', {
          symbol,
          name: quote?.name || fundProfile?.name || matchedQuick?.name || symbol,
          asset_type: isFund ? (isOfund ? 'ofund' : 'fund') : 'stock',
        });
        setIsWatchlisted(true);
        message.success('已成功加入重点监控池');
      }
    } catch {
      message.error('操作自选池失败');
    }
  };

  // 标的当前价格与涨跌。
  // 2026-09-24 返工：原先最后两级兜底是 `matchedQuick?.defaultPrice || 0.533`，
  // 也就是行情、净值、K 线全部拿不到时，页面会**显示一个写死的价格**，
  // 并拿它去算预测中枢、区间与点位 —— 与真行情完全无法区分。现在明确归零，
  // 由渲染层显示 '—'（buildForecast 本身有 basePrice > 0 的保护）。
  const lastKlineClose = klineData[klineData.length - 1]?.close;
  const rawPrice =
    quote?.price ||
    quote?.last ||
    quote?.estimate_nav ||
    quote?.nav ||
    lastKlineClose ||
    0;

  const currentPrice = Number(rawPrice);
  const hasPrice = Number.isFinite(currentPrice) && currentPrice > 0;
  // 涨跌幅缺失时必须保持 null。旧实现兜底成 0.0，界面就渲染出一个
  // "+0.00%" 的涨跌徽章 —— 一个从未被观测到的数字，却带着和真实行情
  // 完全相同的样式和配色。
  const changePct: number | null =
    typeof quote?.change_pct === 'number'
      ? quote.change_pct
      : typeof quote?.estimate_pct === 'number'
      ? quote.estimate_pct
      : null;
  const isUp = changePct !== null && changePct > 0;
  const isDown = changePct !== null && changePct < 0;

  // 标的名称
  const targetName =
    quote?.name ||
    fundProfile?.name ||
    matchedQuick?.name ||
    symbol;

  const decimals = currentPrice < 10 ? 3 : 2;

  // 操作点位：**只认研判记录里真实给出的数值**。
  // 旧实现会在有记录但字段缺失时用 `现价 × 0.98 / 1.08 / 0.94` 现造三个点位顶上，
  // 界面上与真实建议完全无法区分 —— 用户会照着执行。缺就是缺，显示 '—'。
  const hasPred = !!prediction;
  const entryPrice = prediction?.action?.entry || '—';
  const targetPrice = prediction?.action?.target || '—';
  const stopLossPrice = prediction?.action?.stop_loss || '—';

  // ─── 预测区口径：起算基准 + 各周期到期日 + 日期网格 ──────────────────────────
  // 起算日必须是研判自己的 base_date（后端就是按它往后数交易日算到期日的）。
  // 旧实现拿"行情最后一日"当起算日：一条 base_date=09-24 的研判，
  // 图上「次日 T+1」会钉在 09-23 —— 一个**已经过去**的日期。
  // 到期日优先用后端给的 due_dates（含真实交易日历），前端只兜底排周末。
  const forecastPlan = useMemo(
    () => buildForecastPlan(
      prediction?.horizons as any,
      prediction?.base_date,
      (klineData[klineData.length - 1]?.date as any) || '',
      (prediction as any)?.due_dates,
      (prediction as any)?.due_source,
    ),
    [prediction, klineData],
  );

  // 未来预测曲线数据（叠加在走势图上）
  const forecast = useMemo(
    () => buildForecast(prediction?.horizons, currentPrice, forecastPlan),
    [prediction, currentPrice, forecastPlan],
  );

  // ─── 净值数据截止日与滞后天数 ───────────────────────────────────────────────
  // 场外基金（尤其 QDII）净值由基金公司延后公布，行情库里最新一条通常不是"今天"。
  // 不把这个事实写出来，用户就会读成"23 号的数据丢了/页面有 bug"。
  const dataAsOf = klineData[klineData.length - 1]?.date
    ? String(klineData[klineData.length - 1].date).slice(0, 10)
    : '';
  const dataLagDays = useMemo(() => {
    const m = dataAsOf.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!m) return 0;
    const d = new Date(+m[1], +m[2] - 1, +m[3]).getTime();
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    return Math.max(0, Math.round((today - d) / 86400000));
  }, [dataAsOf]);

  // ─── 校准是否真的生效 ───────────────────────────────────────────────────────
  // 后端 calibration.apply 在无可用样本时返回 method='identity'（恒等映射，概率原样返回）。
  // 这种情况下若仍挂"自学习已校准"徽标，等于用未校准的数字冒充已校准，必须按真实 method 显示。
  const calibrated = useMemo(() => {
    const c: any = (prediction as any)?.input_snapshot?.calibration;
    if (!c) return false;
    const arr = Array.isArray(c) ? c : Object.values(c);
    return arr.some((x: any) => x && typeof x.method === 'string' && x.method !== 'identity');
  }, [prediction]);

  // ─── 历史脏数据：涨/跌概率被对调（旧引擎标签顺序缺陷）──────────────────────
  // 这类记录的概率方向是反的，必须在界面上明确标注，不能当成正常结论展示。
  const probsInverted = useMemo(
    () => detectInvertedProbs((prediction as any)?.input_snapshot),
    [prediction],
  );

  // ─── 图上方向标注 ──────────────────────────────────────────────────────────
  // 中枢线在价格刻度上不足 3 像素（概率均衡时必然如此），方向只能靠文字落到图上。
  // 日期取自 forecastPlan —— 与曲线同源；若两处各算一遍，标注会因日期取不到值而静默消失。
  const marks = useMemo(
    () => forecastMarks(prediction?.horizons as any, forecastPlan),
    [prediction, forecastPlan],
  );

  // 后端情景字段是 probability/target_range/triggers，映射成卡片用的 prob/range/condition
  const scenarioCardsData = useMemo(() => {
    const s = prediction?.scenarios;
    if (!s) return undefined;
    const map = (x: any) =>
      x && {
        prob: x.prob ?? x.probability,
        range: x.range || x.target_range,
        condition: x.condition || (Array.isArray(x.triggers) ? x.triggers.join('；') : x.triggers),
      };
    return {
      optimistic: map(s.optimistic),
      base: map(s.base),
      pessimistic: map(s.pessimistic),
    };
  }, [prediction]);

  const horizonRange = (h: any) => h?.range_pct || h?.range || '';

  /**
   * 周期概率胶囊的数据源 —— 缺哪个周期就不渲染哪个周期。
   *
   * 旧实现给四个周期各挂了 `?? 0.52 / 0.28 / 0.2` 这类兜底数字，振幅缺失时
   * 还会补一个假的 '±1.5%'。后果：一条只有次日数据的记录，界面上会显示
   * **四个周期的完整概率**，其中三个是凭空填的，且与真实数据长得一模一样。
   * 这里改成纯派生：有就有，没有就不出现。
   */
  const probRows = useMemo(() => {
    const hs: any = prediction?.horizons;
    if (!hs) return [];
    const specs: Array<[string, string]> = [
      ['next_day', '次日超短 (T+1)'],
      ['one_week', '一周短期 (5日)'],
      ['one_month', '一月中期 (20日)'],
      ['quarter', '季度趋势 (60日)'],
    ];
    return specs.flatMap(([key, label]) => {
      const h = hs[key];
      if (!h || typeof h.up !== 'number' || typeof h.down !== 'number') return [];
      const flat = typeof h.flat === 'number' ? h.flat : 0;
      const total = h.up + flat + h.down;
      if (!(total > 0)) return [];
      return [{
        key,
        label,
        up: h.up / total,
        flat: flat / total,
        down: h.down / total,
        range: horizonRange(h),
      }];
    });
  }, [prediction]);

  // 舆情：只保留相关度达标的条目（标记 relevant 或分数≥0.3）
  const newsItems: any[] = useMemo(() => {
    const items = newsData?.items || [];
    return items
      .filter((it: any) => it.relevant !== false && (it.relevance ?? 0) >= 0.3)
      .sort((a: any, b: any) => (b.relevance ?? 0) - (a.relevance ?? 0));
  }, [newsData]);

  const holdings: any[] = fundProfile?.holdings || [];
  const stageReturns: Record<string, number> = fundProfile?.stage_returns || {};

  return (
    <div className="max-w-[1600px] mx-auto px-6 sm:px-8 py-6 space-y-6">

      {/* ── 顶部标的切换分段控制器 (Apple Segmented Tray) ── */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        <span className="text-xs font-semibold text-[#86868B] mr-1 flex items-center gap-1 shrink-0">
          <Compass className="w-3.5 h-3.5" />
          标的直达：
        </span>
        <div className="p-1 rounded-xl bg-[#E8E8ED]/70 border border-black/[0.04] inline-flex items-center gap-1 shrink-0">
          {QUICK_TARGETS.map((t) => {
            const isSelected = t.symbol === symbol;
            return (
              <button
                key={t.symbol}
                onClick={() => setSearchParams({ symbol: t.symbol, type: t.asset_type })}
                className={`apple-press flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all duration-150 select-none ${
                  isSelected
                    ? 'bg-white text-[#1D1D1F] font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.08)]'
                    : 'text-[#86868B] hover:text-[#1D1D1F] hover:bg-white/40'
                }`}
              >
                <span>{t.name}</span>
                <span className={`font-mono text-[11px] ${isSelected ? 'text-[#86868B]' : 'text-[#A1A1A6]'}`}>
                  {t.symbol}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── 标的行情信息卡片 (Apple Stocks Style Card) ── */}
      <div className="apple-card p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-5">
          {/* 左侧：标的名称 + 标签 + 核心资料 */}
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-[#1D1D1F] tracking-tight">
                {targetName}
              </h1>
              <span className="font-mono text-xs font-semibold text-[#86868B] bg-[#F5F5F7] px-2 py-0.5 rounded border border-black/[0.04]">
                {symbol}
              </span>
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-[#EBF5FF] text-[#0071E3] border border-[#CCE4FF]">
                {isOfund ? '场外开放基金' : isFund ? '场内基金 · ETF/LOF' : 'A股主板标的'}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-5 text-xs text-[#86868B] mt-2.5">
              {fundProfile?.estab_date && (
                <span className="flex items-center gap-1.5">
                  <CalendarDays className="w-3.5 h-3.5 text-[#86868B]" />
                  成立日期：<span className="font-mono text-[#1D1D1F]">{String(fundProfile.estab_date).slice(0, 10)}</span>
                </span>
              )}
              {fundProfile?.type && (
                <span className="flex items-center gap-1.5">
                  基金类型：<span className="text-[#1D1D1F] font-medium">{fundProfile.type}</span>
                </span>
              )}
              {fundProfile?.manager && (
                <span className="flex items-center gap-1.5">
                  <User className="w-3.5 h-3.5 text-[#0071E3]" />
                  基金经理：<b className="text-[#1D1D1F] font-medium">{fundProfile.manager}</b>
                </span>
              )}
              {fundProfile?.company && (
                <span className="flex items-center gap-1.5">
                  <Building2 className="w-3.5 h-3.5 text-[#86868B]" />
                  管理人：<span className="text-[#1D1D1F]">{fundProfile.company}</span>
                </span>
              )}
              {fundProfile?.size && (
                <span className="flex items-center gap-1.5">
                  <Coins className="w-3.5 h-3.5 text-[#86868B]" />
                  资产规模：<span className="font-mono text-[#1D1D1F]">{fundProfile.size}</span>
                </span>
              )}
              {quote?.turnover_rate !== undefined && quote.turnover_rate > 0 && (
                <span>换手率：<span className="font-mono text-[#1D1D1F] font-medium">{quote.turnover_rate.toFixed(2)}%</span></span>
              )}
            </div>
          </div>

          {/* 右侧：实时价格 + 涨跌幅 + Apple 极简操作按钮 */}
          <div className="flex items-center gap-5 self-start md:self-auto">
            <div className="text-right">
              <div className={`text-3xl font-extrabold font-mono tracking-tight ${hasPrice ? 'text-[#1D1D1F]' : 'text-[#A1A1A6]'}`}>
                {hasPrice ? currentPrice.toFixed(decimals) : '—'}
              </div>
              <div className="flex items-center justify-end mt-0.5">
                <span
                  className={`text-xs font-semibold font-mono flex items-center gap-0.5 px-2 py-0.5 rounded-full ${
                    isUp
                      ? 'bg-[#FFECEB] text-[#E03E3E]'
                      : isDown
                      ? 'bg-[#E8F8EE] text-[#34C759]'
                      : 'bg-[#F5F5F7] text-[#86868B]'
                  }`}
                >
                  {isUp && <ArrowUpRight className="w-3.5 h-3.5" />}
                  {isDown && <ArrowDownRight className="w-3.5 h-3.5" />}
                  {changePct === null && <Minus className="w-3.5 h-3.5" />}
                  {changePct === null ? '—' : `${changePct > 0 ? '+' : ''}${changePct.toFixed(2)}%`}
                </span>
              </div>
              {(quote as any)?.note && (
                <div className="text-[10px] text-[#A1A1A6] mt-1">{(quote as any).note}</div>
              )}
            </div>

            <div className="flex items-center gap-2">
              <Button
                icon={<Star className={`w-3.5 h-3.5 ${isWatchlisted ? 'text-[#FF9500] fill-[#FF9500]' : 'text-[#86868B]'}`} />}
                onClick={toggleWatchlist}
                className="apple-press !rounded-xl !border-black/[0.08] !h-9 !px-3.5 !bg-[#F5F5F7] hover:!bg-[#E8E8ED] !text-[#1D1D1F] !text-xs !font-medium"
              >
                {isWatchlisted ? '已关注' : '加入自选'}
              </Button>
              <Button
                type="primary"
                loading={predicting}
                icon={<Play className="w-3.5 h-3.5 text-white fill-current" />}
                onClick={handlePredict}
                className="apple-press !bg-[#0071E3] hover:!bg-[#0077ED] !rounded-full !font-medium !h-9 !px-5 !text-white !text-xs shadow-sm border-none"
              >
                生成时序研判
              </Button>
            </div>
          </div>
        </div>
      </div>

            {/* ── 视图分段控制器 (Apple Segmented Tray) ── */}
      <div className="flex items-center overflow-x-auto pb-1">
        <AppleSegmented
          options={[
            { value: 'overview', label: '综合概览', icon: <Activity className="w-3.5 h-3.5" /> },
            { value: 'technical', label: '技术指标与盘口', icon: <Activity className="w-3.5 h-3.5" /> },
            { value: 'fundamental', label: '基本面与持仓', icon: <PieChart className="w-3.5 h-3.5" /> },
            { value: 'news', label: '舆情与相关性', icon: <FileText className="w-3.5 h-3.5" /> },
            { value: 'predict_detail', label: '推理链与深度研报', icon: <BrainCircuit className="w-3.5 h-3.5" /> },
            { value: 'backtest', label: '历史预测对账', icon: <History className="w-3.5 h-3.5" /> },
          ]}
          value={activeTab}
          onChange={setActiveTab}
        />
      </div>

      <div className="apple-card p-6">
        <ErrorBoundary key={symbol + activeTab} label="该视图">
        {activeTab === 'overview' && (
          <div className="space-y-6 pt-3">
                  {/* 第一排：左侧大图 (2/3) + 右侧 置信度+裁决卡 (1/3)，高度对齐 */}
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                    <div className="lg:col-span-8">
                      <KlineChart
                        key={symbol}
                        data={klineData}
                        title={`${targetName} · 行情走势`}
                        isNav={isOfund || (isFund && quote?.estimate_nav !== undefined && assetTypeHint !== 'fund')}
                        forecast={forecast}
                        forecastMarks={marks}
                        badge={forecast ? '含 AI 未来预测' : undefined}
                        height={430}
                      />
                      {!forecast && (
                        <div className="mt-2 text-[11px] text-[#A1A1A6] flex items-center gap-1.5">
                          <TrendingUp className="w-3 h-3" />
                          点击右上角「生成时序研判」后，走势图右端会叠加未来 1/5/20/60 个交易日的预测中枢与波动区间
                        </div>
                      )}
                      {!!dataAsOf && (
                        <div className="mt-2 text-[11px] text-[#A1A1A6] flex items-start gap-1.5">
                          <CalendarDays className="w-3 h-3 mt-px shrink-0" />
                          <span>
                            行情/净值数据截至 <span className="font-mono text-[#6E6E73]">{dataAsOf}</span>
                            {dataLagDays > 0 && <>（距今 {dataLagDays} 天）</>}
                            {(isOfund || isFund) &&
                              ' · 场外基金净值由基金公司 T+1～T+2 公布，当日净值通常要到次日才可见，因此最新一条不是「今天」属正常，不是数据缺失'}
                          </span>
                        </div>
                      )}
                      {/* 预测区的起算基准与各周期到期日：这两个日期必须写在页面上。
                          否则用户只能看到几条标注，无从判断"预测的是哪段时间"，
                          也无法核对"图上说这天到期"与对账记录是否同一个日期。 */}
                      {forecastPlan && (
                        <div className="mt-2 text-[11px] text-[#A1A1A6] flex items-start gap-1.5">
                          <Flag className="w-3 h-3 mt-px shrink-0" />
                          <span>
                            预测自 <span className="font-mono text-[#6E6E73]">{forecastPlan.anchor.date}</span> 起算
                            （{forecastPlan.anchor.source === 'base_date' ? '研判基准日' : '行情最后一日兜底'}）
                            {forecastPlan.gapDays > 0 && (
                              <>，比行情最后一日晚 {forecastPlan.gapDays} 天（这几个交易日的数据未纳入本次研判）</>
                            )}
                            {'；各周期按'}
                            <span className="font-mono text-[#6E6E73]">交易日</span>
                            计到期日 ——{' '}
                            {HORIZON_TDAYS.map(([key]) => {
                              const d = forecastPlan.dueDates[key];
                              if (!d) return null;
                              return (
                                <span key={key} className="mr-2.5 whitespace-nowrap">
                                  {HORIZON_LABELS[key]?.replace(/\s.*/, '') ?? key}{' '}
                                  <span className="font-mono text-[#6E6E73]">{d.slice(5)}</span>
                                </span>
                              );
                            })}
                            <span className="text-[#A1A1A6]">
                              （{forecastPlan.dueOrigin === 'backend' && forecastPlan.dueCalendar === 'index'
                                ? '已按真实交易日历跳过节假日'
                                : '未来的法定节假日尚未可知，仅排除了周末，实际交割日可能早 1~几天'}）
                            </span>
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="lg:col-span-4 space-y-4 flex flex-col">
                      <AIGauge
                        score={prediction?.confidence ?? null}
                        title="预测置信度"
                      />
                      <div className="flex-1">
                        <VerdictCard
                          advice={prediction?.action?.advice}
                          entry={entryPrice}
                          target={targetPrice}
                          stopLoss={stopLossPrice}
                          risks={prediction?.risks || []}
                        />
                      </div>
                    </div>
                  </div>

                  {/* 方向裁决条：把 K 线上那条"看不懂的紫线"翻译成明确的涨/跌/震荡结论 */}
                  <DirectionVerdict
                    horizons={prediction?.horizons as any}
                    baseDate={prediction?.base_date}
                    dataAsOf={dataAsOf}
                    dueDates={forecastPlan?.dueDates}
                    dueOrigin={forecastPlan?.dueOrigin}
                    dueCalendar={forecastPlan?.dueCalendar}
                    inverted={probsInverted}
                  />

                  {/* 第二排：三情景推演 + 右侧概率分布与雷达并排，消除左空右挤 */}
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
                    <div className="lg:col-span-8">
                      <div className="text-xs font-semibold text-[#86868B] uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                        <Layers className="w-3.5 h-3.5 text-[#0071E3]" />
                        情景路径推演
                      </div>
                      {hasPred ? (
                        <ScenarioCards scenarios={scenarioCardsData} />
                      ) : (
                        <div className="apple-card p-8 text-center text-xs text-[#86868B]">
                          尚未生成该标的的 AI 研判 —— 点击右上角「生成时序研判」后展示三情景路径
                        </div>
                      )}
                    </div>
                    <div className="lg:col-span-4">
                      {/* 原先这里挂的是 <RadarChart />（五个写死的常数 68/75/82/60/70），
                          但后端从未提供过任何五维分值 —— 那张图百分之百是装饰性的假数据，
                          却标着「五维多因子雷达 / 综合量化分值」。
                          改为展示这条研判的**真实出身**：引擎、模式、模型、生成时间、
                          基准日/价、校准口径。没有研判就直说没有。 */}
                      <div className="apple-card p-5">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-xs font-semibold text-[#86868B] uppercase tracking-wider flex items-center gap-1.5">
                            <Cpu className="w-3.5 h-3.5 text-[#0071E3]" />
                            研判来源
                          </span>
                          {hasPred && (
                            <span className="text-[11px] text-[#86868B] font-mono">#{prediction?.id}</span>
                          )}
                        </div>
                        {hasPred ? (
                          <dl className="space-y-2.5">
                            <ProvRow label="推演引擎" value={prediction?.engine} />
                            <ProvRow label="运行模式" value={prediction?.mode} />
                            <ProvRow label="语言模型" value={prediction?.llm_model} />
                            <ProvRow
                              label="生成时间"
                              value={prediction?.created_at
                                ? String(prediction.created_at).replace('T', ' ').slice(0, 19)
                                : ''}
                            />
                            <ProvRow
                              label="预测基准日"
                              value={prediction?.base_date ? String(prediction.base_date).slice(0, 10) : ''}
                            />
                            <ProvRow label="基准价" value={prediction?.base_price} />
                            <ProvRow
                              label="概率校准"
                              value={calibrated ? 'walk-forward 已生效' : 'identity（无样本，未校准）'}
                            />
                          </dl>
                        ) : (
                          <div className="py-12 text-center text-xs text-[#86868B]">
                            尚无研判记录 —— 生成后此处展示该条研判的引擎、模型与基准日
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 第三排：四周期涨跌概率胶囊（整行宽） */}
                  <div className="p-5 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-semibold text-[#86868B] uppercase tracking-wider flex items-center gap-1.5">
                        <TrendingUp className="w-3.5 h-3.5 text-[#0071E3]" />
                        周期概率分布
                      </span>
                      {prediction?.base_date && (
                        <span className="text-[11px] text-[#A1A1A6] font-mono">
                          基准日 {String(prediction.base_date).slice(0, 10)}
                        </span>
                      )}
                    </div>
                    {probRows.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
                      {probRows.map((r) => (
                        <ProbBar
                          key={r.key}
                          label={r.label}
                          up={r.up}
                          flat={r.flat}
                          down={r.down}
                          range={r.range}
                          calibrated={calibrated}
                        />
                      ))}
                    </div>
                    ) : (
                      <div className="py-6 text-center text-xs text-[#86868B]">
                        该条研判没有可用的周期概率 —— 生成时序研判后展示 T+1 / 5日 / 20日 / 60日 涨跌概率
                      </div>
                    )}
                  </div>
                </div>
        )}
        {activeTab === 'technical' && (
          <div className="space-y-5 pt-3">
                  {/* 技术指标矩阵 */}
                  <div className="p-5 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-xs font-semibold text-[#86868B] uppercase tracking-wider flex items-center gap-1.5">
                        <Gauge className="w-3.5 h-3.5 text-[#0071E3]" />
                        技术指标快照
                      </h4>
                      <span className="text-[10px] text-[#A1A1A6] font-mono">
                        {indicators?.date ? `截至 ${indicators.date}` : ''}
                      </span>
                    </div>
                    {indicators ? (
                      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5">
                        <IndTile label="均线形态" value={indicators.ma_alignment || '—'}
                          sub={`MA5 ${indicators.ma5 ?? '—'} / MA20 ${indicators.ma20 ?? '—'} / MA60 ${indicators.ma60 ?? '—'}`}
                          tone={indicators.ma_alignment === '多头排列' ? 'up' : indicators.ma_alignment === '空头排列' ? 'down' : undefined} />
                        <IndTile label="MACD" value={indicators.macd_hist != null ? (indicators.macd_hist > 0 ? '红柱' : indicators.macd_hist < 0 ? '绿柱' : '走平') : '—'}
                          sub={`DIF ${indicators.macd_dif ?? '—'} / DEA ${indicators.macd_dea ?? '—'}`}
                          tone={indicators.macd_hist != null ? (indicators.macd_hist > 0 ? 'up' : indicators.macd_hist < 0 ? 'down' : undefined) : undefined} />
                        <IndTile label="RSI (6/14)" value={indicators.rsi6 != null ? `${indicators.rsi6} / ${indicators.rsi14 ?? '—'}` : '—'}
                          sub={indicators.rsi14 != null ? (indicators.rsi14 > 70 ? '偏超买' : indicators.rsi14 < 30 ? '偏超卖' : '中性区间') : ''}
                          tone={indicators.rsi14 != null ? (indicators.rsi14 > 70 ? 'down' : indicators.rsi14 < 30 ? 'up' : 'flat') : undefined} />
                        <IndTile label="KDJ (K/D/J)" value={indicators.kdj_k != null ? `${indicators.kdj_k} / ${indicators.kdj_d ?? '—'} / ${indicators.kdj_j ?? '—'}` : '—'}
                          tone={indicators.kdj_j != null ? (indicators.kdj_j > 80 ? 'up' : indicators.kdj_j < 20 ? 'down' : 'flat') : undefined} />
                        <IndTile label="布林带 (20,2)" value={indicators.boll_mid != null ? `${indicators.boll_dn} ~ ${indicators.boll_up}` : '—'}
                          sub={`中轨 ${indicators.boll_mid ?? '—'}`} />
                        <IndTile label="近 20 日区间" value={indicators.high_20d != null ? `${indicators.low_20d} ~ ${indicators.high_20d}` : '—'}
                          sub="支撑 / 压力参考位" />
                        <IndTile label="近 1 日 / 5 日动量" value={indicators.chg_1d_pct != null ? `${indicators.chg_1d_pct}% / ${indicators.chg_5d_pct ?? '—'}%` : '—'}
                          tone={indicators.chg_5d_pct != null ? (indicators.chg_5d_pct > 0 ? 'up' : indicators.chg_5d_pct < 0 ? 'down' : 'flat') : undefined} />
                        <IndTile label="近 20 日 / 60 日动量" value={indicators.chg_20d_pct != null ? `${indicators.chg_20d_pct}% / ${indicators.chg_60d_pct ?? '—'}%` : '—'}
                          tone={indicators.chg_20d_pct != null ? (indicators.chg_20d_pct > 0 ? 'up' : indicators.chg_20d_pct < 0 ? 'down' : 'flat') : undefined} />
                        {indicators.vol_ratio != null && (
                          <IndTile label="量比 (5日)" value={indicators.vol_ratio}
                            sub={indicators.vol_ratio > 1.5 ? '明显放量' : indicators.vol_ratio < 0.7 ? '明显缩量' : '量能平稳'} />
                        )}
                      </div>
                    ) : (
                      <div className="py-10 text-center text-xs text-[#86868B]">指标数据加载中…（需有可用净值/K线序列）</div>
                    )}
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                    <div className="lg:col-span-8">
                      <KlineChart key={`t-${symbol}`} data={klineData} title="高低点均线系统与通道" forecast={forecast} />
                    </div>
                    <div className="lg:col-span-4 p-5 bg-[#F5F5F7] rounded-xl border border-black/[0.03] flex flex-col justify-between">
                      <div>
                        <h4 className="text-xs font-semibold text-[#86868B] uppercase tracking-wider mb-3">
                          五档买卖深度盘口
                        </h4>
                        {quote?.asks && quote.asks.length > 0 ? (
                          <div className="space-y-2 text-xs font-mono">
                            {quote.asks.slice().reverse().map((a, i) => (
                              <div key={i} className="flex justify-between text-[#34C759]">
                                <span>卖 {a.level}</span>
                                <span className="font-bold">{a.price?.toFixed(decimals)}</span>
                                <span className="text-[#86868B]">{a.volume_lots} 手</span>
                              </div>
                            ))}
                            <div className="h-px bg-black/[0.06] my-2" />
                            {quote.bids?.map((b, i) => (
                              <div key={i} className="flex justify-between text-[#E03E3E]">
                                <span>买 {b.level}</span>
                                <span className="font-bold">{b.price?.toFixed(decimals)}</span>
                                <span className="text-[#86868B]">{b.volume_lots} 手</span>
                              </div>
                            ))}
                          </div>
                        ) : isOfund ? (
                          // 场外基金没有盘口，改展示估值/净值卡
                          <div className="space-y-3">
                            <div className="p-3.5 bg-white rounded-xl border border-black/[0.05]">
                              <div className="text-[10px] text-[#86868B] mb-1">盘中估值 / 最新净值</div>
                              <div className={`text-xl font-extrabold font-mono ${hasPrice ? 'text-[#1D1D1F]' : 'text-[#A1A1A6]'}`}>
                                {hasPrice ? currentPrice.toFixed(decimals) : '—'}
                              </div>
                              <div className={`text-xs font-mono font-semibold mt-0.5 ${isUp ? 'text-[#E03E3E]' : isDown ? 'text-[#34C759]' : 'text-[#86868B]'}`}>
                                {changePct === null ? '—' : `${changePct > 0 ? '+' : ''}${changePct.toFixed(2)}%`}
                              </div>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-center">
                              <div className="p-2.5 bg-white rounded-lg border border-black/[0.05]">
                                <div className="text-[10px] text-[#86868B]">净值日期</div>
                                <div className="text-xs font-mono font-semibold text-[#1D1D1F] mt-0.5">{(quote as any)?.nav_date || (quote as any)?.estimate_time?.slice(0, 10) || '—'}</div>
                              </div>
                              <div className="p-2.5 bg-white rounded-lg border border-black/[0.05]">
                                <div className="text-[10px] text-[#86868B]">前十大集中度</div>
                                <div className="text-xs font-mono font-semibold text-[#0071E3] mt-0.5">
                                  {fundProfile?.top10_weight ? `${fundProfile.top10_weight}%` : '—'}
                                </div>
                              </div>
                            </div>
                            <div className="text-[10px] text-[#86868B] leading-relaxed">
                              场外基金按净值申赎，没有五档盘口；上表改以估值/净值与持仓集中度替代。
                            </div>
                          </div>
                        ) : (
                          <div className="py-16 text-center text-xs text-[#86868B] space-y-1">
                            <div>当前非交易时段或盘口源暂不可用</div>
                            <div className="text-[11px] text-[#A1A1A6]">（交易时段自动呈现毫秒级五档挂单）</div>
                          </div>
                        )}
                      </div>

                      <div className="pt-3 border-t border-black/[0.04] text-[11px] text-[#86868B] text-center">
                        快照时间：{(quote as any)?.update_time || (quote as any)?.estimate_time || '今日收盘快照'}
                      </div>
                    </div>
                  </div>
                </div>
        )}
        {activeTab === 'fundamental' && (
          <div className="space-y-5 pt-3">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-5 bg-[#F5F5F7] rounded-xl border border-black/[0.03] text-center">
                    <div>
                      <div className="text-xs text-[#86868B] font-medium mb-1">基金管理人</div>
                      <div className="text-sm font-bold text-[#1D1D1F]">{fundProfile?.company || '—'}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[#86868B] font-medium mb-1">现任基金经理</div>
                      <div className="text-sm font-bold text-[#0071E3]">{fundProfile?.manager || '—'}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[#86868B] font-medium mb-1">资产净值规模</div>
                      <div className="text-sm font-bold text-[#1D1D1F] font-mono">{fundProfile?.size || '—'}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[#86868B] font-medium mb-1">跟踪基准指数</div>
                      <div className="text-sm font-bold text-[#1D1D1F]">{fundProfile?.benchmark || (fundProfile?.tracking?.index ?? '—')}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[#86868B] font-medium mb-1">基金类型</div>
                      <div className="text-sm font-bold text-[#1D1D1F]">{fundProfile?.type || '—'}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[#86868B] font-medium mb-1">成立日期</div>
                      <div className="text-sm font-bold text-[#1D1D1F] font-mono">{fundProfile?.estab_date ? String(fundProfile.estab_date).slice(0, 10) : '—'}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[#86868B] font-medium mb-1">前十大持仓集中度</div>
                      <div className="text-sm font-bold text-[#0071E3] font-mono">
                        {fundProfile?.top10_weight ? `${fundProfile.top10_weight}%` : '—'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-[#86868B] font-medium mb-1">跟踪主题</div>
                      <div className="text-sm font-bold text-[#1D1D1F]">
                        {(fundProfile?.tracking?.themes || []).join('、') || '—'}
                      </div>
                    </div>
                  </div>

                  {/* 阶段收益：用真实净值序列计算，直接回答"最近涨没涨" */}
                  <div className="p-5 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
                    <h4 className="text-xs font-semibold text-[#86868B] uppercase tracking-wider mb-3 flex items-center gap-1.5">
                      <Wallet className="w-3.5 h-3.5 text-[#0071E3]" />
                      阶段收益（按净值序列计算）
                    </h4>
                    {Object.keys(stageReturns).length > 0 ? (
                      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2.5 text-center">
                        {Object.entries(stageReturns).map(([k, v]) => (
                          <div key={k} className="p-3 bg-white rounded-xl border border-black/[0.05]">
                            <div className="text-[10px] text-[#86868B] mb-1">{k}</div>
                            <div className={`text-sm font-bold font-mono ${v > 0 ? 'text-[#E03E3E]' : v < 0 ? 'text-[#34C759]' : 'text-[#1D1D1F]'}`}>
                              {v > 0 ? `+${v}%` : `${v}%`}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="py-6 text-center text-xs text-[#86868B]">暂无阶段收益数据（净值序列不可用）</div>
                    )}
                  </div>

                  <div className="apple-card p-5">
                    <h4 className="text-xs font-semibold text-[#86868B] uppercase tracking-wider mb-3">
                      前十大重仓股票持仓明细（季报披露{holdings[0]?.quarter ? ` · ${holdings[0].quarter}` : ''}）
                    </h4>
                    {holdings.length > 0 ? (
                      <Table
                        dataSource={holdings.map((h: any, i: number) => ({ ...h, key: i, rank: i + 1 }))}
                        pagination={false}
                        size="middle"
                        columns={[
                          { title: '序号', dataIndex: 'rank', key: 'rank', width: 70 },
                          { title: '股票代码', dataIndex: 'code', key: 'code', render: (s) => <span className="font-mono text-[#86868B]">{s || '—'}</span> },
                          { title: '股票名称', dataIndex: 'name', key: 'name', render: (t) => <b className="text-[#1D1D1F]">{t || '—'}</b> },
                          {
                            title: '占净值比',
                            dataIndex: 'weight',
                            key: 'weight',
                            render: (v) => (v != null ? <span className="font-mono font-bold text-[#0071E3]">{Number(v).toFixed(2)}%</span> : '—'),
                          },
                          {
                            title: '持仓市值',
                            dataIndex: 'value',
                            key: 'value',
                            render: (v) => (v != null ? <span className="font-mono text-xs text-[#86868B]">{Number(v).toFixed(0)}万</span> : '—'),
                          },
                        ]}
                      />
                    ) : (
                      <div className="py-10 text-center text-xs text-[#86868B]">
                        暂未获取到该标的的季报持仓明细（非基金标的或披露期未更新）
                      </div>
                    )}
                  </div>
                </div>
        )}
        {activeTab === 'news' && (
          <div className="space-y-4 pt-3">
                  <div className="bg-[#EBF5FF] rounded-xl border border-[#CCE4FF] p-4 flex items-center justify-between text-xs gap-3">
                    <div className="min-w-0">
                      <span className="font-semibold text-[#0071E3]">舆情范围穿透策略：</span>
                      <span className="text-[#1D1D1F]">
                        {newsData?.scope || (isFund ? '穿透至基金重仓股与跟踪主题，泛市场噪声自动剔除' : '仅保留与该标的直接相关的资讯')}
                      </span>
                      {newsData?.signal && (
                        <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[10px]">
                          <span className="text-[#0071E3] bg-white px-2 py-0.5 rounded-full border border-[#CCE4FF] font-semibold">
                            标的舆情分 {newsData.signal.score}
                          </span>
                          <span className="text-[#86868B] bg-white px-2 py-0.5 rounded-full border border-black/[0.05]">
                            全市场情绪 {newsData.signal.universe_score}
                          </span>
                          <span className="text-[#86868B] bg-white px-2 py-0.5 rounded-full border border-black/[0.05]">
                            入选 {newsData.signal.relevant_count ?? newsItems.length} / 剔除噪声 {newsData.signal.dropped_count ?? 0} 条
                          </span>
                        </div>
                      )}
                    </div>
                    <span className="text-[11px] font-semibold text-[#0071E3] bg-white px-2.5 py-0.5 rounded-full border border-[#CCE4FF] shrink-0">
                      入选 {newsItems.length} 条
                    </span>
                  </div>

                  <div className="space-y-2.5">
                    {newsItems.length > 0 ? (
                      newsItems.map((it: any, i: number) => {
                        const senti = it.sentiment || '中性';
                        const isBull = senti === '乐观';
                        const isBear = senti === '悲观';
                        return (
                          <div
                            key={i}
                            className="p-4 apple-card apple-card-hover flex items-start justify-between gap-3 text-xs"
                          >
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                                <span
                                  className={`px-2 py-0.5 rounded-full font-semibold text-[10px] ${
                                    isBull
                                      ? 'bg-[#FFECEB] text-[#E03E3E] border border-[#FFD3D0]'
                                      : isBear
                                      ? 'bg-[#E8F8EE] text-[#107C41] border border-[#C8F0D5]'
                                      : 'bg-[#F5F5F7] text-[#86868B]'
                                  }`}
                                >
                                  {isBull ? '偏多利好' : isBear ? '偏空承压' : '中性资讯'}
                                </span>
                                {it.relevance != null && (
                                  <span className="text-[10px] text-[#0071E3] font-mono font-semibold bg-[#EBF5FF] px-2 py-0.5 rounded-full">
                                    相关度 {Number(it.relevance).toFixed(2)}
                                  </span>
                                )}
                                <span className="text-[#86868B] text-[11px]">
                                  {it.source} · {String(it.publish_time || '').slice(0, 16)}
                                </span>
                              </div>
                              <p className="text-[#1D1D1F] font-medium text-xs leading-relaxed">
                                {it.title}
                              </p>
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="py-12 text-center text-xs text-[#86868B] space-y-1.5">
                        <div>暂无与「{targetName}」相关度达标的舆情</div>
                        <div className="text-[11px] text-[#A1A1A6]">
                          系统已按舆情范围穿透抓取，低于相关度门槛的泛市场快讯不会展示，避免噪声误导
                        </div>
                      </div>
                    )}
                  </div>
                </div>
        )}
        {activeTab === 'predict_detail' && (
          <div className="space-y-5 pt-3">
                  <div className="p-5 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-xs font-semibold text-[#86868B] uppercase tracking-wider">
                        综合研报
                      </h4>

                    </div>
                    <div className="p-4 bg-white/60 rounded-xl border border-black/[0.04]">
                      {prediction?.summary ? (
                        <MarkdownView content={prediction.summary} />
                      ) : (
                        <div className="py-8 text-center text-xs text-[#86868B]">
                          尚无该标的的研判报告，点击右上角「生成时序研判」生成
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="apple-card p-5">
                    <h4 className="text-xs font-semibold text-[#86868B] uppercase tracking-wider mb-3 flex items-center gap-1.5">
                      <TrendingUp className="w-3.5 h-3.5 text-[#0071E3]" />
                      核心驱动因子
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {(prediction?.key_factors || []).map((kf: string, i: number) => (
                        <div key={i} className="p-3 bg-[#F5F5F7] rounded-xl border border-black/[0.03] flex items-start gap-2.5 text-xs text-[#1D1D1F]">
                          <span className="w-1.5 h-1.5 rounded-full bg-[#0071E3] mt-1.5 shrink-0" />
                          <span>{kf}</span>
                        </div>
                      ))}
                      {!prediction?.key_factors?.length && (
                        <div className="text-xs text-[#86868B] py-4">生成研判后展示</div>
                      )}
                    </div>
                  </div>

                  <div className="bg-[#FFECEB]/50 p-5 rounded-xl border border-[#FFD3D0]/60">
                    <h4 className="text-xs font-semibold text-[#E03E3E] uppercase tracking-wider mb-3 flex items-center gap-1.5">
                      <Shield className="w-3.5 h-3.5 text-[#E03E3E]" />
                      潜在风险提示与证伪边界
                    </h4>
                    <div className="space-y-2">
                      {(prediction?.risks || []).map((rk: string, i: number) => (
                        <div key={i} className="flex items-start gap-2 text-xs text-[#E03E3E]">
                          <span className="w-1.5 h-1.5 rounded-full bg-[#E03E3E] mt-1.5 shrink-0" />
                          <span>{rk}</span>
                        </div>
                      ))}
                      {!prediction?.risks?.length && (
                        <div className="text-xs text-[#E03E3E]/70 py-2">生成研判后展示</div>
                      )}
                    </div>
                  </div>
                </div>
        )}
        {activeTab === 'backtest' && (
          <div className="space-y-4 pt-3">
                  <div className="apple-card p-5">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-xs font-semibold text-[#86868B] uppercase tracking-wider">
                        该标的历史预测到期履约对账流水
                      </h4>
                      <span className="text-[11px] text-[#86868B]">客观收盘核销记录</span>
                    </div>

                    {accuracyHistory.length > 0 ? (
                      <Table
                        dataSource={accuracyHistory.map((a, i) => ({ ...a, key: i }))}
                        size="middle"
                        pagination={false}
                        columns={[
                          { title: '对账到期日', dataIndex: 'date', key: 'date', render: (d) => <span className="font-mono text-xs text-[#86868B]">{d}</span> },
                          { title: '预测周期', dataIndex: 'horizon', key: 'horizon', render: (h) => <span className="font-semibold text-xs text-[#1D1D1F]">{h}</span> },
                          { title: '模型推演方向', dataIndex: 'predicted_direction', key: 'predicted_direction', render: (dir) => <span className="text-xs font-medium">{dir || '—'}</span> },
                          {
                            title: '到期真实涨跌',
                            dataIndex: 'actual_change_pct',
                            key: 'actual_change_pct',
                            render: (v: number) => (
                              <span className={`font-mono font-bold text-xs ${v > 0 ? 'text-[#E03E3E]' : 'text-[#34C759]'}`}>
                                {v > 0 ? `+${v.toFixed(2)}%` : `${v.toFixed(2)}%`}
                              </span>
                            ),
                          },
                          {
                            title: '履约对账结论',
                            dataIndex: 'hit',
                            key: 'hit',
                            render: (h: boolean) => (
                              <span className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${h ? 'bg-[#E8F8EE] text-[#107C41] border-[#C8F0D5]' : 'bg-[#FFECEB] text-[#E03E3E] border-[#FFD3D0]'}`}>
                                {h ? '✓ 命中预期' : '✕ 未命中'}
                              </span>
                            ),
                          },
                          {
                            title: 'Brier 校准分',
                            dataIndex: 'brier_score',
                            key: 'brier_score',
                            render: (b: number) => <span className="font-mono text-xs font-bold text-[#1D1D1F]">{b != null ? b.toFixed(3) : '—'}</span>,
                          },
                        ]}
                      />
                    ) : (
                      <div className="py-10 text-center text-xs text-[#86868B]">
                        该标的还没有到期的预测对账记录（生成研判后，到期自动核销回写）
                      </div>
                    )}
                  </div>
                </div>
        )}
        </ErrorBoundary>
      </div>
    </div>
  );
};
