import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Spin, message } from 'antd';
import {
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Briefcase,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
  Clock,
  ChevronRight,
  Plus,
  Radio,
  Newspaper,
  Minus,
  FileText,
  Search,
} from 'lucide-react';
import { api, WatchlistItem } from '../api/client';
import { AppleModal } from '../components/AppleModal';
import { MarkdownView } from '../components/MarkdownView';

// ─── 已删除的四组「演示兜底数据」(2026-09-24) ────────────────────────────────────
// 这里原先有 MOCK_INDICES（8 条指数点位与涨跌幅）、CURATED_DEFAULT_TARGETS
// （4 只真实标的的价格 + 5 点走势）、MOCK_BRIEFING（一整篇"晨会内参"，里面甚至
// 写了「招商白酒 近 5 日净值偏离率收窄至 +0.3%，模型给出'温和加仓'信号，
// 置信度 72.4%」「北向资金当前值 -28.5 亿，已进入黄色预警区间」这类**根本不存在的
// 模型输出**）、MOCK_FLASH_NEWS（8 条编造的财经快讯，带假的时间戳）。
//
// 这四组数据的兜底条件都是「真数据为空」，也就是说：接口一挂，页面反而开始
// 稳定输出一套完整的假行情 + 假研报 + 假快讯，且与真数据在视觉上无法区分。
// 用户指出的「露馅文字」在首页最典型的表现就是这条。
// 现在全部改为如实空态。


// 微型迷你走势图组件 (Apple Stocks Mini Sparkline)
const MiniSparkline: React.FC<{ points: number[]; isUp: boolean }> = ({ points, isUp }) => {
  if (!points || points.length < 2) return null;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const width = 64;
  const height = 24;

  const coords = points.map((p, i) => {
    const x = (i / (points.length - 1)) * width;
    const y = height - ((p - min) / range) * (height - 6) - 3;
    return `${x},${y}`;
  });

  const pathD = `M ${coords.join(' L ')}`;
  const strokeColor = isUp ? '#E03E3E' : '#34C759';

  return (
    <svg width={width} height={height} className="overflow-visible">
      <path
        d={pathD}
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

export const Overview: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [briefing, setBriefing] = useState<{ content: string; created_at: string }>({
    content: '',
    created_at: '',
  });
  const [stats, setStats] = useState<any>(null);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [flashNews, setFlashNews] = useState<any[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  // 快捷添加弹窗
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [quickSymbol, setQuickSymbol] = useState('');
  const [quickName, setQuickName] = useState('');
  const [adding, setAdding] = useState(false);

  const fetchData = async () => {
    try {
      setRefreshing(true);
      const [statsRes, briefRes, alertsRes, watchRes] = await Promise.all([
        api.get('/overview/stats').catch(() => null),
        api.get('/overview/briefing').catch(() => null),
        api.get('/overview/alerts').catch(() => []),
        api.get('/watchlist').catch(() => []),
      ]);

      if (statsRes) setStats(statsRes);
      if (briefRes && (briefRes as any).content) setBriefing(briefRes as any);
      if (Array.isArray(alertsRes)) setAlerts(alertsRes);
      if (Array.isArray(watchRes)) setWatchlist(watchRes);

      const mktOv: any = (statsRes as any)?.market;
      if (mktOv?.flash_news?.length > 0) setFlashNews(mktOv.flash_news);
    } catch {
      // ignore
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, 60000);
    return () => clearInterval(timer);
  }, []);

  const handleQuickAdd = async (symbol: string, name?: string) => {
    try {
      await api.post('/watchlist', { symbol, name: name || '' });
      message.success(`已关注 ${name || symbol}`);
      fetchData();
    } catch {
      message.error('关注失败');
    }
  };

  const handleModalAdd = async () => {
    if (!quickSymbol.trim()) {
      message.warning('请输入标的代码');
      return;
    }
    setAdding(true);
    try {
      await api.post('/watchlist', { symbol: quickSymbol.trim(), name: quickName.trim() });
      message.success('已成功加入自选池');
      setAddModalOpen(false);
      setQuickSymbol('');
      setQuickName('');
      fetchData();
    } catch {
      message.error('添加失败');
    } finally {
      setAdding(false);
    }
  };

  const rawIndices = stats?.market?.indices?.indices || {};
  // 只渲染**确实带了数值**的指数；没有任何一条就渲染空态跑马灯，
  // 不再用一组编造的点位把这条带子填满。
  const indicesArr: Array<{ name: string; price: number; change_pct: number }> =
    Object.entries(rawIndices)
      .filter(([, d]: [string, any]) => typeof d?.price === 'number' && Number.isFinite(d.price))
      .map(([name, d]: [string, any]) => ({
        name,
        price: d.price,
        change_pct: typeof d.change_pct === 'number' ? d.change_pct : 0,
      }));

  const displayedFlash: any[] = flashNews;
  const displayedBriefing: string = briefing.content || '';

  const senti   = stats?.market?.sentiment  || {};
  const breadth  = stats?.market?.breadth    || {};
  const flow     = stats?.market?.fund_flow  || {};
  const margin   = stats?.market?.margin     || {};
  const accuracy = stats?.accuracy           || {};

  // 情绪分：缺失时不再兜底成 12（那会让指针稳定停在一个看起来"偏低"的位置）
  const hasSenti = typeof senti.score === 'number' && Number.isFinite(senti.score);
  const sentiVal = hasSenti ? (senti.score as number) : 0;
  // 滑柄位置：左端=涨(红)、右端=跌(绿)，所以**负分（偏空）必须落在右侧**。
  // 原式是 `(sentiVal + 50) / 100`：score=-24.8（后端 level=bearish，偏空）
  // 被算成 25.2%，滑柄停在左侧的"涨"端 —— 光谱方向与后端结论完全相反，
  // 用户看到的是"偏多"。
  // 值域来自后端 core/data/market.py::sentiment_score 的 docstring：
  // 涨跌家数(-40~40) + 涨跌停(-20~20) + 资金流(-40~40)，合计 **-100~100**。
  // 故线性映射为 (100 - score) / 2：+100→0%(涨端)、0→50%、-100→100%(跌端)。
  const sentiPercent = hasSenti ? Math.max(2, Math.min(98, (100 - sentiVal) / 2)) : 50;

  // 到期胜率：三个字段各自可能缺失，全部如实反映，不再用 62.5% / 12 笔顶上。
  //
  // 2026-09-24 修正字段名漂移：后端 `/overview/stats → accuracy` 来自
  // core/verify/evaluator.py::accuracy_report()，返回的键是
  //   hit_rate / avg_brier / total
  // 而这里原先读的是 overall_hit_rate / total_evaluations —— 两个键后端都不存在，
  // 恒为 undefined。后果：hasHitRate 永远 false，即使真实对账数据已经产生，
  // 这张卡也永远显示「—」+「待对账」。
  //
  // 同时去掉 `sample_stats.total` 这个兜底：它是**自学习样本池**的规模
  // （后端 docstring：校准与元模型的训练数据），不是"到期预测已验证笔数"。
  // 原先它在 accuracy.total_evaluations 缺失时顶上来，把 1852 这个不相干的数字
  // 显示在「已验证」标签下 —— 比显示 0 更容易被误读。
  const hasHitRate = typeof accuracy.hit_rate === 'number' && Number.isFinite(accuracy.hit_rate);
  const hitRateValue = hasHitRate ? Math.min(Math.max(accuracy.hit_rate as number, 0), 1) : 0;
  const evalCount = typeof accuracy.total === 'number' ? accuracy.total : 0;

  // 仿真账户总权益：直接取后端聚合值；取不到就显示 '—'。
  // 旧实现是 `stats?.paper_total_value || 1000000`，取不到时凭空显示 ¥100.0 万。
  const paperTotalValue =
    typeof stats?.paper_total_value === 'number' && Number.isFinite(stats.paper_total_value)
      ? stats.paper_total_value
      : null;
  // 后端目前不返回权益曲线序列，因此没有可画的趋势线。不编一条假的。
  const paperTrend: number[] = [];

  return (
    <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

      {/* ── 顶部市场指数跑马灯 (无缝循环滚动，悬停暂停) ── */}
      <div className="apple-card px-4 py-3 overflow-hidden">
        {indicesArr.length === 0 ? (
          <div className="text-xs text-[#A1A1A6] flex items-center gap-2">
            <Minus className="w-3.5 h-3.5" />
            指数行情暂不可用 —— 接口未返回数据时，此处不展示任何替代数据
          </div>
        ) : (
        <div className="ticker-track">
          {[0, 1].map((dup) => (
            <div key={dup} className="flex items-center gap-0 shrink-0" aria-hidden={dup === 1}>
              {indicesArr.map((idx, i) => {
                const isUp   = idx.change_pct > 0;
                const isDown = idx.change_pct < 0;
                return (
                  <div
                    key={`${dup}-${idx.name}`}
                    className={`flex items-baseline gap-2.5 px-4 sm:px-5 ${i < indicesArr.length - 1 ? 'border-r border-black/[0.05]' : 'border-r border-black/[0.05]'}`}
                  >
                    <span className="text-[12px] font-medium text-[#86868B] whitespace-nowrap">{idx.name}</span>
                    <span className="text-[13px] font-semibold text-[#1D1D1F] font-mono">
                      {idx.price.toFixed(2)}
                    </span>
                    <span
                      className={`text-[12px] font-medium font-mono flex items-center gap-0.5 ${
                        isUp ? 'text-[#E03E3E]' : isDown ? 'text-[#34C759]' : 'text-[#86868B]'
                      }`}
                    >
                      {isUp   && <ArrowUpRight   className="w-3 h-3" />}
                      {isDown && <ArrowDownRight className="w-3 h-3" />}
                      {!isUp && !isDown && <Minus className="w-3 h-3" />}
                      {idx.change_pct > 0 ? `+${idx.change_pct.toFixed(2)}%` : `${idx.change_pct.toFixed(2)}%`}
                    </span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        )}
      </div>

      {/* ── 区域 A：每日研判内参简报 (Apple Editorial) ── */}
      <div className="apple-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-2xl bg-black/[0.04] border border-black/[0.04] flex items-center justify-center text-[#1D1D1F]">
              <FileText className="w-4 h-4 text-[#0071E3]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-[#1D1D1F] tracking-tight">
                  每日市场研判内参
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#0071E3]/10 text-[#0071E3]">
                  盘前早报
                </span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-[#86868B] flex items-center gap-1 font-mono">
              <Clock className="w-3.5 h-3.5" />
              {briefing.created_at ? briefing.created_at.slice(0, 10) + ' 09:00' : '尚未生成'}
            </span>
            <Button
              type="text"
              size="small"
              icon={<RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />}
              onClick={fetchData}
              loading={refreshing}
              className="apple-press !rounded-full !text-[#1D1D1F] !text-xs !bg-black/[0.04] hover:!bg-black/[0.08] !h-8 !px-3.5"
            >
              刷新
            </Button>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center gap-2.5 py-6">
            <Spin size="small" />
            <span className="text-xs text-[#86868B]">正在聚合全市场盘面数据…</span>
          </div>
        ) : displayedBriefing ? (
          <div className="p-4 sm:p-5 bg-[#F5F5F7] rounded-2xl border border-black/[0.03]">
            <MarkdownView content={displayedBriefing} />
          </div>
        ) : (
          <div className="py-9 text-center text-xs text-[#86868B] leading-relaxed">
            今日尚无研判内参 —— 内参由定时任务盘前生成，
            <br className="hidden sm:block" />
            未生成时此处保持为空，不会以任何预制文本填充。
          </div>
        )}
      </div>

      {/* ── 区域 B：核心指标四卡片 (创新 Apple iOS 18 / macOS Sequoia Widgets) ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Widget 1: 市场情绪晴雨表 */}
        <div className="apple-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">大盘多空情绪</span>
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-black/[0.04] text-[#1D1D1F]">
              {!hasSenti ? '暂无数据'
                : senti.level === 'bullish' ? '偏多态势'
                  : senti.level === 'bearish' ? '偏空格局'
                    : '中性均衡'}
            </span>
          </div>
          <div className="my-3">
            <div className={`text-3xl font-semibold font-mono tracking-tight ${hasSenti ? 'text-[#1D1D1F]' : 'text-[#A1A1A6]'}`}>
              {!hasSenti ? '—' : sentiVal > 0 ? `+${sentiVal}` : sentiVal}
            </div>
            {/* 连续光谱滑动刻度。
                配色必须是「红涨绿跌」（A 股习惯，见 regional_conventions）：
                左端=涨=红，右端=跌=绿。原先是 from-绿 via-黄 to-红，正好推反 ——
                标签写着「涨」的那一侧染的是绿，等于用颜色告诉用户看多代表下跌。
                无数据时整条光谱与滑柄都不渲染：一根带滑块的彩色刻度条会被读成
                "当前读数是中性"，而实际上我们一个数都没有。 */}
            {hasSenti && (
              <div className="relative mt-2.5 h-1.5 w-full bg-black/[0.06] rounded-full overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 bg-gradient-to-r from-[#E03E3E] via-[#FFCC00] to-[#34C759] w-full"
                />
                <div
                  className="absolute top-0 bottom-0 w-2.5 bg-white border border-black/20 rounded-full shadow-sm -ml-1 transition-all duration-500"
                  style={{ left: `${sentiPercent}%` }}
                />
              </div>
            )}
          </div>
          {/* 涨跌家数：后端 breadth.source 明确写着「概念板块汇总(估算)」，
              并且可能带 breadth_error（当前就是 "No tables found"）。
              这两个数字精确到个位，看起来像逐只统计的结果，实际是估算——
              数据来源必须披露，否则用户会按精确统计去理解它。 */}
          <div
            className="text-[11px] text-[#86868B] flex items-center justify-between font-mono pt-1"
            title={
              breadth.source
                ? `数据来源：${breadth.source}${breadth.breadth_error ? `（${breadth.breadth_error}）` : ''}｜非逐只统计，为估算值`
                : '数据来源未标注'
            }
          >
            <span>涨 {breadth.up ?? '—'}</span>
            {breadth.source && (
              <span className="text-[10px] text-[#A1A1A6] px-1.5 py-0.5 rounded bg-black/[0.03]">估算</span>
            )}
            <span>跌 {breadth.down ?? '—'}</span>
          </div>
        </div>

        {/* Widget 2: 自选异动雷达 */}
        <div className="apple-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">自选异动预警</span>
            {/* 原先是常亮的绿色脉冲点 +「实时监听」。
                两个问题：① 实际是 60 秒轮询（见下方 setInterval），不是实时流，
                写「实时」是对能力的夸大；② 自选为 0 只时脉冲点照样在跳，
                像正在监控一个空的列表。
                判据取 watchlist.length（监控对象的数量），不是 alerts.length ——
                alerts 是"已发现几条异动"，自选有 10 只但都没异动时它是 0，
                用它当判据会把"监控中但无异常"误报成"待监控"。 */}
            <div className="flex items-center gap-1.5">
              {watchlist.length > 0 ? (
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#34C759] opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#34C759]" />
                </span>
              ) : (
                <span className="inline-flex rounded-full h-2 w-2 bg-black/[0.12]" />
              )}
              <span className="text-[11px] text-[#86868B]">
                {watchlist.length > 0 ? `60 秒轮询 · 监控 ${watchlist.length} 只` : '待添加自选'}
              </span>
            </div>
          </div>
          <div className="my-3">
            <div className="text-3xl font-semibold text-[#1D1D1F] font-mono tracking-tight">
              {alerts.length} <span className="text-xs font-normal text-[#86868B]">只</span>
            </div>
            <div className="mt-2 text-xs text-[#86868B] truncate">
              {alerts.length > 0
                ? `${alerts[0].name} ${alerts[0].change_pct > 0 ? '+' : ''}${alerts[0].change_pct?.toFixed(1)}%`
                : '暂无异常波动'}
            </div>
          </div>
        </div>

        {/* Widget 3: 到期预测胜率 (Apple Activity Ring 质感)
            2026-09-24 返工：原先命中率缺失时兜底显示 '62.5%'、笔数缺失时兜底显示 12、
            环形的 strokeDasharray 直接写死 "62.5, 100"、徽标永远显示「已对账」——
            即使一笔对账都没有，这张卡看上去也像是已经统计过了。现在一律如实：没数据就是「—」。 */}
        <div className="apple-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">到期预测胜率</span>
            {hasHitRate ? (
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-[#34C759]/10 text-[#248A3D]">
                已对账
              </span>
            ) : (
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-black/[0.04] text-[#86868B]">
                待对账
              </span>
            )}
          </div>
          <div className="my-3 flex items-center justify-between">
            <div>
              {/* 数值配色同 Verify.tsx 的判据：命中率不是涨跌幅，不能用红绿价格语义。
                  跑赢 50% 随机基准=品牌蓝，未跑赢/无数据=中性灰。
                  原先是 `hasHitRate ? 绿色 : 灰` —— 只要对账过就是绿的，
                  0.0% 命中率（比基准低 50 个百分点）也会被染成"良好绿"。 */}
              <div className={`text-3xl font-semibold font-mono tracking-tight ${
                hasHitRate && hitRateValue >= 0.5 ? 'text-[#0071E3]' : hasHitRate ? 'text-[#1D1D1F]' : 'text-[#A1A1A6]'
              }`}>
                {hasHitRate ? `${(hitRateValue * 100).toFixed(1)}%` : '—'}
              </div>
              <div className="text-xs text-[#86868B] mt-1 font-mono">
                {evalCount > 0 ? `${evalCount} 笔已验证` : '暂无到期样本'}
              </div>
            </div>
            {/* 微型 SVG 环形进度条 */}
            <svg className="w-12 h-12 -rotate-90 transform" viewBox="0 0 36 36">
              <path
                className="text-black/[0.05]"
                strokeWidth="3.5"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className="text-[#34C759]"
                strokeDasharray={`${(hitRateValue * 100).toFixed(1)}, 100`}
                strokeWidth="3.5"
                strokeLinecap="round"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
          </div>
        </div>

        {/* Widget 4: 仿真账户权益 (Apple Wallet Card Sparkline) */}
        <div className="apple-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">仿真账户权益</span>
            {/* 原先这里的「+1.2% 今日」是硬编码的常字符串：无论账户当天是赚是亏，
                徽标永远显示上涨 1.2%。仿真是用来验证策略的，徽标撒谎等于结论失效，
                所以直接去掉这个数字，只保留"模拟盘"这个不涉及数据的定性标注。 */}
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-black/[0.04] text-[#86868B] font-mono">
              模拟盘
            </span>
          </div>
          <div className="my-3 flex items-baseline justify-between">
            <div>
              <div className="text-3xl font-semibold text-[#1D1D1F] font-mono tracking-tight">
                {paperTotalValue !== null ? (
                  <>¥{(paperTotalValue / 10000).toFixed(1)}<span className="text-xs font-normal text-[#86868B]">万</span></>
                ) : (
                  <span className="text-[#A1A1A6]">—</span>
                )}
              </div>
            </div>
            {/* 微趋势线只在有真实净值序列时绘制；原先是七个写死的常数 */}
            {paperTrend.length >= 2 && (
              <MiniSparkline points={paperTrend} isUp={paperTrend[paperTrend.length - 1] >= paperTrend[0]} />
            )}
          </div>
        </div>
      </div>

      {/* ── 区域 C：自选监控池 + 7×24 快讯 ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* 左侧：自选池 (Apple Stocks Layout & Curated Grid) */}
        <div className="lg:col-span-8 apple-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight">
                  核心监控池
                </h3>
                <span className="text-xs text-[#86868B] bg-black/[0.04] px-2 py-0.5 rounded-full">
                  {watchlist.length > 0 ? `${watchlist.length} 只标的` : '精选推荐'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  type="text"
                  size="small"
                  onClick={() => setAddModalOpen(true)}
                  className="!text-[#0071E3] !text-xs !font-medium flex items-center gap-1 hover:!bg-[#0071E3]/10 !rounded-full !px-3"
                >
                  <Plus className="w-3.5 h-3.5" /> 快速添加
                </Button>
                <Button
                  type="text"
                  size="small"
                  onClick={() => navigate('/watchlist')}
                  className="!text-[#86868B] hover:!text-[#1D1D1F] !text-xs !font-medium flex items-center gap-0.5"
                >
                  全部自选 <ChevronRight className="w-3.5 h-3.5" />
                </Button>
              </div>
            </div>

            {/* 自选为空时只做引导，不预填任何"精选资产"卡片。
                原先是 CURATED_DEFAULT_TARGETS：4 只真实标的的写死价格与 5 点走势线，
                看起来像是实时行情，实际是常量。空仓用户会以为自己看到的是市场数据。 */}
            {watchlist.length === 0 ? (
              <div className="p-8 rounded-2xl bg-[#F5F5F7] border border-dashed border-black/[0.08] text-center space-y-2">
                <div className="text-sm font-medium text-[#1D1D1F]">自选池还是空的</div>
                <p className="text-xs text-[#86868B] leading-relaxed">
                  添加标的后会在这里显示实时涨跌与迷你走势。
                  <br className="hidden sm:block" />
                  在此之前不展示任何占位行情，避免把示例数据误读成真实行情。
                </p>
                <div className="pt-1">
                  <Button
                    type="text"
                    size="small"
                    onClick={() => navigate('/watchlist')}
                    className="!text-[#0071E3] !text-xs !font-medium !rounded-full hover:!bg-[#0071E3]/10 !px-4"
                  >
                    去添加标的 <ChevronRight className="w-3.5 h-3.5 inline" />
                  </Button>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                {watchlist.map((item) => {
                  const chg  = item.change_pct || 0;
                  const isUp = chg > 0;
                  const isDn = chg < 0;
                  return (
                    <div
                      key={item.symbol}
                      onClick={() => navigate(`/analysis?symbol=${item.symbol}`)}
                      className="apple-press p-4 rounded-2xl border border-black/[0.04] bg-[#F5F5F7] hover:bg-white hover:shadow-apple transition-all duration-200 cursor-pointer group"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-[#1D1D1F] text-xs truncate max-w-[130px] group-hover:text-[#0071E3] transition-colors">
                          {item.name}
                        </span>
                        <span className="text-[10px] font-mono text-[#86868B]">
                          {item.symbol}
                        </span>
                      </div>
                      <div className="flex items-baseline justify-between">
                        <span className="text-lg font-semibold font-mono text-[#1D1D1F]">
                          {item.price ? Number(item.price).toFixed(item.price < 10 ? 3 : 2) : '—'}
                        </span>
                        <span className={`text-[11px] font-medium font-mono px-2 py-0.5 rounded-full ${
                          isUp ? 'text-[#E03E3E] bg-[#E03E3E]/10'
                            : isDn ? 'text-[#248A3D] bg-[#34C759]/10'
                            : 'text-[#86868B] bg-black/[0.04]'
                        }`}>
                          {isUp ? '+' : ''}{chg.toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
                <div
                  onClick={() => setAddModalOpen(true)}
                  className="apple-press p-4 rounded-2xl border border-dashed border-black/[0.12] hover:border-[#0071E3] hover:bg-[#0071E3]/5 transition-all duration-200 cursor-pointer flex items-center justify-center min-h-[80px] text-[#86868B] hover:text-[#0071E3] gap-2"
                >
                  <Plus className="w-4 h-4" />
                  <span className="text-xs font-medium">添加自选标的</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 右侧：7×24 快讯流 (Apple News Editorial Style - 彻底移除生硬灰盒子) */}
        <div className="lg:col-span-4 apple-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Radio className="w-4 h-4 text-[#0071E3]" />
                <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight">
                  7×24 财经快讯
                </h3>
              </div>
              {/* 「实时抓取」→「60 秒轮询」：数据来自 60 秒一次的整体轮询，
                  不是实时推送。如实标注刷新频率，用户才知道看到的快讯有多新。 */}
              <span className="text-[11px] text-[#86868B] font-medium bg-black/[0.04] px-2 py-0.5 rounded-full">
                60 秒轮询
              </span>
            </div>

            {/* Apple News 排版：纯粹排印与发丝分割线 */}
            {displayedFlash.length === 0 ? (
              <div className="py-12 text-center text-xs text-[#86868B] leading-relaxed">
                暂无快讯
                <br />
                快讯接口未返回数据时此处保持为空，不会以预制新闻填充
              </div>
            ) : (
            <div className="divide-y divide-black/[0.05] overflow-y-auto max-h-[380px] pr-1">
              {displayedFlash.slice(0, 8).map((news: any, idx: number) => (
                <div
                  key={idx}
                  className="py-3 first:pt-0 last:pb-0 group cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-semibold text-[#0071E3]">
                      {news.source || '快讯'}
                    </span>
                    <span className="text-[10px] text-[#86868B] font-mono">
                      {news.publish_time
                        ? (typeof news.publish_time === 'string' && news.publish_time.length > 5
                            ? news.publish_time.slice(11, 16)
                            : news.publish_time)
                        : ''}
                    </span>
                  </div>
                  <p className="text-xs text-[#1D1D1F] leading-snug group-hover:text-[#0071E3] transition-colors font-normal">
                    {news.title}
                  </p>
                </div>
              ))}
            </div>
            )}
          </div>
        </div>
      </div>

      {/* ── 区域 D：市场微观结构与流动性矩阵 (Apple Bento Grid) ── */}
      <div className="apple-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight">
            市场微观结构与流动性矩阵
          </h3>
        </div>
        {/* 2026-09-24 返工：这四格原先各带一个写死的兜底值（'54.2%'、'-28.5 亿'、
            '+42 亿'、4 个池）和一句写死的定性结论（「偏暖格局」「预警区间」「杠杆增持」）。
            更糟的是判断用的是 `flow.main_net_inflow ? ... : ...` —— 净流入恰好为 0
            （一个完全正常的取值）也会被当成"没有数据"而弹出假数字。
            现在统一改成：有值就显示真值 + 派生结论，无值就显示 '—' 且不给结论。 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-[#F5F5F7] rounded-2xl border border-black/[0.03]">
            <div className="text-[11px] font-medium text-[#86868B] mb-1">全市场上涨占比</div>
            <div className={`text-2xl font-semibold font-mono ${typeof breadth.up_ratio === 'number' ? 'text-[#1D1D1F]' : 'text-[#A1A1A6]'}`}>
              {typeof breadth.up_ratio === 'number' ? `${(breadth.up_ratio * 100).toFixed(1)}%` : '—'}
            </div>
            {typeof breadth.up_ratio === 'number' && (
              <div className={`text-[11px] font-medium mt-1 ${breadth.up_ratio >= 0.5 ? 'text-[#E03E3E]' : 'text-[#34C759]'}`}>
                {breadth.up_ratio >= 0.5 ? '上涨家数占优' : '下跌家数占优'}
              </div>
            )}
          </div>
          <div className="p-4 bg-[#F5F5F7] rounded-2xl border border-black/[0.03]">
            <div className="text-[11px] font-medium text-[#86868B] mb-1">主力资金净流入</div>
            <div className={`text-2xl font-semibold font-mono ${typeof flow.main_net_inflow === 'number' ? (flow.main_net_inflow >= 0 ? 'text-[#E03E3E]' : 'text-[#34C759]') : 'text-[#A1A1A6]'}`}>
              {typeof flow.main_net_inflow === 'number'
                ? `${flow.main_net_inflow >= 0 ? '+' : ''}${(flow.main_net_inflow / 1e8).toFixed(1)} 亿`
                : '—'}
            </div>
          </div>
          <div className="p-4 bg-[#F5F5F7] rounded-2xl border border-black/[0.03]">
            <div className="text-[11px] font-medium text-[#86868B] mb-1">两融余额周变动</div>
            <div className={`text-2xl font-semibold font-mono ${typeof margin.weekly_change === 'number' ? 'text-[#0071E3]' : 'text-[#A1A1A6]'}`}>
              {typeof margin.weekly_change === 'number'
                ? `${margin.weekly_change >= 0 ? '+' : ''}${(margin.weekly_change / 1e8).toFixed(0)} 亿`
                : '—'}
            </div>
          </div>
          <div className="p-4 bg-[#F5F5F7] rounded-2xl border border-black/[0.03]">
            <div className="text-[11px] font-medium text-[#86868B] mb-1">已激活校准器</div>
            <div className={`text-2xl font-semibold font-mono ${typeof stats?.active_calibrators === 'number' ? 'text-[#248A3D]' : 'text-[#A1A1A6]'}`}>
              {typeof stats?.active_calibrators === 'number' ? `${stats.active_calibrators} 个池` : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* ── 快捷添加标的模态框 ── */}
      <AppleModal
        open={addModalOpen}
        onClose={() => setAddModalOpen(false)}
        title="添加监控标的"
        icon={<Search className="w-5 h-5 text-[#0071E3]" />}
        footer={
          <div className="flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={() => setAddModalOpen(false)}
              className="px-5 py-2.5 rounded-full text-xs font-medium text-[#1D1D1F] bg-black/[0.05] hover:bg-black/[0.09] transition-all"
            >
              取消
            </button>
            <button
              type="button"
              disabled={adding}
              onClick={handleModalAdd}
              className="apple-press px-6 py-2.5 rounded-full text-xs font-medium text-white bg-[#0071E3] hover:bg-[#0077ED] shadow-sm transition-all"
            >
              {adding ? '正在加入…' : '确认添加'}
            </button>
          </div>
        }
      >
        <div className="space-y-4">
          <div className="bg-[#F5F5F7] rounded-2xl p-4 border border-black/[0.04] space-y-3">
            <div>
              <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                标的代码
              </label>
              <input
                type="text"
                value={quickSymbol}
                onChange={(e) => setQuickSymbol(e.target.value)}
                placeholder="例如：161725 或 000997"
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3] font-mono"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                标的名称 (选填，自动联网识别)
              </label>
              <input
                type="text"
                value={quickName}
                onChange={(e) => setQuickName(e.target.value)}
                placeholder="例如：招商白酒"
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3]"
              />
            </div>
          </div>
        </div>
      </AppleModal>
    </div>
  );
};
