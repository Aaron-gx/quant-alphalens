import React, { useEffect, useState } from 'react';
import { Table, message } from 'antd';
import { EChart } from '../components/EChart';
import {
  Briefcase,
  Plus,
  Play,
  TrendingUp,
  DollarSign,
  PieChart,
  Wallet,
  Clock,
  Send,
  UserCheck,
} from 'lucide-react';
import { api, PaperAccountItem } from '../api/client';
import { AppleModal } from '../components/AppleModal';

export const PaperTradePage: React.FC = () => {
  const [accounts, setAccounts] = useState<PaperAccountItem[]>([]);
  const [currentAid, setCurrentAid] = useState<number | null>(null);
  const [portfolio, setPortfolio] = useState<any>(null);
  const [navCurve, setNavCurve] = useState<Array<{ date: string; total_value: number }>>([]);
  const [loading, setLoading] = useState(false);
  const [autoTrading, setAutoTrading] = useState(false);
  const [seedingSymbol, setSeedingSymbol] = useState<string | null>(null);

  // 新建账户 Modal
  const [newAccModal, setNewAccModal] = useState(false);
  const [newAccName, setNewAccName] = useState('AI先锋实盘组合');
  const [newAccCash, setNewAccCash] = useState(1000000);
  const [newAccMode, setNewAccMode] = useState('ai_signal');

  // 下单 Modal
  const [orderModal, setOrderModal] = useState(false);
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [orderSymbol, setOrderSymbol] = useState('');
  const [orderAmount, setOrderAmount] = useState<number>(50000);
  const [orderReason, setOrderReason] = useState('AI信号触发建仓');

  const fetchAccounts = async () => {
    try {
      const res: any = await api.get('/paper/accounts');
      if (Array.isArray(res) && res.length > 0) {
        setAccounts(res);
        if (!currentAid) {
          setCurrentAid(res[0].id);
        }
      }
    } catch {
      message.error('获取虚拟账户列表失败');
    }
  };

  const fetchPortfolio = async (aid: number) => {
    setLoading(true);
    // 切账户前先清空，避免旧账户的净值曲线残留到新账户上
    setPortfolio(null);
    setNavCurve([]);
    try {
      const [res, navRes] = await Promise.all([
        api.get(`/paper/accounts/${aid}`),
        // 净值曲线走独立的 /nav 接口（后端 paper.nav_curve 的真实 NavHistory 记录）
        api.get(`/paper/accounts/${aid}/nav`).catch(() => null),
      ]);
      setPortfolio(res);
      const curve = (navRes as any)?.curve;
      setNavCurve(Array.isArray(curve) ? curve : []);
    } catch {
      message.error('获取账户持仓失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAccounts();
  }, []);

  useEffect(() => {
    if (currentAid) {
      fetchPortfolio(currentAid);
    }
  }, [currentAid]);

  // 新建账户
  const handleCreateAccount = async () => {
    try {
      const res: any = await api.post('/paper/accounts', {
        name: newAccName,
        initial_cash: newAccCash,
        mode: newAccMode,
      });
      message.success('虚拟账户创建成功！');
      setNewAccModal(false);
      await fetchAccounts();
      if (res?.id) setCurrentAid(res.id);
    } catch {
      message.error('创建账户失败');
    }
  };

  // 触发 AI 自动调仓 (Apple Non-blocking Flow)
  const handleAutoTrade = async () => {
    if (!currentAid || autoTrading) return;
    setAutoTrading(true);
    try {
      await api.post(`/paper/accounts/${currentAid}/auto_trade`);
      message.success('自动调仓撮合已完成');
      fetchPortfolio(currentAid);
    } catch {
      message.error('自动调仓执行异常');
    } finally {
      setAutoTrading(false);
    }
  };

  // 执行下单
  const handleOrder = async () => {
    if (!currentAid || !orderSymbol.trim()) {
      message.warning('请填写正确的代码');
      return;
    }
    try {
      if (orderSide === 'buy') {
        await api.post(`/paper/accounts/${currentAid}/buy`, {
          symbol: orderSymbol.trim(),
          amount: orderAmount,
          reason: orderReason,
        });
        message.success('模拟买入委托已成交！');
      } else {
        await api.post(`/paper/accounts/${currentAid}/sell`, {
          symbol: orderSymbol.trim(),
          ratio: 1.0,
          reason: orderReason,
        });
        message.success('模拟卖出已全额平仓！');
      }
      setOrderModal(false);
      fetchPortfolio(currentAid);
    } catch (e: any) {
      message.error(e.response?.data?.detail || '下单失败');
    }
  };

  // 一键买入示例组合
  const handleQuickSeed = async (symbol: string, amount: number, name: string) => {
    if (!currentAid) return;
    message.loading({ content: `正在买入 ${name} (${amount}元)…`, key: 'seed' });
    try {
      await api.post(`/paper/accounts/${currentAid}/buy`, {
        symbol,
        amount,
        reason: 'AI推荐底仓配置',
      });
      message.success({ content: `${name} 买入成交！`, key: 'seed' });
      fetchPortfolio(currentAid);
    } catch (e: any) {
      message.error({ content: e.response?.data?.detail || '买入失败', key: 'seed' });
    }
  };

  const currentAcc = accounts.find((a) => a.id === currentAid);
  // 2026-09-24 返工：原先三级兜底都以 1000000 收尾，也就是接口失败时会凭空
  // 显示「¥1,000,000.00 总资产 / ¥1,000,000.00 现金 / 本金 ¥1,000,000」。
  // 模拟盘的数字是用来判断策略好坏的，兜出一个整数等于给结论注水。缺就是 null。
  const totalValue = portfolio?.total_value ?? currentAcc?.total_value ?? null;
  const cash = portfolio?.cash ?? currentAcc?.cash ?? null;
  const initialCash = currentAcc?.initial_cash ?? null;
  const positions = portfolio?.positions || [];
  const hasMoney = typeof totalValue === 'number' && typeof cash === 'number';
  const marketValue = hasMoney ? totalValue - cash : null;
  const returnRate =
    hasMoney && typeof initialCash === 'number' && initialCash > 0
      ? ((totalValue - initialCash) / initialCash) * 100
      : null;

  // 净值走势：只用后端 NavHistory 的真实序列（/paper/accounts/{aid}/nav）。
  // 2026-09-24：原先这条线的前 6 个点写死成 [1.000, 1.004, 1.008, 1.006, 1.012, 1.015]、
  // 沪深300基准 7 个点也全是常量，再拼上「今天」一个真值 —— 图上看起来是
  // "一条真实净值曲线跑赢基准"，实际整段历史都是编的。
  const firstNav = navCurve.length > 0 ? navCurve[0].total_value : 0;
  const navSeries = navCurve.length > 1 && firstNav > 0
    ? navCurve.map((c) => Number((c.total_value / firstNav).toFixed(4)))
    : [];
  const navDates = navCurve.map((c) => String(c.date).slice(5, 10));

  // 净值走势图配置 (Apple Stocks Area)
  const navChartOption = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.96)',
      borderColor: 'rgba(0, 0, 0, 0.08)',
      shadowBlur: 10,
      shadowColor: 'rgba(0, 0, 0, 0.06)',
      textStyle: { color: '#1D1D1F', fontSize: 11 },
    },
    grid: { left: 45, right: 15, top: 25, bottom: 25 },
    xAxis: {
      type: 'category',
      data: navDates,
      axisLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.08)' } },
      axisTick: { show: false },
      axisLabel: { color: '#86868B', fontSize: 11 },
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.04)', type: 'dashed' } },
      axisLabel: { color: '#86868B', fontSize: 11 },
    },
    series: [
      {
        name: '组合净值',
        type: 'line',
        smooth: 0.35,
        itemStyle: { color: '#0071E3' },
        lineStyle: { width: 2.5, color: '#0071E3' },
        data: navSeries,
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(0, 113, 227, 0.16)' },
              { offset: 1, color: 'rgba(0, 113, 227, 0.00)' },
            ],
          },
        },
      },
      // 原先这里有一条写死的「沪深300基准」序列（7 个常量点），
      // 等于给每只账户都配上一条永远向上的假基准，用来"衬托"组合表现。
      // 后端目前不提供基准净值序列，直接移除；要加回来必须先有真实数据源。
    ],
  };

  return (
    <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* ── 头部：账户选择与创建 (Apple HIG Card) ── */}
      <div className="apple-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-2xl bg-black/[0.04] border border-black/[0.04] flex items-center justify-center text-[#1D1D1F]">
            <Briefcase className="w-5 h-5 text-[#0071E3]" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-lg font-semibold text-[#1D1D1F] tracking-tight">
                仿真模拟交易盘
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-black/[0.04] text-[#1D1D1F] border border-black/[0.06]">
                T+1
              </span>
              <span className="hidden sm:inline-block px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#0071E3]/10 text-[#0071E3]">
                实盘费率
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 self-start md:self-auto">
          {accounts.length > 0 && (
            <select
              value={currentAid || ''}
              onChange={(e) => setCurrentAid(Number(e.target.value))}
              className="bg-[#F5F5F7] text-xs font-medium text-[#1D1D1F] rounded-full px-3.5 py-2 border border-black/[0.06] focus:outline-none"
            >
              {accounts.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name} [{a.mode === 'ai_signal' ? 'AI调仓' : '自主'}]
                </option>
              ))}
            </select>
          )}

          <button
            type="button"
            onClick={() => setNewAccModal(true)}
            className="apple-press flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-medium text-[#1D1D1F] bg-black/[0.04] hover:bg-black/[0.08] transition-all"
          >
            <Plus className="w-3.5 h-3.5" />
            开立新账户
          </button>

          {currentAcc?.mode === 'ai_signal' && (
            <button
              type="button"
              disabled={autoTrading}
              onClick={handleAutoTrade}
              className={`apple-press flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-medium text-white ${
                autoTrading ? 'bg-[#0071E3]/70 cursor-not-allowed' : 'bg-[#0071E3] hover:bg-[#0077ED]'
              } shadow-sm transition-all`}
            >
              {autoTrading ? (
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5 fill-current" />
              )}
              <span>{autoTrading ? '正在调仓…' : '触发 AI 调仓'}</span>
            </button>
          )}

          <button
            type="button"
            onClick={() => setOrderModal(true)}
            className="apple-press flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-medium text-white bg-[#1D1D1F] hover:bg-[#333336] shadow-sm transition-all"
          >
            <Send className="w-3.5 h-3.5" />
            委托下单
          </button>
        </div>
      </div>

      {/* ── 核心资产指标四卡片 (Apple Wallet Metric Tiles) ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="apple-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">总资产净值</span>
            <div className="w-8 h-8 rounded-xl bg-black/[0.03] flex items-center justify-center text-[#1D1D1F]">
              <Wallet className="w-4 h-4 text-[#0071E3]" />
            </div>
          </div>
          <div className={`text-2xl font-semibold font-mono tracking-tight mt-2.5 ${hasMoney ? 'text-[#1D1D1F]' : 'text-[#A1A1A6]'}`}>
            {hasMoney
              ? `¥${Number(totalValue).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
              : '—'}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-black/[0.04] text-[#86868B]">
              {typeof initialCash === 'number' ? `本金 ¥${Number(initialCash).toLocaleString()}` : '本金 —'}
            </span>
          </div>
        </div>

        <div className="apple-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">可用现金余额</span>
            <div className="w-8 h-8 rounded-xl bg-black/[0.03] flex items-center justify-center text-[#0071E3]">
              <DollarSign className="w-4 h-4" />
            </div>
          </div>
          <div className={`text-2xl font-semibold font-mono tracking-tight mt-2.5 ${hasMoney ? 'text-[#0071E3]' : 'text-[#A1A1A6]'}`}>
            {hasMoney
              ? `¥${Number(cash).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
              : '—'}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-[#0071E3]/10 text-[#0071E3]">
              现金占比 {hasMoney && totalValue > 0 ? `${((cash / totalValue) * 100).toFixed(1)}%` : '—'}
            </span>
          </div>
        </div>

        <div className="apple-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">持仓总市值</span>
            <div className="w-8 h-8 rounded-xl bg-black/[0.03] flex items-center justify-center text-[#34C759]">
              <PieChart className="w-4 h-4" />
            </div>
          </div>
          <div className={`text-2xl font-semibold font-mono tracking-tight mt-2.5 ${hasMoney ? 'text-[#1D1D1F]' : 'text-[#A1A1A6]'}`}>
            {hasMoney
              ? `¥${Number(marketValue).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
              : '—'}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-[#34C759]/10 text-[#248A3D]">
              占比 {marketValue !== null && totalValue !== null && totalValue > 0
                ? `${((marketValue / totalValue) * 100).toFixed(1)}%`
                : '—'} ({positions.length} 只标的)
            </span>
          </div>
        </div>

        <div className="apple-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#86868B]">累计收益表现</span>
            <div className="w-8 h-8 rounded-xl bg-black/[0.03] flex items-center justify-center text-[#E03E3E]">
              <TrendingUp className="w-4 h-4" />
            </div>
          </div>
          <div className={`text-2xl font-semibold font-mono tracking-tight mt-2.5 ${
            returnRate === null ? 'text-[#A1A1A6]' : returnRate >= 0 ? 'text-[#E03E3E]' : 'text-[#248A3D]'
          }`}>
            {returnRate === null
              ? '—'
              : returnRate >= 0 ? `+${returnRate.toFixed(2)}%` : `${returnRate.toFixed(2)}%`}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${
              returnRate === null
                ? 'bg-black/[0.04] text-[#86868B]'
                : returnRate >= 0 ? 'bg-[#E03E3E]/10 text-[#E03E3E]' : 'bg-[#34C759]/10 text-[#248A3D]'
            }`}>
              浮动盈亏 {hasMoney && typeof initialCash === 'number'
                ? `¥${(totalValue - initialCash).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                : '—'}
            </span>
          </div>
        </div>
      </div>

      {/* ── 净值走势曲线 ── */}
      <div className="apple-card p-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight">
            组合历史净值走势
          </h3>
          {/* 原先这里写着「基准：沪深300指数」，但图上那条"基准线"是 7 个写死的常量，
              后端也没有基准序列接口。没有真基准就不宣称有。 */}
          <span className="text-xs text-[#86868B]">
            {navCurve.length > 1 ? `净值记录 ${navCurve.length} 天（已归一化）` : '暂无净值记录'}
          </span>
        </div>
        {navCurve.length > 1 ? (
          <EChart option={navChartOption} style={{ height: 260 }} />
        ) : (
          <div className="h-[260px] flex flex-col items-center justify-center gap-1.5 rounded-xl border border-dashed border-black/[0.08] bg-[#F5F5F7]/60">
            <span className="text-xs font-medium text-[#6E6E73]">暂无净值曲线</span>
            <span className="text-[11px] text-[#A1A1A6] text-center leading-relaxed px-6">
              账户还没有累计到足够的每日净值记录（需 ≥ 2 天）。
              <br />
              此处不会绘制任何模拟曲线或占位基准。
            </span>
          </div>
        )}
      </div>

      {/* ── 持仓清单表格 ── */}
      <div className="apple-card p-6">
        <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight mb-4">
          当前账户持仓明细 ({positions.length} 只)
        </h3>
        {positions.length === 0 ? (
          <div className="py-10 text-center space-y-3">
            <p className="text-xs text-[#86868B]">暂无持仓标的</p>
            <div className="flex flex-wrap justify-center gap-2.5">
              <button
                type="button"
                onClick={() => handleQuickSeed('161725', 100000, '招商白酒')}
                className="px-3.5 py-1.5 rounded-full text-xs font-medium bg-[#F5F5F7] hover:bg-[#0071E3] hover:text-white transition-all"
              >
                + 建仓 招商白酒 10万
              </button>
              <button
                type="button"
                onClick={() => handleQuickSeed('510300', 150000, '沪深300ETF')}
                className="px-3.5 py-1.5 rounded-full text-xs font-medium bg-[#F5F5F7] hover:bg-[#0071E3] hover:text-white transition-all"
              >
                + 建仓 沪深300ETF 15万
              </button>
              <button
                type="button"
                onClick={() => handleQuickSeed('300750', 100000, '宁德时代')}
                className="px-3.5 py-1.5 rounded-full text-xs font-medium bg-[#F5F5F7] hover:bg-[#0071E3] hover:text-white transition-all"
              >
                + 建仓 宁德时代 10万
              </button>
            </div>
          </div>
        ) : (
          <Table
            dataSource={positions.map((p: any) => ({ ...p, key: p.symbol }))}
            pagination={false}
            rowClassName="hover:bg-black/[0.015] transition-colors"
            columns={[
              {
                title: '标的名称 / 代码',
                key: 'name',
                render: (_, r: any) => (
                  <div>
                    <div className="font-semibold text-xs text-[#1D1D1F]">{r.name || r.symbol}</div>
                    <div className="font-mono text-[11px] text-[#86868B]">{r.symbol}</div>
                  </div>
                ),
              },
              {
                title: '持仓股数 / 份额',
                dataIndex: 'volume',
                key: 'volume',
                render: (v) => <span className="font-mono text-xs text-[#1D1D1F]">{Number(v).toLocaleString()}</span>,
              },
              {
                title: '持仓均价',
                dataIndex: 'cost_price',
                key: 'cost_price',
                render: (p) => <span className="font-mono text-xs text-[#1D1D1F]">¥{Number(p).toFixed(3)}</span>,
              },
              {
                title: '最新市价',
                dataIndex: 'current_price',
                key: 'current_price',
                render: (p) => <span className="font-mono text-xs font-semibold text-[#1D1D1F]">¥{Number(p || 0).toFixed(3)}</span>,
              },
              {
                title: '浮动盈亏',
                key: 'pnl',
                render: (_, r: any) => {
                  const pnl = ((r.current_price || r.cost_price) - r.cost_price) * r.volume;
                  const pnlPct = r.cost_price > 0 ? (((r.current_price || r.cost_price) - r.cost_price) / r.cost_price) * 100 : 0;
                  const isUp = pnl >= 0;
                  return (
                    <span className={`font-mono text-xs font-semibold ${isUp ? 'text-[#E03E3E]' : 'text-[#248A3D]'}`}>
                      {isUp ? '+' : ''}¥{pnl.toFixed(2)} ({isUp ? '+' : ''}{pnlPct.toFixed(2)}%)
                    </span>
                  );
                },
              },
            ]}
          />
        )}
      </div>

      {/* ── 开立新仿真账户 ── */}
      <AppleModal
        open={newAccModal}
        onClose={() => setNewAccModal(false)}
        title="开立新账户"
        icon={<UserCheck className="w-5 h-5 text-[#0071E3]" />}
        footer={
          <div className="flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={() => setNewAccModal(false)}
              className="px-5 py-2.5 rounded-full text-xs font-medium text-[#1D1D1F] bg-black/[0.05] hover:bg-black/[0.09] transition-all"
            >
              取消
            </button>
            <button
              type="button"
              onClick={handleCreateAccount}
              className="px-6 py-2.5 rounded-full text-xs font-medium text-white bg-[#0071E3] hover:bg-[#0077ED] shadow-sm transition-all"
            >
              立即开立
            </button>
          </div>
        }
      >
        <div className="space-y-4">
          <div className="bg-[#F5F5F7] rounded-2xl p-4 border border-black/[0.04] space-y-3">
            <div>
              <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                账户名称
              </label>
              <input
                type="text"
                value={newAccName}
                onChange={(e) => setNewAccName(e.target.value)}
                placeholder="例如：AI时序组合"
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3]"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                初始虚拟资金 (元)
              </label>
              <input
                type="number"
                value={newAccCash}
                onChange={(e) => setNewAccCash(Number(e.target.value))}
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3] font-mono"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                运作模式
              </label>
              <select
                value={newAccMode}
                onChange={(e) => setNewAccMode(e.target.value)}
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none"
              >
                <option value="ai_signal">AI信号驱动（尾盘自动撮合）</option>
                <option value="manual">自主操盘（手动委托模式）</option>
              </select>
            </div>
          </div>
        </div>
      </AppleModal>

      {/* ── 模拟委托下单 ── */}
      <AppleModal
        open={orderModal}
        onClose={() => setOrderModal(false)}
        title="委托下单"
        icon={<Send className="w-5 h-5 text-[#0071E3]" />}
        footer={
          <div className="flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={() => setOrderModal(false)}
              className="px-5 py-2.5 rounded-full text-xs font-medium text-[#1D1D1F] bg-black/[0.05] hover:bg-black/[0.09] transition-all"
            >
              取消
            </button>
            <button
              type="button"
              onClick={handleOrder}
              className={`px-6 py-2.5 rounded-full text-xs font-medium text-white shadow-sm transition-all ${
                orderSide === 'buy' ? 'bg-[#E03E3E] hover:bg-[#D32F2F]' : 'bg-[#248A3D] hover:bg-[#1E7233]'
              }`}
            >
              确认{orderSide === 'buy' ? '买入建仓' : '全额平仓'}
            </button>
          </div>
        }
      >
        <div className="space-y-4">
          {/* Apple Segmented Tray 切换交易方向 */}
          <div className="apple-segmented-tray flex items-center p-1 w-full">
            <button
              type="button"
              onClick={() => setOrderSide('buy')}
              className={`flex-1 py-1.5 rounded-full text-xs font-medium transition-all ${
                orderSide === 'buy'
                  ? 'bg-[#E03E3E] text-white shadow-xs'
                  : 'text-[#86868B] hover:text-[#1D1D1F]'
              }`}
            >
              买入建仓
            </button>
            <button
              type="button"
              onClick={() => setOrderSide('sell')}
              className={`flex-1 py-1.5 rounded-full text-xs font-medium transition-all ${
                orderSide === 'sell'
                  ? 'bg-[#248A3D] text-white shadow-xs'
                  : 'text-[#86868B] hover:text-[#1D1D1F]'
              }`}
            >
              全额平仓
            </button>
          </div>

          <div className="bg-[#F5F5F7] rounded-2xl p-4 border border-black/[0.04] space-y-3">
            <div>
              <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                标的代码
              </label>
              <input
                type="text"
                value={orderSymbol}
                onChange={(e) => setOrderSymbol(e.target.value)}
                placeholder="例如: 161725 或 000997"
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3] font-mono"
              />
            </div>
            {orderSide === 'buy' && (
              <div>
                <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                  委托金额 (元)
                </label>
                <input
                  type="number"
                  step={10000}
                  value={orderAmount}
                  onChange={(e) => setOrderAmount(Number(e.target.value))}
                  className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3] font-mono"
                />
              </div>
            )}
            <div>
              <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                建仓逻辑说明
              </label>
              <input
                type="text"
                value={orderReason}
                onChange={(e) => setOrderReason(e.target.value)}
                placeholder="例如：AI推荐底仓配置"
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3]"
              />
            </div>
          </div>
        </div>
      </AppleModal>
    </div>
  );
};
