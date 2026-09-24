import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, message, Table, Tooltip } from 'antd';
import {
  Star,
  Plus,
  Trash2,
  TrendingUp,
  TrendingDown,
  Search,
  Minus,
} from 'lucide-react';
import { api, WatchlistItem } from '../api/client';
import { AppleModal } from '../components/AppleModal';

// 推荐标的池：这里**只保留代码/名称/类型/一句简介**。
// 原先每条还带 `price` 与 `change_pct` 两个字段（如 300750 的 242.50 / 1.80），
// 虽然当前渲染没用到，但它们是写死的行情数字，一旦后续有人顺手渲染出来，
// 就会变成"没有任何数据源的真行情"。宁可不留。
const RECOMMENDED_TARGETS = [
  { symbol: '161725', name: '招商中证白酒指数(LOF)', asset_type: 'fund', desc: '全网规模最大的白酒被动指数基金' },
  { symbol: '005827', name: '易方达蓝筹精选混合', asset_type: 'ofund', desc: '张坤管理，重仓港股互联网及白酒龙头' },
  { symbol: '510300', name: '华泰柏瑞沪深300ETF', asset_type: 'fund', desc: 'A股核心宽基指数代表性风向标' },
  { symbol: '000997', name: '新大陆', asset_type: 'stock', desc: '数字货币与物联网智能终端标杆' },
  { symbol: '300750', name: '宁德时代', asset_type: 'stock', desc: '全球动力电池龙头，创业板权重第一' },
];

// ─── 行级数据：真实近期净值/收盘 + 最新一次 AI 预测（模块级缓存，一行一次请求）─
interface RowIntel {
  closes: number[];
  weekPct: number | null;  // one_week 预期涨跌幅（up - down）
  hasPred: boolean;
}
const _intelCache = new Map<string, Promise<RowIntel>>();

function getRowIntel(symbol: string, assetType: string): Promise<RowIntel> {
  const key = `${symbol}:${assetType}`;
  if (!_intelCache.has(key)) {
    _intelCache.set(
      key,
      (async () => {
        const p = assetType ? { asset_type: assetType } : {};
        const [kl, preds] = await Promise.all([
          api.get(`/kline/${symbol}`, { params: { count: 30, ...p } }).catch(() => []),
          api.get('/predictions', { params: { symbol, limit: 1 } }).catch(() => []),
        ]);
        const closes = (Array.isArray(kl) ? kl : [])
          .map((d: any) => Number(d.close ?? d.nav ?? 0))
          .filter((v: number) => v > 0)
          .slice(-14);
        let weekPct: number | null = null;
        const h = Array.isArray(preds) && preds[0]?.horizons?.one_week;
        if (h && typeof h.up === 'number' && typeof h.down === 'number') {
          // 预期中枢 = 涨跌概率差 × 半幅振幅（如 ±4% 振幅、概率差 -0.17 → -0.34%）
          const m = String(h.range_pct || h.range || '').match(/([\d.]+)\s*%/);
          // 旧实现是 `m ? ... : 0.02` —— 模型没给振幅时凭空补一个 2% 半幅，
          // 等于替模型编了一个波动率。振幅缺失就保持 null，由 UI 渲染"暂无预期"空态。
          const halfRange = m ? parseFloat(m[1]) / 200 : null;
          if (halfRange != null && halfRange > 0) {
            weekPct = (h.up - h.down) * halfRange;
          }
        }
        return { closes, weekPct, hasPred: !!h };
      })(),
    );
  }
  return _intelCache.get(key)!;
}

function useRowIntel(symbol: string, assetType: string) {
  const [intel, setIntel] = useState<RowIntel | null>(null);
  useEffect(() => {
    let live = true;
    getRowIntel(symbol, assetType).then((r) => live && setIntel(r)).catch(() => null);
    return () => { live = false; };
  }, [symbol, assetType]);
  return intel;
}

// 真实近期走势 + AI 预测虚线尾巴（一周预期终点）
const RowSparkline: React.FC<{ symbol: string; assetType: string }> = ({ symbol, assetType }) => {
  const intel = useRowIntel(symbol, assetType);
  const W = 84;
  const H = 22;
  if (!intel || intel.closes.length < 2) {
    return <div className="w-[84px] h-[22px] rounded bg-black/[0.03] animate-pulse" />;
  }
  const { closes, weekPct } = intel;
  const last = closes[closes.length - 1];
  const fcEnd = weekPct != null ? last * (1 + weekPct) : null;
  const all = fcEnd != null ? closes.concat([fcEnd]) : closes;
  const min = Math.min(...all);
  const max = Math.max(...all);
  const range = max - min || 1;
  const fcSlots = fcEnd != null ? 3 : 0; // 预测尾巴占 3 个槽位
  const totalSlots = closes.length - 1 + fcSlots;
  const pt = (v: number, i: number) => {
    const x = (i / totalSlots) * W;
    const y = H - ((v - min) / range) * (H - 6) - 3;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };
  const histPath = closes.map((c, i) => pt(c, i)).join(' L ');
  const lastXY = pt(last, closes.length - 1);
  const endXY = fcEnd != null ? pt(fcEnd, totalSlots) : null;
  const up = closes[closes.length - 1] >= closes[0];
  const stroke = up ? '#E03E3E' : '#34C759';

  return (
    <svg width={W} height={H} className="overflow-visible">
      <path d={`M ${histPath}`} fill="none" stroke={stroke} strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" />
      {endXY && (
        <>
          <line
            x1={lastXY.split(',')[0]} y1={lastXY.split(',')[1]}
            x2={endXY.split(',')[0]} y2={endXY.split(',')[1]}
            stroke="#5856D6" strokeWidth="1.5" strokeDasharray="3 2" strokeLinecap="round"
          />
          <circle cx={endXY.split(',')[0]} cy={endXY.split(',')[1]} r="1.8" fill="#5856D6" />
        </>
      )}
    </svg>
  );
};

// AI 预测（一周）预期方向胶囊
const ForecastCell: React.FC<{ symbol: string; assetType: string }> = ({ symbol, assetType }) => {
  const intel = useRowIntel(symbol, assetType);
  if (!intel) return <span className="text-[11px] text-[#C7C7CC]">…</span>;
  if (!intel.hasPred || intel.weekPct == null) {
    return (
      <Tooltip title="该标的还没有 AI 研判记录，进入详情页点击「生成时序研判」">
        <span className="text-[11px] text-[#A1A1A6] bg-black/[0.04] px-2 py-0.5 rounded-full">未研判</span>
      </Tooltip>
    );
  }
  const pct = intel.weekPct * 100;
  const up = pct > 0.15;
  const down = pct < -0.15;
  return (
    <Tooltip title={`AI 一周预期中枢 ${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%（涨跌概率差 × 预期振幅推算）`}>
      <span
        className={`inline-flex items-center gap-0.5 font-mono font-medium text-xs px-2.5 py-0.5 rounded-full ${
          up
            ? 'text-[#E03E3E] bg-[#E03E3E]/10'
            : down
            ? 'text-[#248A3D] bg-[#34C759]/10'
            : 'text-[#86868B] bg-black/[0.04]'
        }`}
      >
        {up ? <TrendingUp className="w-3 h-3" /> : down ? <TrendingDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
        {pct >= 0 ? '+' : ''}{pct.toFixed(2)}%
      </span>
    </Tooltip>
  );
};

export const WatchlistPage: React.FC = () => {
  const navigate = useNavigate();
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [newName, setNewName] = useState('');
  const [resolvedName, setResolvedName] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [importingAll, setImportingAll] = useState(false);
  const [undoItem, setUndoItem] = useState<any>(null);
  const undoTimerRef = useRef<any>(null);

  // 输入代码后自动联网解析标的名称（防抖），基金/股票都适用
  useEffect(() => {
    const sym = newSymbol.trim();
    if (!sym) {
      setResolvedName('');
      setNewName('');
      return;
    }
    const t = setTimeout(async () => {
      try {
        const q: any = await api.get(`/quote/${sym}`);
        if (q?.name) {
          setResolvedName(q.name);
          setNewName(q.name);
          return;
        }
      } catch { /* 非交易时段或接口失败则走基金档案 */ }
      try {
        const fp: any = await api.get('/fund/profile', { params: { symbol: sym } });
        if (fp?.name) {
          setResolvedName(fp.name);
          setNewName(fp.name);
          return;
        }
      } catch { /* 非基金 */ }
      setResolvedName('');
    }, 450);
    return () => clearTimeout(t);
  }, [newSymbol]);

  const fetchWatchlist = async () => {
    setLoading(true);
    try {
      const res: any = await api.get('/watchlist');
      if (Array.isArray(res)) {
        setItems(res);
      }
    } catch {
      message.error('获取自选池列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWatchlist();
  }, []);

  const handleAdd = async () => {
    if (!newSymbol.trim()) {
      message.warning('请输入代码或名称');
      return;
    }
    setSubmitting(true);
    try {
      await api.post('/watchlist', {
        symbol: newSymbol.trim(),
        name: newName.trim(),
      });
      message.success('已加入监控池');
      setAddModalOpen(false);
      setNewSymbol('');
      setNewName('');
      fetchWatchlist();
    } catch (e: any) {
      message.error(e.response?.data?.detail || '添加失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleQuickAdd = async (t: any) => {
    try {
      await api.post('/watchlist', {
        symbol: t.symbol,
        name: t.name,
        asset_type: t.asset_type,
      });
      message.success(`已关注 ${t.name}`);
      fetchWatchlist();
    } catch {
      message.error('添加失败');
    }
  };

  const handleAddAllRecommended = async () => {
    setImportingAll(true);
    try {
      for (const t of RECOMMENDED_TARGETS) {
        await api.post('/watchlist', {
          symbol: t.symbol,
          name: t.name,
          asset_type: t.asset_type,
        }).catch(() => null);
      }
      message.success('已成功批量导入推荐资产');
      fetchWatchlist();
    } catch {
      message.error('批量导入异常');
    } finally {
      setImportingAll(false);
    }
  };

  const handleDelete = async (item: any) => {
    try {
      await api.delete(`/watchlist/${item.symbol}`);
      setUndoItem(item);
      if (undoTimerRef.current) clearTimeout(undoTimerRef.current);
      undoTimerRef.current = setTimeout(() => {
        setUndoItem(null);
      }, 5000);
      fetchWatchlist();
    } catch {
      message.error('移出标的失败');
    }
  };

  const handleUndo = async () => {
    if (!undoItem) return;
    try {
      await api.post('/watchlist', {
        symbol: undoItem.symbol,
        name: undoItem.name,
        asset_type: undoItem.asset_type,
      });
      setUndoItem(null);
      if (undoTimerRef.current) clearTimeout(undoTimerRef.current);
      message.success(`已恢复 ${undoItem.name || undoItem.symbol}`);
      fetchWatchlist();
    } catch {
      message.error('撤销失败');
    }
  };

  const upCount = items.filter((it) => (it.change_pct ?? 0) > 0).length;
  const downCount = items.filter((it) => (it.change_pct ?? 0) < 0).length;
  const avgChg = items.length > 0
    ? items.reduce((acc, cur) => acc + (cur.change_pct ?? 0), 0) / items.length
    : 0;

  return (
    <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* ── 头部卡片 (Apple HIG) ── */}
      <div className="apple-card p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-2xl bg-black/[0.04] border border-black/[0.04] flex items-center justify-center text-[#1D1D1F]">
            <Star className="w-5 h-5 text-[#FF9500] fill-[#FF9500]" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-lg font-semibold text-[#1D1D1F] tracking-tight">
                自选监控池
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-black/[0.04] text-[#1D1D1F] border border-black/[0.06]">
                {items.length} 只标的
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2.5 self-start sm:self-auto">
          {items.length === 0 && (
            <button
              type="button"
              disabled={importingAll}
              onClick={handleAddAllRecommended}
              className="apple-press px-4 py-2 rounded-full text-xs font-medium text-[#1D1D1F] bg-black/[0.04] hover:bg-black/[0.08] transition-all"
            >
              {importingAll ? '正在导入…' : '一键导入精选宽基'}
            </button>
          )}
          <button
            type="button"
            onClick={() => setAddModalOpen(true)}
            className="apple-press flex items-center gap-1.5 px-5 py-2 rounded-full text-xs font-medium text-white bg-[#0071E3] hover:bg-[#0077ED] shadow-sm transition-all"
          >
            <Plus className="w-3.5 h-3.5" />
            添加标的
          </button>
        </div>
      </div>

      {/* ── 统计三卡片 ── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="apple-card p-5">
          <div className="text-xs font-medium text-[#86868B] mb-1.5">池内平均收益</div>
          <div className={`text-2xl font-semibold font-mono tracking-tight ${avgChg >= 0 ? 'text-[#E03E3E]' : 'text-[#248A3D]'}`}>
            {avgChg >= 0 ? `+${avgChg.toFixed(2)}%` : `${avgChg.toFixed(2)}%`}
          </div>
        </div>
        <div className="apple-card p-5">
          <div className="text-xs font-medium text-[#86868B] mb-1.5">今日上涨占比</div>
          <div className="text-2xl font-semibold font-mono tracking-tight text-[#1D1D1F]">
            {items.length > 0 ? `${((upCount / items.length) * 100).toFixed(0)}%` : '0%'}
            <span className="text-xs font-normal text-[#86868B] ml-2">({upCount} 涨 / {downCount} 跌)</span>
          </div>
        </div>
        <div className="apple-card p-5">
          <div className="text-xs font-medium text-[#86868B] mb-1.5">自动轮巡状态</div>
          <div className="text-sm font-semibold text-[#248A3D] tracking-tight flex items-center gap-2 mt-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#34C759] opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#34C759]" />
            </span>
            轮巡就绪
          </div>
        </div>
      </div>

      {/* ── 核心自选列表 (Apple Stocks Table) ── */}
      <div className="apple-card p-6">
        <Table
          dataSource={items.map((it) => ({ ...it, key: it.symbol }))}
          loading={loading || importingAll}
          pagination={false}
          size="middle"
          rowClassName="hover:bg-black/[0.015] transition-colors"
          columns={[
            {
              title: '标的名称 / 代码',
              key: 'name',
              render: (_, r) => (
                <div
                  className="cursor-pointer group flex items-center gap-2"
                  onClick={() => navigate(`/analysis?symbol=${r.symbol}&type=${r.asset_type || ''}`)}
                >
                  <div>
                    <div className="font-semibold text-[#1D1D1F] text-xs group-hover:text-[#0071E3] transition-colors">
                      {r.name}
                    </div>
                    <div className="font-mono text-[11px] text-[#86868B]">{r.symbol}</div>
                  </div>
                </div>
              ),
            },
            {
              title: '资产类别',
              dataIndex: 'asset_type',
              key: 'asset_type',
              render: (t) => (
                <span className="text-[11px] font-medium text-[#86868B] bg-black/[0.04] px-2.5 py-0.5 rounded-full">
                  {t === 'fund' ? '场内ETF/LOF' : t === 'ofund' ? '场外开放基金' : 'A股股票'}
                </span>
              ),
            },
            {
              title: '近两周走势 / AI 预测',
              key: 'sparkline',
              render: (_, r) => (
                <RowSparkline symbol={r.symbol} assetType={r.asset_type || ''} />
              ),
            },
            {
              title: 'AI 预测 (1周)',
              key: 'forecast',
              render: (_, r) => (
                <ForecastCell symbol={r.symbol} assetType={r.asset_type || ''} />
              ),
            },
            {
              title: '最新价 / 净值',
              dataIndex: 'price',
              key: 'price',
              render: (p) => (
                <span className="font-mono font-semibold text-xs text-[#1D1D1F]">
                  {p ? Number(p).toFixed(p < 10 ? 3 : 2) : '—'}
                </span>
              ),
            },
            {
              title: '今日涨跌幅',
              dataIndex: 'change_pct',
              key: 'change_pct',
              render: (v) => {
                // 缺失时不能兜底成 0 —— 会渲染出 "0.00%" 这个从未被观测到的涨跌幅，
                // 而且配色还是"日涨跌"的语义色，看起来和真实数据毫无区别。
                const num = typeof v === 'number' ? v : null;
                const isUp = num !== null && num > 0;
                const isDn = num !== null && num < 0;
                return (
                  <span
                    className={`font-mono font-medium text-xs px-2.5 py-0.5 rounded-full inline-block ${
                      isUp
                        ? 'text-[#E03E3E] bg-[#E03E3E]/10'
                        : isDn
                        ? 'text-[#248A3D] bg-[#34C759]/10'
                        : 'text-[#86868B] bg-black/[0.04]'
                    }`}
                  >
                    {num === null ? '—' : `${isUp ? '+' : ''}${num.toFixed(2)}%`}
                  </span>
                );
              },
            },
            {
              title: '操作',
              key: 'action',
              render: (_, r) => (
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => navigate(`/analysis?symbol=${r.symbol}&type=${r.asset_type || ''}`)}
                    className="apple-press px-3 py-1 rounded-full text-xs font-medium text-[#0071E3] bg-[#0071E3]/10 hover:bg-[#0071E3] hover:text-white transition-all"
                  >
                    进入研判
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDelete(r)}
                    className="apple-press w-7 h-7 rounded-full flex items-center justify-center text-[#86868B] hover:text-[#E03E3E] hover:bg-black/[0.04] transition-all cursor-pointer"
                    title="移出监控池"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ),
            },
          ]}
        />
      </div>

      {/* ── 推荐标的池 ── */}
      <div className="apple-card p-6">
        <h4 className="text-xs font-semibold text-[#86868B] uppercase tracking-wider mb-3">
          精选核心资产一键关注
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {RECOMMENDED_TARGETS.map((t) => (
            <div
              key={t.symbol}
              className="p-4 bg-[#F5F5F7] rounded-2xl border border-black/[0.03] flex flex-col justify-between"
            >
              <div>
                <div className="font-semibold text-xs text-[#1D1D1F]">{t.name}</div>
                <div className="font-mono text-[10px] text-[#86868B] mb-1.5">{t.symbol}</div>
                <p className="text-[11px] text-[#86868B] line-clamp-2 leading-relaxed">{t.desc}</p>
              </div>
              <button
                type="button"
                onClick={() => handleQuickAdd(t)}
                className="apple-press mt-3 w-full py-1.5 rounded-full text-xs font-medium text-[#1D1D1F] bg-white hover:bg-[#0071E3] hover:text-white border border-black/[0.06] transition-all shadow-2xs"
              >
                + 加入关注
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* ── 添加弹窗 ── */}
      <AppleModal
        open={addModalOpen}
        onClose={() => setAddModalOpen(false)}
        title="添加标的"
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
              disabled={submitting}
              onClick={handleAdd}
              className="px-6 py-2.5 rounded-full text-xs font-medium text-white bg-[#0071E3] hover:bg-[#0077ED] shadow-sm transition-all"
            >
              {submitting ? '正在加入…' : '确认添加'}
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
                placeholder="例如：161725 或 000997"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value)}
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3] font-mono"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#1D1D1F] mb-1">
                标的名称 (选填，输入代码后自动识别)
              </label>
              <input
                type="text"
                placeholder="输入代码后自动填充，也可手动填写"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="w-full bg-white rounded-xl px-3.5 py-2.5 text-xs text-[#1D1D1F] border border-black/[0.06] focus:outline-none focus:border-[#0071E3]"
              />
              {resolvedName ? (
                <div className="mt-1.5 flex items-center gap-1.5 text-[11px] font-medium text-[#248A3D]">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#34C759]" />
                  已识别：{resolvedName}
                </div>
              ) : (
                newSymbol.trim() && (
                  <div className="mt-1.5 text-[11px] text-[#86868B]">正在识别该代码对应的标的名…</div>
                )
              )}
            </div>
          </div>
        </div>
      </AppleModal>

      {/* ── Apple Agency & Forgiveness: 悬浮撤销胶囊 (Undo Capsule) ── */}
      {undoItem && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 animate-in fade-in slide-in-from-bottom-3 duration-200">
          <div className="apple-glass-card px-4 py-2 rounded-full shadow-[0_12px_32px_rgba(0,0,0,0.12)] flex items-center gap-3 text-xs border border-black/[0.06]">
            <span className="text-[#1D1D1F]">已移出标的 <b>{undoItem.name || undoItem.symbol}</b></span>
            <button
              type="button"
              onClick={handleUndo}
              className="apple-press text-[#0071E3] font-semibold hover:text-[#0077ED] border-none bg-transparent cursor-pointer pl-1"
            >
              撤销
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
