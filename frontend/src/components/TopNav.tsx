import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import { AutoComplete, Input } from 'antd';
import {
  Compass,
  LineChart,
  Star,
  Briefcase,
  CheckCircle2,
  BrainCircuit,
  Settings,
  Search,
  Activity,
} from 'lucide-react';
import { api } from '../api/client';

export const TopNav: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchOptions, setSearchOptions] = useState<any[]>([]);
  const [searchValue, setSearchValue] = useState('');
  const [llmReady, setLlmReady] = useState(true);

  useEffect(() => {
    api.get('/health')
      .then((res: any) => {
        setLlmReady(Boolean(res?.llm_configured));
      })
      .catch(() => setLlmReady(false));
  }, []);

  const handleSearch = async (val: string) => {
    setSearchValue(val);
    if (!val || val.trim().length === 0) {
      setSearchOptions([]);
      return;
    }
    try {
      const results: any = await api.get('/search', { params: { q: val.trim(), limit: 8 } });
      if (Array.isArray(results)) {
        setSearchOptions(
          results.map((item: any) => ({
            value: item.symbol,
            label: (
              <div className="flex items-center justify-between py-1.5 px-1 apple-press">
                <span className="font-semibold text-[#1D1D1F]">{item.name}</span>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-mono text-[#86868B] bg-black/[0.04] px-1.5 py-0.5 rounded">
                    {item.symbol}
                  </span>
                  <span className="text-[11px] text-[#0071E3] font-medium">
                    {item.type}
                  </span>
                </div>
              </div>
            ),
          }))
        );
      }
    } catch {
      setSearchOptions([]);
    }
  };

  const onSelect = (symbol: string) => {
    setSearchValue('');
    setSearchOptions([]);
    navigate(`/analysis?symbol=${symbol}`);
  };

  const navItems = [
    { label: '概览', path: '/', icon: <Compass className="w-4 h-4 md:w-3.5 md:h-3.5" /> },
    { label: '研判', path: '/analysis', icon: <LineChart className="w-4 h-4 md:w-3.5 md:h-3.5" /> },
    { label: '自选', path: '/watchlist', icon: <Star className="w-4 h-4 md:w-3.5 md:h-3.5" /> },
    { label: '模拟', path: '/paper', icon: <Briefcase className="w-4 h-4 md:w-3.5 md:h-3.5" /> },
    { label: '对账', path: '/verify', icon: <CheckCircle2 className="w-4 h-4 md:w-3.5 md:h-3.5" /> },
    { label: '自学', path: '/learn', icon: <BrainCircuit className="w-4 h-4 md:w-3.5 md:h-3.5" /> },
    { label: '设置', path: '/settings', icon: <Settings className="w-4 h-4 md:w-3.5 md:h-3.5" /> },
  ];

  return (
    <>
      {/* ── 顶部主导航栏 (macOS Glass Header) ── */}
      <header className="sticky top-0 z-40 apple-glass">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-8 h-14 flex items-center justify-between gap-4">
          {/* Logo 与品牌 (Apple Minimalist Typography) */}
          <div
            onClick={() => navigate('/')}
            className="apple-press flex items-center gap-2.5 cursor-pointer select-none group shrink-0"
          >
            <div className="w-7 h-7 rounded-lg bg-[#1D1D1F] flex items-center justify-center text-white shadow-xs">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[15px] font-bold tracking-tight text-[#1D1D1F] group-hover:text-[#0071E3] transition-colors">
                AlphaLens
              </span>
              <span className="text-[10px] font-bold tracking-wider px-1.5 py-0.5 rounded bg-black/[0.05] text-[#86868B] uppercase">
                PRO
              </span>
            </div>
          </div>

          {/* 桌面端：主导航 Tab (Apple Segmented Tray) */}
          <nav className="hidden md:flex items-center p-1 rounded-xl bg-[#E8E8ED]/70 border border-black/[0.04]">
            {navItems.map((item) => {
              const isActive =
                item.path === '/'
                  ? location.pathname === '/'
                  : location.pathname.startsWith(item.path);
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={`apple-press flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 select-none ${
                    isActive
                      ? 'bg-white text-[#1D1D1F] font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.08)]'
                      : 'text-[#86868B] hover:text-[#1D1D1F] hover:bg-white/40'
                  }`}
                >
                  <span className={isActive ? 'text-[#0071E3]' : 'text-[#86868B]'}>
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>

          {/* 右侧：Spotlight 搜索框 + 状态 */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-44 sm:w-56 lg:w-64">
              <AutoComplete
                options={searchOptions}
                onSearch={handleSearch}
                onSelect={onSelect}
                value={searchValue}
                className="w-full"
                popupMatchSelectWidth={300}
              >
                <Input
                  placeholder="搜索代码或名称..."
                  prefix={<Search className="w-3.5 h-3.5 text-[#86868B] mr-1" />}
                  allowClear
                  size="small"
                  className="!rounded-full !bg-[#E8E8ED]/60 hover:!bg-white focus:!bg-white !border-black/[0.06] hover:!border-[#0071E3]/50 focus:!border-[#0071E3] !text-xs !py-1 !px-3 transition-colors"
                />
              </AutoComplete>
            </div>

            {/* 状态指示胶囊 */}
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium text-[#86868B] bg-black/[0.03] border border-black/[0.04]">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  llmReady ? 'bg-[#34C759]' : 'bg-[#FF9500]'
                }`}
              />
              <span>{llmReady ? '量化引擎就绪' : '离线状态'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* ── 移动端 iOS 悬浮毛玻璃底栏 (Floating Bottom Bar for < md) ── */}
      <div className="md:hidden fixed bottom-3 inset-x-3 z-50">
        <nav className="apple-glass-card rounded-2xl px-2 py-1.5 flex items-center justify-around shadow-[0_12px_32px_rgba(0,0,0,0.12)]">
          {navItems.map((item) => {
            const isActive =
              item.path === '/'
                ? location.pathname === '/'
                : location.pathname.startsWith(item.path);
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={`apple-press flex flex-col items-center justify-center py-1 px-2.5 rounded-xl transition-all duration-200 select-none ${
                  isActive
                    ? 'text-[#0071E3] font-semibold'
                    : 'text-[#86868B] hover:text-[#1D1D1F]'
                }`}
              >
                <span className={`transition-transform duration-200 ${isActive ? 'scale-110' : ''}`}>
                  {item.icon}
                </span>
                <span className="text-[10px] mt-0.5 tracking-tight font-medium">
                  {item.label}
                </span>
              </NavLink>
            );
          })}
        </nav>
      </div>
    </>
  );
};
