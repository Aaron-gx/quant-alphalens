import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { TopNav } from './components/TopNav';
import { Overview } from './pages/Overview';
import { Analysis } from './pages/Analysis';
import { WatchlistPage } from './pages/Watchlist';
import { PaperTradePage } from './pages/PaperTrade';
import { VerifyPage } from './pages/Verify';
import { LearnCenterPage } from './pages/LearnCenter';
import { SettingsPage } from './pages/Settings';

export const App: React.FC = () => {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#0071E3',
          colorLink: '#0071E3',
          colorSuccess: '#34C759',
          colorError: '#E03E3E',
          colorWarning: '#FF9500',
          colorTextBase: '#1D1D1F',
          colorTextSecondary: '#86868B',
          colorBgBase: '#FFFFFF',
          colorBgLayout: '#F5F5F7',
          colorBorder: 'rgba(0, 0, 0, 0.08)',
          colorBorderSecondary: 'rgba(0, 0, 0, 0.04)',
          borderRadius: 12,
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
        },
      }}
    >
      <BrowserRouter>
        <div className="min-h-screen bg-[#F5F5F7] text-[#1D1D1F] flex flex-col selection:bg-[#0071E3]/15 selection:text-[#0071E3]">
          <TopNav />
          <main className="flex-1 pb-16">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/watchlist" element={<WatchlistPage />} />
              <Route path="/paper" element={<PaperTradePage />} />
              <Route path="/verify" element={<VerifyPage />} />
              <Route path="/learn" element={<LearnCenterPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
          <footer className="py-6 border-t border-black/[0.05] bg-[#F5F5F7] text-center text-xs text-[#86868B]">
            AlphaLens · 专业量化投研与决策辅助系统 · 所有模型测算与模拟交易记录仅供学术研究，不构成任何投资建议
          </footer>
        </div>
      </BrowserRouter>
    </ConfigProvider>
  );
};

export default App;
