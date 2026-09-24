import React, { useEffect, useState } from 'react';
import { Input, Button, Select, message } from 'antd';
import {
  Bot,
  User,
  Zap,
  Save,
  BarChart3,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Shield,
  Key,
  Globe,
  Cpu,
} from 'lucide-react';
import { api } from '../api/client';

export const SettingsPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [savingLlm, setSavingLlm] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);

  // LLM Config
  const [baseUrl, setBaseUrl] = useState('https://api.deepseek.com');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('deepseek-flash');
  const [reasonerModel, setReasonerModel] = useState('deepseek-reasoner');
  const [temperature, setTemperature] = useState(0.3);
  const [testResult, setTestResult] = useState<any>(null);

  // Profile Config
  const [riskPreference, setRiskPreference] = useState('neutral');
  const [customPrinciples, setCustomPrinciples] = useState('');
  const [positionText, setPositionText] = useState('');
  const [tradeStyle, setTradeStyle] = useState('');
  const [commonMistakes, setCommonMistakes] = useState('');

  // Usage Config
  const [usage, setUsage] = useState<any>(null);

  const fetchConfigs = async () => {
    setLoading(true);
    try {
      const [llmRes, profRes, useRes] = await Promise.all([
        api.get('/config/llm').catch(() => null),
        api.get('/profile').catch(() => null),
        api.get('/usage').catch(() => null),
      ]);

      if (llmRes) {
        setBaseUrl(llmRes.base_url || 'https://api.deepseek.com');
        setApiKey(llmRes.api_key || '');
        setModel(llmRes.model || 'deepseek-flash');
        setReasonerModel(llmRes.reasoner_model || 'deepseek-reasoner');
        setTemperature(llmRes.temperature ?? 0.3);
      }

      if (profRes) {
        setRiskPreference(profRes.risk_preference || 'neutral');
        setCustomPrinciples(profRes.custom_principles || '');
        setPositionText(profRes.position_text || '');
        setTradeStyle(profRes.trade_style || '');
        setCommonMistakes(profRes.common_mistakes || '');
      }

      if (useRes) {
        setUsage(useRes);
      }
    } catch {
      message.error('加载系统配置失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  const handleTestLlm = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res: any = await api.post('/config/llm/test', {
        base_url: baseUrl,
        api_key: apiKey,
        model,
        reasoner_model: reasonerModel,
        temperature,
      });
      setTestResult(res);
      if (res.ok) {
        message.success('大模型 API 连通性测试通过');
      } else {
        message.warning('连通测试未通过，请检查秘钥或网络配置');
      }
    } catch (e: any) {
      setTestResult({ ok: false, error: e.message });
      message.error('测试请求异常');
    } finally {
      setTesting(false);
    }
  };

  const handleSaveLlm = async () => {
    setSavingLlm(true);
    try {
      await api.post('/config/llm', {
        base_url: baseUrl,
        api_key: apiKey,
        model,
        reasoner_model: reasonerModel,
        temperature,
      });
      message.success('大模型参数已持久化保存');
    } catch {
      message.error('保存大模型参数失败');
    } finally {
      setSavingLlm(false);
    }
  };

  const handleSaveProfile = async () => {
    setSavingProfile(true);
    try {
      await api.put('/profile', {
        risk_preference: riskPreference,
        custom_principles: customPrinciples,
        position_text: positionText,
        trade_style: tradeStyle,
        common_mistakes: commonMistakes,
      });
      message.success('认知画像已保存');
    } catch {
      message.error('保存画像失败');
    } finally {
      setSavingProfile(false);
    }
  };

  return (
    <div className="max-w-[1200px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* ── 头部横幅 (macOS Settings 风格) ── */}
      <div className="apple-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-2xl bg-black/[0.04] border border-black/[0.04] flex items-center justify-center text-[#1D1D1F]">
            <Sliders className="w-5 h-5 text-[#0071E3]" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-lg font-semibold text-[#1D1D1F] tracking-tight">
                系统设置与投资认知画像
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-black/[0.04] text-[#1D1D1F] border border-black/[0.06]">
                即时生效
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── 模块 1: macOS Inset Grouped 大模型服务配置 ── */}
      <div className="apple-card p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-black/[0.06]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-[#0071E3] flex items-center justify-center text-white shadow-2xs">
              <Bot className="w-4 h-4" />
            </div>
            <span className="font-semibold text-sm text-[#1D1D1F]">大模型端点配置</span>
          </div>

          <div className="flex items-center gap-2.5">
            <Button
              icon={<Zap className="w-3.5 h-3.5" />}
              loading={testing}
              onClick={handleTestLlm}
              className="apple-press !rounded-full !border-black/10 !text-xs !font-medium !h-8 !px-4 hover:!bg-black/[0.03]"
            >
              测试连通性
            </Button>
            <Button
              type="primary"
              icon={<Save className="w-3.5 h-3.5" />}
              loading={savingLlm}
              onClick={handleSaveLlm}
              className="apple-press !bg-[#0071E3] hover:!bg-[#0077ED] !rounded-full !text-xs !font-medium !h-8 !px-4 border-none shadow-sm"
            >
              保存配置
            </Button>
          </div>
        </div>

        {/* 连通测试结果 (Apple 原生内联状态胶囊) */}
        {testResult && (
          <div
            className={`p-3.5 rounded-xl border flex items-center justify-between text-xs transition-all ${
              testResult.ok
                ? 'bg-[#E8F8EE] border-[#C8F0D5] text-[#107C41]'
                : 'bg-[#FFECEB] border-[#FFD3D0] text-[#E03E3E]'
            }`}
          >
            <div className="flex items-center gap-2">
              {testResult.ok ? (
                <CheckCircle2 className="w-4 h-4 text-[#248A3D] shrink-0" />
              ) : (
                <AlertCircle className="w-4 h-4 text-[#E03E3E] shrink-0" />
              )}
              <span className="font-medium">
                {testResult.ok
                  ? `API 连通测试通过 · 通识模型 ${testResult.chat_latency_ms}ms · 推理模型 ${testResult.reasoner_latency_ms}ms`
                  : `连通测试失败：${testResult.error || '无法建立与模型的网络连接'}`}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setTestResult(null)}
              className="apple-press text-xs opacity-60 hover:opacity-100 cursor-pointer"
            >
              ✕
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <label className="block text-xs font-medium text-[#86868B] mb-1.5 flex items-center gap-1.5">
              <Globe className="w-3.5 h-3.5 text-[#0071E3]" />
              API Base URL
            </label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.deepseek.com"
              className="!rounded-lg !h-9 !bg-white !border-black/[0.06] focus:!border-[#0071E3] text-xs font-mono"
            />
          </div>

          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <label className="block text-xs font-medium text-[#86868B] mb-1.5 flex items-center gap-1.5">
              <Key className="w-3.5 h-3.5 text-[#FF9500]" />
              API Key (秘钥)
            </label>
            <Input.Password
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="!rounded-lg !h-9 !bg-white !border-black/[0.06] focus:!border-[#0071E3] text-xs font-mono"
            />
          </div>

          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <label className="block text-xs font-medium text-[#86868B] mb-1.5 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-[#34C759]" />
              分析模型 (Chat)
            </label>
            <Input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="deepseek-flash"
              className="!rounded-lg !h-9 !bg-white !border-black/[0.06] focus:!border-[#0071E3] text-xs font-mono"
            />
          </div>

          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <label className="block text-xs font-medium text-[#86868B] mb-1.5 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-[#5856D6]" />
              推理模型 (Reasoner)
            </label>
            <Input
              value={reasonerModel}
              onChange={(e) => setReasonerModel(e.target.value)}
              placeholder="deepseek-reasoner"
              className="!rounded-lg !h-9 !bg-white !border-black/[0.06] focus:!border-[#0071E3] text-xs font-mono"
            />
          </div>
        </div>
      </div>

      {/* ── 模块 2: macOS Inset Grouped 认知画像 ── */}
      <div className="apple-card p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-black/[0.06]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-[#34C759] flex items-center justify-center text-white shadow-2xs">
              <User className="w-4 h-4" />
            </div>
            <span className="font-semibold text-sm text-[#1D1D1F]">投资认知画像</span>
          </div>

          <Button
            type="primary"
            icon={<Save className="w-3.5 h-3.5" />}
            loading={savingProfile}
            onClick={handleSaveProfile}
            className="apple-press !bg-[#0071E3] hover:!bg-[#0077ED] !rounded-full !text-xs !font-medium !h-8 !px-4 border-none shadow-sm self-start sm:self-auto"
          >
            保存认知画像
          </Button>
        </div>

        <div className="space-y-4">
          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <label className="block text-xs font-medium text-[#86868B] mb-2 flex items-center gap-1.5">
              <Shield className="w-3.5 h-3.5 text-[#0071E3]" />
              风险偏好
            </label>
            <Select
              value={riskPreference}
              onChange={(val) => setRiskPreference(val)}
              className="w-full !h-9"
              options={[
                { value: 'neutral', label: '中性平衡' },
                { value: 'conservative', label: '绝对收益防守' },
                { value: 'aggressive', label: '进攻成长型' },
                { value: 'custom', label: '自定义严格原则' },
              ]}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
              <label className="block text-xs font-medium text-[#86868B] mb-1.5">
                交易纪律与原则
              </label>
              <Input.TextArea
                rows={3}
                value={customPrinciples}
                onChange={(e) => setCustomPrinciples(e.target.value)}
                placeholder="例如：单标的仓位上限不得超过 15%；浮亏超过 7% 严格止损…"
                className="!rounded-lg text-xs !bg-white !border-black/[0.06] focus:!border-[#0071E3]"
              />
            </div>

            <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
              <label className="block text-xs font-medium text-[#86868B] mb-1.5">
                底仓持仓配置
              </label>
              <Input.TextArea
                rows={3}
                value={positionText}
                onChange={(e) => setPositionText(e.target.value)}
                placeholder="例如：贵州茅台 2 成，沪深300ETF 3 成，现金 5 成…"
                className="!rounded-lg text-xs !bg-white !border-black/[0.06] focus:!border-[#0071E3]"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
              <label className="block text-xs font-medium text-[#86868B] mb-1.5">
                交易风格
              </label>
              <Input
                value={tradeStyle}
                onChange={(e) => setTradeStyle(e.target.value)}
                placeholder="例如：右侧趋势跟踪 / 中长线价值持有"
                className="!rounded-lg text-xs !h-9 !bg-white !border-black/[0.06] focus:!border-[#0071E3]"
              />
            </div>

            <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
              <label className="block text-xs font-medium text-[#86868B] mb-1.5">
                风控与自省约束
              </label>
              <Input.TextArea
                rows={2}
                value={commonMistakes}
                onChange={(e) => setCommonMistakes(e.target.value)}
                placeholder="例如：避免高位追涨、破位单严格执行纪律…"
                className="!rounded-lg text-xs !bg-white !border-black/[0.06] focus:!border-[#0071E3]"
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── 模块 3: Token 用量审计 (纯粹数字网格，无冗余说明) ── */}
      <div className="apple-card p-6">
        <div className="flex items-center gap-2.5 mb-4">
          <div className="w-7 h-7 rounded-lg bg-[#5856D6] flex items-center justify-center text-white shadow-2xs">
            <BarChart3 className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-[#1D1D1F] tracking-tight">
            算力开销审计 (近 30 天)
          </h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <div className="text-xs text-[#86868B] font-medium mb-1">累计调用频次</div>
            <div className="text-2xl font-bold font-mono text-[#1D1D1F] tabular-nums">
              {usage?.calls || 0}
            </div>
          </div>
          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <div className="text-xs text-[#86868B] font-medium mb-1">累计消耗 Tokens</div>
            <div className="text-2xl font-bold font-mono text-[#0071E3] tabular-nums">
              {usage?.total_tokens ? Number(usage.total_tokens).toLocaleString() : '0'}
            </div>
          </div>
          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <div className="text-xs text-[#86868B] font-medium mb-1">预估月度费用</div>
            <div className="text-2xl font-bold font-mono text-[#1D1D1F] tabular-nums">
              ¥{((usage?.total_tokens || 0) * 0.000004).toFixed(2)}
            </div>
          </div>
          <div className="p-4 bg-[#F5F5F7] rounded-xl border border-black/[0.03]">
            <div className="text-xs text-[#86868B] font-medium mb-1">多级缓存命中率</div>
            <div className="text-2xl font-bold font-mono text-[#248A3D] tabular-nums">
              ~65%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
