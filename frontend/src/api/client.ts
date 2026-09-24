import axios from 'axios';

const axiosInstance = axios.create({
  baseURL: '/api',
  timeout: 300000, // 预测链路涉及东财数据+新闻打分+LLM 裁决，放宽到 5 分钟
  headers: {
    'Content-Type': 'application/json',
  },
});

axiosInstance.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const msg = error.response?.data?.detail || error.message || '请求服务发生异常';
    console.error('API Error:', msg);
    return Promise.reject(error);
  }
);

// in-flight GET 去重：React StrictMode dev 下 effect 双执行会导致同一接口连打两次
//（/news 这类重接口会触发 LLM 打分，重复请求非常昂贵）。同 url+params 并发时合并为一个请求。
const _inflight = new Map<string, Promise<any>>();

export const api = {
  get: <T = any>(url: string, config?: any): Promise<T> => axiosInstance.get(url, config) as Promise<T>,
  /** 去重 GET：并发期同 key 请求共享同一个 Promise */
  getDedup: <T = any>(url: string, config?: any): Promise<T> => {
    const key = url + '|' + JSON.stringify(config?.params || {});
    let p = _inflight.get(key);
    if (!p) {
      p = axiosInstance.get(url, config).finally(() => _inflight.delete(key));
      _inflight.set(key, p);
    }
    return p as Promise<T>;
  },
  post: <T = any>(url: string, data?: any, config?: any): Promise<T> => axiosInstance.post(url, data, config) as Promise<T>,
  put: <T = any>(url: string, data?: any, config?: any): Promise<T> => axiosInstance.put(url, data, config) as Promise<T>,
  delete: <T = any>(url: string, config?: any): Promise<T> => axiosInstance.delete(url, config) as Promise<T>,
};

export interface QuoteData {
  ok?: boolean;
  symbol: string;
  name?: string;
  asset_type?: string;
  last?: number;
  price?: number;
  change_pct?: number;
  high?: number;
  low?: number;
  open?: number;
  prev_close?: number;
  volume_lots?: number;
  amount?: number;
  turnover_rate?: number;
  pe?: number;
  pb?: number;
  total_mv?: number;
  float_mv?: number;
  inner?: number;
  outer?: number;
  asks?: Array<{ level: number; price: number; volume_lots: number }>;
  bids?: Array<{ level: number; price: number; volume_lots: number }>;
  estimate_nav?: number;
  estimate_pct?: number;
  estimate_time?: string;
  nav?: number;
  nav_date?: string;
  source?: string;
  error?: string;
}

export interface WatchlistItem {
  id: number;
  symbol: string;
  name: string;
  asset_type: string;
  auto_predict: boolean;
  price?: number | null;
  change_pct?: number | null;
  added_at: string;
}

export interface PredictionData {
  id: number;
  symbol: string;
  name: string;
  base_date: string;
  base_price?: number;
  confidence: number;
  // ── 口径留痕字段（后端 /predictions/{pid} 原样返回）──────────────────────
  // 界面必须能说清「这条研判是谁、用什么、什么时候产出的」，
  // 才有资格把结论摆给用户看。
  asset_type?: string;
  engine?: string;
  mode?: string;
  llm_model?: string;
  created_at?: string;
  input_snapshot?: any;
  // ── 各周期到期日（后端按**交易日**口径算好，含真实交易日历）────────────────
  // 前端**不要**自己拿 base_date 加天数推日期：前端只有"排除周末"这一档精度，
  // 碰到国庆/春节就会算错，于是图上标的到期日和对账记录里的到期日对不上。
  // due_source 是口径留痕：index = 真实交易日历，weekday = 仅排除周末（可能偏早）。
  due_dates?: Record<string, string>;
  due_source?: 'index' | 'weekday' | 'empty';
  horizons: Record<string, { up: number; flat: number; down: number; range?: string; range_pct?: string }>;
  scenarios?: {
    optimistic?: { prob: number; range: string; target?: number; condition: string };
    base?: { prob: number; range: string; target?: number; condition: string };
    pessimistic?: { prob: number; range: string; target?: number; condition: string };
  };
  action?: {
    advice?: string;
    entry?: number | string;
    target?: number | string;
    stop_loss?: number | string;
  };
  risks?: string[];
  key_factors?: string[];
  summary?: string;
  evals?: Array<{
    horizon: string;
    actual_change_pct: number;
    hit: boolean;
    brier: number;
  }>;
}

export interface PaperAccountItem {
  id: number;
  name: string;
  initial_cash: number;
  cash: number;
  mode: string;
  note: string;
  created_at: string;
  total_value: number;
  total_return: number;
  positions_count: number;
}
