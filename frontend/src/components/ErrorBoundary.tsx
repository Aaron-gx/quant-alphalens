import React from 'react';
import { AlertTriangle } from 'lucide-react';

interface Props {
  children: React.ReactNode;
  /** 错误提示标题 */
  label?: string;
}

interface State {
  error: Error | null;
}

/** 局部渲染兜底：子树抛错时显示友好卡片而不是整页白屏 */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="apple-card p-8 text-center space-y-3">
          <AlertTriangle className="w-6 h-6 text-[#FF9500] mx-auto" />
          <div className="text-sm font-semibold text-[#1D1D1F]">
            {this.props.label || '该模块'}渲染异常，已自动隔离
          </div>
          <div className="text-[11px] text-[#86868B] font-mono break-all max-w-xl mx-auto">
            {String(this.state.error.message || this.state.error)}
          </div>
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            className="apple-press px-4 py-1.5 rounded-full text-xs font-medium text-white bg-[#0071E3] hover:bg-[#0077ED]"
          >
            重试
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
