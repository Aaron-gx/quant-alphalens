import React, { useEffect } from 'react';
import { X } from 'lucide-react';

interface AppleModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  maxWidth?: string;
}

export const AppleModal: React.FC<AppleModalProps> = ({
  open,
  onClose,
  title,
  subtitle,
  icon,
  children,
  footer,
  maxWidth = 'max-w-lg',
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto animate-in fade-in duration-200">
      {/* ── macOS 深度毛玻璃背景遮罩 ── */}
      <div
        className="fixed inset-0 bg-black/35 backdrop-blur-xl transition-opacity"
        onClick={onClose}
      />

      {/* ── Apple Sheet 模态窗口 ── */}
      <div
        className={`relative w-full ${maxWidth} bg-[#FFFFFF] rounded-3xl border border-black/[0.08] shadow-[0_25px_60px_-15px_rgba(0,0,0,0.18)] p-6 sm:p-7 z-10 my-auto transform transition-all animate-in zoom-in-95 duration-200`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* macOS Sheet 顶部把手指示条 */}
        <div className="w-10 h-1 rounded-full bg-black/[0.12] mx-auto mb-4" />

        {/* 头部区域 */}
        <div className="flex items-start justify-between gap-4 mb-5">
          <div className="flex items-center gap-3">
            {icon && (
              <div className="w-10 h-10 rounded-2xl bg-black/[0.04] border border-black/[0.04] flex items-center justify-center text-[#1D1D1F] shrink-0">
                {icon}
              </div>
            )}
            <div>
              <h3 className="text-base font-semibold text-[#1D1D1F] tracking-tight">
                {title}
              </h3>
              {subtitle && (
                <p className="text-xs text-[#86868B] mt-0.5 font-normal">
                  {subtitle}
                </p>
              )}
            </div>
          </div>

          {/* 苹果原生圆形关闭按钮 */}
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-black/[0.05] hover:bg-black/[0.09] active:scale-95 flex items-center justify-center text-[#86868B] hover:text-[#1D1D1F] transition-all"
            aria-label="关闭"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* 内容主体 */}
        <div className="mb-6">{children}</div>

        {/* 底部操作栏 */}
        {footer && <div className="pt-2">{footer}</div>}
      </div>
    </div>
  );
};
