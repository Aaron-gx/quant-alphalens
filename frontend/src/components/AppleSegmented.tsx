import React from 'react';

export interface SegmentedOption {
  value: string;
  label: string;
  icon?: React.ReactNode;
}

interface AppleSegmentedProps {
  options: SegmentedOption[];
  value: string;
  onChange: (value: string) => void;
  className?: string;
  size?: 'sm' | 'md';
}

export const AppleSegmented: React.FC<AppleSegmentedProps> = ({
  options,
  value,
  onChange,
  className = '',
  size = 'md',
}) => {
  return (
    <div
      className={`inline-flex items-center p-1 rounded-xl bg-[#E8E8ED]/75 border border-black/[0.04] select-none ${className}`}
      role="tablist"
    >
      {options.map((opt) => {
        const isSelected = opt.value === value;
        const paddingClass = size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-1.5 text-xs';

        return (
          <button
            key={opt.value}
            type="button"
            role="tab"
            aria-selected={isSelected}
            onClick={() => onChange(opt.value)}
            className={`apple-press flex items-center justify-center gap-1.5 rounded-lg font-medium transition-all duration-200 cursor-pointer border-none outline-none ${paddingClass} ${
              isSelected
                ? 'bg-white text-[#1D1D1F] font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.08)]'
                : 'text-[#86868B] hover:text-[#1D1D1F] hover:bg-white/40'
            }`}
          >
            {opt.icon && (
              <span className={`w-3.5 h-3.5 flex items-center justify-center transition-colors ${
                isSelected ? 'text-[#0071E3]' : 'text-[#86868B]'
              }`}>
                {opt.icon}
              </span>
            )}
            <span>{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
};
