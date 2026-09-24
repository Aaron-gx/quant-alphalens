import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownViewProps {
  content?: string;
  className?: string;
}

export const MarkdownView: React.FC<MarkdownViewProps> = ({ content = '', className = '' }) => {
  if (!content) {
    return <span className="text-xs text-[#86868B]">暂无分析报告内容</span>;
  }

  return (
    <div className={`markdown-body text-[#1D1D1F] text-xs sm:text-[13px] leading-relaxed ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-base sm:text-[17px] font-bold text-[#1D1D1F] mt-4 mb-2 pb-1.5 border-b border-black/[0.06] tracking-tight">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-sm sm:text-[15px] font-bold text-[#1D1D1F] mt-3.5 mb-2 tracking-tight flex items-center gap-1.5">
              <span className="w-1.5 h-3.5 rounded-full bg-[#0071E3] inline-block" />
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-xs sm:text-[13px] font-semibold text-[#1D1D1F] mt-3 mb-1.5 tracking-tight">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-xs font-semibold text-[#86868B] mt-2 mb-1 uppercase tracking-wider">
              {children}
            </h4>
          ),
          p: ({ children }) => (
            <p className="my-2 text-xs sm:text-[13px] text-[#1D1D1F] leading-7 font-normal">
              {children}
            </p>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-[#1D1D1F] bg-black/[0.03] px-1 py-0.5 rounded">
              {children}
            </strong>
          ),
          ul: ({ children }) => (
            <ul className="my-2.5 pl-4 space-y-1.5 list-disc marker:text-[#0071E3]">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="my-2.5 pl-4 space-y-1.5 list-decimal marker:text-[#0071E3]">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="text-xs sm:text-[13px] text-[#1D1D1F] leading-6 pl-1">
              {children}
            </li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="my-3 pl-3.5 py-1.5 border-l-2 border-[#0071E3] bg-black/[0.02] rounded-r-lg text-xs text-[#86868B] italic">
              {children}
            </blockquote>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 rounded-lg border border-black/[0.06]">
              <table className="w-full text-left border-collapse text-xs">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-[#F5F5F7] text-[#86868B] font-semibold border-b border-black/[0.06]">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-black/[0.04]">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-black/[0.015] transition-colors">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 font-medium">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-[#1D1D1F]">
              {children}
            </td>
          ),
          code: ({ children }) => (
            <code className="font-mono text-[11px] px-1.5 py-0.5 bg-black/[0.05] rounded text-[#0071E3]">
              {children}
            </code>
          ),
          hr: () => (
            <hr className="my-4 border-black/[0.06]" />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
