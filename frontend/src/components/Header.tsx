import React from 'react';
import { RefreshCw, Activity } from 'lucide-react';
import { SystemStatus } from '../types';

interface HeaderProps {
  status: SystemStatus | null;
  connected: boolean;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ status, connected, onRefresh }) => {
  const capital = status?.capital || 15000;
  const pnl = status ? status.realized_pnl + status.unrealized_pnl : 0;
  const pnlPct = (pnl / capital) * 100;
  const isUp = pnl >= 0;

  return (
    <header className="border-b border-[#27272a] bg-[#09090b] sticky top-0 z-50 px-6 py-2.5">
      <div className="max-w-[1680px] mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Left Identity */}
        <div className="flex items-center space-x-3">
          <div className="w-7 h-7 rounded bg-white flex items-center justify-center shadow-sm">
            <span className="font-mono font-bold text-xs text-black">A</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-sm text-white tracking-tight">
              AEGIS TERMINAL
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#18181b] text-[#a1a1aa] border border-[#27272a]">
              PRO
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/40 text-emerald-400 border border-emerald-800/40 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              AUTOPILOT ACTIVE
            </span>
          </div>

          <div className="h-4 w-px bg-[#27272a] hidden sm:block" />

          {/* Connection */}
          <div className="flex items-center space-x-2 text-[11px] font-mono text-[#71717a]">
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
            <span className="text-[#a1a1aa]">{connected ? 'FEED CONNECTED' : 'INITIALIZING'}</span>
            <span>•</span>
            <span>{status?.timestamp || '--:--:--'}</span>
          </div>
        </div>

        {/* Right Financial Strip */}
        <div className="flex flex-wrap items-center gap-2.5 font-mono text-xs">
          {/* Capital */}
          <div className="bg-[#121215] border border-[#27272a] rounded px-3 py-1">
            <span className="text-[10px] text-[#71717a] uppercase mr-2">Capital</span>
            <span className="font-semibold text-white">
              ₹{status?.equity ? status.equity.toLocaleString('en-IN', { minimumFractionDigits: 2 }) : '15,000.00'}
            </span>
          </div>

          {/* Realized P&L */}
          <div className={`border rounded px-3 py-1 ${
            isUp ? 'bg-emerald-950/30 border-emerald-800/50 text-emerald-400' : 'bg-rose-950/30 border-rose-800/50 text-rose-400'
          }`}>
            <span className="text-[10px] uppercase opacity-75 mr-2">Net P&L</span>
            <span className="font-semibold">
              {isUp ? '+' : ''}₹{pnl.toFixed(2)} ({isUp ? '+' : ''}{pnlPct.toFixed(2)}%)
            </span>
          </div>

          {/* Regime */}
          <div className="hidden md:block bg-[#121215] border border-[#27272a] rounded px-3 py-1">
            <span className="text-[10px] text-[#71717a] uppercase mr-2">Regime</span>
            <span className="text-amber-400 font-semibold">{status?.regime || 'BULLISH'}</span>
          </div>

          {/* Mode */}
          <div className="hidden lg:block bg-[#121215] border border-[#27272a] rounded px-3 py-1">
            <span className="text-[10px] text-[#71717a] uppercase mr-2">Mode</span>
            <span className="text-[#e4e4e7] font-semibold">{status?.trade_mode || 'PAPER'} (₹15K)</span>
          </div>

          {/* Refresh */}
          <button
            onClick={onRefresh}
            className="p-1.5 rounded bg-[#121215] hover:bg-[#18181b] border border-[#27272a] text-[#a1a1aa] hover:text-white transition-colors cursor-pointer"
            title="Refresh State"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </header>
  );
};