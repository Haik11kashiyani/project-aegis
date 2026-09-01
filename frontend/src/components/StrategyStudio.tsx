import React, { useState, useEffect, useRef } from 'react';
import { StrategyInfo } from '../types';
import { CheckCircle2, Sliders } from 'lucide-react';

interface StrategyStudioProps {
  strategies: Record<string, StrategyInfo>;
  onToggle: (name: string, enabled: boolean) => void;
  onWeightChange: (name: string, weight: number) => void;
}

export const StrategyStudio: React.FC<StrategyStudioProps> = ({
  strategies,
  onToggle,
  onWeightChange,
}) => {
  const [localWeights, setLocalWeights] = useState<Record<string, number>>({});
  const [toast, setToast] = useState<string | null>(null);
  const debounceTimers = useRef<Record<string, any>>({});
  const baseBullet = 3000;

  useEffect(() => {
    const initial: Record<string, number> = {};
    for (const [k, v] of Object.entries(strategies)) {
      initial[k] = v.weight;
    }
    setLocalWeights(initial);
  }, [strategies]);

  const handleSliderChange = (name: string, val: number) => {
    // 1. Update UI immediately (0ms response)
    setLocalWeights((prev) => ({ ...prev, [name]: val }));

    // 2. Debounce API network call (300ms) to prevent 40 requests/sec flooding
    if (debounceTimers.current[name]) {
      clearTimeout(debounceTimers.current[name]);
    }
    debounceTimers.current[name] = setTimeout(() => {
      onWeightChange(name, val);
      setToast(`Capital allocation for ${name} set to ₹${Math.round(baseBullet * val)} / trade`);
      setTimeout(() => setToast(null), 2500);
    }, 300);
  };

  const list = Object.entries(strategies);
  const activeCount = list.filter(([_, s]) => s.enabled).length;

  return (
    <div className="space-y-4 font-sans">
      {/* Context Banner */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-white">
                Strategy Capital Share & Position Sizing
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#18181b] border border-[#27272a] text-[#a1a1aa]">
                {activeCount} / {list.length} ACTIVE
              </span>
            </div>
            <p className="text-xs text-[#a1a1aa] mt-1">
              With your ₹15,000 capital, each bullet is base ₹3,000. Use the sliders to adjust the exact cash deployed per trade.
            </p>
          </div>
        </div>

        {toast && (
          <div className="mt-3 p-2 bg-[#18181b] border border-emerald-700/40 rounded text-emerald-400 text-xs font-mono flex items-center space-x-2">
            <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
            <span>{toast}</span>
          </div>
        )}
      </div>

      {/* Modern Monochrome Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3.5">
        {list.map(([name, s]) => {
          const currentWeight = localWeights[name] !== undefined ? localWeights[name] : s.weight;
          const winRate = s.win_rate || (s.trades > 0 && s.wins ? (s.wins / s.trades) * 100 : 55.0);
          const isHigh = winRate >= 55;
          const cashPerTrade = Math.round(baseBullet * currentWeight);
          const pctShare = Math.round((cashPerTrade / 15000) * 100);

          return (
            <div
              key={name}
              className={`rounded-lg p-4 border transition-all ${
                s.enabled
                  ? 'bg-[#121215] border-[#27272a] hover:border-[#3f3f46]'
                  : 'bg-[#09090b] border-[#18181b] opacity-40'
              }`}
            >
              {/* Header */}
              <div className="flex items-start justify-between gap-3 mb-2.5">
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="font-bold text-white text-sm tracking-tight">{name}</h3>
                    <span className={`text-[9px] font-mono px-1.5 py-0.2 rounded font-semibold ${
                      s.enabled ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-800/40' : 'bg-[#18181b] text-[#71717a]'
                    }`}>
                      {s.enabled ? 'ACTIVE' : 'PAUSED'}
                    </span>
                  </div>
                  <p className="text-[11px] text-[#71717a] mt-0.5 line-clamp-2">
                    {s.description || 'Quantitative algorithm'}
                  </p>
                </div>

                {/* Custom Monochrome Switch */}
                <button
                  type="button"
                  onClick={() => onToggle(name, !s.enabled)}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                    s.enabled ? 'bg-white' : 'bg-[#27272a]'
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-black shadow transition duration-200 ease-in-out ${
                      s.enabled ? 'translate-x-4' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-2 py-2.5 border-y border-[#1c1c20] my-2.5 font-mono text-center">
                <div className="bg-[#18181b] rounded p-1.5">
                  <div className="text-[9px] text-[#71717a] uppercase">Win Rate</div>
                  <div className={`text-xs font-bold ${isHigh ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {winRate.toFixed(1)}%
                  </div>
                </div>
                <div className="bg-[#18181b] rounded p-1.5">
                  <div className="text-[9px] text-[#71717a] uppercase">Trades</div>
                  <div className="text-xs font-bold text-white">{s.trades}</div>
                </div>
                <div className="bg-[#18181b] rounded p-1.5">
                  <div className="text-[9px] text-[#71717a] uppercase">Net P&L</div>
                  <div className="text-xs font-bold text-emerald-400">
                    +₹{s.total_pnl ? s.total_pnl.toFixed(1) : '0.0'}
                  </div>
                </div>
              </div>

              {/* Capital Share Slider */}
              <div className="space-y-1.5 pt-1 font-mono">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[#71717a] text-[11px]">Trade Size:</span>
                  <div className="flex items-center space-x-1.5">
                    <span className="font-bold text-white text-xs">₹{cashPerTrade.toLocaleString('en-IN')}</span>
                    <span className="text-[10px] text-[#71717a]">({pctShare}%)</span>
                  </div>
                </div>

                <input
                  type="range"
                  min="0.3"
                  max="1.6"
                  step="0.05"
                  value={currentWeight}
                  disabled={!s.enabled}
                  onChange={(e) => handleSliderChange(name, parseFloat(e.target.value))}
                  className="w-full h-1 bg-[#27272a] rounded appearance-none cursor-pointer accent-white disabled:opacity-30"
                />

                <div className="flex justify-between text-[9px] text-[#71717a]">
                  <span>₹900 (Defensive)</span>
                  <span>₹3,000 (Base)</span>
                  <span>₹4,800 (Aggressive)</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};