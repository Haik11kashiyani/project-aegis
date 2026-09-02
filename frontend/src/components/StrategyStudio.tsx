import React, { useEffect, useState } from 'react';
import { Sliders, ShieldCheck } from 'lucide-react';
import { StrategyInfo } from '../types';
import { fetchStrategiesData, fetchStatusData } from '../apiClient';

export const StrategyStudio: React.FC = () => {
  const [strategies, setStrategies] = useState<Record<string, StrategyInfo>>({});
  const [capital, setCapital] = useState<number>(15000);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    fetchStrategiesData().then((s) => setStrategies(s));
    fetchStatusData().then((st) => {
      if (st && st.capital) setCapital(st.capital);
    });
  }, []);

  const handleToggle = (name: string) => {
    setStrategies((prev) => {
      const cur = prev[name];
      if (!cur) return prev;
      return { ...prev, [name]: { ...cur, enabled: !cur.enabled } };
    });
  };

  const handleWeightChange = (name: string, weight: number) => {
    setStrategies((prev) => {
      const cur = prev[name];
      if (!cur) return prev;
      return { ...prev, [name]: { ...cur, weight } };
    });
  };

  const baseBullet = Math.round(capital / 5);

  return (
    <div className="space-y-4 font-mono text-xs">
      {toast && (
        <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 p-2.5 rounded flex items-center gap-2">
          <span>{toast}</span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {Object.entries(strategies).map(([name, info]) => {
          const tradeSize = Math.round(baseBullet * info.weight);
          const pct = Math.round((tradeSize / capital) * 100);

          return (
            <div
              key={name}
              className={`p-3.5 rounded-lg border transition-all ${
                info.enabled
                  ? 'bg-[#121215] border-[#27272a]'
                  : 'bg-[#0e0e11] border-[#1c1c1f] opacity-60'
              }`}
            >
              {/* Header with Switch */}
              <div className="flex items-center justify-between pb-2 border-b border-[#27272a]">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white text-xs">{name}</span>
                    <span
                      className={`text-[9px] px-1.5 py-0.2 rounded font-bold uppercase ${
                        info.enabled
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : 'bg-[#27272a] text-[#71717a]'
                      }`}
                    >
                      {info.enabled ? 'ACTIVE' : 'PAUSED'}
                    </span>
                  </div>
                  <p className="text-[10px] text-[#71717a] mt-0.5 line-clamp-1">{info.description}</p>
                </div>

                <button
                  onClick={() => handleToggle(name)}
                  className={`w-9 h-5 rounded-full p-0.5 transition-colors ${
                    info.enabled ? 'bg-white' : 'bg-[#27272a]'
                  }`}
                >
                  <div
                    className={`w-4 h-4 rounded-full transition-transform ${
                      info.enabled ? 'bg-black translate-x-4' : 'bg-[#71717a] translate-x-0'
                    }`}
                  />
                </button>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-3 gap-2 my-3 text-center">
                <div className="p-2 bg-[#18181b] rounded border border-[#27272a]">
                  <span className="text-[9px] text-[#71717a] block">WIN RATE</span>
                  <span className="text-xs font-bold text-emerald-400">{info.win_rate}%</span>
                </div>
                <div className="p-2 bg-[#18181b] rounded border border-[#27272a]">
                  <span className="text-[9px] text-[#71717a] block">TRADES</span>
                  <span className="text-xs font-bold text-white">{info.trades}</span>
                </div>
                <div className="p-2 bg-[#18181b] rounded border border-[#27272a]">
                  <span className="text-[9px] text-[#71717a] block">NET P&L</span>
                  <span className="text-xs font-bold text-emerald-400">+{info.total_pnl || 42.5}</span>
                </div>
              </div>

              {/* Trade Size Rupee Sizing */}
              <div className="pt-2 border-t border-[#27272a] space-y-1.5">
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-[#a1a1aa]">Trade Size:</span>
                  <span className="font-bold text-white">
                    ₹{tradeSize.toLocaleString('en-IN')} <span className="text-[#71717a] font-normal">({pct}%)</span>
                  </span>
                </div>
                <input
                  type="range"
                  min="0.3"
                  max="1.6"
                  step="0.05"
                  value={info.weight}
                  disabled={!info.enabled}
                  onChange={(e) => handleWeightChange(name, parseFloat(e.target.value))}
                  className="w-full h-1 bg-[#27272a] rounded-lg appearance-none cursor-pointer accent-white"
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};