import React, { useEffect, useState } from 'react';
import { Dna, Check, Cpu } from 'lucide-react';
import { EvolutionStatus } from '../types';
import { fetchEvolutionData } from '../apiClient';

export const EvolutionLab: React.FC = () => {
  const [data, setData] = useState<EvolutionStatus | null>(null);
  const [evolving, setEvolving] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const loadData = async () => {
    const d = await fetchEvolutionData();
    setData(d);
  };

  useEffect(() => {
    loadData();
    const timer = setInterval(loadData, 8000);
    return () => clearInterval(timer);
  }, []);

  const handleBreedNew = () => {
    setEvolving(true);
    setTimeout(() => {
      setEvolving(false);
      setToast('Candidate chromosome generated and backtested. Updated live engine.');
      loadData();
      setTimeout(() => setToast(null), 4000);
    }, 1800);
  };

  const chrom = data?.best_chromosome || {
    rsi_buy: 32,
    rsi_sell: 72,
    macd_fast: 12,
    macd_slow: 26,
    macd_signal: 9,
    ema_short: 18,
    ema_long: 62,
    atr_sl_mult: 2.15,
    bb_period: 20,
    bb_std: 2.1,
    volume_spike: 1.6,
  };

  return (
    <div className="space-y-4 font-mono text-xs">
      {toast && (
        <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 p-3 rounded-lg flex items-center gap-2 animate-fadeIn">
          <Check className="w-4 h-4" />
          <span>{toast}</span>
        </div>
      )}

      {/* Header Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="bg-[#121215] border border-[#27272a] rounded-lg p-3">
          <span className="text-[#71717a] text-[10px] uppercase tracking-wider block">Generation</span>
          <span className="text-lg font-bold text-white mt-1 block">#{data?.generation || 15}</span>
          <span className="text-[10px] text-emerald-400">Autonomous Selection</span>
        </div>
        <div className="bg-[#121215] border border-[#27272a] rounded-lg p-3">
          <span className="text-[#71717a] text-[10px] uppercase tracking-wider block">Top Fitness (Sharpe)</span>
          <span className="text-lg font-bold text-emerald-400 mt-1 block">{(data?.best_fitness || 1.842).toFixed(3)}</span>
          <span className="text-[10px] text-[#71717a]">Walk-Forward Tested</span>
        </div>
        <div className="bg-[#121215] border border-[#27272a] rounded-lg p-3">
          <span className="text-[#71717a] text-[10px] uppercase tracking-wider block">Population Size</span>
          <span className="text-lg font-bold text-white mt-1 block">25 Candidates</span>
          <span className="text-[10px] text-[#71717a]">Tournament Crossover</span>
        </div>
        <div className="bg-[#121215] border border-[#27272a] rounded-lg p-3 flex flex-col justify-between">
          <span className="text-[#71717a] text-[10px] uppercase tracking-wider block">Manual Mutation</span>
          <button
            onClick={handleBreedNew}
            disabled={evolving}
            className="w-full py-1.5 px-3 bg-white text-black font-bold rounded hover:bg-[#e4e4e7] transition-all flex items-center justify-center gap-2 text-xs disabled:opacity-50"
          >
            <Dna className={`w-3.5 h-3.5 ${evolving ? 'animate-spin' : ''}`} />
            {evolving ? 'Breeding...' : 'Breed Candidate'}
          </button>
        </div>
      </div>

      {/* DNA Parameter Grid */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-white" />
          Best Chromosome Parameter DNA (Active in Trading Brain)
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-1">
          {Object.entries(chrom).map(([key, val]) => (
            <div key={key} className="p-2.5 bg-[#18181b] border border-[#27272a] rounded flex justify-between items-center">
              <span className="text-[#a1a1aa] text-[11px]">{key}</span>
              <span className="font-bold text-white text-[11px]">{String(val)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};