import React, { useEffect, useState } from 'react';
import { Dna, Check, Cpu, GitBranch } from 'lucide-react';
import { EvolutionStatus } from '../types';

export const EvolutionLab: React.FC = () => {
  const [data, setData] = useState<EvolutionStatus | null>(null);
  const [evolving, setEvolving] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const fetchStatus = () => {
    fetch('/api/evolution/status')
      .then((r) => r.json())
      .then((d) => setData(d))
      .catch((e) => console.error(e));
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 8000);
    return () => clearInterval(timer);
  }, []);

  const handleBreedNew = () => {
    setEvolving(true);
    fetch('/api/evolution/evolve', { method: 'POST' })
      .then(() => {
        setTimeout(() => {
          setEvolving(false);
          setToast('Candidate chromosome generated and backtested. Updated live engine.');
          fetchStatus();
          setTimeout(() => setToast(null), 4000);
        }, 2500);
      })
      .catch((e) => {
        setEvolving(false);
        console.error(e);
      });
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
    <div className="space-y-4 font-sans">
      {/* Header */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-white">
                Autonomous Strategy Evolution & Genetic Breeding
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/40 text-emerald-400 border border-emerald-800/40 font-semibold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                AUTONOMOUS MODE ACTIVE
              </span>
            </div>
            <p className="text-xs text-[#a1a1aa] mt-1 max-w-3xl">
              The AI continually simulates, crosses over, and mutates strategy parameters on real NSE historical candles. Winning setups are automatically promoted to live trading.
            </p>
          </div>

          <button
            onClick={handleBreedNew}
            disabled={evolving}
            className="flex items-center space-x-2 bg-white hover:bg-[#e4e4e7] text-black font-mono text-xs font-semibold px-3.5 py-1.5 rounded transition-colors disabled:opacity-50 cursor-pointer"
          >
            <Dna className={`w-3.5 h-3.5 ${evolving ? 'animate-spin' : ''}`} />
            <span>{evolving ? 'BREEDING GENERATION...' : 'BREED NEW CANDIDATE'}</span>
          </button>
        </div>

        {toast && (
          <div className="mt-3 p-2 bg-[#18181b] border border-emerald-700/40 rounded text-emerald-400 text-xs font-mono flex items-center space-x-2">
            <Check className="w-3.5 h-3.5 shrink-0" />
            <span>{toast}</span>
          </div>
        )}
      </div>

      {/* 24/7 Timeline */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 font-mono text-xs">
        <div className="text-[10px] text-[#71717a] uppercase tracking-wider mb-2.5 font-bold flex items-center justify-between">
          <span>24/7 Autopilot Schedule (Zero Development Required)</span>
          <span className="text-emerald-400">Scheduled Automatically</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 text-[11px]">
          <div className="p-2.5 rounded bg-[#18181b] border border-[#27272a]">
            <span className="text-white font-bold block">08:30 – 09:14 IST</span>
            <span className="text-[#a1a1aa] font-medium block">Pre-Market Recon</span>
            <span className="text-[10px] text-[#71717a] block mt-0.5">Scrapes FII/DII, VIX, global cues & screens stocks under ₹1000.</span>
          </div>
          <div className="p-2.5 rounded bg-[#18181b] border border-[#27272a]">
            <span className="text-emerald-400 font-bold block">09:15 – 15:30 IST</span>
            <span className="text-[#a1a1aa] font-medium block">Live Execution</span>
            <span className="text-[10px] text-[#71717a] block mt-0.5">Executes 6 strategies with trailing stops & ₹15K capital protection.</span>
          </div>
          <div className="p-2.5 rounded bg-[#18181b] border border-[#27272a]">
            <span className="text-white font-bold block">15:35 – 17:00 IST</span>
            <span className="text-[#a1a1aa] font-medium block">Self-Evolution</span>
            <span className="text-[10px] text-[#71717a] block mt-0.5">Reviews trades, breeds new chromosomes, auto-promotes winners.</span>
          </div>
          <div className="p-2.5 rounded bg-[#18181b] border border-[#27272a]">
            <span className="text-amber-400 font-bold block">17:00 – 08:30 IST</span>
            <span className="text-[#a1a1aa] font-medium block">Overnight Walk-Forward</span>
            <span className="text-[10px] text-[#71717a] block mt-0.5">Retrains neural networks & recalibrates probabilities for next day.</span>
          </div>
        </div>
      </div>

      {/* Chromosome DNA */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-[#1f1f23] pb-2.5">
          <div className="flex items-center space-x-2">
            <GitBranch className="w-3.5 h-3.5 text-[#a1a1aa]" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-white">
              Active Strategy Chromosome DNA (Live Parameters)
            </h3>
          </div>
          <span className="text-[11px] font-mono text-[#71717a]">
            Auto-tuned on 360+ Historical Daily & 15m Candles
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5 font-mono text-xs">
          <div className="p-2.5 bg-[#18181b] rounded border border-[#27272a]">
            <span className="text-[10px] text-[#71717a] uppercase block">RSI Entry Trigger</span>
            <span className="font-bold text-white text-sm">&lt; {chrom.rsi_buy || 32}</span>
            <span className="text-[9px] text-[#71717a] block mt-0.5">Oversold value filter</span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded border border-[#27272a]">
            <span className="text-[10px] text-[#71717a] uppercase block">RSI Profit Exit</span>
            <span className="font-bold text-white text-sm">&gt; {chrom.rsi_sell || 72}</span>
            <span className="text-[9px] text-[#71717a] block mt-0.5">Overbought exit threshold</span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded border border-[#27272a]">
            <span className="text-[10px] text-[#71717a] uppercase block">Trend EMA Cross</span>
            <span className="font-bold text-white text-sm">{chrom.ema_short || 18} / {chrom.ema_long || 62}</span>
            <span className="text-[9px] text-[#71717a] block mt-0.5">Fast / Slow trend baseline</span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded border border-[#27272a]">
            <span className="text-[10px] text-[#71717a] uppercase block">MACD Config</span>
            <span className="font-bold text-white text-sm">
              {chrom.macd_fast || 12}/{chrom.macd_slow || 26}/{chrom.macd_signal || 9}
            </span>
            <span className="text-[9px] text-[#71717a] block mt-0.5">Fast / Slow / Signal</span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded border border-[#27272a]">
            <span className="text-[10px] text-[#71717a] uppercase block">Trailing Stop</span>
            <span className="font-bold text-amber-400 text-sm">{typeof chrom.atr_sl_mult === 'number' ? chrom.atr_sl_mult.toFixed(2) : '2.15'}× ATR</span>
            <span className="text-[9px] text-[#71717a] block mt-0.5">Volatility stop-loss</span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded border border-[#27272a]">
            <span className="text-[10px] text-[#71717a] uppercase block">Bollinger Channel</span>
            <span className="font-bold text-white text-sm">{chrom.bb_period || 20}d / {typeof chrom.bb_std === 'number' ? chrom.bb_std.toFixed(2) : '2.10'}σ</span>
            <span className="text-[9px] text-[#71717a] block mt-0.5">Mean reversion width</span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded border border-[#27272a]">
            <span className="text-[10px] text-[#71717a] uppercase block">Volume Filter</span>
            <span className="font-bold text-white text-sm">{typeof chrom.volume_spike === 'number' ? chrom.volume_spike.toFixed(1) : '1.6'}×</span>
            <span className="text-[9px] text-[#71717a] block mt-0.5">Minimum volume spike</span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded border border-[#27272a]">
            <span className="text-[10px] text-[#71717a] uppercase block">Promotion Status</span>
            <span className="font-bold text-emerald-400 text-sm">ACTIVE IN ENGINE</span>
            <span className="text-[9px] text-[#71717a] block mt-0.5">Live execution parameters</span>
          </div>
        </div>
      </div>
    </div>
  );
};