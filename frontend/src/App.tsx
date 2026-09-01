import React, { useEffect, useState } from 'react';
import {
  Activity,
  Cpu,
  BarChart2,
  Settings,
  Radar,
  RotateCw,
  Clock,
  ShieldCheck,
  TrendingUp,
  FileText,
  Sliders,
} from 'lucide-react';
import { SystemStatus, StrategyInfo, TradeRecord } from './types';
import { LiveChart } from './components/LiveChart';
import { StrategyStudio } from './components/StrategyStudio';
import { EvolutionLab } from './components/EvolutionLab';
import { MarketRadar } from './components/MarketRadar';
import { TradesHistory } from './components/TradesHistory';
import { SettingsStudio } from './components/SettingsStudio';
import { isLocalhost, fetchStatusData, fetchStrategiesData, fetchTradesData } from './apiClient';

export const App: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [strategies, setStrategies] = useState<Record<string, StrategyInfo>>({});
  const [trades, setTrades] = useState<TradeRecord[]>([]);
  const [connected, setConnected] = useState<boolean>(!isLocalhost);
  const [activeTab, setActiveTab] = useState<'cockpit' | 'strategies' | 'evolution' | 'radar' | 'audit' | 'settings'>('cockpit');

  const refreshAll = async () => {
    const s = await fetchStatusData();
    setStatus(s);
    const strats = await fetchStrategiesData();
    setStrategies(strats);
    const t = await fetchTradesData();
    setTrades(t);
  };

  useEffect(() => {
    refreshAll();
    const poll = setInterval(refreshAll, 4000);

    // Only connect local WebSocket if running locally on localhost
    if (isLocalhost) {
      let ws: WebSocket | null = null;
      let timer: any = null;

      const connectWS = () => {
        try {
          ws = new WebSocket('ws://127.0.0.1:8000/ws/stream');
          ws.onopen = () => setConnected(true);
          ws.onclose = () => {
            setConnected(false);
            timer = setTimeout(connectWS, 5000);
          };
          ws.onerror = () => {
            setConnected(false);
            try { ws?.close(); } catch(e) {}
          };
          ws.onmessage = (evt) => {
            try {
              const msg = JSON.parse(evt.data);
              if (msg.type === 'TICK') {
                setStatus((prev) => (prev ? { ...prev, ...msg } : msg));
              }
            } catch (e) {}
          };
        } catch (e) {
          setConnected(false);
        }
      };

      connectWS();
      return () => {
        clearInterval(poll);
        if (timer) clearTimeout(timer);
        try { ws?.close(); } catch(e) {}
      };
    }

    // On Vercel cloud, setConnected to true and rely on cloud polling
    setConnected(true);
    return () => clearInterval(poll);
  }, []);

  const pnl = status?.realized_pnl || 0;
  const isPositive = pnl >= 0;

  return (
    <div className="flex h-screen w-screen bg-[#09090b] text-[#f4f4f5] overflow-hidden font-sans select-none antialiased">
      {/* ── Left Sidebar ────────────────────────────────────────────── */}
      <aside className="w-64 flex-shrink-0 bg-[#0c0c0e] border-r border-[#1e1e24] flex flex-col justify-between z-20">
        <div>
          {/* Brand Header */}
          <div className="p-4 border-b border-[#1e1e24] flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded bg-white flex items-center justify-center font-black text-black text-xs tracking-tighter shadow-sm">
                AE
              </div>
              <div>
                <h1 className="text-xs font-bold text-white tracking-wider font-mono">PROJECT AEGIS</h1>
                <p className="text-[10px] text-[#71717a] font-mono">QUANT AUTOPILOT v4</p>
              </div>
            </div>
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]' : 'bg-amber-500'}`} />
          </div>

          {/* Navigation Items */}
          <nav className="p-2 space-y-1">
            {[
              { id: 'cockpit', label: 'Overview Cockpit', icon: Activity },
              { id: 'strategies', label: 'Strategy Studio', icon: Sliders },
              { id: 'evolution', label: 'Evolution Lab', icon: Cpu },
              { id: 'radar', label: 'Market & News Radar', icon: Radar },
              { id: 'audit', label: 'Trade Audit Ledger', icon: FileText },
              { id: 'settings', label: 'Settings & Config', icon: Settings },
            ].map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-xs font-medium transition-all ${
                    active
                      ? 'bg-white text-black font-semibold shadow-sm'
                      : 'text-[#a1a1aa] hover:text-white hover:bg-[#18181b]'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${active ? 'text-black' : 'text-[#71717a]'}`} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Sidebar Status */}
        <div className="p-3 border-t border-[#1e1e24] bg-[#09090b]">
          <div className="p-3 rounded-lg bg-[#121215] border border-[#27272a] space-y-2 font-mono">
            <div>
              <span className="text-[10px] text-[#71717a] uppercase tracking-wider block">Trading Capital</span>
              <span className="text-sm font-bold text-white tracking-tight">₹{status?.capital?.toLocaleString('en-IN') || '15,000.00'}</span>
            </div>
            <div>
              <span className="text-[10px] text-[#71717a] uppercase tracking-wider block">Net Profit/Loss</span>
              <span className={`text-xs font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                {isPositive ? '+' : ''}₹{pnl.toFixed(2)}
              </span>
            </div>
            <div className="pt-2 border-t border-[#27272a] flex items-center justify-between text-[10px]">
              <span className="text-[#71717a]">Autopilot</span>
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                24/7 ACTIVE
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main Canvas ─────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#09090b] overflow-hidden">
        {/* Top Header Strip */}
        <header className="h-12 border-b border-[#1e1e24] bg-[#0c0c0e] px-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-6 font-mono text-xs">
            <div className="flex items-center gap-2">
              <span className="text-[#71717a]">REGIME:</span>
              <span className="px-2 py-0.5 rounded bg-[#18181b] text-white font-semibold border border-[#27272a]">
                {status?.regime || 'BULLISH'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[#71717a]">EXECUTION:</span>
              <span className="text-emerald-400 font-semibold">{status?.trade_mode || 'PAPER ₹15K'}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[#71717a]">AI HEALTH:</span>
              <span className="text-white font-semibold">{status?.model_health || 'OPTIMAL'}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono text-[#71717a] flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-[#52525b]" />
              {status?.timestamp || new Date().toLocaleTimeString('en-IN') + ' IST'}
            </span>
            <button
              onClick={refreshAll}
              className="p-1.5 rounded bg-[#18181b] border border-[#27272a] text-[#a1a1aa] hover:text-white hover:bg-[#27272a] transition-all"
              title="Refresh State"
            >
              <RotateCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </header>

        {/* Viewport Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {activeTab === 'cockpit' && (
            <div className="space-y-4">
              <LiveChart />
              <div className="pt-2">
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-white mb-3 flex items-center gap-2">
                  <Sliders className="w-3.5 h-3.5 text-white" />
                  Active Multi-Strategy Weights & Real-Time Performance
                </h3>
                <StrategyStudio />
              </div>
            </div>
          )}

          {activeTab === 'strategies' && <StrategyStudio />}
          {activeTab === 'evolution' && <EvolutionLab />}
          {activeTab === 'radar' && <MarketRadar />}
          {activeTab === 'audit' && <TradesHistory trades={trades} />}
          {activeTab === 'settings' && <SettingsStudio />}
        </div>
      </main>
    </div>
  );
};
export default App;