import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { LiveChart } from './components/LiveChart';
import { StrategyStudio } from './components/StrategyStudio';
import { EvolutionLab } from './components/EvolutionLab';
import { MarketRadar } from './components/MarketRadar';
import { TradesHistory } from './components/TradesHistory';
import { SettingsStudio } from './components/SettingsStudio';
import { SystemStatus, StrategyInfo, TradeRecord } from './types';
import { LayoutDashboard, Sliders, Dna, Radio, FileText, Settings, ShieldCheck, Activity } from 'lucide-react';

export const App: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [strategies, setStrategies] = useState<Record<string, StrategyInfo>>({});
  const [trades, setTrades] = useState<TradeRecord[]>([]);
  const [activeTab, setActiveTab] = useState<'terminal' | 'strategies' | 'evolution' | 'radar' | 'trades' | 'settings'>('terminal');
  const [selectedStock, setSelectedStock] = useState('SBIN.NS');
  const [connected, setConnected] = useState(false);

  const fetchStatus = () => {
    fetch('/api/status')
      .then((r) => r.json())
      .then((d) => setStatus(d))
      .catch(() => {});
  };

  const fetchStrategies = () => {
    fetch('/api/strategies')
      .then((r) => r.json())
      .then((d) => setStrategies(d))
      .catch(() => {});
  };

  const fetchTrades = () => {
    fetch('/api/trades')
      .then((r) => r.json())
      .then((d) => setTrades(d.trades || []))
      .catch(() => {});
  };

  useEffect(() => {
    fetchStatus();
    fetchStrategies();
    fetchTrades();

    const poll = setInterval(fetchStatus, 3000);

    let ws: WebSocket | null = null;
    let timer: any = null;

    const connectWS = () => {
      try {
        ws = new WebSocket('ws://127.0.0.1:8000/ws/stream');
        ws.onopen = () => setConnected(true);
        ws.onclose = () => {
          setConnected(false);
          timer = setTimeout(connectWS, 4000);
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
        timer = setTimeout(connectWS, 4000);
      }
    };

    connectWS();

    return () => {
      clearInterval(poll);
      if (timer) clearTimeout(timer);
      if (ws) {
        ws.onclose = null;
        ws.onerror = null;
        ws.close();
      }
    };
  }, []);

  const handleToggle = (name: string, enabled: boolean) => {
    fetch('/api/strategies/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, enabled }),
    })
      .then((r) => r.json())
      .then(() => {
        setStrategies((prev) => ({
          ...prev,
          [name]: { ...prev[name], enabled },
        }));
      })
      .catch((err) => console.error(err));
  };

  const handleWeightChange = (name: string, weight: number) => {
    fetch('/api/strategies/weight', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, weight }),
    })
      .then((r) => r.json())
      .then(() => {
        setStrategies((prev) => ({
          ...prev,
          [name]: { ...prev[name], weight },
        }));
      })
      .catch((err) => console.error(err));
  };

  const pnl = status ? status.realized_pnl + status.unrealized_pnl : 0;
  const isUp = pnl >= 0;

  return (
    <div className="min-h-screen bg-[#09090b] text-[#f4f4f5] flex font-sans antialiased">
      {/* Left Modern Docked Sidebar */}
      <aside className="w-56 shrink-0 bg-[#0c0c0e] border-r border-[#27272a] flex flex-col justify-between p-4 hidden md:flex">
        <div className="space-y-6">
          {/* Logo */}
          <div className="flex items-center space-x-2.5 px-1">
            <div className="w-7 h-7 rounded bg-white flex items-center justify-center shadow">
              <span className="font-mono font-bold text-xs text-black">A</span>
            </div>
            <div>
              <div className="font-bold text-sm text-white tracking-tight leading-none">AEGIS QUANT</div>
              <div className="text-[9px] font-mono text-[#71717a] mt-0.5">INSTITUTIONAL v3</div>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1 font-mono text-xs">
            {[
              { id: 'terminal', label: 'Overview Cockpit', icon: LayoutDashboard },
              { id: 'strategies', label: 'Strategy Studio', icon: Sliders },
              { id: 'evolution', label: 'Evolution Lab', icon: Dna },
              { id: 'radar', label: 'Market & News Radar', icon: Radio },
              { id: 'trades', label: 'Trade Audit Ledger', icon: FileText },
              { id: 'settings', label: 'Settings & Config', icon: Settings },
            ].map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded transition-colors cursor-pointer text-left ${
                    active
                      ? 'bg-white text-black font-semibold shadow-sm'
                      : 'text-[#a1a1aa] hover:text-white hover:bg-[#18181b]'
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Sidebar Status */}
        <div className="p-3 bg-[#121215] border border-[#27272a] rounded-lg space-y-2 font-mono text-xs">
          <div>
            <div className="text-[10px] text-[#71717a] uppercase">Trading Capital</div>
            <div className="font-bold text-white">
              ₹{status?.equity ? status.equity.toLocaleString('en-IN', { minimumFractionDigits: 2 }) : '15,000.00'}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-[#71717a] uppercase">Net Profit/Loss</div>
            <div className={`font-bold ${isUp ? 'text-emerald-400' : 'text-rose-400'}`}>
              {isUp ? '+' : ''}₹{pnl.toFixed(2)}
            </div>
          </div>
          <div className="pt-1 border-t border-[#1f1f23] flex items-center justify-between text-[10px]">
            <span className="text-[#71717a]">Autopilot</span>
            <span className="text-emerald-400 font-semibold flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              24/7 ACTIVE
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header status={status} connected={connected} onRefresh={fetchStatus} />

        <main className="flex-1 p-5 space-y-4 max-w-[1600px] w-full mx-auto">
          {/* Mobile Tab Selector */}
          <div className="flex md:hidden overflow-x-auto space-x-1 pb-2 border-b border-[#27272a] font-mono text-xs">
            {[
              { id: 'terminal', label: 'Cockpit' },
              { id: 'strategies', label: 'Strategies' },
              { id: 'evolution', label: 'Evolution' },
              { id: 'radar', label: 'Radar' },
              { id: 'trades', label: 'Audit' },
              { id: 'settings', label: 'Settings' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-3 py-1.5 rounded whitespace-nowrap ${
                  activeTab === tab.id ? 'bg-white text-black font-bold' : 'text-[#a1a1aa] bg-[#121215]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Views */}
          {activeTab === 'terminal' && (
            <div className="space-y-4">
              <LiveChart symbol={selectedStock} onSymbolChange={setSelectedStock} />
              <StrategyStudio
                strategies={strategies}
                onToggle={handleToggle}
                onWeightChange={handleWeightChange}
              />
            </div>
          )}

          {activeTab === 'strategies' && (
            <StrategyStudio
              strategies={strategies}
              onToggle={handleToggle}
              onWeightChange={handleWeightChange}
            />
          )}

          {activeTab === 'evolution' && <EvolutionLab />}

          {activeTab === 'radar' && <MarketRadar />}

          {activeTab === 'trades' && <TradesHistory trades={trades} />}

          {activeTab === 'settings' && <SettingsStudio />}
        </main>
      </div>
    </div>
  );
};