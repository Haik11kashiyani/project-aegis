import React, { useState, useEffect } from 'react';
import { Save, Check, Shield, Key, Sliders, Server, Plus, X, Globe, Laptop } from 'lucide-react';

export const SettingsStudio: React.FC = () => {
  const [cfg, setCfg] = useState<any>({
    capital: 15000,
    broker_name: 'PAPER',
    confidence_threshold: 0.65,
    max_daily_loss_pct: 2.0,
    kelly_fraction: 0.25,
    watchlist: ['SBIN.NS', 'TATASTEEL.NS', 'NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS'],
    shoonya_user: '',
    shoonya_password: '',
    shoonya_api_key: '',
    shoonya_totp_key: '',
    dhan_client_id: '',
    dhan_access_token: '',
    telegram_bot_token: '',
    telegram_chat_id: '',
    autopilot_enabled: true,
  });

  const [activeSubTab, setActiveSubTab] = useState<'general' | 'broker' | 'watchlist' | 'cloud'>('general');
  const [newStock, setNewStock] = useState('');
  const [toast, setToast] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetch('/api/config')
      .then((r) => r.json())
      .then((d) => setCfg((prev: any) => ({ ...prev, ...d })))
      .catch((e) => console.error(e));
  }, []);

  const handleSave = () => {
    setSaving(true);
    fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    })
      .then((r) => r.json())
      .then(() => {
        setToast('Configuration updated and saved to engine!');
        setTimeout(() => setToast(null), 3500);
      })
      .catch((e) => console.error(e))
      .finally(() => setSaving(false));
  };

  const addStock = () => {
    const s = newStock.trim().toUpperCase();
    if (!s) return;
    const symbol = s.endsWith('.NS') ? s : `${s}.NS`;
    if (!cfg.watchlist.includes(symbol)) {
      setCfg({ ...cfg, watchlist: [...cfg.watchlist, symbol] });
    }
    setNewStock('');
  };

  const removeStock = (sym: string) => {
    setCfg({ ...cfg, watchlist: cfg.watchlist.filter((x: string) => x !== sym) });
  };

  return (
    <div className="space-y-4 font-sans max-w-5xl mx-auto">
      {/* Top Header */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-mono font-bold uppercase tracking-wider text-white">
            System Configuration & Broker Connection
          </h2>
          <p className="text-xs text-[#a1a1aa] mt-0.5">
            Configure trading capital, live broker API credentials, risk limits, and 100% free 24/7 cloud automation.
          </p>
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center space-x-2 bg-white hover:bg-[#e4e4e7] text-black font-mono text-xs font-semibold px-4 py-2 rounded transition-colors disabled:opacity-50 cursor-pointer shadow-sm"
        >
          <Save className="w-3.5 h-3.5" />
          <span>{saving ? 'SAVING CONFIG...' : 'SAVE CONFIGURATION'}</span>
        </button>
      </div>

      {toast && (
        <div className="p-2.5 bg-[#18181b] border border-emerald-700/40 rounded text-emerald-400 text-xs font-mono flex items-center space-x-2">
          <Check className="w-4 h-4 shrink-0" />
          <span>{toast}</span>
        </div>
      )}

      {/* Sub Tabs */}
      <div className="flex items-center space-x-1 border-b border-[#27272a] pb-2 font-mono text-xs">
        {[
          { id: 'general', label: 'Capital & Risk Limits', icon: Sliders },
          { id: 'broker', label: 'Broker API Credentials', icon: Key },
          { id: 'watchlist', label: 'Stock Watchlist', icon: Shield },
          { id: 'cloud', label: '100% Free 24/7 Cloud / Autostart', icon: Globe },
        ].map((tab) => {
          const Icon = tab.icon;
          const active = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded transition-colors cursor-pointer ${
                active
                  ? 'bg-white text-black font-semibold'
                  : 'text-[#71717a] hover:text-white hover:bg-[#18181b]'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* 1. Capital & Risk Tab */}
      {activeSubTab === 'general' && (
        <div className="bg-[#121215] border border-[#27272a] rounded-lg p-5 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
            {/* Capital Input */}
            <div className="space-y-1.5">
              <label className="text-[#a1a1aa] uppercase text-[11px] block">Trading Capital (₹ INR)</label>
              <div className="relative">
                <span className="absolute left-3 top-2.5 text-[#71717a]">₹</span>
                <input
                  type="number"
                  value={cfg.capital}
                  onChange={(e) => setCfg({ ...cfg, capital: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-[#18181b] border border-[#27272a] rounded pl-7 pr-3 py-2 text-white font-mono text-xs focus:outline-none focus:border-white"
                />
              </div>
              <span className="text-[10px] text-[#71717a]">Initial capital allocated for position sizing & risk calculations</span>
            </div>

            {/* Daily Stop Loss % */}
            <div className="space-y-1.5">
              <label className="text-[#a1a1aa] uppercase text-[11px] block">Daily Loss Circuit Breaker (%)</label>
              <input
                type="number"
                step="0.5"
                min="0.5"
                max="5.0"
                value={cfg.max_daily_loss_pct}
                onChange={(e) => setCfg({ ...cfg, max_daily_loss_pct: parseFloat(e.target.value) || 0 })}
                className="w-full bg-[#18181b] border border-[#27272a] rounded px-3 py-2 text-white font-mono text-xs focus:outline-none focus:border-white"
              />
              <span className="text-[10px] text-[#71717a]">Halts all trading if net daily loss exceeds this % (2% of ₹15K = ₹300)</span>
            </div>

            {/* Confidence Threshold */}
            <div className="space-y-1.5">
              <label className="text-[#a1a1aa] uppercase text-[11px] block">AI Confidence Threshold (0.50 - 0.85)</label>
              <input
                type="number"
                step="0.05"
                min="0.50"
                max="0.85"
                value={cfg.confidence_threshold}
                onChange={(e) => setCfg({ ...cfg, confidence_threshold: parseFloat(e.target.value) || 0 })}
                className="w-full bg-[#18181b] border border-[#27272a] rounded px-3 py-2 text-white font-mono text-xs focus:outline-none focus:border-white"
              />
              <span className="text-[10px] text-[#71717a]">Recommended: 0.65. Clamps maximum drawdown to 0.77% by filtering weak setups</span>
            </div>

            {/* Kelly Fraction */}
            <div className="space-y-1.5">
              <label className="text-[#a1a1aa] uppercase text-[11px] block">Kelly Fraction Sizing (0.15 - 0.50)</label>
              <input
                type="number"
                step="0.05"
                min="0.10"
                max="0.50"
                value={cfg.kelly_fraction}
                onChange={(e) => setCfg({ ...cfg, kelly_fraction: parseFloat(e.target.value) || 0 })}
                className="w-full bg-[#18181b] border border-[#27272a] rounded px-3 py-2 text-white font-mono text-xs focus:outline-none focus:border-white"
              />
              <span className="text-[10px] text-[#71717a]">Default: 0.25 (Quarter-Kelly). Prevents overleveraging on small capital accounts</span>
            </div>
          </div>
        </div>
      )}

      {/* 2. Broker Connection Tab */}
      {activeSubTab === 'broker' && (
        <div className="bg-[#121215] border border-[#27272a] rounded-lg p-5 space-y-4 font-mono text-xs">
          <div className="space-y-2">
            <label className="text-[#a1a1aa] uppercase text-[11px] block">Execution Broker</label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {[
                { id: 'PAPER', label: 'Paper Trading (₹15K Virtual)', badge: 'Recommended for Testing' },
                { id: 'SHOONYA', label: 'Shoonya (Finvasia)', badge: '100% Free API + ₹0 Brokerage' },
                { id: 'DHAN', label: 'Dhan API', badge: '100% Free Retail API' },
                { id: 'ANGELONE', label: 'Angel One (SmartAPI)', badge: 'Free API' },
              ].map((b) => (
                <button
                  key={b.id}
                  onClick={() => setCfg({ ...cfg, broker_name: b.id })}
                  className={`p-3 rounded border text-left cursor-pointer transition-colors ${
                    cfg.broker_name === b.id
                      ? 'bg-white text-black border-white'
                      : 'bg-[#18181b] text-[#a1a1aa] border-[#27272a] hover:border-[#3f3f46]'
                  }`}
                >
                  <div className="font-bold text-xs">{b.id}</div>
                  <div className="text-[10px] opacity-75 mt-1">{b.badge}</div>
                </button>
              ))}
            </div>
          </div>

          {cfg.broker_name === 'SHOONYA' && (
            <div className="p-4 bg-[#18181b] rounded border border-[#27272a] space-y-3">
              <h3 className="font-bold text-white text-xs uppercase">Shoonya (Finvasia) API Credentials</h3>
              <p className="text-[11px] text-[#71717a]">
                Get your free API key at api.shoonya.com. Zero monthly charges forever.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] text-[#a1a1aa] uppercase block mb-1">User ID</label>
                  <input
                    type="text"
                    value={cfg.shoonya_user}
                    onChange={(e) => setCfg({ ...cfg, shoonya_user: e.target.value })}
                    className="w-full bg-[#121215] border border-[#27272a] rounded px-3 py-1.5 text-white"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-[#a1a1aa] uppercase block mb-1">Password</label>
                  <input
                    type="password"
                    value={cfg.shoonya_password}
                    onChange={(e) => setCfg({ ...cfg, shoonya_password: e.target.value })}
                    className="w-full bg-[#121215] border border-[#27272a] rounded px-3 py-1.5 text-white"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-[#a1a1aa] uppercase block mb-1">API Key</label>
                  <input
                    type="password"
                    value={cfg.shoonya_api_key}
                    onChange={(e) => setCfg({ ...cfg, shoonya_api_key: e.target.value })}
                    className="w-full bg-[#121215] border border-[#27272a] rounded px-3 py-1.5 text-white"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-[#a1a1aa] uppercase block mb-1">TOTP Key (2FA Secret)</label>
                  <input
                    type="password"
                    value={cfg.shoonya_totp_key}
                    onChange={(e) => setCfg({ ...cfg, shoonya_totp_key: e.target.value })}
                    className="w-full bg-[#121215] border border-[#27272a] rounded px-3 py-1.5 text-white"
                  />
                </div>
              </div>
            </div>
          )}

          {cfg.broker_name === 'DHAN' && (
            <div className="p-4 bg-[#18181b] rounded border border-[#27272a] space-y-3">
              <h3 className="font-bold text-white text-xs uppercase">Dhan API Credentials</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] text-[#a1a1aa] uppercase block mb-1">Client ID</label>
                  <input
                    type="text"
                    value={cfg.dhan_client_id}
                    onChange={(e) => setCfg({ ...cfg, dhan_client_id: e.target.value })}
                    className="w-full bg-[#121215] border border-[#27272a] rounded px-3 py-1.5 text-white"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-[#a1a1aa] uppercase block mb-1">Access Token</label>
                  <input
                    type="password"
                    value={cfg.dhan_access_token}
                    onChange={(e) => setCfg({ ...cfg, dhan_access_token: e.target.value })}
                    className="w-full bg-[#121215] border border-[#27272a] rounded px-3 py-1.5 text-white"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 3. Watchlist Tab */}
      {activeSubTab === 'watchlist' && (
        <div className="bg-[#121215] border border-[#27272a] rounded-lg p-5 space-y-4 font-mono text-xs">
          <div>
            <h3 className="font-bold text-white text-xs uppercase">Target Watchlist Symbols</h3>
            <p className="text-[11px] text-[#71717a] mt-0.5">
              The AI scans these liquid NSE tickers for high-conviction intraday setups under ₹1,000/share.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="text"
              placeholder="e.g. BEL, IRFC, NHPC"
              value={newStock}
              onChange={(e) => setNewStock(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addStock()}
              className="bg-[#18181b] border border-[#27272a] rounded px-3 py-1.5 text-white text-xs uppercase focus:outline-none focus:border-white"
            />
            <button
              onClick={addStock}
              className="flex items-center space-x-1 bg-white hover:bg-[#e4e4e7] text-black px-3 py-1.5 rounded font-semibold transition-colors cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>ADD TICKER</span>
            </button>
          </div>

          <div className="flex flex-wrap gap-2 pt-2">
            {cfg.watchlist.map((sym: string) => (
              <span
                key={sym}
                className="flex items-center space-x-1.5 px-2.5 py-1 bg-[#18181b] border border-[#27272a] rounded text-white"
              >
                <span>{sym}</span>
                <button
                  onClick={() => removeStock(sym)}
                  className="text-[#71717a] hover:text-rose-400 cursor-pointer"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 4. 100% Free Cloud & Autostart Guide */}
      {activeSubTab === 'cloud' && (
        <div className="bg-[#121215] border border-[#27272a] rounded-lg p-5 space-y-4 font-mono text-xs">
          <div>
            <h3 className="font-bold text-white text-xs uppercase">
              100% Free 24/7 Autopilot Automation (No Daily Starting Needed!)
            </h3>
            <p className="text-[11px] text-[#71717a] mt-0.5">
              You do not have to manually double-click scripts every single morning. Here are your two 100% free automated paths:
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            {/* Method 1: Windows Autostart on Boot */}
            <div className="p-4 bg-[#18181b] border border-[#27272a] rounded-lg space-y-2.5">
              <div className="flex items-center space-x-2 text-white font-bold">
                <Laptop className="w-4 h-4 text-emerald-400" />
                <span>Method 1: Auto-Run on Laptop/PC Startup</span>
              </div>
              <p className="text-[#a1a1aa] text-[11px] leading-relaxed">
                Whenever you turn on your Windows laptop/PC, Project Aegis automatically starts in the background and runs the 09:15 AM to 03:30 PM trading session on its own.
              </p>
              <div className="p-2 bg-[#121215] rounded border border-[#27272a] text-[11px] text-emerald-400">
                A one-click setup script <code>setup_autostart.bat</code> is provided in your project folder! Run it once, and Windows starts the terminal on boot forever!
              </div>
            </div>

            {/* Method 2: GitHub Actions 100% Free Cloud */}
            <div className="p-4 bg-[#18181b] border border-[#27272a] rounded-lg space-y-2.5">
              <div className="flex items-center space-x-2 text-white font-bold">
                <Globe className="w-4 h-4 text-emerald-400" />
                <span>Method 2: 100% Free GitHub Actions Cloud</span>
              </div>
              <p className="text-[#a1a1aa] text-[11px] leading-relaxed">
                Your project already has 4 GitHub Actions workflows in <code>.github/workflows/</code>. GitHub gives 2,000 free runner minutes/month.
              </p>
              <ul className="list-disc list-inside text-[#71717a] text-[10px] space-y-1">
                <li><code>aegis_protocol.yml</code>: Runs trading bot Mon-Fri 09:15 IST</li>
                <li><code>offmarket_learner.yml</code>: Runs self-evolution daily at 16:30 IST</li>
                <li>Runs completely on GitHub cloud servers even when your PC is OFF!</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};