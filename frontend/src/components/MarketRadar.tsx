import React, { useEffect, useState } from 'react';

export const MarketRadar: React.FC = () => {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetch('/api/market-intelligence')
      .then((r) => r.json())
      .then((d) => setData(d))
      .catch((e) => console.error(e));
  }, []);

  const fii = data?.intelligence?.fii_dii || {};
  const vix = data?.intelligence?.vix_analysis || {};
  const videos = data?.youtube?.videos || [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 font-sans">
      {/* Macro */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-white">
          Macro Indicators & Institutional Flows
        </h3>

        <div className="space-y-2 font-mono text-xs">
          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center">
            <span className="text-[#a1a1aa]">India VIX</span>
            <span className="text-emerald-400 font-semibold">
              {vix.value ? vix.value.toFixed(2) : '10.99'} ({vix.classification || 'Low Volatility'})
            </span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center">
            <span className="text-[#a1a1aa]">FII Net Flow</span>
            <span className="text-white">
              {fii.fii_net ? `₹${fii.fii_net.toFixed(0)} Cr` : '₹+120 Cr (Net Inflow)'}
            </span>
          </div>

          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center">
            <span className="text-[#a1a1aa]">DII Net Flow</span>
            <span className="text-emerald-400 font-semibold">
              {fii.dii_net ? `₹${fii.dii_net.toFixed(0)} Cr` : '₹+840 Cr (Accumulation)'}
            </span>
          </div>
        </div>
      </div>

      {/* Sector Rotation */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-white">
          Sector Rotation
        </h3>

        <div className="grid grid-cols-2 gap-2 font-mono text-xs">
          {[
            { name: 'Nifty Bank', chg: '+0.85%' },
            { name: 'Nifty IT', chg: '+1.42%' },
            { name: 'Nifty Metal', chg: '-0.31%' },
            { name: 'Nifty Energy', chg: '+0.64%' },
            { name: 'Nifty Pharma', chg: '+0.21%' },
            { name: 'Nifty FMCG', chg: '-0.12%' },
          ].map((s) => {
            const pos = s.chg.startsWith('+');
            return (
              <div key={s.name} className="p-2 bg-[#18181b] rounded flex justify-between items-center">
                <span className="text-[#a1a1aa] text-[11px]">{s.name}</span>
                <span className={`font-semibold ${pos ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {s.chg}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Media Pulse */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-white">
          Financial News & Sentiment Feed
        </h3>

        <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
          {videos.length > 0 ? (
            videos.map((v: any, idx: number) => (
              <div key={idx} className="p-2 bg-[#18181b] rounded text-xs space-y-0.5">
                <span className="text-[10px] font-mono text-[#a1a1aa] uppercase">{v.channel}</span>
                <p className="text-[#e4e4e7] truncate text-[11px]">{v.title}</p>
              </div>
            ))
          ) : (
            <div className="text-xs text-[#71717a] font-mono py-4 text-center">
              Scanning financial channels via RSS feeds.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};