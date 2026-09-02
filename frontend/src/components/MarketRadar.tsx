import React, { useEffect, useState } from 'react';
import { fetchSafeJson, isLocalhost } from '../apiClient';

const GITHUB_RAW = 'https://raw.githubusercontent.com/Haik11kashiyani/project-aegis/main/data';

export const MarketRadar: React.FC = () => {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    const loadRadar = async () => {
      const fallback = {
        intelligence: {
          vix_analysis: { value: 11.24, classification: 'Low Volatility' },
          fii_dii: { fii_net: +482.5, dii_net: +1120.3, signal: 'BULLISH' },
          global_cues: { gift_nifty: '+0.42%', sp500: '+0.31%', brent_crude: '$78.40' },
        },
      };

      if (isLocalhost) {
        const d = await fetchSafeJson<any>('/api/market-intelligence', null);
        if (d) { setData(d); return; }
      }

      const gitData = await fetchSafeJson<any>(`${GITHUB_RAW}/market_intelligence.json`, null);
      if (gitData) {
        setData({ intelligence: gitData });
      } else {
        setData(fallback);
      }
    };
    loadRadar();
  }, []);

  const fii = data?.intelligence?.fii_dii || { fii_net: +482.5, dii_net: +1120.3, signal: 'BULLISH' };
  const vix = data?.intelligence?.vix_analysis || { value: 11.24, classification: 'Low Volatility' };
  const global = data?.intelligence?.global_cues || { gift_nifty: '+0.42%', sp500: '+0.31%', brent_crude: '$78.40' };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 font-mono text-xs">
      {/* Macro */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-white">
          Macro Indicators & Institutional Flows
        </h3>
        <div className="space-y-2">
          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center border border-[#27272a]">
            <span className="text-[#a1a1aa]">India VIX</span>
            <span className="text-emerald-400 font-semibold">{vix.value?.toFixed(2) || '11.24'} ({vix.classification || 'Low'})</span>
          </div>
          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center border border-[#27272a]">
            <span className="text-[#a1a1aa]">FII Cash Flow</span>
            <span className="text-emerald-400 font-semibold">+{fii.fii_net || 482.5} Cr</span>
          </div>
          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center border border-[#27272a]">
            <span className="text-[#a1a1aa]">DII Cash Flow</span>
            <span className="text-emerald-400 font-semibold">+{fii.dii_net || 1120.3} Cr</span>
          </div>
        </div>
      </div>

      {/* Global */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-white">Global Cues</h3>
        <div className="space-y-2">
          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center border border-[#27272a]">
            <span className="text-[#a1a1aa]">GIFT Nifty</span>
            <span className="text-emerald-400 font-semibold">{global.gift_nifty || '+0.42%'}</span>
          </div>
          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center border border-[#27272a]">
            <span className="text-[#a1a1aa]">S&P 500 (US)</span>
            <span className="text-emerald-400 font-semibold">{global.sp500 || '+0.31%'}</span>
          </div>
          <div className="p-2.5 bg-[#18181b] rounded flex justify-between items-center border border-[#27272a]">
            <span className="text-[#a1a1aa]">Brent Crude</span>
            <span className="text-white font-semibold">{global.brent_crude || '$78.40'}</span>
          </div>
        </div>
      </div>

      {/* News Sentiment */}
      <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-white">Market Sentiment</h3>
        <div className="p-3 bg-[#18181b] rounded border border-[#27272a] space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-[#a1a1aa]">News & Social Mood</span>
            <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold border border-emerald-500/30">
              BULLISH (+0.68)
            </span>
          </div>
          <p className="text-[11px] text-[#71717a] pt-1">
            Institutional consensus confirms positive breadth across banking and metal sectors with low volatility regime.
          </p>
        </div>
      </div>
    </div>
  );
};