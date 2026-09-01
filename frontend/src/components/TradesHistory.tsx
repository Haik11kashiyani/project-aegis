import React from 'react';
import { TradeRecord } from '../types';

interface TradesHistoryProps {
  trades: TradeRecord[];
}

export const TradesHistory: React.FC<TradesHistoryProps> = ({ trades }) => {
  return (
    <div className="bg-[#0e111a] border border-[#1b202e] rounded-lg overflow-hidden">
      <div className="p-3 bg-[#131722] border-b border-[#1b202e] flex justify-between items-center">
        <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300">
          Executed Trades ({trades.length})
        </span>
        <span className="text-[11px] font-mono text-slate-500">Live Audit Log</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-[#10141e] text-slate-400 text-[10px] uppercase border-b border-[#1b202e]">
            <tr>
              <th className="py-2 px-3">Timestamp</th>
              <th className="py-2 px-3">Symbol</th>
              <th className="py-2 px-3 text-right">Qty</th>
              <th className="py-2 px-3 text-right">Entry</th>
              <th className="py-2 px-3 text-right">Exit</th>
              <th className="py-2 px-3 text-right">Net P&L</th>
              <th className="py-2 px-3">Exit Reason</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1b202e] text-slate-300">
            {trades.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-6 text-center text-slate-500">
                  No trades recorded yet.
                </td>
              </tr>
            ) : (
              trades.slice(0, 25).map((t, idx) => {
                const pnl = parseFloat(t.PnL || t.pnl || '0');
                const isWin = pnl >= 0;
                return (
                  <tr key={idx} className="hover:bg-[#131722]/60 transition-colors">
                    <td className="py-2 px-3 text-slate-400">{t.Date || t.date || 'Recent'}</td>
                    <td className="py-2 px-3 font-semibold text-slate-100">{t.Stock || t.symbol || 'SBIN.NS'}</td>
                    <td className="py-2 px-3 text-right">{t.Qty || t.qty || 2}</td>
                    <td className="py-2 px-3 text-right">₹{parseFloat(t.Entry || t.entry_price || '0').toFixed(2)}</td>
                    <td className="py-2 px-3 text-right">₹{parseFloat(t.Exit || t.exit_price || '0').toFixed(2)}</td>
                    <td className={`py-2 px-3 text-right font-semibold ${isWin ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {isWin ? '+' : ''}₹{pnl.toFixed(2)}
                    </td>
                    <td className="py-2 px-3 text-slate-400">{t.Reason || t.exit_reason || 'TARGET_HIT'}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};