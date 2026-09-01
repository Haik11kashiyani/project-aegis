import React, { useEffect, useRef, useState } from 'react';
import { createChart, CandlestickSeries, LineSeries, IChartApi, ISeriesApi } from 'lightweight-charts';
import { CandleData } from '../types';

interface LiveChartProps {
  symbol: string;
  onSymbolChange: (sym: string) => void;
}

const STOCKS = [
  { sym: 'SBIN.NS', label: 'SBIN' },
  { sym: 'TATASTEEL.NS', label: 'TATASTEEL' },
  { sym: 'NTPC.NS', label: 'NTPC' },
  { sym: 'POWERGRID.NS', label: 'POWERGRID' },
  { sym: 'COALINDIA.NS', label: 'COALINDIA' },
];

export const LiveChart: React.FC<LiveChartProps> = ({ symbol, onSymbolChange }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const vwapSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const emaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  const [candles, setCandles] = useState<CandleData[]>([]);
  const [loading, setLoading] = useState(true);
  const [price, setPrice] = useState(0);
  const [chg, setChg] = useState(0);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/chart/${symbol}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.candles && d.candles.length > 0) {
          setCandles(d.candles);
          const last = d.candles[d.candles.length - 1];
          const first = d.candles[0];
          setPrice(last.close);
          setChg(((last.close - first.open) / first.open) * 100);
        }
      })
      .catch((e) => console.error(e))
      .finally(() => setLoading(false));
  }, [symbol]);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 420,
      layout: {
        background: { color: '#09090b' },
        textColor: '#71717a',
        fontSize: 11,
        fontFamily: 'JetBrains Mono, monospace',
      },
      grid: {
        vertLines: { color: '#18181b' },
        horzLines: { color: '#18181b' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#27272a',
      },
      rightPriceScale: {
        borderColor: '#27272a',
      },
      crosshair: {
        vertLine: { color: '#3f3f46', width: 1, style: 2 },
        horzLine: { color: '#3f3f46', width: 1, style: 2 },
      },
    });

    chartRef.current = chart;

    candleSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    vwapSeriesRef.current = chart.addSeries(LineSeries, {
      color: '#f59e0b',
      lineWidth: 1,
      title: 'VWAP',
    });

    emaSeriesRef.current = chart.addSeries(LineSeries, {
      color: '#e4e4e7',
      lineWidth: 1,
      title: 'EMA 20',
    });

    const onResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (e) {}
        chartRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!candleSeriesRef.current || candles.length === 0) return;
    try {
      candleSeriesRef.current.setData(
        candles.map((c) => ({
          time: c.time as any,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        }))
      );
      if (vwapSeriesRef.current) {
        vwapSeriesRef.current.setData(
          candles.map((c) => ({ time: c.time as any, value: c.vwap || c.close }))
        );
      }
      if (emaSeriesRef.current) {
        emaSeriesRef.current.setData(
          candles.map((c) => ({ time: c.time as any, value: c.ema20 || c.close }))
        );
      }
      chartRef.current?.timeScale().fitContent();
    } catch (e) {
      console.error(e);
    }
  }, [candles]);

  return (
    <div className="bg-[#121215] border border-[#27272a] rounded-lg p-4 space-y-3 font-sans">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1f1f23] pb-3">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1 font-mono">
            {STOCKS.map((s) => (
              <button
                key={s.sym}
                onClick={() => onSymbolChange(s.sym)}
                className={`px-2.5 py-1 text-xs font-semibold rounded transition-colors cursor-pointer ${
                  symbol === s.sym
                    ? 'bg-white text-black'
                    : 'bg-[#18181b] text-[#a1a1aa] hover:text-white hover:bg-[#27272a]'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>

          <div className="h-4 w-px bg-[#27272a]" />

          <div className="flex items-baseline space-x-2 font-mono">
            <span className="text-sm font-bold text-white">₹{price.toFixed(2)}</span>
            <span className={`text-xs font-semibold ${chg >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {chg >= 0 ? '+' : ''}{chg.toFixed(2)}%
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-4 text-[11px] font-mono text-[#71717a]">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 bg-amber-400 rounded-full" />
            <span>VWAP</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 bg-zinc-300 rounded-full" />
            <span>EMA 20</span>
          </span>
          <span>15m Candles</span>
        </div>
      </div>

      <div className="relative w-full h-[420px] rounded overflow-hidden">
        {loading && (
          <div className="absolute inset-0 bg-[#09090b]/80 flex items-center justify-center z-10 font-mono text-xs text-[#a1a1aa]">
            Streaming Candles...
          </div>
        )}
        <div ref={containerRef} className="w-full h-full" />
      </div>
    </div>
  );
};