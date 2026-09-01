export interface SystemStatus {
  status: string;
  trade_mode: string;
  capital: number;
  realized_pnl: number;
  unrealized_pnl: number;
  equity: number;
  regime: string;
  model_health: string;
  active_bullets: number;
  max_bullets: number;
  timestamp: string;
}

export interface StrategyInfo {
  weight: number;
  enabled: boolean;
  trades: number;
  wins?: number;
  win_rate?: number;
  total_pnl: number;
  description?: string;
  category?: string;
  capital_per_trade?: number;
}

export interface EvolutionStatus {
  status: string;
  generation: number;
  best_fitness: number;
  best_chromosome: Record<string, any>;
  fitness_history: number[];
  generation_stats: any[];
  best_backtest: Record<string, any>;
  timestamp: string;
  autonomous_mode: boolean;
  next_cycle_in: string;
}

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
  ema20?: number;
}

export interface TradeRecord {
  Date?: string;
  Stock?: string;
  Entry?: number;
  Exit?: number;
  Qty?: number;
  PnL?: number;
  Strategy?: string;
  Action?: string;
  Reason?: string;
  [key: string]: any;
}

export interface AutopilotStatus {
  autopilot_active: boolean;
  current_phase: string;
  phase_description: string;
  last_action: string;
  next_action: string;
  evolution_count: number;
  auto_promoted_strategies: number;
  last_updated: string;
}