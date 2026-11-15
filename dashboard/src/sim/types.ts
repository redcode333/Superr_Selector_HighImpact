// public/data/sim.json (scripts/build_data.py -> simulator()).
export interface SimBowler {
  player_id: number; name: string; rank: number; group: 'top 10' | 'alternate';
  counts: Record<string, number>; n: number; p: number[];
  death_balls: number; death_econ: number; dot_pct: number; boundary_pct: number; balls_per_wkt: number; yorker_pct: number;
  defend: Record<string, number>;
}
export interface Sim {
  outcomes: string[]; runs: number[]; K: number; league: number[]; league_counts: Record<string, number>;
  mean_abs_wpa: number; base_defend: Record<string, number>; few_death_balls: number;
  situations: { need: number; wk: number }[]; need_range: [number, number]; wk_range: [number, number];
  bowlers: SimBowler[];
}
