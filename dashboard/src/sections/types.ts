// public/data/{matchups,specialists,sandbox}.json (scripts/build_data.py)
export interface RosterRow { player_id: number; name: string; rank: number }
export interface Pair { batter_id: number; bowler_id: number; balls: number; runs: number; outs: number; dots: number; fours: number; sixes: number; p_balls: number; p_runs: number; matches: number }
export interface Split { player_id: number; role: 'batter' | 'bowler'; phase: string; vs: string; balls: number; runs: number; outs: number; rate: number | null; par: number; batter_edge: number | null }
export interface Matchups { min_balls: number; league_sr: number; batters: RosterRow[]; bowlers: RosterRow[]; pairs: Pair[]; splits: Split[] }

export interface SpecRow { role: 'batter' | 'bowler'; chip: string; label: string; metric: string; player_id: number; name: string; value: number; par: number; edge: number; balls: number; runs: number; outs: number; rank: number; min_balls: number }
export interface Specialists { rows: SpecRow[] }

export interface SandboxTop { player_id: number; name: string; value: number; high_pressure_balls: number }
export interface Sandbox {
  own_share_pct: number[]; matches_2025_26: number[]; keys: ('cautious' | 'projected')[]; floors: number[]; pressure_rule: number[];
  ours: { own: number; m: number; key: 'cautious' | 'projected'; floor: number; pp: number };
  lists: Record<string, { eligible: number; top: SandboxTop[] }>;
}
