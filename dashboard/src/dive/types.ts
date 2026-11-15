// public/data/players.json (scripts/build_data.py batter_dive / bowler_dive + scripts/callouts.py finish_dive)
export interface Region {
  name: string; from: number; to: number; balls: number; runs: number; fours: number; sixes: number; boundary_runs: number;
  outs: number; caught: number; run_share: number | null; sr: number | null; league_sr: number | null;
}
export interface Callout { slot: string; id: string; text: string }
export interface Flag { few: boolean; total: number; min: number; what: string }
export interface Ground { ground: string; city: string; balls: number; vs_par: number; tag: 'best' | 'worst'; matches: number;
  sr?: number; par_sr?: number; runs?: number; outs?: number; econ?: number; par_econ?: number; wkts?: number }
export interface TypeShare { type: string; n: number; pct: number; league_pct: number }
interface DiveBase {
  player_id: number; name: string; rank: number; group: 'top 10' | 'alternate'; matches: number; balls: number;
  regions: Region[]; role_label: string; flags: Record<string, Flag>; callouts: Callout[]; grounds: Ground[];
  grounds_qualifying: number; grounds_min_balls: number; opponents_min_balls: number; overs_min_balls: number;
  phase_balls: Record<string, number>; mins: Record<string, number>;
}
export interface BatterDive extends DiveBase {
  role: 'batter'; runs: number; sr: number; fours: number; sixes: number; outs: number;
  death: { balls: number; sr: number | null; par_sr: number | null; boundary_pct: number | null };
  setting: { balls: number; sr: number | null }; chasing: { balls: number; sr: number | null };
  region_totals: { boundary_runs: number; fours: number; sixes: number; caught: number; outs: number; runs: number; balls: number };
  best_field: { regions: string[]; boundary_runs: number };
  overs: { over: number; balls: number; sr: number | null; par_sr: number }[];
  dismissals: TypeShare[];
  styles: { style: string; balls: number; runs: number; sr: number | null; outs: number; balls_per_out: number | null; par_sr: number }[];
  toughest: Opp[]; easiest: Opp[];
}
export interface Opp { player_id: number; name: string; balls: number; runs: number; outs: number; sr: number; style?: string | null }
export interface BowlerOver { over: number; balls: number; econ: number | null; par_econ: number; wkts: number; times: number; saving: number }
export interface BowlerDive extends DiveBase {
  role: 'bowler'; wickets: number; runs: number; econ: number; dot_pct: number; balls_per_wkt: number;
  death: { balls: number; econ: number | null; par_econ: number | null; yorker_pct: number | null; wickets: number };
  region_totals: { runs_all: number; wickets_in_regions: number };
  overs: BowlerOver[];
  planner: { shrink_balls: number; few_balls: number; avg_par_unbowled: number;
    best: { overs: number[]; saving: number; his: number; avg: number }; usual: { overs: number[]; times: number } };
  wicket_types: TypeShare[];
  deliveries: { type: string; balls: number; usage_pct: number; econ: number | null; wkts: number; balls_per_wkt: number | null; par_econ: number }[];
  deliveries_labelled_pct: number;
  hands: { hand: string; balls: number; econ: number | null; wkts: number; par_econ: number }[];
  owns: Opp[]; troubled_by: Opp[];
}
export type Dive = BatterDive | BowlerDive;
export type Players = Record<string, Dive>;
