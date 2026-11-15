// Types for public/data/*.json (written by scripts/build_data.py) and a small loader.
import { useEffect, useState } from 'react';

export type Role = 'batter' | 'bowler';
export type Verdict = 'better under pressure' | 'worse under pressure' | 'no clear difference';

export interface Pressure {
  normal: number; pressure: number; gap: number; lo80: number; hi80: number; verdict: Verdict;
  balls_normal: number; balls_pressure: number;
}
export interface Pick {
  player_id: number; name: string; name_full: string; group: 'top 10' | 'alternate'; rank: number;
  initials: string; badge: Badge;
  cautious: number; best: number; range_lo: number; range_hi: number;
  pct_top10: number; median_rank: number; own_pct: number; matches: number; matches_2025_26: number;
  high_pressure_balls: number; strict_record: number; share_pp: number; share_mid: number; share_death: number;
  ipl_pct: number; missed_season: string; close_call: boolean; pressure: Pressure;
}
export interface BadgeWedge { name: string; from: number; to: number; share: number; len: number; color: string }
export interface BadgeTick { over: number; balls: number; angle: number; len: number; empty: boolean; color: string | null }
export type Badge =
  | { kind: 'batter'; k: number; r_in: number; disc_r: number; wedges: BadgeWedge[]; label: string }
  | { kind: 'bowler'; r_in: number; l_max: number; disc_r: number; ticks: BadgeTick[]; label: string };
export interface FieldGap { player_id: number; name: string; gap: number; lo80: number; hi80: number; verdict: Verdict }
export interface RolePicks {
  players: Pick[]; field: FieldGap[];
  league: { normal: number; pressure: number; gap: number; lo80: number; hi80: number; verdict: Verdict; players: number };
}
export interface Picks { unit: string; roles: Record<Role, RolePicks> }

export interface ForwardRow { players: number; rho: number; lo95: number; hi95: number }
export interface VerdictCounts { n: number; pct: number }
export interface SensRole { eligible: number; top10: string[]; out: string[]; in: string[] }
export interface Meta {
  title: string; data_range: string; last_built: string; unit: string; rating_unit: string; floor: number; li_pressure: number;
  rules: { own_share_min_pct: number; matches_2025_26_min: number; high_pressure_balls_min: number };
  eligibility: Record<'bat' | 'bowl', { ranked: number; eligible: number; fail_own_share: number; fail_matches_2025_26: number }>;
  verification: { passed: number; total: number };
  forward_test: Record<Role, Record<string, ForwardRow>>;
  forward_confirmed: Record<Role, boolean>;
  verdicts: Record<Role, Record<'top 10' | 'top 10 + 2 alternates' | 'all eligible', Record<Verdict, VerdictCounts> & { players: number }>>;
  chance_share_pct: number;
  changes: {
    swaps: Record<Role, { out: { name: string; v1_rank: number; high_pressure_balls: number }[]; in: { name: string; v2_rank: number; high_pressure_balls: number }[] }>;
    sensitivity: ({ floor: number } & Record<Role, SensRole>)[];
  };
  data_source: string;
  balls: { total: number; without_importance: number; without_importance_pct: number };
  how_it_works_lines: Record<string, string>;
}

export const FORWARD_POOL = 'eligible only, >= 20 each period';

const cache = new Map<string, Promise<unknown>>();
export function loadJSON<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(name, fetch(`${import.meta.env.BASE_URL}data/${name}`).then((r) => {
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(name) as Promise<T>;
}

export function useJSON<T>(name: string): { data: T | null; error: string | null } {
  const [state, setState] = useState<{ data: T | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    let live = true;
    loadJSON<T>(name).then((data) => live && setState({ data, error: null }), (e: Error) => live && setState({ data: null, error: e.message }));
    return () => { live = false; };
  }, [name]);
  return state;
}

