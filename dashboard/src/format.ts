// Number formatting and shared colour meaning. No data values live here.
import type { Role, Verdict } from './data';

const MINUS = '−';

export function signed(x: number, digits = 3): string {
  const s = Math.abs(x).toFixed(digits);
  if (Number(s) === 0) return (0).toFixed(digits);
  return (x < 0 ? MINUS : '+') + s;
}
export const fixed = (x: number, digits = 1) => (x < 0 ? MINUS : '') + Math.abs(x).toFixed(digits);
export const int = (x: number) => Math.round(x).toLocaleString('en-IN');
export const pct = (x: number, digits = 0) => `${x.toFixed(digits)}%`;
export const range = (lo: number, hi: number, digits = 3) => `${signed(lo, digits)} to ${signed(hi, digits)}`;

// Colour meaning, never broken: pink = batter on top, blue = bowler on top.
export const INK = '#0f1a3c';
export const PINK = '#e4007c';
export const BLUE = '#254aa5';
export const NEUTRAL = '#f1f1f4';

/** Who is on top when this player's value is positive. */
export const sideColour = (role: Role, positive: boolean) => ((role === 'batter') === positive ? PINK : BLUE);

export function verdictStyle(role: Role, v: Verdict): { bg: string; fg: string; border: string } {
  if (v === 'no clear difference') return { bg: NEUTRAL, fg: INK, border: '#d6dbe8' };
  const c = sideColour(role, v === 'better under pressure');
  return { bg: c, fg: '#ffffff', border: c };
}

export function verdictLabel(v: Verdict, range: string): string {
  return v === 'no clear difference' ? v : `${v} in ${range}`;
}
