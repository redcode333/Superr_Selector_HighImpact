// Fixed wording and text templates. Templates only rearrange numbers passed in from the data files.
import type { Meta, Pick, Role } from './data';
import { FORWARD_POOL } from './data';
import { range, signed } from './format';

export const TAGLINES = {
  A: 'When the match tightens, who holds?',
  B: 'Pressure, measured ball by ball.',
  C: "The 19th over doesn't lie.",
} as const;
/** Swap the tagline here. */
export const TAGLINE: keyof typeof TAGLINES = 'A';

export const ROLE_PLURAL: Record<Role, string> = { batter: 'Batters', bowler: 'Bowlers' };

/** One plain sentence for every player: how he rates under pressure, how that compares with his usual level, and so what the gap is.
 *  Shown wherever a verdict chip is, so "worse under pressure" is never printed without it. */
export function pressureLine(p: Pick): string {
  const q = p.pressure;
  const under = q.pressure > 0 ? 'above average' : 'below average';
  const level = Math.abs(q.gap) < 0.0005 ? 'in line with his usual level' : q.gap < 0 ? 'his usual level is higher' : 'above his usual level';
  const joiner = q.gap < 0 ? '; ' : ', and ';
  return `Rated ${under} under pressure (${signed(q.pressure)})${joiner}${level} (${signed(q.normal)}), so the gap is ${signed(q.gap)}.`;
}

export function honestNote(meta: Meta) {
  const chance = meta.chance_share_pct;
  const lines = (['batter', 'bowler'] as Role[]).map((r) => {
    const v = meta.verdicts[r]['all eligible'];
    const clear = v['better under pressure'].n + v['worse under pressure'].n;
    const f = meta.forward_test[r][FORWARD_POOL];
    return {
      role: r,
      clear: `${clear} of ${v.players} eligible ${r}s (${Math.round((100 * clear) / v.players)}%) land clearly above or below zero`,
      forward: `${ROLE_PLURAL[r]}: ${f.players} players, rank correlation ${signed(f.rho)} (95% range ${range(f.lo95, f.hi95)})`,
    };
  });
  return {
    chance: `With 80% ranges, about 1 in ${Math.round(100 / chance)} players lands clearly above or below zero by luck alone.`,
    lines,
    noCarry: (['batter', 'bowler'] as Role[]).every((r) => !meta.forward_confirmed[r]),
  };
}
