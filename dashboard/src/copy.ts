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

/** Note for a top-10 player whose record is clearly worse under pressure (Phil Salt in the v2 lists). */
export function worseNote(p: Pick, role: Role, range_: string): string | null {
  if (p.group !== 'top 10' || p.pressure.verdict !== 'worse under pressure') return null;
  return `The one top-10 ${role} who was clearly worse under pressure in ${range_} (gap ${signed(p.pressure.gap)}). `
    + `He is picked for his overall value: a cautious rating of ${signed(p.cautious)}, in the top 10 in ${p.pct_top10.toFixed(0)}% of re-rankings.`;
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
