// The exact 19th-over engine: a line-for-line port of _handoff/design/Main.dc.html (chaseWin / playOne), the same engine
// that scripts/check_data.py re-implements to verify sim.json. Outcome order: 0 1 2 3 4 6 W WD NB.
// The 19th over (balls 12..7) uses the chosen bowler's odds; the 20th (balls 6..1) the average death bowler.

export type Odds = number[];
export const RUN = [0, 1, 2, 3, 4, 6];

/** Chance the chasing side wins from b balls left, w wickets in hand, r runs needed. A tie counts as half. */
export function chaseWin(pX: Odds, league: Odds, b0: number, w0: number, r0: number): number {
  const memo = new Map<number, number>();
  const V = (b: number, w: number, r: number): number => {
    if (r <= 0) return 1;
    if (b === 0 || w === 0) return r === 1 ? 0.5 : 0;
    const k = b * 10000 + w * 100 + r;
    const hit = memo.get(k);
    if (hit !== undefined) return hit;
    const p = b > 6 ? pX : league;
    let v = 0;
    for (let i = 0; i < 6; i++) v += p[i] * V(b - 1, w, r - RUN[i]);
    v += p[6] * V(b - 1, w - 1, r) + (p[7] + p[8]) * V(b, w, r - 1);
    memo.set(k, v);
    return v;
  };
  return V(b0, w0, r0);
}

/** Exact defend % for the bowling side, 12 balls left. */
export const defendPct = (pX: Odds, league: Odds, wk: number, need: number) => 100 * (1 - chaseWin(pX, league, 12, wk, need));

/** Expected absolute swing in chase win probability from the next ball, against the average ball in the data. */
export function nextBallSwing(league: Odds, wk: number, need: number, meanAbsWpa: number): number {
  const v0 = chaseWin(league, league, 12, wk, need);
  let s = 0;
  for (let i = 0; i < 6; i++) s += league[i] * Math.abs(chaseWin(league, league, 11, wk, need - RUN[i]) - v0);
  s += league[6] * Math.abs(chaseWin(league, league, 11, wk - 1, need) - v0)
    + (league[7] + league[8]) * Math.abs(chaseWin(league, league, 12, wk, need - 1) - v0);
  return s / meanAbsWpa;
}

export interface Ball { i: number; by: 19 | 20 }
export interface Played { out: Ball[]; r: number; w: number; legal: number }

function draw(p: Odds, rand: () => number): number {
  let x = rand(), i = 0;
  for (; i < 8; i++) { x -= p[i]; if (x < 0) return i; }
  return 8;
}

/** One random playing of the last two overs, drawn from the same odds the exact figure uses. */
export function playOne(pX: Odds, league: Odds, need: number, wk: number, rand: () => number = Math.random): Played {
  let legal = 0, r = need, w = wk;
  const out: Ball[] = [];
  while (r > 0 && w > 0 && legal < 12) {
    const p = legal < 6 ? pX : league, i = draw(p, rand);
    out.push({ i, by: legal < 6 ? 19 : 20 });
    if (i <= 5) { r -= RUN[i]; legal++; } else if (i === 6) { w--; legal++; } else { r -= 1; }
  }
  return { out, r, w, legal };
}

/** Outcome of one played match for the bowling side: 2 defended, 1 tie, 0 chased down. */
export const outcome = (g: Played) => (g.r <= 0 ? 0 : g.r === 1 ? 1 : 2);
