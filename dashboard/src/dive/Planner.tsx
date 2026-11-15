import { useState } from 'react';
import type { BowlerDive } from './types';
import { Callouts, f1 } from './common';

const LIMIT = 4;

/** "Try it: plan his four overs" (Bumrah.dc.html). Saving per over = (par - his economy) x balls / (balls + 60), from the data. */
export function Planner({ dive }: { dive: BowlerDive }) {
  const P = dive.planner, OV = dive.overs;
  const [o, setO] = useState<number[]>(P.usual.overs);
  const [msg, setMsg] = useState('');
  const pick = (k: number) => {
    if (o.includes(k)) { setO(o.filter((x) => x !== k)); setMsg(''); return; }
    if (o.length >= LIMIT) { setMsg('Four overs is the limit. Tap a chosen over to free it first.'); return; }
    if (o.includes(k - 1) || o.includes(k + 1)) { setMsg(`He cannot bowl two overs in a row: over ${k} sits next to one you picked.`); return; }
    setO([...o, k].sort((a, b) => a - b)); setMsg('');
  };
  const same = (a: number[], b: number[]) => a.length === b.length && a.every((x) => b.includes(x));
  const chosen = OV.filter((x) => o.includes(x.over));
  const saved = chosen.reduce((t, x) => t + x.saving, 0);
  const avg = chosen.reduce((t, x) => t + (x.balls ? x.par_econ : P.avg_par_unbowled), 0);
  const wk = chosen.reduce((t, x) => t + x.wkts, 0), wb = chosen.reduce((t, x) => t + x.balls, 0);
  const isBest = same(o, P.best.overs);
  let vHead: string, vText: string;
  if (chosen.length < LIMIT) { vHead = `Pick ${LIMIT - chosen.length} more`; vText = `The best plan the data allows saves about ${f1(P.best.saving)} runs.`; }
  else if (isBest) { vHead = 'This is the best plan in the data'; vText = ''; }
  else {
    vHead = `${f1(Math.max(0, P.best.saving - saved))} runs short of the best plan`;
    vText = `The best is overs ${P.best.overs.join(', ')}.`;
  }
  const presets: [string, number[]][] = [['His usual overs', P.usual.overs], ['Best plan', P.best.overs], ['Clear', []]];
  return (
    <div className="planner">
      <div className="seg-group" role="group" aria-label="Plan presets">
        {presets.map(([label, list]) => (
          <button type="button" key={label} className={`seg${same(o, list) ? ' on' : ''}`} aria-pressed={same(o, list)} onClick={() => { setO([...list]); setMsg(''); }}>{label}</button>
        ))}
      </div>
      <p className="field-note">"His usual overs" are the four non-adjacent overs he has bowled most often ({P.usual.times} times in all, full 20-over innings).</p>
      <p className="field-note">Dashed tiles: he has bowled fewer than {P.few_balls} balls in that over, so its saving is held back further.</p>
      <div className="planner-grid">
        <ol className="tiles" aria-label="Overs: what he concedes an over and what an average bowler concedes">
          {OV.map((x) => {
            const on = o.includes(x.over), blocked = !on && (o.includes(x.over - 1) || o.includes(x.over + 1));
            const few = x.balls < P.few_balls;
            return (
              <li key={x.over}>
                <button type="button" className={`tile${on ? ' on' : ''}${blocked ? ' blocked' : ''}${few && x.balls ? ' few' : ''}`} aria-pressed={on} onClick={() => pick(x.over)}
                  title={`Over ${x.over}: ${x.balls ? `he concedes ${f1(x.econ)} an over, average ${f1(x.par_econ)}, ${x.balls} balls${few ? ' (few balls)' : ''}, ${x.wkts} wickets` : 'never bowled'}`}>
                  <span className="t-n">Over {x.over}</span>
                  <span className="t-v num">{x.balls ? f1(x.econ) : '—'}</span>
                  <span className="t-par">avg {f1(x.balls ? x.par_econ : P.avg_par_unbowled)}</span>
                  <span className="t-tag">{!x.balls ? 'never bowled' : few ? '' : `${x.wkts} wkts`}</span>
                </button>
              </li>
            );
          })}
        </ol>
        <div className="plan-panel">
          <span className="fp-l">Runs your plan saves</span>
          <span className="num fp-pct">{f1(saved)}</span>
          <span className="fp-sub">across {chosen.length} of {LIMIT} overs</span>
          <dl className="fp-rows">
            <div><dt className="num">{f1(avg - saved)}</dt><dd>{dive.name}, allowing for few balls</dd></div>
            <div><dt className="num">{f1(avg)}</dt><dd>Average bowler</dd></div>
            <div><dt className="num">{wb ? `${wk} in ${wb}` : '—'}</dt><dd>his wickets in these overs (balls)</dd></div>
          </dl>
          <div className={`fp-gap${isBest ? ' best' : ''}`} aria-live="polite"><b>{vHead}</b>{vText && <span>{vText}</span>}</div>
          {msg && <p className="fp-msg" role="status">{msg}</p>}
          {isBest && <Callouts dive={dive} slot="planner" />}
        </div>
      </div>
      <p className="dcard-foot">Each over's saving is held back toward zero, more where he has bowled few balls (dashed tiles), so one lucky over cannot win the plan. Wickets are his real count in those overs.</p>
    </div>
  );
}
