import { useMemo, useState } from 'react';
import type { Picks, Role } from '../data';
import { Section } from '../components/SectionBand';
import { col, inkFor } from '../dive/common';
import type { Matchups } from './types';

const PHASES = ['Powerplay', 'Middle', 'Death'];
const PLABEL: Record<string, string> = { Powerplay: 'Powerplay', Middle: 'Middle overs', Death: 'Death overs' };
const TLABEL: Record<string, string> = { Pace: 'Against pace', Spin: 'Against spin', RHB: 'To right-handers', LHB: 'To left-handers' };

/** Batter x bowler grid and phase heat map (ideas from design/live_dashboard; design-system colours). */
export function HeadToHead({ mu, picks }: { mu: Matchups; picks: Picks }) {
  const bats = picks.roles.batter.players, bowls = picks.roles.bowler.players;
  const pairs = useMemo(() => new Map(mu.pairs.map((p) => [`${p.batter_id}-${p.bowler_id}`, p])), [mu]);
  const [hl, setHl] = useState<{ b: number; w: number } | null>(null);
  const [heatRole, setHeatRole] = useState<Role>('batter');
  const [heatId, setHeatId] = useState(bats[0].player_id);
  const list = heatRole === 'batter' ? bats : bowls;
  const heatPlayer = list.find((p) => p.player_id === heatId) ?? list[0];
  const types = heatRole === 'batter' ? ['Pace', 'Spin'] : ['RHB', 'LHB'];
  const cell = (b: number, w: number) => pairs.get(`${b}-${w}`);
  const counted = bats.flatMap((b) => bowls.map((w) => cell(b.player_id, w.player_id))).filter(Boolean);
  const readable = counted.filter((p) => p!.balls >= mu.min_balls).length;
  const sel = hl ? cell(hl.b, hl.w) : undefined;
  const selNames = hl ? [bats.find((x) => x.player_id === hl.b)!.name, bowls.find((x) => x.player_id === hl.w)!.name] : null;
  return (
    <Section id="h2h" title="Head-to-head" lede={`Our batters against our bowlers, and each player's phase record. Cells with fewer than ${mu.min_balls} balls are greyed out.`}>
      <div className="h2h-wrap">
        <div className="card h2h-grid-card">
          <div className="h2h-head">
            <h3>Batter against bowler</h3>
            <span>Strike rate when they met · par {mu.league_sr.toFixed(1)} (every batter, every ball) · {readable} of {bats.length * bowls.length} pairs have {mu.min_balls}+ balls</span>
          </div>
          <div className="h2h-scroll" tabIndex={0} aria-label="Head-to-head grid, scrolls sideways">
            <table className="h2h">
              <caption className="sr-only">Strike rate of each batter against each bowler; balls, runs and outs in each cell's label</caption>
              <thead>
                <tr><th scope="col" className="h2h-corner">Batter ↓ · bowler →</th>
                  {bowls.map((w) => <th scope="col" key={w.player_id} className="h2h-col"><span>{w.name}</span></th>)}</tr>
              </thead>
              <tbody>
                {bats.map((b) => (
                  <tr key={b.player_id}>
                    <th scope="row" className="h2h-row">{b.name}</th>
                    {bowls.map((w) => {
                      const p = cell(b.player_id, w.player_id);
                      const few = !p || p.balls < mu.min_balls;
                      const sr = p && p.balls ? (100 * p.runs) / p.balls : null;
                      const bg = few || sr == null ? undefined : col(sr - mu.league_sr, 60);
                      const label = p ? `${b.name} against ${w.name}: ${p.runs} runs off ${p.balls} balls, out ${p.outs}${few ? ' (too few balls to read)' : `, strike rate ${Math.round(sr!)}`}` : `${b.name} against ${w.name}: never met`;
                      const on = hl?.b === b.player_id && hl?.w === w.player_id;
                      return (
                        <td key={w.player_id}>
                          <button type="button" className={`h2h-cell${few ? ' few' : ''}${on ? ' on' : ''}`} style={bg ? { background: bg, color: inkFor(bg) } : undefined}
                            title={label} aria-label={label} aria-pressed={on} onClick={() => setHl(on ? null : { b: b.player_id, w: w.player_id })}>
                            {few ? (p ? <><span className="h2h-n">{p.balls}</span> balls</> : '·') : Math.round(sr!)}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="h2h-read" aria-live="polite">
            {sel && selNames ? `${selNames[0]} against ${selNames[1]}: ${sel.runs} runs off ${sel.balls} balls, out ${sel.outs}, ${sel.fours} ${sel.fours === 1 ? 'four' : 'fours'}, ${sel.sixes} ${sel.sixes === 1 ? 'six' : 'sixes'}, ${sel.dots} ${sel.dots === 1 ? 'dot' : 'dots'}, over ${sel.matches} ${sel.matches === 1 ? 'match' : 'matches'}.`
              : hl && selNames ? `${selNames[0]} and ${selNames[1]} have not met in the data.` : 'Select a cell for the full record. Pink: batter on top; blue: bowler on top. Grey: under the ball minimum, shown as a ball count.'}
          </p>
        </div>

        <div className="card heat-card">
          <div className="h2h-head"><h3>Phase heat map</h3><span>{heatRole === 'batter' ? 'Strike rate' : 'Economy'} by phase against par for the same kind of ball</span></div>
          <div className="seg-group" role="group" aria-label="Role">
            {(['batter', 'bowler'] as Role[]).map((r) => (
              <button type="button" key={r} className={`seg${heatRole === r ? ' on' : ''}`} aria-pressed={heatRole === r}
                onClick={() => { setHeatRole(r); setHeatId(picks.roles[r].players[0].player_id); }}>{r === 'batter' ? 'Batters' : 'Bowlers'}</button>
            ))}
          </div>
          <label className="heat-l" htmlFor="heat-pick">Player</label>
          <select id="heat-pick" className="heat-select" value={heatPlayer.player_id} onChange={(e) => setHeatId(Number(e.target.value))}>
            {list.map((p) => <option key={p.player_id} value={p.player_id}>{p.group === 'alternate' ? 'Alt' : p.rank}. {p.name}</option>)}
          </select>
          <div className="heat" role="table" aria-label={`Phase heat map for ${heatPlayer.name}`}>
            <div role="row" className="heat-r"><span role="columnheader" />{types.map((t) => <span role="columnheader" key={t} className="heat-h">{TLABEL[t]}</span>)}</div>
            {PHASES.map((ph) => (
              <div role="row" className="heat-r" key={ph}>
                <span role="rowheader" className="heat-ph">{PLABEL[ph]}</span>
                {types.map((t) => {
                  const s = mu.splits.find((x) => x.player_id === heatPlayer.player_id && x.phase === ph && x.vs === t);
                  const few = !s || s.balls < mu.min_balls || s.rate == null;
                  const bg = few ? undefined : col(s!.batter_edge!, 40);
                  const bat = heatRole === 'batter';
                  const better = !few && (bat ? s!.batter_edge! >= 0 : s!.batter_edge! <= 0);
                  return (
                    <div role="cell" key={t} className={`heat-c${few ? ' few' : ''}`} style={bg ? { background: bg, color: inkFor(bg) } : undefined}
                      title={s ? `${s.balls} balls, ${s.runs} runs, ${s.outs} ${bat ? 'outs' : 'wickets'}` : 'No balls'}>
                      {few ? (<><b className="num">{s?.balls ? `${s.balls} balls` : 'No balls'}</b><span>Too few to read</span></>) : (
                        <><b className="num">{bat ? Math.round(s!.rate!) : s!.rate!.toFixed(2)}</b>
                          <span>{bat ? 'strike rate' : 'economy'} · par {bat ? s!.par.toFixed(1) : s!.par.toFixed(2)} · {better ? 'ahead of par' : 'behind par'}</span>
                          <span>{s!.balls} balls · {s!.runs} runs · {s!.outs} {bat ? 'out' : 'wkts'}</span></>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}
