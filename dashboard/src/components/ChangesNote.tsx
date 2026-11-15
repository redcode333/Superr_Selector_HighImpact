import { useState } from 'react';
import type { Meta, Role } from '../data';
import { ROLE_PLURAL } from '../copy';

const ROLES: Role[] = ['batter', 'bowler'];

export function ChangesNote({ meta }: { meta: Meta }) {
  const fl = meta.rules.high_pressure_balls_min;
  const sw = meta.changes.swaps;
  const outs = ROLES.flatMap((r) => sw[r].out.map((o) => ({ ...o, role: r })));
  const ins = ROLES.flatMap((r) => sw[r].in.map((o) => ({ ...o, role: r })));
  const unchanged = ROLES.filter((r) => !sw[r].out.length);
  // open on wide screens; on phones start folded so the picks come first (the summary line still names the change)
  const [open, setOpen] = useState(() => typeof window === 'undefined' || window.matchMedia('(min-width: 720px)').matches);
  return (
    <details className="changes" open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary><span className="changes-tag">What changed from v1</span> A new eligibility rule: at least {fl} high-pressure balls</summary>
      <div className="changes-body">
        <div className="changes-text">
          <p>
            A player must now have faced (batters) or bowled (bowlers) at least {fl} high-pressure balls in {meta.data_range},
            on top of the two v1 rules ({meta.rules.own_share_min_pct}% own data, {meta.rules.matches_2025_26_min} matches in 2025–26).
            A pressure pick should rest on a real pressure record. The ratings and the order are unchanged; only the pool is smaller.
          </p>
          <ul className="swaps">
            {outs.map((o) => <li key={o.name} className="swap-out"><b>Out:</b> {o.name}, v1 {o.role} #{o.v1_rank}, {o.high_pressure_balls} high-pressure balls</li>)}
            {ins.map((o) => <li key={o.name} className="swap-in"><b>In:</b> {o.name}, now #{o.v2_rank}, {o.high_pressure_balls} high-pressure balls</li>)}
          </ul>
          {unchanged.map((r) => <p key={r}>{ROLE_PLURAL[r]}: the top 10 is the same as v1.</p>)}
        </div>
        <div className="table-scroll">
          <table className="sens">
            <caption>Floor sensitivity: changes to the top 10 against no floor</caption>
            <thead>
              <tr><th scope="col">Floor</th>{ROLES.map((r) => <th scope="col" key={r}>{ROLE_PLURAL[r]}</th>)}</tr>
            </thead>
            <tbody>
              {meta.changes.sensitivity.map((row) => (
                <tr key={row.floor} className={row.floor === fl ? 'is-ours' : undefined}>
                  <th scope="row">{row.floor}{row.floor === fl ? ' (ours)' : ''}</th>
                  {ROLES.map((r) => (
                    <td key={r}>
                      <span className="sens-n">{row[r].eligible} eligible</span>
                      {row[r].out.length
                        ? <span className="sens-d">out {row[r].out.join(', ')}; in {row[r].in.join(', ')}</span>
                        : <span className="sens-d sens-same">{row.floor === 0 ? 'v1 list' : 'same top 10'}</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  );
}
