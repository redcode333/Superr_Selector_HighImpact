import { useState } from 'react';
import type { Meta, Role } from '../data';
import { ROLE_PLURAL } from '../copy';
import { signed } from '../format';

const ROLES: Role[] = ['batter', 'bowler'];

export function ChangesNote({ meta }: { meta: Meta }) {
  const fl = meta.rules.high_pressure_balls_min;
  const pr = meta.changes.pressure_rule;
  const unchanged = ROLES.filter((r) => !pr.left[r].length);
  // open on wide screens; on phones start folded so the picks come first (the summary line still names the change)
  const [open, setOpen] = useState(() => typeof window === 'undefined' || window.matchMedia('(min-width: 720px)').matches);
  return (
    <details className="changes" open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary><span className="changes-tag">What changed from v2</span> One more eligibility rule: a pressure value above average</summary>
      <div className="changes-body">
        <div className="changes-text">
          <p>
            A player is now eligible only if his value on high-pressure balls (importance {meta.li_pressure} or more) is above average, that is, above zero in {meta.unit}.
            We added this rule on {pr.decided}, after reviewing the v2 list against the problem statement. It is a judgement made with the results in view,
            in the same way as the {fl}-ball floor, not a test we set in advance.
          </p>
          <p>
            Every player is treated the same, and the ratings and the order are unchanged. It removes {pr.failed.batter} of the {pr.pool_before.batter} batters
            and {pr.failed.bowler} of the {pr.pool_before.bowler} bowlers who passed the earlier rules.
          </p>
          <ul className="swaps">
            {ROLES.flatMap((r) => pr.left[r].map((o) => <li key={o.name} className="swap-out"><b>Out:</b> {o.name}, v2 {r} #{o.v2_rank}, pressure value {signed(o.pressure_value)} over {o.high_pressure_balls} balls</li>))}
            {ROLES.flatMap((r) => pr.entered[r].map((o) => <li key={o.name} className="swap-in"><b>In:</b> {o.name}, now #{o.v3_rank} (was #{o.v2_rank}), pressure value {signed(o.pressure_value)}</li>))}
          </ul>
          {unchanged.map((r) => <p key={r}>{ROLE_PLURAL[r]}: the top 10 is the same as v2.</p>)}
        </div>
        <div className="table-scroll">
          <table className="sens">
            <caption>Earlier change (v2): the {fl}-ball floor, changes to the top 10 against no floor</caption>
            <thead>
              <tr><th scope="col">Floor</th>{ROLES.map((r) => <th scope="col" key={r}>{ROLE_PLURAL[r]}</th>)}</tr>
            </thead>
            <tbody>
              {meta.changes.sensitivity.map((row) => (
                <tr key={row.floor} className={row.floor === fl ? 'is-ours' : undefined}>
                  <th scope="row">{row.floor}{row.floor === fl ? ' (v2)' : ''}</th>
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
