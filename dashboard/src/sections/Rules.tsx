import { useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import type { Role } from '../data';
import { Section } from '../components/SectionBand';
import type { Sandbox } from './types';

type Key = 'cautious' | 'projected';

/** "Change our rules, watch the top 10" (Sandbox.dc.html) plus the exposure-floor and pressure-rule controls. Every setting is a real re-run (sandbox.json). */
export function RulesSection({ sb }: { sb: Sandbox }) {
  const rm = useReducedMotion() ?? false;
  const O = sb.ours;
  const [own, setOwn] = useState(O.own), [m, setM] = useState(O.m), [k, setK] = useState<Key>(O.key), [fl, setFl] = useState(O.floor), [pp, setPp] = useState(O.pp);
  const key = (r: Role, o = own, mm = m, kk: Key = k, f = fl, q = pp) => `${r}|${o}|${mm}|${kk}|${f}|${q}`;
  const isOurs = own === O.own && m === O.m && k === O.key && fl === O.floor && pp === O.pp;
  // how often the rule combinations (own share x matches x ranking) leave our list unchanged, at the chosen floor
  const combos = sb.own_share_pct.flatMap((o) => sb.matches_2025_26.flatMap((mm) => sb.keys.map((kk) => [o, mm, kk] as const)));
  const sameCount = (r: Role) => {
    const base = sb.lists[key(r, O.own, O.m, O.key, O.floor, O.pp)].top.slice(0, 10).map((x) => x.player_id);
    return combos.filter(([o, mm, kk]) => {
      const t = sb.lists[key(r, o, mm, kk, fl, pp)].top.slice(0, 10).map((x) => x.player_id);
      return t.length === base.length && t.every((id) => base.includes(id));
    }).length;
  };
  const opt = <T,>(vals: T[], cur: T, set: (v: T) => void, fmt: (v: T) => string, label: string) => (
    <div className="rule-opt">
      <span className="rule-l">{label}</span>
      <div className="seg-group" role="group" aria-label={label}>
        {vals.map((v) => <button type="button" key={String(v)} className={`seg${v === cur ? ' on' : ''}`} aria-pressed={v === cur} onClick={() => set(v)}>{fmt(v)}</button>)}
      </div>
    </div>
  );
  return (
    <Section id="rules" title="Change our rules" lede="Our lists follow the rules below; the 50-ball floor and the pressure rule were added after we had seen results. Move any rule and watch who comes in or drops out. Every setting is a real re-run.">
      <div className="card rules-ctl">
        {opt(sb.own_share_pct, own, setOwn, (v) => `${v}%${v === O.own ? ' (ours)' : ''}`, 'How much of a rating must come from his own matches')}
        {opt(sb.matches_2025_26, m, setM, (v) => `${v}${v === O.m ? ' (ours)' : ''}`, 'Matches needed in 2025–26')}
        {opt(sb.floors, fl, setFl, (v) => `${v}${v === O.floor ? ' (ours)' : ''}`, 'High-pressure balls needed (importance 1.5 or more)')}
        {opt(sb.pressure_rule, pp, setPp, (v) => (v === 1 ? `Required${v === O.pp ? ' (ours)' : ''}` : 'Not required'), 'Pressure value must be above average')}
        {opt(sb.keys, k, setK, (v) => (v === 'cautious' ? 'Cautious rating (ours)' : 'Best estimate'), 'Rank players by')}
        <button type="button" className="btn-text" onClick={() => { setOwn(O.own); setM(O.m); setK(O.key); setFl(O.floor); setPp(O.pp); }} disabled={isOurs}>Back to our rules</button>
      </div>
      <div className="rules-cols">
        {(['batter', 'bowler'] as Role[]).map((r) => {
          const base = sb.lists[key(r, O.own, O.m, O.key, O.floor, O.pp)].top.slice(0, 10);
          const curL = sb.lists[key(r)];
          const top = curL.top.slice(0, 10);
          const outs = base.filter((b) => !top.some((t) => t.player_id === b.player_id));
          const same = 10 - outs.length;
          return (
            <div key={r} className="card rules-col">
              <div className="h2h-head">
                <h3>{r === 'batter' ? 'Batters' : 'Bowlers'}</h3>
                <span className={same === 10 ? 'rules-same' : 'rules-diff'}>{same === 10 ? 'Same 10 players as our list' : `${same} of 10 unchanged`} · {curL.eligible} eligible</span>
              </div>
              <ol className="rrows">
                {top.map((t, p) => {
                  const was = base.findIndex((b) => b.player_id === t.player_id);
                  const isNew = was < 0;
                  return (
                    <motion.li key={t.player_id} layout={!rm} transition={{ type: 'spring', stiffness: 420, damping: 36 }} className={`rrow${isNew ? ' new' : ''}`}>
                      <span className="rr-pos num">{p + 1}</span>
                      <span className="rr-name">{t.name}</span>
                      <span className="rr-note">{isNew ? 'Comes in' : was !== p ? `${was > p ? '▲' : '▼'} from ${was + 1}` : ''}</span>
                      <span className="rr-hp">{t.high_pressure_balls} pressure balls</span>
                    </motion.li>
                  );
                })}
              </ol>
              {outs.length > 0 && <p className="rr-out"><b>Drops out:</b> {outs.map((o) => o.name).join(', ')}</p>}
              <p className="rr-sum">With these floor and pressure settings, {sameCount(r)} of the {combos.length} combinations of the other three rules give the same 10 {r}s as ours.</p>
            </div>
          );
        })}
      </div>
      <p className="rules-note">What the settings mean: "own matches" is how much of a rating comes from the player's own data rather than the pull towards average. "Cautious rating" ranks on the low end of each player's likely range, so short records are not rewarded for luck; "best estimate" ranks on the middle. "High-pressure balls" is the exposure floor added in v2. "Pressure value must be above average" is the rule added in v3: his value on high-pressure balls must be above zero.</p>
    </Section>
  );
}
