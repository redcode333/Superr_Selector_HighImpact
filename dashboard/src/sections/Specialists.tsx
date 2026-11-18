import { useMemo, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import type { Picks, Role } from '../data';
import { Section } from '../components/SectionBand';
import type { Specialists as Spec } from './types';

const SHOW = 8;

/** Role chips recomputed on the eligible pool (build_specialists.py definitions). */
export function SpecialistsSection({ sp, picks }: { sp: Spec; picks: Picks }) {
  const rm = useReducedMotion() ?? false;
  const chips = useMemo(() => {
    const seen = new Map<string, { role: Role; chip: string; label: string; metric: string; min: number; n: number }>();
    for (const r of sp.rows) {
      const k = `${r.role}|${r.chip}`;
      const c = seen.get(k);
      if (c) c.n++; else seen.set(k, { role: r.role, chip: r.chip, label: r.label, metric: r.metric, min: r.min_balls, n: 1 });
    }
    return [...seen.values()];
  }, [sp]);
  const [cur, setCur] = useState(`${chips[0].role}|${chips[0].chip}`);
  const [all, setAll] = useState(false);
  const c = chips.find((x) => `${x.role}|${x.chip}` === cur)!;
  const rows = sp.rows.filter((r) => r.role === c.role && r.chip === c.chip);
  const ours = new Set(picks.roles[c.role].players.filter((p) => p.group === 'top 10').map((p) => p.player_id));
  const alt = new Set(picks.roles[c.role].players.filter((p) => p.group === 'alternate').map((p) => p.player_id));
  const shown = all ? rows : rows.slice(0, SHOW);
  const lowGood = c.metric === 'Economy';
  const oursIn = rows.slice(0, SHOW).filter((r) => ours.has(r.player_id)).length;
  return (
    <Section id="specialists" title="Specialists" lede="Who leads each job among all eligible players, against par for the same kind of ball. Our picks are marked.">
      <div className="spec">
        {(['batter', 'bowler'] as Role[]).map((role) => (
          <div key={role} className="spec-chips" role="group" aria-label={role === 'batter' ? 'Batting roles' : 'Bowling roles'}>
            <span className="spec-gl">{role === 'batter' ? 'Batting' : 'Bowling'}</span>
            {chips.filter((x) => x.role === role).map((x) => {
              const k = `${x.role}|${x.chip}`;
              return <button type="button" key={k} className={`chip-btn${k === cur ? ' on' : ''}`} aria-pressed={k === cur} onClick={() => { setCur(k); setAll(false); }}>{x.label}</button>;
            })}
          </div>
        ))}
        <div className="card spec-list">
          <div className="h2h-head">
            <h3>{c.label}</h3>
            <span>{c.metric} · at least {c.min} balls · par {lowGood ? rows[0]?.par.toFixed(2) : rows[0]?.par.toFixed(1)} · {rows.length} qualify · {oursIn} of our top 10 in the first {Math.min(SHOW, rows.length)}</span>
          </div>
          <ol className="spec-rows">
            <AnimatePresence initial={false}>
              {shown.map((r, i) => {
                const mine = ours.has(r.player_id), a = alt.has(r.player_id);
                const ahead = lowGood ? r.edge < 0 : r.edge > 0;
                return (
                  <motion.li key={`${cur}-${r.player_id}`} layout={!rm} initial={rm ? false : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={rm ? undefined : { opacity: 0 }}
                    className={`spec-row${mine ? ' mine' : ''}`}>
                    <span className="spec-pos num">{i + 1}</span>
                    <span className="spec-name">{r.name}{mine && <span className="spec-tag">Our #{r.rank}</span>}{a && <span className="spec-tag alt">Alternate</span>}</span>
                    <span className="spec-val num">{lowGood ? r.value.toFixed(2) : r.value.toFixed(1)}</span>
                    <span className={`spec-edge ${ahead ? (c.role === 'batter' ? 'tone-bat-text' : 'tone-bowl-text') : 'muted'}`}>
                      {r.edge >= 0 ? '+' : '−'}{Math.abs(r.edge).toFixed(lowGood ? 2 : 1)} vs par</span>
                    <span className="spec-balls">{r.balls} balls</span>
                  </motion.li>
                );
              })}
            </AnimatePresence>
          </ol>
          {rows.length > SHOW && (
            <button type="button" className="btn-text" onClick={() => setAll(!all)} aria-expanded={all}>{all ? `Show the first ${SHOW}` : `Show all ${rows.length}`}</button>
          )}
        </div>
      </div>
    </Section>
  );
}
