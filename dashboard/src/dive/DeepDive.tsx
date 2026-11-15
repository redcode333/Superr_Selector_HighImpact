import { useId, useMemo, useState } from 'react';
import type { Meta, Picks, Role } from '../data';
import { useJSON } from '../data';
import { Section } from '../components/SectionBand';
import type { Players } from './types';
import { BatterPage, BowlerPage } from './Pages';

const ROLES: Role[] = ['batter', 'bowler'];

export function DeepDive({ picks, meta, selected, onSelect }: { picks: Picks; meta: Meta; selected: number; onSelect: (id: number) => void }) {
  const players = useJSON<Players>('players.json');
  const [q, setQ] = useState('');
  const searchId = useId();
  const all = useMemo(() => ROLES.flatMap((r) => picks.roles[r].players.map((p) => ({ p, role: r }))), [picks]);
  const gapDomain = useMemo(() => Math.max(...ROLES.flatMap((r) => picks.roles[r].field.map((f) => Math.abs(f.gap)))) * 1.05, [picks]);
  const shown = all.filter(({ p }) => p.name.toLowerCase().includes(q.trim().toLowerCase()));
  const cur = all.find((x) => x.p.player_id === selected) ?? all[0];
  const d = players.data?.[String(cur.p.player_id)];
  return (
    <Section id="dive" title="Player deep dive" lede="Any top-10 player or alternate: where he scores or concedes, over by over, and two things to try.">
      <div className="picker">
        <label htmlFor={searchId} className="picker-l">Find a player</label>
        <input id={searchId} type="search" className="picker-search" placeholder="Type a name" value={q} onChange={(e) => setQ(e.target.value)} autoComplete="off" />
        {ROLES.map((r) => {
          const list = shown.filter((x) => x.role === r);
          if (!list.length) return null;
          return (
            <div key={r} className="picker-group" role="group" aria-label={r === 'batter' ? 'Batters' : 'Bowlers'}>
              <span className="picker-gl">{r === 'batter' ? 'Batters' : 'Bowlers'}</span>
              <div className="picker-list">
                {list.map(({ p }) => (
                  <button type="button" key={p.player_id} className={`pk${p.player_id === cur.p.player_id ? ' on' : ''}`} aria-pressed={p.player_id === cur.p.player_id}
                    onClick={() => onSelect(p.player_id)}>
                    <span className="pk-r num">{p.group === 'alternate' ? 'Alt' : p.rank}</span>{p.name}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
        {!shown.length && <p className="picker-none" role="status">No top-10 player or alternate matches "{q}".</p>}
      </div>
      {players.error && <p role="alert">Could not load the deep dives: {players.error}</p>}
      {!d && !players.error && <p>Loading…</p>}
      {d && (
        <div key={d.player_id}>
          {d.role === 'batter'
            ? <BatterPage d={d} ctx={{ pick: cur.p, field: picks.roles.batter.field, gapDomain, dataRange: meta.data_range }} />
            : <BowlerPage d={d} ctx={{ pick: cur.p, field: picks.roles.bowler.field, gapDomain, dataRange: meta.data_range }} />}
        </div>
      )}
    </Section>
  );
}
