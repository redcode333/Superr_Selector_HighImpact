import { scaleLinear } from 'd3-scale';
import { motion } from 'framer-motion';
import type { FieldGap, Pick, Role } from '../data';
import { Badge } from './Badge';
import { worseNote } from '../copy';
import { range, signed } from '../format';
import { PressureStrip } from './PressureStrip';

/** Cards keep their big rank number, so their badge shows only the ALT chip; the deep-dive badge shows the rank chip too. */
export function PlayerBadge({ p, size, rankChip = true }: { p: Pick; size: number; rankChip?: boolean }) {
  return <Badge badge={p.badge} initials={p.initials} rank={p.rank} alt={p.group === 'alternate'} size={size} rankChip={rankChip} />;
}

interface Props {
  p: Pick; role: Role; field: FieldGap[]; gapDomain: number; ratingMax: number; dataRange: string; onOpen: (p: Pick) => void;
}

export function PlayerCard({ p, role, field, gapDomain, ratingMax, dataRange, onOpen }: Props) {
  const alt = p.group === 'alternate';
  const note = worseNote(p, role, dataRange);
  const x = scaleLinear().domain([0, ratingMax]).range([0, 100]).clamp(true);
  return (
    <motion.article className={`pcard${alt ? ' pcard-alt' : ''}`} whileHover={{ y: -2 }} transition={{ duration: 0.15 }}>
      <button type="button" className="pcard-open" onClick={() => onOpen(p)}
        aria-label={`${p.name}, ${alt ? 'alternate' : `${role} rank ${p.rank}`}. Open his deep dive.`} />
      <div className="pcard-head">
        <span className="pcard-rank num" aria-hidden="true">{alt ? 'Alt' : p.rank}</span>
        <PlayerBadge p={p} size={64} rankChip={false} />
        <div className="pcard-id">
          <h4 className="pcard-name">{p.name}</h4>
          <p className="pcard-sub">{alt ? `Alternate · rank ${p.rank}` : `${role === 'batter' ? 'Batting' : 'Bowling'} rank ${p.rank}`}</p>
        </div>
      </div>
      <div className="pcard-stats">
        <div className="pcard-rating">
          <div className="stat-top"><span className="stat-label">Cautious rating</span><span className="stat-val num">{signed(p.cautious)}</span></div>
          <div className="rbar" aria-hidden="true">
            <span className="rbar-range" style={{ left: `${x(p.range_lo)}%`, width: `${x(p.range_hi) - x(p.range_lo)}%` }} />
            <span className="rbar-best" style={{ left: `${x(p.best)}%` }} />
          </div>
          <span className="stat-note">80% range {range(p.range_lo, p.range_hi)} · best estimate {signed(p.best)}</span>
        </div>
        <div className="pcard-mini">
          <div><span className="stat-val num">{p.pct_top10.toFixed(1)}%</span><span className="stat-label">of re-rankings in the top 10</span></div>
          <div><span className="stat-val num">{p.high_pressure_balls}</span><span className="stat-label">high-pressure balls</span></div>
        </div>
      </div>
      <PressureStrip p={p} role={role} field={field} domain={gapDomain} dataRange={dataRange} />
      {note && <p className="pcard-note">{note}</p>}
    </motion.article>
  );
}
