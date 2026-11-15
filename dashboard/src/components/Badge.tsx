import type { Badge as BadgeData } from '../data';

const C = 50;
const rad = (d: number) => (d * Math.PI) / 180;
const pt = (deg: number, r: number) => [C + r * Math.sin(rad(deg)), C - r * Math.cos(rad(deg))];

/** A picture of the player's own data (scripts/badges.py), never of the player. Batter: where he scores; bowler: which overs he bowls. */
export function Badge({ badge, initials, rank, alt, size, rankChip = true }: { badge: BadgeData; initials: string; rank: number; alt: boolean; size: number; rankChip?: boolean }) {
  const big = size >= 100;
  return (
    <span className={`badge${big ? ' badge-lg' : ''}`} style={{ width: size, height: size }} role="img" aria-label={badge.label}>
      <svg viewBox="0 0 100 100" width={size} height={size} aria-hidden="true" focusable="false">
        {badge.kind === 'batter'
          ? badge.wedges.map((w) => {
              const a1 = pt(w.from, badge.r_in), a2 = pt(w.to, badge.r_in), b1 = pt(w.from, w.len), b2 = pt(w.to, w.len);
              const d = `M${a1[0].toFixed(2)},${a1[1].toFixed(2)} L${b1[0].toFixed(2)},${b1[1].toFixed(2)} A${w.len},${w.len} 0 0 1 ${b2[0].toFixed(2)},${b2[1].toFixed(2)} `
                + `L${a2[0].toFixed(2)},${a2[1].toFixed(2)} A${badge.r_in},${badge.r_in} 0 0 0 ${a1[0].toFixed(2)},${a1[1].toFixed(2)} Z`;
              return <path key={w.name} d={d} fill={w.color} stroke="#ffffff" strokeWidth={0.8} strokeLinejoin="round" />;
            })
          : badge.ticks.map((t) => (
              <g key={t.over} transform={`rotate(${t.angle} ${C} ${C})`}>
                {t.empty
                  ? <rect x={C - 1.6} y={C - badge.r_in - t.len} width={3.2} height={t.len} rx={1.4} fill="none" stroke="#b9c0d6" strokeWidth={0.7} />
                  : <rect x={C - 1.9} y={C - badge.r_in - t.len} width={3.8} height={t.len} rx={1.6} fill={t.color!} />}
              </g>
            ))}
        <circle cx={C} cy={C} r={badge.disc_r} fill="#0f1a3c" />
        <text x={C} y={C + 6.6} textAnchor="middle" fontFamily="Barlow Condensed, 'Arial Narrow', sans-serif" fontWeight={700} fontSize={19} fill="#ffffff">{initials}</text>
      </svg>
      {(alt || rankChip) && <span className="badge-chip" aria-hidden="true">{alt ? 'ALT' : rank}</span>}
    </span>
  );
}
