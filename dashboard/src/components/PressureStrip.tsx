import { scaleLinear } from 'd3-scale';
import type { FieldGap, Pick, Role } from '../data';
import { range, sideColour, signed, verdictLabel, verdictStyle } from '../format';
import { pressureLine } from '../copy';

interface Props { p: Pick; role: Role; field: FieldGap[]; domain: number; dataRange: string }

/** Normal vs pressure: the two values, the gap with its 80% interval, a verdict chip, and where he sits among all eligible players. */
export function PressureStrip({ p, role, field, domain, dataRange }: Props) {
  const q = p.pressure;
  const vs = verdictStyle(role, q.verdict);
  const H = 30;
  // percentage coordinates: the strip stretches with the card without distorting the marker
  const x = scaleLinear().domain([-domain, domain]).range([3, 97]).clamp(true);
  const at = (v: number) => `${x(v).toFixed(2)}%`;
  const rankInField = [...field].sort((a, b) => b.gap - a.gap).findIndex((f) => f.player_id === p.player_id) + 1;
  const mark = sideColour(role, q.gap >= 0);
  const label = `${p.name}: pressure gap ${signed(q.gap)}, 80% range ${range(q.lo80, q.hi80)}. `
    + `Number ${rankInField} of ${field.length} eligible ${role}s by gap, largest first.`;
  return (
    <div className="pstrip">
      <dl className="pstrip-vals">
        <div><dt>Normal</dt><dd><span className="num">{signed(q.normal)}</span><span className="pstrip-balls">{q.balls_normal.toLocaleString('en-IN')} balls</span></dd></div>
        <div><dt>Pressure</dt><dd><span className="num">{signed(q.pressure)}</span><span className="pstrip-balls">{q.balls_pressure.toLocaleString('en-IN')} balls</span></dd></div>
        <div className="pstrip-gap"><dt>Gap</dt><dd><span className="num">{signed(q.gap)}</span><span className="pstrip-balls">80% range {range(q.lo80, q.hi80)}</span></dd></div>
      </dl>
      <p className="pstrip-plain">{pressureLine(p)}</p>
      <div className="pstrip-foot">
        <span className="chip" style={{ background: vs.bg, color: vs.fg, borderColor: vs.border }}>{verdictLabel(q.verdict, dataRange, q.balls_pressure)}</span>
        <figure className="pstrip-field">
          <svg width="100%" height={H} role="img" aria-label={label}>
            <line x1={at(0)} x2={at(0)} y1={2} y2={H - 2} stroke="#586080" strokeWidth={1} strokeDasharray="2 2" />
            {field.map((f) => f.player_id !== p.player_id && (
              <line key={f.player_id} x1={at(f.gap)} x2={at(f.gap)} y1={9} y2={21} stroke="#8b92aa" strokeOpacity={0.6} strokeWidth={1.2} />
            ))}
            <line x1={at(q.lo80)} x2={at(q.hi80)} y1={15} y2={15} stroke={mark} strokeWidth={3} strokeLinecap="round" />
            <circle cx={at(q.gap)} cy={15} r={5.5} fill={mark} stroke="#ffffff" strokeWidth={1.5} />
          </svg>
          <figcaption className="pstrip-axis">
            <span>worse</span>
            <span>{rankInField} of {field.length} eligible by gap</span>
            <span>better</span>
          </figcaption>
        </figure>
      </div>
    </div>
  );
}
