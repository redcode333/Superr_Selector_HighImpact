import { useState, type ReactNode } from 'react';
import type { Callout, Dive, Flag, Region } from './types';

export const RAMP = ['#254aa5', '#4b6cb7', '#728dc8', '#9baed7', '#c5cfe6', '#f1f1f4', '#f6cddb', '#f6a8c2', '#f381aa', '#ed5593', '#e4007c'];
/** Diverging colour: positive edge = batter on top (pink), negative = bowler on top (blue). */
export const col = (edge: number, span: number) => RAMP[Math.round(5 + 5 * Math.max(-1, Math.min(1, edge / span)))];
export const inkFor = (hex: string) => (['#254aa5', '#4b6cb7', '#e4007c'].includes(hex) ? '#ffffff' : '#0f1a3c');
export const r0 = (x: number | null | undefined) => (x == null ? '—' : String(Math.round(x)));
export const f1 = (x: number | null | undefined) => (x == null ? '—' : x.toFixed(1));
export const f2 = (x: number | null | undefined) => (x == null ? '—' : x.toFixed(2));
export const n = (x: number) => x.toLocaleString('en-IN');

export function FewFlag({ flag }: { flag?: Flag }) {
  if (!flag?.few) return null;
  return <span className="few-flag" title={`Only ${flag.total} ${flag.what}; this section needs ${flag.min}.`}>few balls · {flag.total} of {flag.min}</span>;
}

export function Card({ title, flag, children, className = '', lede }: { title: string; flag?: Flag; children: ReactNode; className?: string; lede?: ReactNode }) {
  return (
    <section className={`dcard ${className}`} aria-label={title}>
      <div className="dcard-head">
        <h4>{title}</h4>
        <FewFlag flag={flag} />
      </div>
      {lede && <p className="dcard-lede">{lede}</p>}
      {children}
    </section>
  );
}

export function Callouts({ dive, slot }: { dive: Dive; slot: string }) {
  const items: Callout[] = dive.callouts.filter((c) => c.slot === slot);
  if (!items.length) return null;      // no qualifying template: show nothing
  return <ul className="callouts">{items.map((c) => <li key={c.id}>{c.text}</li>)}</ul>;
}

// ---------------- wheel (ported from TimDavid.dc.html / Bumrah.dc.html)
export interface Metric { key: string; label: string; value: (r: Region) => number; show: (r: Region) => string }

export function Wheel({ dive, metrics, edge, aria }: { dive: Dive; metrics: Metric[]; edge: (r: Region) => number; aria: string }) {
  const [mk, setMk] = useState(metrics[0].key);
  const m = metrics.find((x) => x.key === mk)!;
  const R = dive.regions;
  const max = Math.max(1, ...R.map((r) => m.value(r)));
  const cx = 220, cy = 218, rad = (d: number) => (d * Math.PI) / 180;
  const pt = (d: number, r: number) => [cx + r * Math.sin(rad(d)), cy - r * Math.cos(rad(d))];
  return (
    <div className="wheel">
      <div className="seg-group" role="group" aria-label="Size slices by">
        {metrics.map((x) => (
          <button type="button" key={x.key} className={`seg${x.key === mk ? ' on' : ''}`} aria-pressed={x.key === mk} onClick={() => setMk(x.key)}>{x.label}</button>
        ))}
      </div>
      <svg viewBox="0 -14 440 470" className="wheel-svg" role="img" aria-label={aria}>
        {[200, 150, 100, 50].map((r) => <circle key={r} cx={cx} cy={cy} r={r} fill="none" stroke="#e4e8f1" />)}
        {R.map((s) => {
          const v = m.value(s), r = 34 + 150 * Math.sqrt(Math.max(0, v) / max);
          const a = pt(s.from, r), b = pt(s.to, r);
          const fill = s.sr == null || s.league_sr == null ? '#f1f1f4' : col(edge(s), 60);
          return <path key={s.name} d={`M${cx},${cy} L${a[0].toFixed(1)},${a[1].toFixed(1)} A${r.toFixed(1)},${r.toFixed(1)} 0 0 1 ${b[0].toFixed(1)},${b[1].toFixed(1)} Z`}
            fill={fill} stroke="#ffffff" strokeWidth={1.5}><title>{`${s.name}: ${m.label.toLowerCase()} ${m.show(s)}, ${s.balls} balls`}</title></path>;
        })}
        <rect x="213" y="190" width="14" height="40" rx="2" fill="#f1f1f4" stroke="#8b92aa" />
        {R.map((s) => {
          const v = m.value(s), r = 34 + 150 * Math.sqrt(Math.max(0, v) / max), mid = (s.from + s.to) / 2;
          const nm = pt(mid, 198), lab = pt(mid, Math.max(26, r * 0.66));
          const fill = s.sr == null || s.league_sr == null ? '#f1f1f4' : col(edge(s), 60);
          return (
            <g key={s.name}>
              <text x={nm[0]} y={nm[1] + 4} textAnchor={(Math.sin(rad(mid)) < -0.6 ? 'start' : Math.sin(rad(mid)) > 0.6 ? 'end' : 'middle')} className="wheel-name">{s.name}</text>
              {r > 60 && <text x={lab[0]} y={lab[1] + 5} textAnchor="middle" className="wheel-val" fill={inkFor(fill)}>{m.show(s)}</text>}
            </g>
          );
        })}
        <text x="220" y="-2" textAnchor="middle" className="wheel-axis">Bowler's end ↑</text>
        <text x="436" y="450" textAnchor="end" className="wheel-axis">Off side →</text>
        <text x="4" y="450" textAnchor="start" className="wheel-axis">← Leg side</text>
      </svg>
      <div className="ramp-key" aria-hidden="true">
        <span>Bowler on top</span><span className="ramp-bar" /><span>Batter on top</span>
      </div>
    </div>
  );
}

export function Kpis({ items }: { items: [string, ReactNode][] }) {
  return <dl className="kpis">{items.map(([k, v]) => <div key={k} className="kpi"><dt>{k}</dt><dd className="num">{v}</dd></div>)}</dl>;
}

export function TypeBars({ rows, noun }: { rows: { type: string; n: number; pct: number; league_pct: number }[]; noun: string }) {
  const max = Math.max(1, ...rows.flatMap((r) => [r.pct, r.league_pct]));
  return (
    <ul className="typebars">
      {rows.map((r) => (
        <li key={r.type}>
          <span className="tb-name">{r.type} · {r.n}</span>
          <span className="tb-track" aria-hidden="true">
            <span className="tb-bar" style={{ width: `${(100 * r.pct) / max}%` }} />
            <span className="tb-league" style={{ left: `${(100 * r.league_pct) / max}%` }} />
          </span>
          <span className="tb-pct num">{Math.round(r.pct)}% <span className="tb-lg">league {Math.round(r.league_pct)}%</span></span>
        </li>
      ))}
      <li className="tb-key">Bar: his share of {noun} · tick: every player</li>
    </ul>
  );
}
