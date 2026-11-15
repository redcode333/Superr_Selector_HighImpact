import { useState } from 'react';
import type { BatterDive } from './types';
import { Callouts, n } from './common';

/** The same common field for every batter, shown as an example (from TimDavid.dc.html). */
export const EXAMPLE_FIELD = ['Third man', 'Cover', 'Long-off', 'Long-on', 'Fine leg'];
const SEQ = ['#f1f1f4', '#f6cddb', '#f6a8c2', '#f381aa', '#ed5593', '#e4007c'];
const LIMIT = 5;

export function FieldSetter({ dive }: { dive: BatterDive }) {
  const [f, setF] = useState<string[]>(EXAMPLE_FIELD);
  const [msg, setMsg] = useState('');
  const B = dive.regions, T = dive.region_totals;
  const maxB = Math.max(1, ...B.map((b) => b.boundary_runs));
  const best = dive.best_field.regions;
  const toggle = (name: string) => {
    if (f.includes(name)) { setF(f.filter((x) => x !== name)); setMsg(''); return; }
    if (f.length >= LIMIT) { setMsg('Five is the limit outside the circle. Tap a placed fielder to move him first.'); return; }
    setF([...f, name]); setMsg('');
  };
  const c = 220, rad = (d: number) => (d * Math.PI) / 180;
  const fp = (d: number, r: number) => [c + r * Math.sin(rad(d)), c - r * Math.cos(rad(d))];
  const cov = B.filter((b) => f.includes(b.name));
  const open = B.filter((b) => !f.includes(b.name)).sort((x, y) => y.boundary_runs - x.boundary_runs);
  const sum = (k: 'boundary_runs' | 'fours' | 'sixes' | 'caught') => cov.reduce((t, b) => t + b[k], 0);
  const covRuns = sum('boundary_runs'), pct = T.boundary_runs ? Math.round((100 * covRuns) / T.boundary_runs) : 0;
  const same = (a: string[], b: string[]) => a.length === b.length && a.every((x) => b.includes(x));
  const isBest = same(f, best);
  const gap = open[0];
  const gapHead = !gap ? 'Every region is covered' : isBest ? 'This is the best five for him' : `Biggest gap: ${gap.name}`;
  const gapText = !gap ? '' : isBest
    ? `No other five regions cover more of his boundary runs. ${gap.name} is still open (${n(gap.boundary_runs)} runs).`
    : `${n(gap.boundary_runs)} boundary runs (${T.boundary_runs ? Math.round((100 * gap.boundary_runs) / T.boundary_runs) : 0}% of his total) go there with nobody in the deep.`;
  const presets: [string, string[]][] = [['Example field', EXAMPLE_FIELD], ['Best field for him', best], ['Clear', []]];
  return (
    <div className="field">
      <div className="seg-group" role="group" aria-label="Field presets">
        {presets.map(([label, list]) => (
          <button type="button" key={label} className={`seg${same(f, list) ? ' on' : ''}`} aria-pressed={same(f, list)} onClick={() => { setF([...list]); setMsg(''); }}>{label}</button>
        ))}
      </div>
      <p className="field-note">"Example field" is the same common field for every batter, shown as a starting point.</p>
      <div className="field-grid">
        <svg viewBox="0 -14 440 454" className="field-svg" role="group" aria-label="Field: tap a boundary region to place or remove a fielder">
          <circle cx={c} cy={c} r={212} fill="#eef1f6" />
          {B.map((b) => {
            const on = f.includes(b.name);
            const o1 = fp(b.from, 212), o2 = fp(b.to, 212), i1 = fp(b.from, 104), i2 = fp(b.to, 104);
            const k = Math.round((5 * b.boundary_runs) / maxB);
            return (
              <g key={b.name} className="band" role="button" tabIndex={0} aria-pressed={on}
                aria-label={`${b.name} ${b.boundary_runs} boundary runs. ${on ? 'Fielder placed; press to remove.' : 'Press to place a fielder.'}`}
                onClick={() => toggle(b.name)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(b.name); } }}>
                <path className="band-hit" d={`M${o1[0].toFixed(1)},${o1[1].toFixed(1)} A212,212 0 0 1 ${o2[0].toFixed(1)},${o2[1].toFixed(1)} L${i2[0].toFixed(1)},${i2[1].toFixed(1)} A104,104 0 0 0 ${i1[0].toFixed(1)},${i1[1].toFixed(1)} Z`}
                  fill={SEQ[k]} stroke={on ? '#0f1a3c' : '#ffffff'} strokeWidth={on ? 3 : 1.5} />
              </g>
            );
          })}
          {/* labels and fielder markers sit in their own layer above the tappable regions: the controls hold no text of their own */}
          <g aria-hidden="true" pointerEvents="none">
            {B.map((b) => {
              const on = f.includes(b.name), mid = (b.from + b.to) / 2;
              const k = Math.round((5 * b.boundary_runs) / maxB), nm = fp(mid, 188), v = fp(mid, 140), q = fp(mid, 163);
              const side = Math.sin(rad(mid));
              return (
                <g key={b.name}>
                  <text x={nm[0]} y={nm[1] + 4} textAnchor={side < -0.6 ? 'start' : side > 0.6 ? 'end' : 'middle'} className="band-name" fill={k === 5 ? '#ffffff' : '#0f1a3c'}>{b.name}</text>
                  <text x={v[0]} y={v[1] + 6} textAnchor="middle" className="band-val num" fill={k === 5 ? '#ffffff' : '#0f1a3c'}>{b.boundary_runs}</text>
                  {on && <g><circle cx={q[0]} cy={q[1]} r={8} fill="#0f1a3c" stroke="#ffffff" strokeWidth={1.5} /><circle cx={q[0]} cy={q[1]} r={3} fill="#ffffff" /></g>}
                </g>
              );
            })}
          </g>
          <circle cx={c} cy={c} r={100} fill="#f6f7fb" stroke="#d6dbe8" />
          <rect x="213" y="196" width="14" height="48" rx="2" fill="#f1f1f4" stroke="#8b92aa" />
          <text x={c} y="282" textAnchor="middle" className="wheel-axis">Inner circle</text>
          <text x={c} y="-2" textAnchor="middle" className="wheel-axis">Bowler's end ↑</text>
        </svg>
        <div className="field-panel">
          <div className="slots" aria-hidden="true">{Array.from({ length: LIMIT }, (_, i) => <span key={i} className={i < f.length ? 'on' : ''} />)}</div>
          <span className="field-count">{f.length} of {LIMIT} boundary fielders placed</span>
          <div className="field-big">
            <span className="fp-l">Share of his boundary runs hit towards your fielders</span>
            <span className="num fp-pct">{pct}%</span>
            <span className="fp-sub">{n(covRuns)} of {n(T.boundary_runs)} runs</span>
            <div className="fp-track" aria-hidden="true"><div style={{ width: `${pct}%` }} /></div>
          </div>
          <dl className="fp-rows">
            <div><dt className="num">{sum('fours')} / {T.fours}</dt><dd>fours into your fielders: these can be cut off</dd></div>
            <div><dt className="num">{sum('sixes')} / {T.sixes}</dt><dd>sixes over them: a fielder only helps with a catch</dd></div>
            <div><dt className="num">{sum('caught')} / {T.caught}</dt><dd>of his catches out went to these regions</dd></div>
          </dl>
          <div className={`fp-gap${isBest ? ' best' : ''}`} aria-live="polite"><b>{gapHead}</b><span>{gapText}</span></div>
          {msg && <p className="fp-msg" role="status">{msg}</p>}
          <Callouts dive={dive} slot="field" />
        </div>
      </div>
    </div>
  );
}
