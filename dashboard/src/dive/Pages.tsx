import type { FieldGap, Pick } from '../data';
import { PressureStrip } from '../components/PressureStrip';
import { PlayerBadge } from '../components/PlayerCard';
import type { BatterDive, BowlerDive, Ground, Opp } from './types';
import { Callouts, Card, col, f1, f2, inkFor, Kpis, n, r0, TypeBars, Wheel, type Metric } from './common';
import { FieldSetter } from './FieldSetter';
import { Planner } from './Planner';

interface Ctx { pick: Pick; field: FieldGap[]; gapDomain: number; dataRange: string }

function Head({ ctx, role, sub, chips, kpis }: { ctx: Ctx; role: 'batter' | 'bowler'; sub: string; chips: string[]; kpis: [string, React.ReactNode][] }) {
  const p = ctx.pick;
  return (
    <header className="dive-head">
      <div className="dh-id">
        <PlayerBadge p={p} size={140} />
        <div>
          <span className="dh-role">{role === 'batter' ? 'Batter' : 'Bowler'}{p.group === 'alternate' ? ' · alternate' : ''}</span>
          <h3 className="num">{p.name}</h3>
          <p className="dh-sub">{sub}</p>
          <div className="dh-chips">{chips.map((c) => <span key={c}>{c}</span>)}</div>
        </div>
      </div>
      <Kpis items={kpis} />
      <div className="dh-press"><span className="dh-press-l">Normal vs pressure</span>
        <PressureStrip p={p} role={role} field={ctx.field} domain={ctx.gapDomain} dataRange={ctx.dataRange} /></div>
    </header>
  );
}

function OverStrip({ cells, minBalls, label }: { cells: { n: number; v: string; balls: number; edge: number | null; w?: string }[]; minBalls: number; label: string }) {
  return (
    <>
      <ol className="ostrip" aria-label={label}>
        {cells.map((c) => {
          const solid = c.balls >= minBalls && c.edge != null;
          const bg = solid ? col(c.edge!, 1) : 'transparent';
          return (
            <li key={c.n} className={`ocell${solid ? '' : ' thin'}`} style={{ background: bg, color: solid ? inkFor(bg) : undefined }}
              title={`Over ${c.n}: ${c.v}, ${c.balls} balls`}>
              <span className="oc-n">{c.n}</span><span className="oc-v num">{c.v}</span>{c.w !== undefined && <span className="oc-w">{c.w}</span>}
            </li>
          );
        })}
      </ol>
      <div className="ophase" aria-hidden="true"><span>Powerplay · overs 1–6</span><span>Middle · 7–15</span><span>Death · 16–20</span></div>
    </>
  );
}

function GroundList({ grounds, fmt, unit }: { grounds: Ground[]; fmt: (g: Ground) => [string, string]; unit: string }) {
  if (!grounds.length) return <p className="dcard-lede">No ground with enough balls.</p>;
  return (
    <ul className="glist">
      {grounds.map((g) => {
        const [v, par] = fmt(g);
        return (
          <li key={g.ground} className={g.tag === 'worst' ? 'g-worst' : ''}>
            <span className="g-name">{g.ground}<span className="g-tag">{g.tag === 'best' ? 'best against par' : 'worst against par'} · {g.balls} balls</span></span>
            <b className="num">{v} <span className="g-par">{unit} · par {par}</span></b>
          </li>
        );
      })}
    </ul>
  );
}

function OppList({ title, rows }: { title: string; rows: Opp[] }) {
  return (
    <div className="opp">
      <h5>{title}</h5>
      {rows.length ? <ul>{rows.map((o) => <li key={o.player_id}><span>{o.name}</span><span className="num">{o.runs} off {o.balls} · out {o.outs}</span></li>)}</ul>
        : <p className="dcard-lede">Nobody with enough balls.</p>}
    </div>
  );
}

// ------------------------------------------------------------------ batter (TimDavid.dc.html)
export function BatterPage({ d, ctx }: { d: BatterDive; ctx: Ctx }) {
  const metrics: Metric[] = [
    { key: 'runs', label: 'Runs', value: (r) => r.runs, show: (r) => `${r0(r.run_share)}%` },
    { key: 'sr', label: 'Strike rate', value: (r) => r.sr ?? 0, show: (r) => r0(r.sr) },
    { key: 'sixes', label: 'Sixes', value: (r) => r.sixes, show: (r) => String(r.sixes) },
    { key: 'outs', label: 'Dismissals', value: (r) => r.outs, show: (r) => String(r.outs) },
  ];
  const table = [...d.regions].sort((a, b) => b.runs - a.runs);
  const maxRuns = Math.max(1, ...table.map((t) => t.runs));
  const p = ctx.pick;
  return (
    <div className="dive">
      <Head ctx={ctx} role="batter"
        sub={`${d.role_label} · batting rank ${p.rank} · ${d.matches} matches · ${n(d.runs)} runs off ${n(d.balls)} balls`}
        chips={[`Top 10 in ${Math.round(p.pct_top10)}% of re-rankings`, `Death strike rate ${r0(d.death.sr)} · par ${r0(d.death.par_sr)}`]}
        kpis={[['Strike rate', f1(d.sr)], ['Sixes', d.sixes], ['Death balls hit for 4 or 6', `${r0(d.death.boundary_pct)}%`], ['Chasing / setting', `${r0(d.chasing.sr)} / ${r0(d.setting.sr)}`]]} />
      <div className="dive-two">
        <Card title="Where he hits it" flag={d.flags.wheel} lede="Each slice is a region of the field. Size: the measure you pick. Colour: his strike rate there against every batter's.">
          <Wheel dive={d} metrics={metrics} edge={(r) => (r.sr ?? 0) - (r.league_sr ?? 0)} aria={`Wagon wheel of ${p.name}'s runs by field region, inferred from the shot played`} />
          <p className="dcard-foot">Region inferred from the shot recorded; the data has no hit coordinates.</p>
        </Card>
        <Card title="What the wheel says" flag={d.flags.wheel}>
          <Callouts dive={d} slot="wheel" />
          <ul className="rtable">
            {table.map((t) => (
              <li key={t.name}><span>{t.name}</span><span className="rt-track" aria-hidden="true"><span style={{ width: `${(100 * t.runs) / maxRuns}%`, background: col((t.sr ?? 0) - (t.league_sr ?? 0), 60) }} /></span>
                <span className="num">{t.runs}</span><span className="rt-sr">SR {r0(t.sr)}</span></li>
            ))}
          </ul>
        </Card>
      </div>
      <Card title="Try it: set the field for his last overs" flag={d.flags.field}
        lede="After the powerplay you may put five fielders on the boundary. Tap a region to place or remove one. The numbers on the field are the boundary runs he has scored into each region (2022–26).">
        <FieldSetter dive={d} />
      </Card>
      <Card title="Over by over" flag={d.flags.overs} lede={`Strike rate in each over against par for that over · full 20-over innings · outlined: under ${d.overs_min_balls} balls`}>
        <OverStrip label="Strike rate by over" minBalls={d.overs_min_balls}
          cells={d.overs.map((o) => ({ n: o.over, v: o.balls ? r0(o.sr) : '—', balls: o.balls, edge: o.sr == null ? null : (o.sr - o.par_sr) / 60 }))} />
        <Callouts dive={d} slot="overs" />
      </Card>
      <div className="dive-three">
        <Card title="How he gets out" flag={d.flags.dismissals}>
          <TypeBars rows={d.dismissals} noun="dismissals" />
          <Callouts dive={d} slot="dismissals" />
        </Card>
        <Card title="Against each type of bowler" flag={d.flags.styles}>
          <ul className="styles">
            {d.styles.map((s) => {
              const few = s.balls < d.mins.style;
              return (
                <li key={s.style} className={few ? 'few' : ''}>
                  <span className="st-name">{s.style}</span>
                  <span className="st-v num" style={{ color: !few && s.sr != null ? (s.sr >= s.par_sr ? '#c4006b' : '#254aa5') : undefined }}>{r0(s.sr)}</span>
                  <span className="st-meta">par {r0(s.par_sr)} · {s.balls} balls{s.balls_per_out ? ` · out every ${Math.round(s.balls_per_out)}` : ''}{few ? ' · few balls' : ''}</span>
                </li>
              );
            })}
          </ul>
          <Callouts dive={d} slot="styles" />
        </Card>
        <Card title="Grounds" flag={d.flags.grounds} lede={`Grounds with at least ${d.grounds_min_balls} balls: his three best against par and his worst.`}>
          <GroundList grounds={d.grounds} unit="SR" fmt={(g) => [r0(g.sr), r0(g.par_sr)]} />
          <Callouts dive={d} slot="grounds" />
        </Card>
      </div>
      <Card title="Against individual bowlers" flag={d.flags.opponents} lede={`At least ${d.opponents_min_balls} balls faced against each.`}>
        <div className="opps"><OppList title="Lowest strike rate" rows={d.toughest} /><OppList title="Highest strike rate" rows={d.easiest} /></div>
        <Callouts dive={d} slot="opponents" />
      </Card>
      <p className="dive-foot">All counts from the supplied ball-by-ball file, 2022–26, IPL and T20 internationals. Field regions are inferred from the stroke recorded on each ball; defensive shots and leaves are left out. Drawn for a right-hander. Par is every batter on the same kind of ball.</p>
    </div>
  );
}

// ------------------------------------------------------------------ bowler (Bumrah.dc.html)
export function BowlerPage({ d, ctx }: { d: BowlerDive; ctx: Ctx }) {
  const metrics: Metric[] = [
    { key: 'runs', label: 'Runs off him', value: (r) => r.runs, show: (r) => `${r0(r.run_share)}%` },
    { key: 'wk', label: 'His wickets', value: (r) => r.outs, show: (r) => String(r.outs) },
    { key: 'sr', label: 'Strike rate', value: (r) => r.sr ?? 0, show: (r) => r0(r.sr) },
  ];
  const table = [...d.regions].sort((a, b) => b.outs - a.outs || b.runs - a.runs);
  const maxRuns = Math.max(1, ...table.map((t) => t.runs));
  const p = ctx.pick;
  return (
    <div className="dive">
      <Head ctx={ctx} role="bowler"
        sub={`${d.role_label} · bowling rank ${p.rank} · ${d.matches} matches · ${d.wickets} wickets from ${n(d.balls)} balls`}
        chips={[`Top 10 in ${Math.round(p.pct_top10)}% of re-rankings`, `Death economy ${f2(d.death.econ)} · par ${f2(d.death.par_econ)}`]}
        kpis={[['Economy', f2(d.econ)], ['Dot balls', `${r0(d.dot_pct)}%`], ['Balls per wicket', f1(d.balls_per_wkt)], ['Yorkers at the death', `${f1(d.death.yorker_pct)}%`]]} />
      <div className="dive-two">
        <Card title="Where batters try to hit him" flag={d.flags.wheel} lede="Each slice is a region of the field, from the shot the batter played. Colour: batters' strike rate there against him, compared with against every bowler. Blue means he wins.">
          <Wheel dive={d} metrics={metrics} edge={(r) => (r.sr ?? 0) - (r.league_sr ?? 0)} aria={`Field regions where batters hit ${p.name}, inferred from the shot played`} />
          <p className="dcard-foot">Off and leg side are relative to the batter.</p>
        </Card>
        <Card title="What the field says" flag={d.flags.wheel}>
          <Callouts dive={d} slot="wheel" />
          <ul className="rtable">
            {table.map((t) => (
              <li key={t.name}><span>{t.name}</span><span className="rt-track" aria-hidden="true"><span style={{ width: `${(100 * t.runs) / maxRuns}%`, background: col((t.sr ?? 0) - (t.league_sr ?? 0), 60) }} /></span>
                <span className="rt-sr">SR {r0(t.sr)}</span><span className="num">{t.outs} wkts</span></li>
            ))}
          </ul>
        </Card>
      </div>
      <Card title="Over by over" flag={d.flags.overs} lede={`Economy in each over against par for that over · wickets underneath · full 20-over innings · outlined: under ${d.overs_min_balls} balls`}>
        <OverStrip label="Economy by over" minBalls={d.overs_min_balls}
          cells={d.overs.map((o) => ({ n: o.over, v: o.balls ? f1(o.econ) : '—', balls: o.balls, edge: o.econ == null ? null : ((o.econ - o.par_econ) * 100) / 6 / 40,
            w: o.wkts ? `${o.wkts} wkt${o.wkts > 1 ? 's' : ''}` : '' }))} />
        <Callouts dive={d} slot="overs" />
      </Card>
      <Card title="Try it: plan his four overs" flag={d.flags.planner}
        lede="Pick four overs for him. As in a real match, he cannot bowl two overs in a row. Each tile shows what he has conceded in that over and what an average bowler concedes there.">
        <Planner dive={d} />
      </Card>
      <div className="dive-three">
        <Card title="How he takes wickets" flag={d.flags.wicket_types}>
          <TypeBars rows={d.wicket_types} noun="wickets" />
          <Callouts dive={d} slot="wicket_types" />
        </Card>
        <Card title="What he bowls, and what it costs" flag={d.flags.deliveries} lede={`Delivery labels recorded on ${Math.round(d.deliveries_labelled_pct)}% of his balls.`}>
          <ul className="dels">
            {d.deliveries.map((v) => {
              const few = v.balls < d.mins.delivery;
              return (
                <li key={v.type} className={few ? 'few' : ''}>
                  <span className="dl-name">{v.type}</span><span className="dl-use num">{r0(v.usage_pct)}%</span>
                  <span className="dl-econ num" style={{ color: !few && v.econ != null ? (v.econ <= v.par_econ ? '#254aa5' : '#c4006b') : undefined }}>{f2(v.econ)}</span>
                  <span className="dl-meta">par {f2(v.par_econ)} · {v.wkts} wkts · {v.balls} balls{few ? ' · few balls' : ''}</span>
                </li>
              );
            })}
          </ul>
          <Callouts dive={d} slot="deliveries" />
        </Card>
        <Card title="Right- and left-handers" flag={d.flags.hands}>
          <ul className="dels">
            {d.hands.map((h) => (
              <li key={h.hand}><span className="dl-name">{h.hand === 'RHB' ? 'Right-handers' : 'Left-handers'}</span><span className="dl-use" />
                <span className="dl-econ num">{f2(h.econ)}</span><span className="dl-meta">par {f2(h.par_econ)} · {h.wkts} wkts · {h.balls} balls</span></li>
            ))}
          </ul>
          <Callouts dive={d} slot="hands" />
        </Card>
      </div>
      <div className="dive-two">
        <Card title="Grounds" flag={d.flags.grounds} lede={`Grounds with at least ${d.grounds_min_balls} balls: his three best against par and his worst.`}>
          <GroundList grounds={d.grounds} unit="econ" fmt={(g) => [f2(g.econ), f2(g.par_econ)]} />
          <Callouts dive={d} slot="grounds" />
        </Card>
        <Card title="Against individual batters" flag={d.flags.opponents} lede={`At least ${d.opponents_min_balls} balls faced against him.`}>
          <div className="opps"><OppList title="Most dismissals" rows={d.owns} /><OppList title="Highest strike rate" rows={d.troubled_by} /></div>
          <Callouts dive={d} slot="opponents" />
        </Card>
      </div>
      <p className="dive-foot">All counts from the supplied ball-by-ball file, 2022–26, IPL and T20 internationals. Field regions come from the stroke the batter played on each ball; defensive shots and leaves are left out. Delivery types use the ball label recorded; the data has no line or length, so there is no pitch map. Par is every bowler on the same kind of ball.</p>
    </div>
  );
}
