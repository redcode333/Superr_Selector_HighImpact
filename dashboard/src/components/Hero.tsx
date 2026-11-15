import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import type { Sim } from '../sim/types';
import { defendPct, nextBallSwing, outcome, playOne, RUN, type Ball, type Played } from '../sim/engine';

// Ported from _handoff/design/Main.dc.html. Every number comes from sim.json and the engine checked by check_data.py.

const LBL = ['•', '1', '2', '3', '4', '6', 'W', 'wd', 'nb'];
const TIP = ['Dot ball', '1 run', '2 runs', '3 runs', 'Four', 'Six', 'Wicket', 'Wide: 1 run, bowled again', 'No-ball: 1 run, bowled again'];
// colour meaning: blue = bowler on top (dot, wicket), pink = batter on top (runs, boundaries)
const BALL_CLASS = ['b-dot', 'b-run', 'b-run', 'b-run', 'b-bnd', 'b-bnd', 'b-wkt', 'b-extra', 'b-extra'];
const STEP_MS = 650;
const SIM_N = 1000;

interface SimsState { def: number; tie: number; marks: number[]; who: number; need: number; wk: number }

const fmt1 = (x: number) => x.toFixed(1);
const signedPts = (d: number) => `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(1)}`;

export function Hero({ sim }: { sim: Sim }) {
  const rm = useReducedMotion() ?? false;
  const L = sim.league;
  const S = sim.situations;
  const topBowler = useMemo(() => sim.bowlers.find((b) => b.group === 'top 10' && b.rank === Math.min(...sim.bowlers.filter((x) => x.group === 'top 10').map((x) => x.rank)))!, [sim]);
  const altBowler = sim.bowlers.find((b) => b.group === 'alternate') ?? sim.bowlers[sim.bowlers.length - 1];
  const byId = useMemo(() => new Map(sim.bowlers.map((b) => [b.player_id, b])), [sim]);

  const [need, setNeed] = useState(S[0].need);
  const [wk, setWk] = useState(S[0].wk);
  const [sel, setSel] = useState(topBowler.player_id);
  const [balls, setBalls] = useState<Ball[]>([]);
  const [playing, setPlaying] = useState(false);
  const [res, setRes] = useState<Played | null>(null);
  const [sims, setSims] = useState<SimsState | null>(null);
  const [simShown, setSimShown] = useState(0);
  const [demo, setDemo] = useState(-1);

  // state mirror for timer callbacks (the walkthrough calls bowl/runSims later)
  const st = useRef({ need, wk, sel, playing });
  st.current = { need, wk, sel, playing };
  const timers = useRef<number[]>([]);
  const clearTimers = useCallback(() => { timers.current.forEach((t) => window.clearTimeout(t)); timers.current = []; }, []);
  const later = (fn: () => void, ms: number) => { timers.current.push(window.setTimeout(fn, ms)); };
  useEffect(() => clearTimers, [clearTimers]);

  // self-check: the exact figures shown equal the numbers check_data.py verified in sim.json
  useEffect(() => {
    const bad: string[] = [];
    for (const s of S) {
      const k = `${s.need}_${s.wk}`;
      if (fmt1(defendPct(L, L, s.wk, s.need)) !== fmt1(sim.base_defend[k])) bad.push(`average ${k}`);
      for (const b of sim.bowlers) if (fmt1(defendPct(b.p, L, s.wk, s.need)) !== fmt1(b.defend[k])) bad.push(`${b.name} ${k}`);
    }
    if (bad.length) console.error('Simulator figures differ from sim.json:', bad.join(', '));
  }, [sim, S, L]);

  const selB = byId.get(sel)!;
  const selPct = defendPct(selB.p, L, wk, need);
  const base = defendPct(L, L, wk, need);
  const pressure = nextBallSwing(L, wk, need, sim.mean_abs_wpa);

  // ---------------- actions
  const bowl = useCallback(() => {
    if (st.current.playing) return;
    const s = st.current;
    const g = playOne(byId.get(s.sel)!.p, L, s.need, s.wk);
    if (rm) { setBalls(g.out); setRes(g); setPlaying(false); return; }
    setBalls([]); setRes(null); setPlaying(true);
    g.out.forEach((_, k) => later(() => setBalls(g.out.slice(0, k + 1)), STEP_MS * (k + 1)));
    later(() => { setPlaying(false); setRes(g); }, STEP_MS * g.out.length + 350);
  }, [byId, L, rm]);

  const runSims = useCallback(() => {
    const s = st.current, p = byId.get(s.sel)!.p;
    let def = 0, tie = 0;
    for (let k = 0; k < SIM_N; k++) { const m = outcome(playOne(p, L, s.need, s.wk)); if (m === 2) def++; if (m === 1) tie++; }
    const nd = Math.round(def / 10), nt = Math.round(tie / 10), marks: number[] = [];
    for (let k = 0; k < 100; k++) marks.push(k < nd ? 2 : k < nd + nt ? 1 : 0);
    setSims({ def, tie, marks, who: s.sel, need: s.need, wk: s.wk });
    if (rm) { setSimShown(100); return; }
    setSimShown(0);
    for (let j = 1; j <= 10; j++) later(() => setSimShown(j * 10), 110 * j);
  }, [byId, L, rm]);

  const resetOver = () => { setBalls([]); setRes(null); };
  const setSit = (n: number, w: number) => {
    setNeed(Math.max(sim.need_range[0], Math.min(sim.need_range[1], n)));
    setWk(Math.max(sim.wk_range[0], Math.min(sim.wk_range[1], w)));
    resetOver();
  };
  const pick = (id: number) => { setSel(id); resetOver(); };

  // ---------------- walkthrough (40 s, 7 steps; timings as in Main.dc.html)
  const CAP = [
    `Step 1: set the chase. ${S[0].need} needed off the last 12 balls, ${S[0].wk} wickets in hand.`,
    'Step 2: every bowler is re-rated for that exact situation. Watch the list reorder when the target changes.',
    `Step 3: pick your bowler. ${topBowler.name} goes on for the 19th.`,
    'Step 4: bowl it. Each ball is drawn from his real death-over record.',
    'Step 5: the 20th over goes to an average death bowler, so only the 19th changes.',
    'Step 6: one match is luck. Play it 1,000 times and count how often you defend.',
    'Your turn: change the target, pick any bowler and bowl.',
  ];
  const HL = ['sit', 'list', 'list', 'bowl', 'bowl', 'sims', 'sit'];
  const stopDemo = useCallback(() => { clearTimers(); setDemo(-1); setPlaying(false); }, [clearTimers]);
  const startDemo = () => {
    clearTimers();
    setDemo(0); setNeed(S[2].need); setWk(S[0].wk); setSel(altBowler.player_id);
    setBalls([]); setRes(null); setSims(null); setSimShown(0); setPlaying(false);
    later(() => setNeed(S[0].need), 1600);
    later(() => setDemo(1), 4500);
    later(() => setNeed(S[1].need), 6000);
    later(() => setNeed(S[0].need), 8500);
    later(() => setDemo(2), 11000);
    later(() => setSel(topBowler.player_id), 12500);
    later(() => { setDemo(3); bowl(); }, 15500);
    later(() => setDemo(4), 19500);
    later(() => { setDemo(5); runSims(); }, 26000);
    later(() => setDemo(6), 33000);
    later(() => setDemo(-1), 40000);
  };
  // any click on a control during the walkthrough stops it first
  const act = <A extends unknown[]>(fn: (...a: A) => void) => (...a: A) => { if (demo >= 0) stopDemo(); fn(...a); };

  // leaving the section cancels the walkthrough
  const sectionRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = sectionRef.current;
    if (!el || demo < 0) return;
    const io = new IntersectionObserver(([e]) => { if (!e.isIntersecting) stopDemo(); }, { threshold: 0 });
    io.observe(el);
    return () => io.disconnect();
  }, [demo >= 0, stopDemo]);

  const hl = demo >= 0 ? HL[demo] : '';
  const hlClass = (key: string) => (hl === key ? (rm ? ' hl-static' : ' hl-pulse') : '');

  // ---------------- live scoreboard from the balls shown so far
  let lr = need, lw = wk, ll = 0;
  balls.forEach((b) => { if (b.i <= 5) { lr -= RUN[b.i]; ll++; } else if (b.i === 6) { lw--; ll++; } else lr -= 1; });
  let statusHead = 'Ready', statusText = 'Press "Bowl the 19th" to play it out', statusTone = '';
  if (playing) { statusHead = 'Bowling'; statusText = lr <= 0 ? 'They are over the line' : `Needs ${lr} off ${12 - ll}`; }
  if (res) {
    statusHead = 'Result of this one match';
    const spare = 12 - res.legal;
    if (res.r <= 0) { statusText = spare ? `Chased down with ${spare} ball${spare === 1 ? '' : 's'} to spare` : 'Chased down off the last ball'; statusTone = 'tone-bat'; }
    else if (res.r === 1) statusText = 'Scores level: a tie';
    else { statusText = `Defended: won by ${res.r - 1} run${res.r - 1 === 1 ? '' : 's'}${res.w === 0 ? ' (all out)' : ''}`; statusTone = 'tone-bowl'; }
  }
  const overRow = (n: 19 | 20, who: string) => {
    const mine = balls.filter((b) => b.by === n);
    const legalMine = mine.filter((b) => b.i <= 6).length;
    return { n, who, mine, pads: Math.max(0, 6 - legalMine) };
  };
  const rows2 = [overRow(19, selB.name), overRow(20, 'Average death bowler')];

  // ---------------- sims text
  let simHead = `${SIM_N.toLocaleString('en-IN')} matches`;
  let simText = 'Each dot is 10 matches: blue defended, pink chased down, light grey tied.';
  if (sims) {
    const fresh = sims.who === sel && sims.need === need && sims.wk === wk;
    simHead = `Defended ${sims.def} times`;
    simText = `in ${SIM_N.toLocaleString('en-IN')} random runs with ${byId.get(sims.who)!.name}, plus ${sims.tie} ties. `
      + (fresh ? `The exact figure is ${fmt1(selPct)}%: the random runs land close to it, never exactly on it.` : 'You have changed the setting since: press again to replay.');
  }

  // ---------------- bowler list
  const list = sim.bowlers.map((b) => ({ b, pct: defendPct(b.p, L, wk, need) })).sort((a, c) => c.pct - a.pct);
  const presets = S.map((s) => ({ ...s, label: `${s.need} off 12 · ${s.wk} wkts`, on: s.need === need && s.wk === wk }));

  const enter = rm ? {} : { initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 } };
  const ent = (i: number) => (rm ? {} : { ...enter, transition: { duration: 0.45, delay: 0.08 + i * 0.09, ease: [0.2, 0.8, 0.2, 1] } });

  return (
    <section className="hero-section" id="nineteenth" aria-labelledby="hero-h" ref={sectionRef}>
      <div className="band-navy hero">
        <div className="wrap hero-inner">
          <motion.div className="demo-bar" {...ent(0)}>
            {demo < 0 ? (
              <>
                <button type="button" className="btn btn-light" onClick={startDemo}>
                  <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="#e4007c" /></svg>
                  Watch how it works (40 seconds)
                </button>
                <span className="demo-hint">A guided run: it sets the chase, picks a bowler, bowls the over and plays it 1,000 times. You can stop it at any point.</span>
              </>
            ) : (
              <>
                <button type="button" className="btn btn-pink" onClick={stopDemo}>Stop</button>
                <div className="demo-cap">
                  <AnimatePresence mode="wait" initial={false}>
                    <motion.span key={demo} className="demo-caption" role="status" aria-live="polite"
                      initial={rm ? false : { opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={rm ? undefined : { opacity: 0 }} transition={{ duration: 0.3 }}>
                      {CAP[demo]}
                    </motion.span>
                  </AnimatePresence>
                  <div className="demo-steps" aria-hidden="true">
                    {CAP.map((_, k) => <span key={k} className={k <= demo ? 'on' : ''} />)}
                  </div>
                </div>
                <span className="demo-n num">Step {demo + 1} of {CAP.length}</span>
              </>
            )}
          </motion.div>

          <div className="hero-grid">
            <div className="hero-left">
              <motion.div className="hero-title" {...ent(1)}>
                <span className="hero-kicker">Decision tool · every ball drawn from real death overs, 2022–26</span>
                <h2 id="hero-h" className="num">Who bowls the 19th over?</h2>
              </motion.div>
              <motion.div className="scoreboard" role="status" aria-live="polite" {...ent(2)}>
                <div className="sb-box"><span className="sb-l">Chasing side needs</span><span className="sb-v num sb-need">{Math.max(0, lr)}</span></div>
                <div className="sb-box"><span className="sb-l">Balls left</span><span className="sb-v num">{12 - ll}</span></div>
                <div className="sb-box"><span className="sb-l">Wickets in hand</span><span className="sb-v num">{lw}</span></div>
                <div className="sb-box sb-status">
                  <span className="sb-l">{statusHead}</span>
                  <span className={`sb-text ${statusTone}`}>{statusText}</span>
                </div>
              </motion.div>
              <motion.div className="overs" {...ent(3)}>
                {rows2.map((o) => (
                  <div className="over-row" key={o.n}>
                    <div className="over-who"><span className="over-n">{o.n}th over</span><span className="over-by">{o.who}</span></div>
                    <ol className="over-balls" aria-label={`${o.n}th over`}>
                      {o.mine.map((b, k) => (
                        <motion.li key={`${o.n}-${k}`} className={`ball ${BALL_CLASS[b.i]}`} title={TIP[b.i]} aria-label={TIP[b.i]}
                          initial={rm ? false : { scale: 0.2, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                          transition={{ type: 'spring', stiffness: 520, damping: 22 }}>
                          {LBL[b.i]}
                        </motion.li>
                      ))}
                      {Array.from({ length: o.pads }, (_, k) => <li key={`p${k}`} className="ball b-pad" aria-label="To come" />)}
                    </ol>
                  </div>
                ))}
              </motion.div>
              <ul className="ball-legend" aria-hidden="true">
                <li><span className="ball-key b-dot" />Dot ball</li>
                <li><span className="ball-key b-run" />1, 2 or 3</li>
                <li><span className="ball-key b-bnd" />Four or six</li>
                <li><span className="ball-key b-wkt" />Wicket</li>
                <li><span className="ball-key b-extra" />Wide or no-ball (bowled again)</li>
              </ul>
            </div>

            <div className="hero-right">
              <motion.div className={`panel panel-bowl${hlClass('bowl')}`} {...ent(2)}>
                <div className="pb-head">
                  <svg className={playing && !rm ? 'spin' : ''} width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">
                    <circle cx="17" cy="17" r="15" fill="#c4006b" />
                    <path d="M9 6c4 7 4 15 0 22M25 6c-4 7-4 15 0 22" fill="none" stroke="#FFFFFF" strokeWidth="1.6" strokeDasharray="2 2.4" />
                  </svg>
                  <div><span className="pb-l">19th over bowled by</span><span className="pb-name">{selB.name}</span></div>
                </div>
                <div className="pb-pct"><span className="num pb-big">{fmt1(selPct)}%</span><span className="pb-l">chance your side defends this</span></div>
                <p className="pb-base">With an average death bowler it is <b>{fmt1(base)}%</b>. {selPct >= base ? `He adds ${fmt1(selPct - base)} points.` : `He costs ${fmt1(base - selPct)} points.`}</p>
                <div className="pb-btns">
                  <button type="button" className="btn btn-pink" onClick={act(bowl)} aria-disabled={playing}>{playing ? 'Bowling…' : res ? 'Bowl it again' : 'Bowl the 19th'}</button>
                  <button type="button" className="btn btn-ghost" onClick={act(runSims)}>Play it 1,000 times</button>
                </div>
              </motion.div>
              <motion.div className={`panel panel-sims${hlClass('sims')}`} {...ent(3)}>
                <div className="dots" aria-hidden="true">
                  {Array.from({ length: 100 }, (_, k) => {
                    const m = sims && k < simShown ? sims.marks[k] : -1;
                    return <span key={k} className={`dot ${m === 2 ? 'd-def' : m === 1 ? 'd-tie' : m === 0 ? 'd-chase' : ''}`} />;
                  })}
                </div>
                <div className="sims-text" aria-live="polite">
                  <span className="num sims-head">{simHead}</span>
                  <span className="sims-body">{simText}</span>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </div>

      <div className="wrap hero-below">
        <section className={`card sit${hlClass('sit')}`} aria-labelledby="sit-h">
          <h3 id="sit-h">Set the chase</h3>
          <div className="stepper-row">
            <span>Runs needed off the last 12 balls</span>
            <div className="stepper">
              <button type="button" className="step" onClick={act(() => setSit(need - 1, wk))} aria-label="Fewer runs needed" disabled={need <= sim.need_range[0]}>−</button>
              <span className="num step-v" aria-live="polite">{need}</span>
              <button type="button" className="step" onClick={act(() => setSit(need + 1, wk))} aria-label="More runs needed" disabled={need >= sim.need_range[1]}>+</button>
            </div>
          </div>
          <div className="stepper-row">
            <span>Wickets in hand</span>
            <div className="stepper">
              <button type="button" className="step" onClick={act(() => setSit(need, wk - 1))} aria-label="Fewer wickets in hand" disabled={wk <= sim.wk_range[0]}>−</button>
              <span className="num step-v" aria-live="polite">{wk}</span>
              <button type="button" className="step" onClick={act(() => setSit(need, wk + 1))} aria-label="More wickets in hand" disabled={wk >= sim.wk_range[1]}>+</button>
            </div>
          </div>
          <div className="chips">
            {presets.map((p) => (
              <button type="button" key={p.label} className={`chip-btn${p.on ? ' on' : ''}`} aria-pressed={p.on} onClick={act(() => setSit(p.need, p.wk))}>{p.label}</button>
            ))}
          </div>
          <hr />
          <div className="swing">
            <span className="swing-l">How much the next ball can swing the match</span>
            <span className="num swing-v">{fmt1(pressure)}× <span className="swing-u">an average ball</span></span>
            <div className="swing-track" aria-hidden="true"><motion.div className="swing-bar" animate={{ width: `${Math.min(100, pressure * 10)}%` }} initial={false} transition={{ duration: rm ? 0 : 0.6 }} /></div>
            <span className="swing-note">Average ball across the whole data = 1×. Scale shown to 10×.</span>
          </div>
        </section>

        <section className={`card blist${hlClass('list')}`} aria-labelledby="blist-h">
          <div className="blist-head">
            <h3 id="blist-h">Chance of defending {need} off 12, by bowler</h3>
            <span>Tap a bowler to put him on · dashed line: average death bowler</span>
          </div>
          <ol className="brows">
            {list.map(({ b, pct }, k) => {
              const on = b.player_id === sel, few = b.death_balls < sim.few_death_balls, d = pct - base;
              return (
                <motion.li key={b.player_id} layout={!rm} transition={{ type: 'spring', stiffness: 380, damping: 34 }}>
                  <button type="button" className={`brow${on ? ' on' : ''}`} aria-pressed={on} onClick={act(() => pick(b.player_id))}>
                    <span className="brow-pos num">{k + 1}</span>
                    <span className="brow-id">
                      <span className="brow-name">{b.name}{on && <span className="brow-on">On</span>}</span>
                      <span className="brow-tag">{b.group === 'top 10' ? `Our rank ${b.rank}` : 'Alternate'} · {b.death_balls} death balls{few ? ' · few, pulled to average' : ''}</span>
                    </span>
                    <span className="brow-mid">
                      <span className="brow-track" aria-hidden="true">
                        <motion.span className={`brow-bar${on ? ' on' : few ? ' few' : ''}`} initial={false} animate={{ width: `${pct}%` }} transition={{ duration: rm ? 0 : 0.7, ease: [0.2, 0.8, 0.2, 1] }} />
                        <svg className="brow-base" style={{ left: `${base}%` }} width="2" height="20" aria-hidden="true"><line x1="1" x2="1" y1="0" y2="20" stroke="#0f1a3c" strokeWidth="2" strokeDasharray="3 2" /></svg>
                      </span>
                      <span className="brow-stats">
                        <span>Death economy <b>{b.death_econ.toFixed(2)}</b></span><span>Dots <b>{fmt1(b.dot_pct)}%</b></span>
                        <span>4s and 6s <b>{fmt1(b.boundary_pct)}%</b></span><span>Wicket every <b>{fmt1(b.balls_per_wkt)}</b> balls</span>
                      </span>
                    </span>
                    <span className="brow-pct">
                      <span className="num brow-big">{fmt1(pct)}%</span>
                      <span className={`brow-delta ${d >= 0 ? 'tone-bowl-text' : 'tone-bat-text'}`}>{signedPts(d)} vs average</span>
                    </span>
                  </button>
                </motion.li>
              );
            })}
          </ol>
        </section>
      </div>

      <div className="wrap hero-notes">
        <p>
          How it works: we counted every death-over ball each bowler bowled from 2022 to 2026 (dots, singles, twos, fours, sixes, wickets, wides, no-balls).
          Those counts set the odds of each ball in the 19th over. Short records are pulled towards the average death bowler (as if he had bowled {sim.K} extra
          average balls), so a bowler with few death overs sits near average. The 20th over is bowled by an average death bowler. The percentage is worked out
          exactly over every possible ball; "Bowl the 19th" and "Play it 1,000 times" draw balls at random from the same odds, so single runs vary.
        </p>
        <p>Kept simple on purpose: the batting side is average, and who is on strike is not tracked. A tie counts as half a win.</p>
      </div>
    </section>
  );
}

