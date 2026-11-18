import type { Meta, Role } from '../data';
import { FORWARD_POOL } from '../data';
import { Section } from '../components/SectionBand';
import { range, signed } from '../format';

const ROLES: Role[] = ['batter', 'bowler'];

/** Plain-English method notes. Every number comes from meta.json. */
export function HowItWorks({ meta }: { meta: Meta }) {
  const R = meta.rules, E = meta.eligibility;
  const pr = meta.changes.pressure_rule;
  const fw = (r: Role) => meta.forward_test[r][FORWARD_POOL];
  const noCarry = !meta.forward_confirmed.batter && !meta.forward_confirmed.bowler;
  return (
    <Section id="how" title="How it works" lede="What the numbers mean, what the rules are, and what the data cannot tell us.">
      <div className="how">
        <article className="card how-item">
          <h3>The question we ask</h3>
          <p>How does a player do when the match is tight, compared with his own usual level? For every player we measure pressure performance and set it beside his usual performance. The gap is pressure minus usual: a plus means he does better when it matters, a minus means worse. It is a description of {meta.data_range}, not a forecast.</p>
        </article>
        <article className="card how-item">
          <h3>A ball's importance</h3>
          <p>Every ball gets an importance score: how much the match's win chances can swing on it, compared with an average ball (importance 1). A dot ball early in a big chase barely matters; the same dot in the 19th over of a tight chase can matter many times over. Balls with importance {meta.li_pressure} or more are "high-pressure balls".</p>
          <p className="how-small">{meta.how_it_works_lines.no_importance}</p>
        </article>
        <article className="card how-item">
          <h3>The cautious rating</h3>
          <p>Each player's rating is in {meta.rating_unit}: the win chance his side gains per ball, against a typical player in the same season, competition, innings and phase, corrected for opponent strength. Short records are pulled towards average. We rank on the low end of each player's likely range (the cautious rating), so a short lucky run is not rewarded.</p>
        </article>
        <article className="card how-item">
          <h3>Who can be picked</h3>
          <ul>
            <li>At least {R.own_share_min_pct}% of the rating must come from his own matches, not the pull towards average.</li>
            <li>At least {R.matches_2025_26_min} matches in 2025–26.</li>
            <li>At least {R.high_pressure_balls_min} high-pressure balls in {meta.data_range} (the exposure floor, added in v2).</li>
            <li>A pressure value above average: his value on high-pressure balls must be above zero (added in v3).</li>
          </ul>
          <p>{E.bat.eligible} of {E.bat.ranked} batters and {E.bowl.eligible} of {E.bowl.ranked} bowlers pass all four. {E.bat.eligible_floor50_before_pressure_rule} batters and {E.bowl.eligible_floor50_before_pressure_rule} bowlers passed the first three.</p>
        </article>
        <article className="card how-item how-wide">
          <h3>The pressure rule, with and without it</h3>
          <p>The last rule was added after the fact. On {pr.decided} we reviewed the v2 list against the problem statement and decided that a pressure pick should be above average under pressure. It is a judgement made with the results in view, in the same way as the {R.high_pressure_balls_min}-ball floor, not a pre-set test. It applies to every player alike; the ratings and the order are unchanged. Here are the lists with and without it.</p>
          <div className="table-scroll">
            <table className="sens">
              <caption>Top 10 and the next two ranks, with and without the pressure rule</caption>
              <thead><tr><th scope="col">List</th>{ROLES.map((r) => <th scope="col" key={r}>{r === 'batter' ? 'Batters' : 'Bowlers'}</th>)}</tr></thead>
              <tbody>
                {([['without_rule', 'Without the rule (v2)'], ['with_rule', 'With the rule (ours)']] as const).map(([k, label]) => (
                  <tr key={k} className={k === 'with_rule' ? 'is-ours' : undefined}>
                    <th scope="row">{label}</th>
                    {ROLES.map((r) => (
                      <td key={r}>
                        <span className="sens-d">{pr.lists[r][k].top10.join(', ')}</span>
                        <span className="sens-n">{k === 'with_rule' ? 'Alternates' : 'Next two'}: {pr.lists[r][k].alternates.join(', ')}</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
        <article className="card how-item">
          <h3>Normal against pressure</h3>
          <p>For every player we split his balls into normal (importance under {meta.li_pressure}) and pressure (importance {meta.li_pressure} or more), in {meta.unit}. The gap is pressure minus normal, with an 80% range from resampling matches. With 80% ranges, about 1 in {Math.round(100 / meta.chance_share_pct)} players lands clearly above or below zero by luck alone.</p>
          <p>A player can be above average under pressure and still have a negative gap, if his usual level is higher. That is why each card says both numbers in words.</p>
        </article>
        <article className="card how-item">
          <h3>Pressure gaps did not carry forward</h3>
          <p>We asked whether a player's gap in 2022–24 predicts his gap in 2025–26, for floor-50 players with at least 20 pressure balls in each period:</p>
          <ul className="how-num">
            {ROLES.map((r) => (
              <li key={r}>{r === 'batter' ? 'Batters' : 'Bowlers'}: {fw(r).players} players, rank correlation {signed(fw(r).rho)} (95% range {range(fw(r).lo95, fw(r).hi95)})</li>
            ))}
          </ul>
          <p>{noCarry ? `Both ranges include zero, so the gap did not carry forward. It describes ${meta.data_range} and is not a forecast. That is why the picks rank on the overall rating, not the gap.` : 'At least one role shows a gap that carries forward; see the method log.'}</p>
          <p className="how-small">We did not repeat this test on the players who pass the pressure rule, because that rule picks players using the same pressure results.</p>
        </article>
        <article className="card how-item">
          <h3>What we cannot measure</h3>
          <ul>
            <li>There is no line or length in the data, so there is no pitch map. Delivery types use the label recorded on each ball.</li>
            <li>There are no hit coordinates: field regions are inferred from the stroke recorded.</li>
            <li>The simulator keeps the batting side average and does not track who is on strike.</li>
          </ul>
        </article>
        <article className="card how-item how-ver">
          <h3>Checked</h3>
          <p>The v3 package is rebuilt from the frozen v2 outputs, twice, and the two runs give identical files. It is compared with a frozen reference: <b className="num">{meta.verification.passed} of {meta.verification.total}</b> checks pass. Every number on this page is read from those files.</p>
        </article>
      </div>
    </Section>
  );
}
