import type { Meta, Pick, Picks as PicksData, Role } from '../data';
import { honestNote, ROLE_PLURAL } from '../copy';
import { signed } from '../format';
import { PlayerCard } from './PlayerCard';
import { ChangesNote } from './ChangesNote';
import { Section } from './SectionBand';

const ROLES: Role[] = ['batter', 'bowler'];

function RoleColumn({ role, data, meta, gapDomain, ratingMax, onOpen }: {
  role: Role; data: PicksData; meta: Meta; gapDomain: number; ratingMax: number; onOpen: (p: Pick) => void;
}) {
  const R = data.roles[role];
  const top = R.players.filter((p) => p.group === 'top 10');
  const alts = R.players.filter((p) => p.group === 'alternate');
  const v = meta.verdicts[role]['top 10'];
  const lg = R.league;
  return (
    <div className={`role-col role-${role}`}>
      <div className="role-head">
        <h3 id={`picks-${role}`}>Top 10 {ROLE_PLURAL[role].toLowerCase()}</h3>
        <p className="role-sum">
          Under pressure: {v['better under pressure'].n} better, {v['worse under pressure'].n} worse, {v['no clear difference'].n} no clear difference.
          {' '}Average eligible {role}: gap {signed(lg.gap)} ({lg.verdict}).
        </p>
      </div>
      <ol className="card-list">
        {top.map((p) => (
          <li key={p.player_id}>
            <PlayerCard p={p} role={role} field={R.field} gapDomain={gapDomain} ratingMax={ratingMax} dataRange={meta.data_range} onOpen={onOpen} />
          </li>
        ))}
      </ol>
      <h4 className="alt-head">Alternate</h4>
      {alts.map((p) => (
        <PlayerCard key={p.player_id} p={p} role={role} field={R.field} gapDomain={gapDomain} ratingMax={ratingMax} dataRange={meta.data_range} onOpen={onOpen} />
      ))}
    </div>
  );
}

function HonestNote({ meta }: { meta: Meta }) {
  const h = honestNote(meta);
  return (
    <aside className="honest" aria-label="How to read the verdicts">
      <h3>How to read the verdicts</h3>
      <p>{h.chance}</p>
      <ul>{h.lines.map((l) => <li key={l.role}>{l.clear}.</li>)}</ul>
      <p>That is close to luck. And pressure gaps did not carry forward: the gap in 2022–24 does not predict the gap in 2025–26 for the floor-50 pool.</p>
      <ul className="honest-num">{h.lines.map((l) => <li key={l.role}>{l.forward}</li>)}</ul>
      <p className="honest-end">So a verdict describes the past, not a promise.</p>
    </aside>
  );
}

export function PicksSection({ data, meta, onOpen }: { data: PicksData; meta: Meta; onOpen: (p: Pick) => void }) {
  // one gap scale per page, so markers compare across cards; one rating scale per page for the same reason
  const all = ROLES.flatMap((r) => data.roles[r].field.map((f) => Math.abs(f.gap)));
  const gapDomain = Math.max(...all) * 1.05;
  const ratingMax = Math.max(...ROLES.flatMap((r) => data.roles[r].players.map((p) => p.range_hi))) * 1.05;
  return (
    <Section id="picks" title="The picks"
      lede={<>Ranked by cautious rating ({meta.rating_unit}); each card adds normal vs pressure ({data.unit}).</>}>
      <ChangesNote meta={meta} />
      <p className="badge-note">Each badge is drawn from the player's own data. Batters show where they score; bowlers show which overs they bowl.</p>
      <div className="picks-grid">
        <HonestNote meta={meta} />
        <RoleColumn role="batter" data={data} meta={meta} gapDomain={gapDomain} ratingMax={ratingMax} onOpen={onOpen} />
        <RoleColumn role="bowler" data={data} meta={meta} gapDomain={gapDomain} ratingMax={ratingMax} onOpen={onOpen} />
      </div>
    </Section>
  );
}
