import { useEffect, useState } from 'react';
import { MotionConfig } from 'framer-motion';
import type { Meta, Pick, Picks } from './data';
import { useJSON } from './data';
import { Header, type NavItem } from './components/Header';
import { PicksSection } from './components/Picks';
import { Hero } from './components/Hero';
import type { Sim } from './sim/types';
import { Footer } from './components/Footer';
import { DeepDive } from './dive/DeepDive';
import { HeadToHead } from './sections/HeadToHead';
import { SpecialistsSection } from './sections/Specialists';
import { RulesSection } from './sections/Rules';
import { HowItWorks } from './sections/HowItWorks';
import type { Matchups, Sandbox, Specialists } from './sections/types';

// Sections are added here phase by phase; the nav follows this list.
const NAV: NavItem[] = [{ id: 'nineteenth', label: 'The 19th over' }, { id: 'picks', label: 'The picks' }, { id: 'dive', label: 'Deep dive' },
  { id: 'h2h', label: 'Head-to-head' }, { id: 'specialists', label: 'Specialists' }, { id: 'rules', label: 'Change our rules' }, { id: 'how', label: 'How it works' }];
const fromHash = () => { const m = /^#player-(\d+)$/.exec(window.location.hash); return m ? Number(m[1]) : null; };

export default function App() {
  const meta = useJSON<Meta>('meta.json');
  const picks = useJSON<Picks>('picks.json');
  const sim = useJSON<Sim>('sim.json');
  const mu = useJSON<Matchups>('matchups.json');
  const sp = useJSON<Specialists>('specialists.json');
  const sb = useJSON<Sandbox>('sandbox.json');
  const err = meta.error || picks.error || sim.error || mu.error || sp.error || sb.error;
  const [selected, setSelected] = useState<number | null>(fromHash);
  useEffect(() => {
    const on = () => { const id = fromHash(); if (id != null) setSelected(id); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const select = (id: number) => { setSelected(id); window.history.replaceState(null, '', `#player-${id}`); };
  // a card opens that player's deep dive
  const open = (p: Pick) => { select(p.player_id); requestAnimationFrame(() => document.getElementById('dive')?.scrollIntoView({ block: 'start' })); };
  if (err) return <p className="load-msg" role="alert">Could not load the data: {err}</p>;
  if (!meta.data || !picks.data || !sim.data || !mu.data || !sp.data || !sb.data) return <p className="load-msg">Loading…</p>;
  return (
    <MotionConfig reducedMotion="user">
      <a className="skip" href="#main">Skip to content</a>
      <Header meta={meta.data} nav={NAV} />
      <main id="main">
        <Hero sim={sim.data} />
        <PicksSection data={picks.data} meta={meta.data} onOpen={open} />
        <DeepDive picks={picks.data} meta={meta.data} onSelect={select}
          selected={selected ?? picks.data.roles.batter.players[0].player_id} />
        <HeadToHead mu={mu.data} picks={picks.data} />
        <SpecialistsSection sp={sp.data} picks={picks.data} />
        <RulesSection sb={sb.data} />
        <HowItWorks meta={meta.data} />
      </main>
      <Footer meta={meta.data} />
    </MotionConfig>
  );
}
