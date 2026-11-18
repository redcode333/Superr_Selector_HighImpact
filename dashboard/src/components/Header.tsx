import type { Meta } from '../data';
import { TAGLINE, TAGLINES } from '../copy';

import { SiteNav, type NavItem } from './SiteNav';
export type { NavItem };

/** Our own typographic monogram: two condensed capitals, the second in pink, on a square rule. Not a crest.
 *  tone "ink" is the version for the light nav bar; decorative hides it from assistive tech (its link carries the label). */
export function Monogram({ size = 56, tone = 'light', decorative = false }: { size?: number; tone?: 'light' | 'ink'; decorative?: boolean }) {
  const c = tone === 'ink' ? '#0f1a3c' : '#ffffff';
  return (
    <svg className="monogram" width={size} height={size} viewBox="0 0 56 56" {...(decorative ? { 'aria-hidden': true } : { role: 'img', 'aria-label': 'SupeRR Selector monogram' })}>
      <rect x="1.5" y="1.5" width="53" height="53" rx="4" fill="none" stroke={c} strokeOpacity={tone === 'ink' ? 1 : 0.9} strokeWidth="1.5" />
      <rect x="1.5" y="46" width="53" height="8.5" fill="#e4007c" />
      <text x="16.5" y="38" textAnchor="middle" fontFamily="Barlow Condensed, sans-serif" fontWeight="700" fontSize="34" fill={c}>R</text>
      <text x="36" y="38" textAnchor="middle" fontFamily="Barlow Condensed, sans-serif" fontWeight="700" fontSize="34" fill="#e4007c">R</text>
    </svg>
  );
}

function builtDate(iso: string) {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
}

export function Header({ meta, nav }: { meta: Meta; nav: NavItem[] }) {
  return (
    <>
      <header className="band-navy header" id="top">
        <div className="wrap header-inner">
          <Monogram />
          <div className="header-text">
            <h1 className="header-title">{meta.title}</h1>
            <p className="header-tagline">{TAGLINES[TAGLINE]}</p>
          </div>
          <dl className="header-facts">
            <div><dt>Data</dt><dd>{meta.data_range}</dd></div>
            <div><dt>Last built</dt><dd><time dateTime={meta.last_built}>{builtDate(meta.last_built)}</time></dd></div>
          </dl>
        </div>
      </header>
      <SiteNav items={nav} monogram={<Monogram size={26} tone="ink" decorative />} />
    </>
  );
}
