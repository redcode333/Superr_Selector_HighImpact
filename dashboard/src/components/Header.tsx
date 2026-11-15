import type { Meta } from '../data';
import { TAGLINE, TAGLINES } from '../copy';

export interface NavItem { id: string; label: string }

/** Our own typographic monogram: two condensed capitals, the second in pink, on a square rule. Not a crest. */
export function Monogram({ size = 56 }: { size?: number }) {
  return (
    <svg className="monogram" width={size} height={size} viewBox="0 0 56 56" role="img" aria-label="SupeRR Selector monogram">
      <rect x="1.5" y="1.5" width="53" height="53" rx="4" fill="none" stroke="#ffffff" strokeOpacity="0.9" strokeWidth="1.5" />
      <rect x="1.5" y="46" width="53" height="8.5" fill="#e4007c" />
      <text x="16.5" y="38" textAnchor="middle" fontFamily="Barlow Condensed, sans-serif" fontWeight="700" fontSize="34" fill="#ffffff">R</text>
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
      <nav className="sticky-nav" aria-label="Sections">
        <div className="wrap nav-inner">
          <a className="nav-home" href="#top" aria-label="SupeRR, back to top">SupeRR</a>
          <ul>
            {nav.map((n) => <li key={n.id}><a href={`#${n.id}`}>{n.label}</a></li>)}
          </ul>
        </div>
      </nav>
    </>
  );
}
