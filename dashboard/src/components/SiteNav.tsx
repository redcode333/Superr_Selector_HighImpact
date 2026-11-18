import { useEffect, useRef, useState } from 'react';

export interface NavItem { id: string; label: string }

const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * One slim sticky bar: monogram, the section links, a 2px reading-progress line along its bottom edge.
 * The active section comes from an IntersectionObserver over the section elements (a band just under the bar);
 * the link for it gets aria-current="true" and a 2px pink underline.
 */
export function SiteNav({ items, monogram }: { items: NavItem[]; monogram: React.ReactNode }) {
  const [active, setActive] = useState<string | null>(null);
  const navRef = useRef<HTMLElement>(null);
  const rowRef = useRef<HTMLUListElement>(null);
  const lineRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const nav = navRef.current;
    if (!nav) return;
    const ids = items.map((i) => i.id);
    const seen = new Set<string>();
    let io: IntersectionObserver | undefined;
    let barH = -1;

    // the section whose heading band has reached the top of the page: the last one in order that touches the band
    const pick = () => {
      const bottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
      if (bottom) { setActive(ids[ids.length - 1]); return; }
      for (let i = ids.length - 1; i >= 0; i--) if (seen.has(ids[i])) { setActive(ids[i]); return; }
      const first = document.getElementById(ids[0]);
      if (first && first.getBoundingClientRect().top > barH) setActive(null);   // still in the page header
    };

    const observe = () => {
      const h = Math.round(nav.getBoundingClientRect().height);
      if (h === barH) return;
      barH = h;
      io?.disconnect();
      seen.clear();
      io = new IntersectionObserver((entries) => {
        for (const e of entries) { if (e.isIntersecting) seen.add(e.target.id); else seen.delete(e.target.id); }
        pick();
      }, { rootMargin: `-${h}px 0px -55% 0px`, threshold: 0 });
      for (const id of ids) { const el = document.getElementById(id); if (el) io.observe(el); }
    };

    let raf = 0;
    const frame = () => {
      raf = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      if (lineRef.current) lineRef.current.style.transform = `scaleX(${p})`;   // no transition: the line follows the scroll exactly
      pick();
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(frame); };
    const onResize = () => { observe(); onScroll(); };

    observe();
    frame();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize);
    return () => {
      io?.disconnect();
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
    };
  }, [items]);

  // on phones the row scrolls sideways: keep the active link in view (the row only, never the page)
  useEffect(() => {
    const row = rowRef.current;
    const link = row?.querySelector<HTMLElement>('[aria-current="true"]');
    if (!row || !link || row.scrollWidth <= row.clientWidth) return;
    row.scrollTo({ left: link.offsetLeft - (row.clientWidth - link.offsetWidth) / 2, behavior: reduced() ? 'auto' : 'smooth' });
  }, [active]);

  const go = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    const el = document.getElementById(id);
    if (!el) return;   // no target: let the browser handle the link
    e.preventDefault();
    el.scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });   // scroll-padding-top = bar height keeps the heading clear
    window.history.replaceState(null, '', `#${id}`);
  };

  return (
    <nav className="sticky-nav" aria-label="Sections" ref={navRef}>
      <div className="wrap nav-inner">
        <a className="nav-home" href="#top" aria-label="RR monogram, SupeRR Selector, back to top" onClick={(e) => go(e, 'top')}>{monogram}</a>
        <ul className="nav-links" ref={rowRef}>
          {items.map((n) => (
            <li key={n.id}>
              <a className="nav-link" href={`#${n.id}`} aria-current={active === n.id ? 'true' : undefined} onClick={(e) => go(e, n.id)}>{n.label}</a>
            </li>
          ))}
        </ul>
      </div>
      <span className="nav-progress" ref={lineRef} aria-hidden="true" />
    </nav>
  );
}
