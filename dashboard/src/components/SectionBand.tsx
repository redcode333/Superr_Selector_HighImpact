import type { ReactNode } from 'react';

/** Each section opens with a heading band: 4px pink left border, 1px bottom border, no numbers. */
export function Section({ id, title, lede, children }: { id: string; title: string; lede?: ReactNode; children: ReactNode }) {
  return (
    <section className="section" id={id} aria-labelledby={`${id}-h`}>
      <div className="section-band">
        <div className="wrap">
          <h2 id={`${id}-h`}>{title}</h2>
          {lede && <p className="section-lede">{lede}</p>}
        </div>
      </div>
      <div className="wrap section-body">{children}</div>
    </section>
  );
}
