import type { Meta } from '../data';
import { Monogram } from './Header';

export function Footer({ meta }: { meta: Meta }) {
  return (
    <footer className="footer">
      <div className="wrap footer-inner">
        <Monogram size={40} />
        <div className="footer-text">
          <p className="footer-strong">© 2026 Abhinav Anand. All rights reserved.</p>
          <p>Made and designed with love by Abhinav.</p>
          <p>Independent analysis for the SupeRR Selector challenge. Not affiliated with or endorsed by Rajasthan Royals.</p>
          <p className="footer-src">Data: {meta.data_source}</p>
        </div>
      </div>
    </footer>
  );
}
