// Rendered impeccable scan of the whole page (every section) and of all 22 deep dives (#player-<id> opens each one).
// Usage: npm run build && node scripts/impeccable_scan.mjs   (no ignores: .impeccable/config.json must stay empty)
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const IMP = process.env.IMPECCABLE || 'impeccable';   // path to the impeccable CLI, or the command if it is on PATH
const PORT = 4177, base = `http://localhost:${PORT}/`;
const cfg = JSON.parse(readFileSync(`${ROOT}.impeccable/config.json`, 'utf8')).detector;
if (cfg.ignoreRules.length || cfg.ignoreFiles.length || cfg.ignoreValues.length) { console.error('ignores present in .impeccable/config.json'); process.exit(2); }
const players = JSON.parse(readFileSync(`${ROOT}public/data/players.json`, 'utf8'));
const server = spawn(process.platform === 'win32' ? 'npx.cmd' : 'npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], { cwd: ROOT, shell: process.platform === 'win32' });
for (let i = 0; i < 60; i++) { try { if ((await fetch(base)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 500)); }
const targets = [['whole page (all sections)', base], ...Object.values(players).map((p) => [`deep dive: ${p.name}`, `${base}#player-${p.player_id}`])];
let total = 0;
try {
  for (const [label, url] of targets) {
    const r = spawnSync(IMP, ['detect', url, '--json'], { encoding: 'utf8', shell: process.platform === 'win32', maxBuffer: 1 << 26 });
    let found = [];
    try {
      const walk = (x) => { if (Array.isArray(x)) x.forEach(walk); else if (x && typeof x === 'object') { if (x.antipattern) found.push(x); else Object.values(x).forEach(walk); } };
      walk(JSON.parse(r.stdout || '[]'));
    } catch { found = [{ antipattern: 'scan-output', snippet: (r.stdout || r.stderr || '').slice(0, 300) }]; }
    total += found.length;
    console.log(`${found.length ? 'FINDINGS' : 'clean   '}  ${label}${found.length ? '' : ''}`);
    for (const f of found) console.log(`    [${f.antipattern}] ${f.snippet || ''}  ${f.selector ? '@ ' + f.selector : ''}`);
  }
} finally {
  server.kill();
  if (process.platform === 'win32') spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F']);
}
console.log(`\nIMPECCABLE: ${targets.length} pages scanned, ${total} findings (advisories included), ignores: none`);
process.exit(total ? 1 : 0);
