// Playwright tests for the live readouts of the two "Try it" tools.
// For Tim David, Abhishek Sharma, Jasprit Bumrah and Kuldeep Yadav: click each preset and assert that the numbers on the page
// equal values computed here, independently, from public/data/players.json (raw regions / overs only; none of the page's own
// derived fields such as best_field, saving or planner are used as the expected values).
// Usage: npm run build && node scripts/readout_tests.mjs        MUTATE=1 makes one expected value wrong (the test must then fail).
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const D = JSON.parse(readFileSync(`${ROOT}public/data/players.json`, 'utf8'));
const byName = (n) => Object.values(D).find((p) => p.name === n);
const PORT = 4178, base = `http://localhost:${PORT}/`;
const inr = (x) => x.toLocaleString('en-IN');
const MUT = process.env.MUTATE ? 1 : 0;
const EXAMPLE = ['Third man', 'Cover', 'Long-off', 'Long-on', 'Fine leg'];   // the design page's example field
let checks = 0; const fails = [];
const eq = (label, got, want) => { checks++; if (String(got) !== String(want)) fails.push(`${label}: page "${got}" vs players.json "${want}"`); };

// ---------------------------------------------------------------- independent expectations: batter field
function bestFive(regs) {                                    // brute force over all 5-region sets, most boundary runs
  let best = null;
  const idx = regs.map((_, i) => i);
  const comb = (start, pick) => {
    if (pick.length === 5) { const v = pick.reduce((t, i) => t + regs[i].boundary_runs, 0); if (!best || v > best.v) best = { v, pick: [...pick] }; return; }
    for (let i = start; i < idx.length; i++) { pick.push(i); comb(i + 1, pick); pick.pop(); }
  };
  comb(0, []);
  return best.pick.map((i) => regs[i].name);
}
function expectField(p, field) {
  const R = p.regions, tot = R.reduce((t, r) => t + r.boundary_runs, 0);
  const cov = R.filter((r) => field.includes(r.name));
  const sum = (k) => cov.reduce((t, r) => t + r[k], 0);
  const covRuns = sum('boundary_runs'), pct = tot ? Math.round((100 * covRuns) / tot) : 0;
  const open = R.filter((r) => !field.includes(r.name)).sort((a, b) => b.boundary_runs - a.boundary_runs);
  const gap = open[0];
  const best = bestFive(R);
  const isBest = field.length === 5 && best.every((n) => field.includes(n));
  const head = !gap ? 'Every region is covered' : isBest ? 'This is the best five for him' : `Biggest gap: ${gap.name}`;
  const text = !gap ? '' : isBest
    ? `No other five regions cover more of his boundary runs. ${gap.name} is still open (${inr(gap.boundary_runs)} runs).`
    : `${inr(gap.boundary_runs)} boundary runs (${tot ? Math.round((100 * gap.boundary_runs) / tot) : 0}% of his total) go there with nobody in the deep.`;
  return { pct: pct + MUT, covRuns, tot, fours: sum('fours'), sixes: sum('sixes'), caught: sum('caught'),
           tf: R.reduce((t, r) => t + r.fours, 0), ts: R.reduce((t, r) => t + r.sixes, 0), tc: R.reduce((t, r) => t + r.caught, 0), head, text, best };
}

// ---------------------------------------------------------------- independent expectations: bowler planner
const f1 = (x) => x.toFixed(1);
function expectPlan(p, plan) {
  const OV = p.overs;
  const saving = (o) => (o.balls ? ((o.par_econ - o.econ) * o.balls) / (o.balls + 60) : 0);
  const bowled = OV.filter((o) => o.balls);
  const avgUnbowled = bowled.reduce((t, o) => t + o.par_econ, 0) / bowled.length;
  const nonAdj = (c) => c.every((x, i) => i === 0 || x - c[i - 1] > 1);
  const combos = []; for (let a = 1; a <= 20; a++) for (let b = a + 1; b <= 20; b++) for (let c = b + 1; c <= 20; c++) for (let d = c + 1; d <= 20; d++) if (nonAdj([a, b, c, d])) combos.push([a, b, c, d]);
  const val = (c, fn) => c.reduce((t, o) => t + fn(OV[o - 1]), 0);
  const bestC = combos.reduce((m, c) => (val(c, saving) > val(m, saving) + 1e-12 ? c : m));
  const usualC = combos.reduce((m, c) => { const vc = val(c, (o) => o.times), vm = val(m, (o) => o.times); return vc > vm || (vc === vm && c.join() < m.join() && false) ? c : m; });
  const chosen = plan === 'usual' ? usualC : plan === 'best' ? bestC : [];
  const saved = val(chosen, saving), avg = chosen.reduce((t, o) => t + (OV[o - 1].balls ? OV[o - 1].par_econ : avgUnbowled), 0);
  const wk = val(chosen, (o) => o.wkts), wb = val(chosen, (o) => o.balls);
  const same = (a, b) => a.length === b.length && a.every((x) => b.includes(x));
  let head;
  if (chosen.length < 4) head = `Pick ${4 - chosen.length} more`;
  else if (same(chosen, bestC)) head = 'This is the best plan in the data';
  else head = `${f1(Math.max(0, val(bestC, saving) - saved))} runs short of the best plan`;
  return { saved: f1(saved) , avg: f1(avg), his: f1(avg - saved), wk: wb ? `${wk} in ${wb}` : '—', head, count: chosen.length, chosen, bestSaving: val(bestC, saving), bestC };
}

// ---------------------------------------------------------------- run
const server = spawn(process.platform === 'win32' ? 'npx.cmd' : 'npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], { cwd: ROOT, shell: process.platform === 'win32' });
for (let i = 0; i < 60; i++) { try { if ((await fetch(base)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 500)); }
const browser = await chromium.launch();
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(e.message));
  for (const name of ['Tim David', 'Abhishek Sharma', 'Jasprit Bumrah', 'Kuldeep Yadav']) {
    const p = byName(name);
    await page.goto(`${base}#player-${p.player_id}`, { waitUntil: 'networkidle' });
    await page.waitForFunction((nm) => document.querySelector('.dive-head h3')?.textContent === nm, name);
    if (p.role === 'batter') {
      const best = bestFive(p.regions);
      for (const [label, field] of [['Example field', EXAMPLE], ['Best field for him', best], ['Clear', []]]) {
        await page.locator('.field .seg-group').getByRole('button', { name: label, exact: true }).click();
        const E = expectField(p, field), L = `${name} / ${label}`;
        eq(`${L} placed`, await page.locator('.field-count').textContent(), `${field.length} of 5 boundary fielders placed`);
        eq(`${L} coverage %`, await page.locator('.field-panel .fp-pct').textContent(), `${E.pct}%`);
        eq(`${L} coverage runs`, await page.locator('.field-panel .fp-sub').textContent(), `${inr(E.covRuns)} of ${inr(E.tot)} runs`);
        const rows = await page.locator('.field-panel .fp-rows dt').allTextContents();
        eq(`${L} fours`, rows[0], `${E.fours} / ${E.tf}`); eq(`${L} sixes`, rows[1], `${E.sixes} / ${E.ts}`); eq(`${L} catches`, rows[2], `${E.caught} / ${E.tc}`);
        eq(`${L} biggest-gap headline`, await page.locator('.field-panel .fp-gap b').textContent(), E.head);
        eq(`${L} biggest-gap text`, (await page.locator('.field-panel .fp-gap span').textContent().catch(() => '')) ?? '', E.text);
      }
      // the page's own "best field" equals the independent one (placed fielders are marked on the map)
      await page.locator('.field .seg-group').getByRole('button', { name: 'Best field for him', exact: true }).click();
      const placed = await page.locator('.band[aria-pressed="true"]').evaluateAll((g) => g.map((x) => x.getAttribute('aria-label').replace(/ \d+ boundary runs.*$/, '')));
      eq(`${name} best field regions`, placed.sort().join(', '), bestFive(p.regions).sort().join(', '));
    } else {
      for (const [label, key] of [['His usual overs', 'usual'], ['Best plan', 'best'], ['Clear', 'clear']]) {
        await page.locator('.planner .seg-group').getByRole('button', { name: label, exact: true }).click();
        const E = expectPlan(p, key), L = `${name} / ${label}`;
        eq(`${L} runs saved`, await page.locator('.plan-panel .fp-pct').textContent(), E.saved);
        eq(`${L} overs chosen`, await page.locator('.plan-panel .fp-sub').textContent(), `across ${E.count} of 4 overs`);
        const rows = await page.locator('.plan-panel .fp-rows dt').allTextContents();
        eq(`${L} his runs`, rows[0], E.his); eq(`${L} average bowler`, rows[1], E.avg); eq(`${L} wickets`, rows[2], E.wk);
        eq(`${L} runs-short headline`, await page.locator('.plan-panel .fp-gap b').textContent(), E.head);
        const on = await page.locator('.tile[aria-pressed="true"]').evaluateAll((t) => t.map((x) => Number(x.querySelector('.t-n').textContent.replace('Over ', ''))));
        eq(`${L} overs picked`, on.join(','), E.chosen.join(','));
        if (key === 'usual') eq(`${L} runs-short text names the best plan`, (await page.locator('.plan-panel .fp-gap span').textContent()), `The best is overs ${E.bestC.join(', ')}.`);
      }
    }
  }
  if (errors.length) fails.push(`console errors: ${errors.join(' | ')}`);
} finally {
  await browser.close();
  server.kill();
  if (process.platform === 'win32') spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F']);
}
console.log(`READOUT TESTS: ${checks - fails.filter((f) => !f.startsWith('console')).length} of ${checks} assertions pass, console errors: ${errors.length}`);
for (const f of fails) console.log('  FAIL', f);
process.exit(fails.length ? 1 : 0);
