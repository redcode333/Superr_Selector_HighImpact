// Playwright screenshots of every section at 1440px and 390px, from the production build (vite preview).
// Usage: npm run build && node scripts/screenshots.mjs [phase-label]
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';

const label = process.argv[2] || 'latest';
const OUT = fileURLToPath(new URL(`../screenshots/${label}/`, import.meta.url));
mkdirSync(OUT, { recursive: true });
const PORT = 4173;
const server = spawn(process.platform === 'win32' ? 'npx.cmd' : 'npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'],
  { cwd: new URL('..', import.meta.url), shell: process.platform === 'win32' });
const url = `http://localhost:${PORT}/`;
for (let i = 0; i < 60; i++) {
  try { if ((await fetch(url)).ok) break; } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 500));
}

const browser = await chromium.launch();
const errors = [];
const shots = [];
const fail = (msg) => errors.push(msg);
const demoOn = (page) => page.locator('.demo-bar button', { hasText: 'Stop' }).count().then((n) => n > 0);

/** Phase 3: the 19th-over hero at rest, mid-over, after 1,000 runs, mid-walkthrough; plus walkthrough behaviour checks. */
async function heroStates(page, width) {
  const hero = page.locator('#nineteenth .hero');
  const shot = async (name, loc = hero) => { const f = `${OUT}${width}-hero-${name}.png`; await loc.screenshot({ path: f }); shots.push(f); };
  await page.locator('#nineteenth').scrollIntoViewIfNeeded();
  await shot('rest');
  await page.getByRole('button', { name: 'Bowl the 19th' }).click();
  await page.waitForTimeout(650 * 3 + 200);
  await shot('mid-over');
  await page.waitForFunction(() => !document.querySelector('.pb-btns .btn[aria-disabled="true"]'), null, { timeout: 15000 });
  await page.getByRole('button', { name: 'Play it 1,000 times' }).click();
  await page.waitForTimeout(1600);
  await shot('after-1000');
  // walkthrough: mid-frame at step 4 (bowling), then Stop
  await page.getByRole('button', { name: /Watch how it works/ }).click();
  for (let t = 0; t < 16; t++) {
    await page.waitForTimeout(1000);
  }
  await page.waitForTimeout(800);
  if (!(await demoOn(page))) fail(`${width}px walkthrough not running at 16.8 s`);
  await shot('walkthrough', page.locator('#nineteenth'));
  await page.getByRole('button', { name: 'Stop' }).click();
  if (await demoOn(page)) fail(`${width}px Stop did not end the walkthrough`);
  // a click on any control stops it
  await page.getByRole('button', { name: /Watch how it works/ }).click();
  await page.waitForTimeout(800);
  await page.getByRole('button', { name: 'More runs needed' }).click();
  if (await demoOn(page)) fail(`${width}px a control click did not stop the walkthrough`);
  // leaving the section stops it and cancels its timers
  await page.getByRole('button', { name: /Watch how it works/ }).click();
  await page.waitForTimeout(500);
  await page.locator('footer').scrollIntoViewIfNeeded();
  await page.waitForTimeout(600);
  if (await demoOn(page)) fail(`${width}px leaving the section did not stop the walkthrough`);
  const capBefore = await page.locator('.pb-name').textContent();
  await page.waitForTimeout(13000);   // a live timer would have switched the bowler at 12.5 s
  if ((await page.locator('.pb-name').textContent()) !== capBefore) fail(`${width}px walkthrough timers kept running after leaving`);
}

/** Phase 4: deep dives opened through the picker search; field-map tap targets >= 44px on phones; a card opens its dive. */
const DIVES = (process.env.DIVES || 'Tim David,Abhishek Sharma,Jasprit Bumrah,Kuldeep Yadav,Glenn Maxwell').split(',');
async function diveStates(page, width) {
  for (const name of DIVES) {
    await page.locator('.picker-search').fill(name);
    await page.locator('.picker .pk', { hasText: name }).first().click();
    await page.waitForFunction((nm) => document.querySelector('.dive-head h3')?.textContent === nm, name, { timeout: 10000 });
    await page.locator('.picker-search').fill('');
    const slug = name.toLowerCase().replace(/[^a-z]+/g, '-');
    const f = `${OUT}${width}-dive-${slug}.png`;
    await page.locator('#dive').screenshot({ path: f }); shots.push(f);
    if (width < 800 && (await page.$('.field-svg'))) {
      const res = await page.evaluate(() => {
        const svg = document.querySelector('.field-svg'); const k = svg.getBoundingClientRect().width / 440;
        // each band spans radius 104..212 in the 440 viewBox; tap target = depth and arc length at the band's middle radius
        return [...svg.querySelectorAll('.band')].map((g) => {
          const label = g.getAttribute('aria-label').replace(/ \d+ boundary runs.*$/, ''); const box = g.querySelector('.band-hit').getBBox();
          return { label, depth: 108 * k, k, box: [box.width * k, box.height * k] };
        });
      });
      const regions = await page.evaluate(async () => (await (await fetch('data/players.json')).json()));
      const any = Object.values(regions).find((p) => p.role === 'batter');
      for (const r of res) {
        const reg = any.regions.find((x) => x.name === r.label);
        const arc = (158 * r.k * Math.abs(reg.to - reg.from) * Math.PI) / 180;
        if (r.depth < 44 || arc < 44) fail(`${width}px field region ${r.label} tap target ${Math.round(r.depth)} x ${Math.round(arc)} px < 44`);
      }
      console.log(`${width}px field tap targets: min depth ${Math.round(Math.min(...res.map((r) => r.depth)))} px`);
    }
  }
  // a card opens the deep dive
  await page.locator('#picks .pcard-open').nth(2).click();
  await page.waitForTimeout(400);
  const want = (await page.locator('#picks .pcard-name').nth(2).textContent());
  if ((await page.locator('.dive-head h3').textContent()) !== want) fail(`${width}px clicking a card did not open its deep dive`);
}

/** Phase 5: rules sandbox floor control and head-to-head readout. */
async function phase5States(page, width) {
  const rules = page.locator('#rules');
  await rules.getByRole('button', { name: '0', exact: true }).click();
  await page.waitForTimeout(700);
  const txt = await rules.textContent();
  if (!txt.includes('Comes in') || !txt.includes('Drops out') || !txt.includes('Sooryavanshi')) fail(`${width}px rules: floor 0 did not bring back the v1 batting list`);
  let f = `${OUT}${width}-rules-floor0.png`; await rules.screenshot({ path: f }); shots.push(f);
  await rules.getByRole('button', { name: 'Back to our rules' }).click();
  const cellBtn = page.locator('#h2h .h2h-cell:not(.few)').first();
  await cellBtn.click();
  const read = await page.locator('#h2h .h2h-read').textContent();
  if (!/runs off \d+ balls, out \d+/.test(read)) fail(`${width}px head-to-head readout missing: ${read}`);
  f = `${OUT}${width}-h2h-selected.png`; await page.locator('#h2h').screenshot({ path: f }); shots.push(f);
}

/** Reduced motion: no ball animation, no pulsing; the walkthrough still advances without motion. */
async function reducedMotionChecks() {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  page.on('console', (m) => { if (m.type() === 'error') fail(`reduced-motion console: ${m.text()}`); });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: 'Bowl the 19th' }).click();
  await page.waitForTimeout(60);
  const n = await page.locator('.over-balls .ball:not(.b-pad)').count();
  if (!n) fail('reduced motion: balls did not appear at once');
  const result = await page.locator('.sb-status .sb-l').textContent();
  if (result !== 'Result of this one match') fail('reduced motion: result not shown at once');
  await page.getByRole('button', { name: /Watch how it works/ }).click();
  await page.waitForTimeout(5000);
  const cap = await page.locator('.demo-caption').textContent();
  if (!cap?.startsWith('Step 2')) fail(`reduced motion: walkthrough did not advance (caption: ${cap})`);
  if (await page.locator('.hl-pulse').count()) fail('reduced motion: pulsing highlight present');
  if (!(await page.locator('.hl-static').count())) fail('reduced motion: no still highlight on the explained part');
  const anim = await page.evaluate(() => [...document.querySelectorAll('.hl-static, .ball')].some((e) => getComputedStyle(e).animationName !== 'none'));
  if (anim) fail('reduced motion: a CSS animation is running');
  await page.screenshot({ path: `${OUT}1440-reduced-motion-walkthrough.png` });
  await ctx.close();
}
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: width > 800 ? 900 : 844 }, deviceScaleFactor: 1 });
    page.on('console', (m) => { if (m.type() === 'error') errors.push(`${width}px console: ${m.text()}`); });
    page.on('pageerror', (e) => errors.push(`${width}px page error: ${e.message}`));
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForSelector('#picks');
    await page.evaluate(() => document.fonts.ready);
    // load lazy images now (screenshots of off-screen elements do not trigger lazy loading); give up after 10 s
    await page.evaluate(() => document.querySelectorAll('img[loading="lazy"]').forEach((i) => { i.loading = 'eager'; }));
    await page.waitForFunction(() => [...document.images].every((i) => i.complete), null, { timeout: 10000 }).catch(() => {});
    const broken = await page.$$eval('img', (ims) => ims.filter((i) => !i.naturalWidth).map((i) => i.src));
    if (broken.length) errors.push(`${width}px images not loaded: ${broken.join(', ')}`);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (overflow > 0) errors.push(`${width}px horizontal overflow ${overflow}px`);
    // element captures stitch tall elements; unstick the nav while they are taken so it is not painted over them
    const unstick = await page.addStyleTag({ content: '.sticky-nav{position:static !important}' });
    const targets = [['header', 'header.header'], ...(await page.$$eval('main section[id]', (s) => s.map((x) => [x.id, `#${x.id}`]))), ['footer', 'footer']];
    for (const [name, sel] of targets) {
      const file = `${OUT}${width}-${name}.png`;
      await page.locator(sel).screenshot({ path: file });
      shots.push(file);
    }
    await unstick.evaluate((el) => el.remove());
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
    await page.waitForTimeout(150);
    await page.screenshot({ path: `${OUT}${width}-first-screen.png` });
    await page.screenshot({ path: `${OUT}${width}-full.png`, fullPage: true });
    if (await page.$('#nineteenth')) {
      const u2 = await page.addStyleTag({ content: '.sticky-nav{position:static !important}' });
      await heroStates(page, width);
      if (await page.$('#dive') && !process.env.SKIP_DIVES) await diveStates(page, width);
      if (await page.$('#rules')) await phase5States(page, width);
      await u2.evaluate((el) => el.remove());
    }
    await page.close();
  }
  if (await (async () => { const p = await browser.newPage(); await p.goto(url); const has = !!(await p.$('#nineteenth')); await p.close(); return has; })()) {
    await reducedMotionChecks();
  }
} finally {
  await browser.close();
  server.kill();
  if (process.platform === 'win32') spawn('taskkill', ['/pid', String(server.pid), '/T', '/F']);
}
console.log(`screenshots: ${shots.length + 2} files in ${OUT}`);
console.log(errors.length ? `PROBLEMS:\n${errors.join('\n')}` : 'console errors: 0; horizontal overflow: none');
process.exit(errors.length ? 1 : 0);
