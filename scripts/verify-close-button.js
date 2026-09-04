/*
 * Regression check: every dismissible panel's close button must stay reachable
 * at every scroll position, in every iOS URL-bar state.
 *
 *   node scripts/verify-close-button.js [baseUrl] [surface] [--ios-vh]
 *
 *   baseUrl   default http://127.0.0.1:8011   (uvicorn main:app --port 8011)
 *   surface   scene | progress | levels | all  (default all)
 *   --ios-vh  model iOS viewport-unit semantics (see below)
 *
 * Exit code 0 = pass, 1 = fail, 2 = inconclusive (content never overflowed, so
 * the check proved nothing). Note the exit code is lost if you pipe to `tail`.
 *
 * Asserts with document.elementFromPoint() that the close control is the
 * TOPMOST element at its own centre. Deliberately NOT is_visible(): that checks
 * CSS, not occlusion, and passes in every failing case this script catches.
 * A first-run consent notice shipped broken here for exactly that reason.
 *
 * Two distinct defects are covered, and they compose -- neither fix alone is
 * enough:
 *   1. The panel box overflowing its fixed container, putting the close button
 *      off screen before any scrolling.
 *   2. The close button living inside the panel's own scroller, so reading a
 *      long result scrolled it away with no way back.
 *
 * --ios-vh: Playwright models the URL bar by resizing the viewport, which moves
 * `vh` and the fixed-position box together. Real iOS does not: `vh` is pinned to
 * the LARGE viewport (URL bar hidden) and never changes, while position:fixed
 * sizes to the CURRENT one. So a vh-capped panel is measured against a box up to
 * ~86px shorter than it thinks. This flag pins each panel's max-height to its
 * old vh value resolved against LARGE, reproducing that mismatch. It is only
 * meaningful against a build that still uses vh -- the current CSS uses percent,
 * which tracks the fixed box correctly and needs no modelling.
 */
const { webkit } = require('playwright');

const args    = process.argv.slice(2).filter(a => a !== '--ios-vh');
const IOS_VH  = process.argv.includes('--ios-vh');
const BASE    = args[0] || 'http://127.0.0.1:8011';
const WHICH   = args[1] || 'all';
const SMALL   = 659, LARGE = 745;   // iPhone 14: URL bar showing / collapsed

const SURFACES = {
  scene: {
    label: 'scene modal (completed result)',
    box: '#modal',              // the capped box
    scroller: '#modal',         // the element that scrolls
    animated: '#modal',         // the element with the entry transition
    close: '#btnClose',
    legacyVh: 0.94,             // pre-fix mobile max-height
    css: '#scorePanel,#phonSection,#pbCompare,#transReveal,#btnTryAgain,#btnChallenge' +
         '{display:block !important;opacity:1 !important;}',
    open: () => {
      document.getElementById('overlay').classList.add('open');
      document.body.style.position = 'fixed'; document.body.style.width = '100%';
      document.getElementById('mTitle').textContent = 'The Dark Knight';
      document.getElementById('mYear').textContent  = '2008';
      document.getElementById('scoreVal').textContent = '96';
      document.getElementById('cmpYou').textContent   = 'Why so serious?';
      document.getElementById('cmpOrig').textContent  = 'Why so serious?';
      document.getElementById('transText').textContent = '¿Por qué tan serio?';
      const pw = document.getElementById('phonWords');
      if (pw) pw.innerHTML = Array(40).fill('<span class="phon-word">serious</span>').join(' ');
    },
  },
  progress: {
    label: 'progress dashboard',
    box: '.progress-modal',
    scroller: '.progress-modal',
    animated: '.progress-modal',
    close: '#btnProgressClose',
    legacyVh: 0.90,
    open: () => {
      document.getElementById('progressOverlay').classList.add('open');
      document.body.style.position = 'fixed'; document.body.style.width = '100%';
      const body = document.querySelector('.progress-modal-body');
      if (body) body.innerHTML = Array(40)
        .fill('<div style="padding:14px 0;border-bottom:1px solid rgba(255,255,255,.06)">Scene attempt</div>').join('');
    },
  },
  levels: {
    label: 'level panel',
    box: '.clv-panel',
    scroller: '.clv-panel-inner',
    animated: '.clv-panel',     // NOT the scroller: the slide is on the parent
    close: '.clv-panel-close',
    legacyVh: 0.85,
    open: () => {
      document.getElementById('clvPanel').classList.add('open');
      // The backdrop is what makes the panel read as opaque; opening the panel
      // without it lets the page show through and misrepresents the screenshot.
      document.getElementById('clvPanelBackdrop').classList.add('open');
      document.getElementById('clvPanelTitle').textContent = 'Beginner';
      const list = document.getElementById('clvClipList');
      if (list) list.innerHTML = Array(30)
        .fill('<div style="padding:18px;margin-bottom:10px;border:1px solid rgba(255,255,255,.08);border-radius:12px">Scene</div>').join('');
    },
  },
};

async function boot(page, surface) {
  await page.goto(BASE + '/', { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(2000);
  if (surface.css) await page.addStyleTag({ content: surface.css });
  if (IOS_VH) {
    await page.addStyleTag({ content:
      surface.box + ' { max-height: ' + (LARGE * surface.legacyVh).toFixed(2) + 'px !important; }' });
  }
  await page.evaluate(surface.open);
  await page.evaluate(() => document.fonts.ready);
  // Wait out the entry transition on whichever element carries it. Measuring
  // through it offsets every rect by the slide distance and reads like a layout
  // bug when it is only the animation.
  await page.waitForFunction((sel) => {
    const el = document.querySelector(sel);
    if (!el) return true;
    const t = getComputedStyle(el).transform;
    if (t === 'none') return true;
    const m = t.match(/matrix\(([^)]+)\)/);
    return m ? Math.abs(parseFloat(m[1].split(',')[5])) < 0.5 : true;
  }, surface.animated, { timeout: 6000 }).catch(() => {});
  await page.waitForTimeout(300);
}

async function check(page, surface, scrollTo) {
  return page.evaluate(({ scroller, close, scrollTo }) => {
    const s = document.querySelector(scroller), b = document.querySelector(close);
    if (!s || !b) return { missing: true };
    if (scrollTo === 'bottom')   s.scrollTop = s.scrollHeight;
    else if (scrollTo === 'mid') s.scrollTop = Math.round((s.scrollHeight - s.clientHeight) / 2);
    else                          s.scrollTop = 0;
    const br = b.getBoundingClientRect();
    const cx = (br.left + br.right) / 2, cy = (br.top + br.bottom) / 2;
    const onScreen = cy >= 0 && cy <= window.innerHeight && cx >= 0 && cx <= window.innerWidth;
    const top = onScreen ? document.elementFromPoint(cx, cy) : null;
    // Children of the button (an icon, an svg) still count as hitting it.
    const hit = !!top && (top === b || b.contains(top));
    return {
      scrollTop: s.scrollTop, scrollMax: s.scrollHeight - s.clientHeight,
      scrolls: s.scrollHeight > s.clientHeight,
      panelTop: s.getBoundingClientRect().top, xTop: br.top,
      topmost: top ? (top.id || (typeof top.className === 'string' && top.className) || top.tagName) : 'OFF-SCREEN',
      pass: hit,
    };
  }, { scroller: surface.scroller, close: surface.close, scrollTo });
}

async function runSurface(name) {
  const surface = SURFACES[name];
  const browser = await webkit.launch();
  const page = await browser.newPage({
    viewport: { width: 390, height: SMALL }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
  });
  await boot(page, surface);

  console.log('\n--- ' + name + ': ' + surface.label + (IOS_VH ? '  [iOS vh modelled]' : '') + ' ---');
  const states = [
    { label: 'top',                     scroll: 'top',    h: SMALL },
    { label: 'mid-scroll',              scroll: 'mid',    h: null  },
    { label: 'BOTTOM',                  scroll: 'bottom', h: null  },
    { label: 'BOTTOM, bar collapsed',   scroll: 'bottom', h: LARGE },
    { label: 'BOTTOM, bar re-expanded', scroll: 'bottom', h: SMALL },
  ];
  let allPass = true, everScrolled = false;
  for (const st of states) {
    if (st.h) { await page.setViewportSize({ width: 390, height: st.h }); await page.waitForTimeout(400); }
    const r = await check(page, surface, st.scroll);
    if (r.missing) { console.log('  SKIP - selector not found'); await browser.close(); return 'skip'; }
    allPass = allPass && r.pass;
    everScrolled = everScrolled || r.scrolls;
    console.log('  ' + st.label.padEnd(24) +
      ' panel top ' + r.panelTop.toFixed(0).padStart(4) +
      ' scroll ' + String(r.scrollTop).padStart(4) + '/' + String(r.scrollMax).padEnd(4) +
      ' xTop ' + r.xTop.toFixed(0).padStart(5) +
      '  topmost: ' + String(r.topmost).slice(0, 18).padEnd(18) + (r.pass ? 'PASS' : 'FAIL'));
  }
  const slug = BASE.replace(/^https?:\/\//, '').replace(/[^a-z0-9]+/gi, '-');
  await page.screenshot({ path: 'verify-' + name + '--' + slug + (IOS_VH ? '--iosvh' : '') + '.png' });
  await browser.close();
  if (!everScrolled) { console.log('  INCONCLUSIVE - content never overflowed.'); return 'inconclusive'; }
  return allPass ? 'pass' : 'fail';
}

(async () => {
  const names = WHICH === 'all' ? Object.keys(SURFACES) : [WHICH];
  console.log('Target: ' + BASE + '   (WebKit, iPhone 14 geometry)');
  const results = {};
  for (const n of names) {
    if (!SURFACES[n]) { console.error('unknown surface: ' + n); process.exit(2); }
    results[n] = await runSurface(n);
  }
  console.log('\n' + Object.entries(results).map(([k, v]) => k + '=' + v).join('  '));
  const vals = Object.values(results);
  process.exit(vals.includes('fail') ? 1 : (vals.includes('inconclusive') ? 2 : 0));
})();
