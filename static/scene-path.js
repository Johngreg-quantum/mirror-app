// Scene path: what happens after a take.
//
// Finishing a scene used to be a dead end: the result card had Try Again and
// Challenge a Friend, and the only way on was to close the modal (the X) and
// pick the next scene from the Levels panel. This owns the forward path:
//
//   - a "Next scene" bar pinned to the bottom of the result card, which opens
//     the next scene the learner still has to do, in level order -- unlocked
//     and not yet passed (SCORE.PASS), so a scene already beaten is skipped;
//   - a Daily Take in progress goes to its next line first;
//   - every 10 scenes, a quick review: the ones in that run still under the
//     pass mark are offered again, one after another, before the path goes on;
//   - when the next scene is locked, the bar says why and where to go instead.
//
// Progress through the run of 10 is a per-device convenience kept in
// localStorage (scores themselves are on the server, and "passed" is always
// read from them), so a cleared browser only restarts the count.
(function() {
  const BLOCK = 10;
  const STORE_PREFIX = 'mirror.path.v1.';

  let ctx = null;
  let memState = null;      // used when localStorage is unavailable
  let lastDaily = null;     // { from, nextId, n, total } from the last submit

  // ── state ────────────────────────────────────────────────────────────────
  function storeKey() {
    const u = ctx.getUser();
    return STORE_PREFIX + ((u && (u.id || u.username)) || 'anon');
  }

  function load() {
    try {
      const raw = localStorage.getItem(storeKey());
      if (raw) {
        const s = JSON.parse(raw);
        if (s && Array.isArray(s.done)) {
          if (s.review && !(Array.isArray(s.review.queue) && s.review.queue.length)) s.review = null;
          return s;
        }
      }
    } catch (e) { /* private mode, blocked storage, bad JSON */ }
    return memState || { done: [], review: null };
  }

  function save(s) {
    memState = s;
    try { localStorage.setItem(storeKey(), JSON.stringify(s)); } catch (e) { /* in memory only */ }
  }

  // ── reading progress ─────────────────────────────────────────────────────
  function pathIds() {
    return (ctx.getLevels() || []).reduce(function(all, lv) { return all.concat(lv.scenes || []); }, []);
  }
  function progress() { return ctx.getProgress() || {}; }
  function bestScore(sid) {
    const b = progress().best_scores || {};
    return typeof b[sid] === 'number' ? b[sid] : null;
  }
  function passed(sid) {
    const b = bestScore(sid);
    return b !== null && b >= ctx.passScore;
  }
  function unlocked(sid) {
    const u = progress().unlocked_scenes;
    return Array.isArray(u) && u.includes(sid);
  }
  function owned(sid) {
    const a = progress().accessible_scenes;
    return !Array.isArray(a) || a.includes(sid);
  }
  // Both, not either: unlocked_scenes is level progression and can fall back
  // to all of Level 1 for an account that owns none of it, which would send
  // the bar to a scene the server then refuses to score.
  function playable(sid) { return unlocked(sid) && owned(sid); }
  function sceneOf(sid) { return (ctx.getScenes() || {})[sid] || null; }

  // The next scene still to do after `sid`: the first unlocked, unpassed scene
  // later in level order, then wrapping round to earlier ones the learner left
  // behind. If nothing playable is left, the first locked one says why.
  function nextInPath(sid) {
    const path = pathIds();
    const i = path.indexOf(sid);
    const order = i < 0 ? path : path.slice(i + 1).concat(path.slice(0, i));
    const todo = order.find(function(id) { return playable(id) && !passed(id); });
    if (todo) return { kind: 'scene', id: todo };
    const locked = order.find(function(id) { return !playable(id) && !passed(id); });
    if (locked) return { kind: 'locked', id: locked, owned: owned(locked) };
    return { kind: 'done' };
  }

  // What the Next button does from `current`, without doing it.
  function plan(current) {
    const s = load();

    if (s.review) {
      const q = s.review.queue, idx = s.review.idx;
      if (q[idx] !== current) {
        return { kind: 'review-resume', id: q[idx], n: idx + 1, total: q.length };
      }
      if (idx + 1 < q.length) {
        return { kind: 'review-next', id: q[idx + 1], n: idx + 2, total: q.length };
      }
      return Object.assign({ endReview: true }, nextInPath(s.review.resumeAfter || current));
    }

    // A Daily Take in progress finishes its set before anything else.
    if (lastDaily && lastDaily.from === current && lastDaily.nextId && lastDaily.nextId !== current) {
      return { kind: 'daily', id: lastDaily.nextId, n: lastDaily.n, total: lastDaily.total };
    }

    if (s.done.length >= BLOCK) {
      const failed = s.done.filter(function(id) { return !passed(id); });
      if (failed.length) return { kind: 'review-start', queue: failed };
      return Object.assign({ blockClear: true }, nextInPath(current));
    }
    return nextInPath(current);
  }

  // ── events from app.js ───────────────────────────────────────────────────
  function onScored(sceneId, data) {
    if (!ctx || !sceneId) return;
    const s = load();
    const inReview = s.review && s.review.queue[s.review.idx] === sceneId;
    if (!inReview && !s.done.includes(sceneId)) s.done.push(sceneId);
    save(s);

    const d = data && data.daily;
    lastDaily = (d && d.next_scene_id && !d.completed_now && !d.already_done)
      ? { from: sceneId, nextId: d.next_scene_id, n: (d.lines_done || 0) + 1, total: d.line_total || 3 }
      : null;
  }

  function onOpen(sceneId) {
    if (!ctx) return;
    hide();
    const banner = document.getElementById('spReviewBanner');
    if (!banner) return;
    const s = load();
    if (s.review && s.review.queue[s.review.idx] === sceneId) {
      const b = bestScore(sceneId);
      banner.innerHTML =
        '<span class="sp-rb-tag">Quick review · ' + (s.review.idx + 1) + ' of ' + s.review.queue.length + '</span>' +
        '<span class="sp-rb-text">' + (b !== null ? 'Last time: ' + Math.round(b) + '%. ' : '') +
        'Score ' + ctx.passScore + '%+ to pass.</span>';
      banner.hidden = false;
    } else {
      banner.hidden = true;
    }
  }

  // ── the bar ──────────────────────────────────────────────────────────────
  function thumbHTML(sid) {
    const sc = sceneOf(sid);
    const src = sc && sc.ui && (sc.ui.art_sm || sc.ui.art);
    return src ? '<img src="' + esc(src) + '" alt="" loading="lazy">' : '';
  }

  function render() {
    if (!ctx) return;
    const bar = document.getElementById('spNext');
    const current = ctx.getActiveScene();
    if (!bar || !current) return;

    const p = plan(current);
    const sc = p.id ? sceneOf(p.id) : null;
    const title = sc ? sc.title : '';
    let eyebrow = 'Next scene', label = title, thumb = p.id ? thumbHTML(p.id) : '', icon = '';

    switch (p.kind) {
      case 'daily':
        eyebrow = 'Daily Take · line ' + p.n + ' of ' + p.total; break;
      case 'review-start':
        eyebrow = 'Quick review';
        label = 'Retake the ' + p.queue.length + ' scene' + (p.queue.length === 1 ? '' : 's') + ' you missed';
        thumb = thumbHTML(p.queue[0]); break;
      case 'review-next':
        eyebrow = 'Review · ' + p.n + ' of ' + p.total; break;
      case 'review-resume':
        eyebrow = 'Back to your review · ' + p.n + ' of ' + p.total; break;
      case 'locked':
        if (p.owned) {
          eyebrow = 'Next: ' + title;
          label = 'Locked — keep scoring to unlock it';
        } else {
          eyebrow = 'Next: ' + title;
          label = 'Unlock more scenes';
        }
        icon = 'lock'; break;
      case 'done':
        eyebrow = 'All caught up';
        label = 'You’ve passed every scene — back to the menu';
        thumb = ''; icon = 'done'; break;
      default:
        if (p.endReview) eyebrow = 'Review done · next scene';
    }

    const btn = document.getElementById('spNextBtn');
    btn.classList.toggle('is-review', p.kind === 'review-start');
    btn.classList.toggle('is-locked', p.kind === 'locked');
    document.getElementById('spNextEyebrow').textContent = eyebrow;
    document.getElementById('spNextTitle').textContent = label;
    const th = document.getElementById('spNextThumb');
    th.innerHTML = thumb || (icon === 'lock'
      ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>'
      : '<span aria-hidden="true">🎬</span>');
    bar.hidden = false;
  }

  function hide() {
    const bar = document.getElementById('spNext');
    if (bar) bar.hidden = true;
  }

  // ── acting on it ─────────────────────────────────────────────────────────
  function go() {
    const current = ctx.getActiveScene();
    if (!current) return;
    const p = plan(current);
    const s = load();

    if (p.kind === 'review-start') { openReviewSheet(p.queue, current); return; }
    if (p.kind === 'review-resume') { open(p.id); return; }
    if (p.kind === 'review-next') { s.review.idx += 1; save(s); open(p.id); return; }

    if (p.endReview || p.blockClear) {
      s.review = null; s.done = []; save(s);
      if (p.blockClear) toast('10 scenes done — every one passed! 🔥');
      if (p.endReview) toast('Review done — back on the path 🎬');
    }

    if (p.kind === 'scene' || p.kind === 'daily') { open(p.id); return; }
    if (p.kind === 'locked') { ctx.close(); if (p.owned) ctx.showLevels(); else ctx.showUpgrade(); return; }
    ctx.close();
  }

  function open(sid) {
    const sc = sceneOf(sid);
    if (!sc) return;
    ctx.openScene(sid, sc);
  }

  // ── the review sheet ─────────────────────────────────────────────────────
  function openReviewSheet(queue, resumeAfter) {
    const ov = document.getElementById('spReviewSheet');
    if (!ov) return;
    document.getElementById('spRsCount').textContent =
      queue.length + ' of your last ' + BLOCK + ' scene' + (queue.length === 1 ? ' is' : 's are') +
      ' still under ' + ctx.passScore + '%.';
    document.getElementById('spRsList').innerHTML = queue.map(function(id) {
      const sc = sceneOf(id) || {};
      const b = bestScore(id);
      return '<li class="sp-rs-item"><span class="sp-rs-thumb">' + thumbHTML(id) + '</span>' +
        '<span class="sp-rs-name">' + esc(sc.title || id) + '</span>' +
        '<span class="sp-rs-score">' + (b !== null ? Math.round(b) + '%' : '—') + '</span></li>';
    }).join('');

    document.getElementById('spRsStart').onclick = function() {
      const s = load();
      s.review = { queue: queue.slice(), idx: 0, resumeAfter: resumeAfter };
      save(s);
      closeReviewSheet();
      open(queue[0]);
    };
    document.getElementById('spRsSkip').onclick = function() {
      const s = load();
      s.review = null; s.done = []; save(s);
      closeReviewSheet();
      const n = nextInPath(resumeAfter);
      if (n.kind === 'scene') open(n.id); else render();
    };
    ov.classList.add('open');
    ov.setAttribute('aria-hidden', 'false');
    setTimeout(function() { document.getElementById('spRsStart').focus(); }, 50);
  }

  function closeReviewSheet() {
    const ov = document.getElementById('spReviewSheet');
    if (!ov) return;
    ov.classList.remove('open');
    ov.setAttribute('aria-hidden', 'true');
  }

  function isReviewSheetOpen() {
    const ov = document.getElementById('spReviewSheet');
    return !!(ov && ov.classList.contains('open'));
  }

  // Its own toast: the app's mission toasts sit at z-index 9999, under the
  // scene overlay (99999) that is open whenever this one fires.
  function toast(msg) {
    let t = document.getElementById('spToast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'spToast';
      t.className = 'sp-toast';
      t.setAttribute('role', 'status');
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.classList.add('on');
    clearTimeout(toast._t);
    toast._t = setTimeout(function() { t.classList.remove('on'); }, 2600);
  }

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function init(options) {
    ctx = options;
    const btn = document.getElementById('spNextBtn');
    if (btn) btn.addEventListener('click', go);
    const ov = document.getElementById('spReviewSheet');
    if (ov) ov.addEventListener('click', function(e) { if (e.target === ov) closeReviewSheet(); });
  }

  window.MirrorScenePath = {
    init: init,
    onScored: onScored,
    onOpen: onOpen,
    render: render,
    hide: hide,
    closeReviewSheet: closeReviewSheet,
    isReviewSheetOpen: isReviewSheetOpen,
    _plan: plan,            // exposed for tests
  };
})();
