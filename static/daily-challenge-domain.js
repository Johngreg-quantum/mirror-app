(function() {
  // Which line to practise next, and how far through the set we are. The server
  // answers both on /api/daily for a signed-in caller -- this only falls back
  // to the first line when the response is the anonymous one, so the "which
  // line is next" rule exists once, on the server, and not twice in two shells
  // that share no code.
  function dailyLines(daily) {
    const ids = Array.isArray(daily && daily.scene_ids) && daily.scene_ids.length
      ? daily.scene_ids
      : (daily && daily.scene_id ? [daily.scene_id] : []);
    const done = Array.isArray(daily && daily.done_scene_ids) ? daily.done_scene_ids : [];
    const nextId = (daily && daily.next_scene_id) || ids.find(id => !done.includes(id)) || ids[0] || '';
    return {
      ids: ids,
      done: done,
      nextId: nextId,
      nextIndex: Math.max(0, ids.indexOf(nextId)),
      doneCount: done.length,
      total: ids.length,
      allDone: ids.length > 0 && done.length >= ids.length,
    };
  }

  function renderDailyCardDisplay(options) {
    const daily = options.daily || {};
    const scenes = options.scenes || {};
    const refs = options.refs || {};

    const lines = dailyLines(daily);
    // The next line, not the first: every daily entry point resumes the set.
    const sceneIndex = lines.nextIndex;
    const scene = (daily.scenes && daily.scenes[sceneIndex])
      || scenes[lines.nextId]
      || daily.scene
      || {};
    refs.movieEl.textContent = scene.movie || lines.nextId;
    refs.quoteEl.textContent = scene.quote ? `\u201c${scene.quote}\u201d` : '';
    refs.actorEl.textContent = scene.actor || '';
    refs.levelEl.textContent = scene.difficulty || '';
    refs.levelEl.className = `badge ${(scene.difficulty || '').toLowerCase()}`;
    refs.sectionEl.classList.toggle('on', true);
  }

  // The dots on the hero chip. Rendered from the same data as everything else,
  // so a line recorded on the other shell shows up here on the next load.
  function renderDailyProgressDisplay(options) {
    const daily = options.daily || {};
    const refs = options.refs || {};
    const createElement = options.createElement;
    const lines = dailyLines(daily);

    if (refs.dotsEl) {
      refs.dotsEl.innerHTML = '';
      lines.ids.forEach((id, i) => {
        const dot = createElement('span');
        const isDone = lines.done.includes(id);
        dot.className = 'hero-chip-dot'
          + (isDone ? ' done' : '')
          + (!isDone && i === lines.nextIndex ? ' next' : '');
        refs.dotsEl.appendChild(dot);
      });
    }

    if (refs.labelEl) {
      refs.labelEl.textContent = lines.total <= 1
        ? 'Daily Challenge'
        : (lines.allDone
            ? 'Daily Take done'
            : `Daily Take \u00b7 ${lines.doneCount} of ${lines.total}`);
    }
  }

  function renderStreakCardDisplay(options) {
    const streak = options.streak;
    const doneToday = options.doneToday;
    const refs = options.refs || {};
    const days = options.days;
    const getNow = options.getNow;
    const createElement = options.createElement;
    const getStreakMessage = options.getStreakMessage;

    refs.numberEl.textContent = streak;

    const now = getNow();
    const dotRow = refs.dotRowEl;
    dotRow.innerHTML = '';

    for (let i = 6; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      const dayLbl = days[d.getDay()];

      let completed = false;
      if (doneToday) completed = i < streak;
      else completed = i >= 1 && i <= streak;
      const isToday = i === 0;

      const dot = createElement('div');
      dot.className = 'streak-dot-col';
      const dotInner = createElement('div');
      dotInner.className = 'streak-dot' + (isToday ? ' today' : completed ? ' done' : '');
      dotInner.textContent = completed ? '\u2714' : (isToday ? '\u2605' : '');
      const dotLbl = createElement('div');
      dotLbl.className = 'streak-dot-lbl';
      dotLbl.textContent = dayLbl;
      dot.appendChild(dotInner);
      dot.appendChild(dotLbl);
      dotRow.appendChild(dot);
    }

    refs.messageEl.textContent = getStreakMessage(streak, doneToday);
  }

  function renderDailyCompleteDisplay(options) {
    const refs = options.refs || {};
    const overlay = refs.overlayEl;
    if (!overlay) return;
    refs.pointsEl.textContent = options.ptsText;
    overlay.style.display = '';
  }

  window.MIRROR_DAILY_CHALLENGE_DOMAIN = {
    dailyLines,
    renderDailyCardDisplay,
    renderDailyCompleteDisplay,
    renderDailyProgressDisplay,
    renderStreakCardDisplay,
  };
})();
