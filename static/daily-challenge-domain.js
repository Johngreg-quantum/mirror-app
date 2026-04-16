(function() {
  function renderDailyCardDisplay(options) {
    const daily = options.daily || {};
    const scenes = options.scenes || {};
    const refs = options.refs || {};

    const scene = daily.scene || scenes[daily.scene_id] || {};
    refs.movieEl.textContent = scene.movie || daily.scene_id;
    refs.quoteEl.textContent = scene.quote ? `\u201c${scene.quote}\u201d` : '';
    refs.actorEl.textContent = scene.actor || '';
    refs.levelEl.textContent = scene.difficulty || '';
    refs.levelEl.className = `badge ${(scene.difficulty || '').toLowerCase()}`;
    refs.sectionEl.classList.toggle('on', true);
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
    renderDailyCardDisplay,
    renderDailyCompleteDisplay,
    renderStreakCardDisplay,
  };
})();
