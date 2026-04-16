(function() {
  function renderDivCardDisplay(options) {
    const profile = options.profile;
    const refs = options.refs || {};
    const setHtml = options.setHtml;

    if (!profile || !profile.division) {
      refs.card.classList.remove('on');
      return;
    }

    refs.card.classList.add('on');
    const division = profile.division;
    refs.badge.textContent = division.name.slice(0, 3).toUpperCase();
    refs.badge.style.color = division.color;
    refs.badge.style.borderColor = division.color;
    refs.badge.style.background = division.color + '18';
    refs.nameEl.textContent = division.name;
    refs.nameEl.style.color = division.color;

    const streakTxt = profile.streak > 0
      ? ` &nbsp;&#128293; ${profile.streak}-day streak`
      : '';
    setHtml('divCardPts', `${profile.total_points} total points${streakTxt}`);

    if (profile.next_division) {
      refs.nextEl.innerHTML = `<strong>${profile.points_to_next}</strong>pts to ${profile.next_division.name}`;
    } else {
      refs.nextEl.innerHTML = `<strong>MAX</strong>rank achieved`;
    }
  }

  function renderProgressDashboardDisplay(options) {
    const history = options.history || [];
    const stats = options.stats || {};
    const refs = options.refs || {};
    const helpers = options.helpers || {};
    const setHtml = options.setHtml;
    const setText = options.setText;

    const avg = stats.avg_score || 0;
    const circleColor = avg >= 70 ? 'var(--green)' : avg >= 40 ? 'var(--gold)' : 'var(--red)';
    setHtml('progCircle', helpers.buildCircleSVG(avg, circleColor));

    setHtml(
      'progBest',
      stats.best_score > 0 ? `${stats.best_score}<sup style="font-size:16px;opacity:.6">%</sup>` : '—'
    );
    setText('progScenes', stats.unique_scenes || 0);

    const improvementSign = stats.improvement > 0 ? '+' : '';
    refs.improvementEl.innerHTML = `${improvementSign}${stats.improvement}<sup style="font-size:16px;opacity:.6">%</sup>`;
    refs.improvementEl.className = `prog-stat-val${stats.improvement > 0 ? ' green' : stats.improvement < 0 ? ' red' : ''}`;

    setText(
      'historyLabel',
      `Score History  —  ${stats.total_attempts} recording${stats.total_attempts !== 1 ? 's' : ''}`
    );

    if (!history.length) {
      refs.historyListEl.innerHTML = `<div class="history-empty">No recordings yet — start acting!</div>`;
      return;
    }

    const improved = helpers.computeImprovedIds(history);
    refs.historyListEl.innerHTML = history.map(function(item) {
      return helpers.buildHistoryItemHTML({
        color: item.sync_score >= 85 ? '#06d6a0' : item.sync_score >= 65 ? '#ffd166' : item.sync_score >= 40 ? '#f4a261' : '#e63946',
        date: item.created_at ? new Date(item.created_at).toLocaleDateString() : '—',
        isImproved: improved.has(item.id),
        movie: item.movie,
        score: item.sync_score,
      });
    }).join('');
  }

  function renderPersonalBestsDisplay(options) {
    const history = options.history || [];
    const bestScores = options.bestScores || {};
    const scenes = options.scenes || {};
    const pbEl = options.pbEl;
    const helpers = options.helpers || {};

    if (!bestScores || !Object.keys(bestScores).length) {
      pbEl.innerHTML = '<div class="pb-empty">No scores yet \u2014 start recording!</div>';
      return;
    }

    const latestByScene = {};
    const prevByScene = {};
    for (const item of history) {
      if (!(item.scene_id in latestByScene)) latestByScene[item.scene_id] = item.sync_score;
      else if (!(item.scene_id in prevByScene)) prevByScene[item.scene_id] = item.sync_score;
    }

    const sorted = Object.entries(bestScores).sort(([, a], [, b]) => b - a);
    pbEl.innerHTML = sorted.map(function(entry, idx) {
      const sid = entry[0];
      const score = entry[1];
      const latest = latestByScene[sid];
      const prev = prevByScene[sid];
      return helpers.buildPersonalBestItemHTML({
        color: helpers.getSceneColor(sid, 'var(--gold)'),
        improved: latest !== undefined && prev !== undefined && latest > prev,
        movie: scenes[sid] && scenes[sid].movie ? scenes[sid].movie : sid,
        rank: idx + 1,
        score: Math.round(score),
      });
    }).join('');
  }

  window.MIRROR_PROGRESS_DASHBOARD_DOMAIN = {
    renderDivCardDisplay,
    renderPersonalBestsDisplay,
    renderProgressDashboardDisplay,
  };
})();
