(function() {
  function updateLevelCardStatsDisplay(options) {
    const levels = options.levels || [];
    const bestScores = options.bestScores || {};
    const unlockedSceneIds = options.unlockedSceneIds || [];
    const formatAvgPb = options.formatAvgPb;
    const getPositiveSceneScores = options.getPositiveSceneScores;
    const hasUnlockedScene = options.hasUnlockedScene;
    const setTextIfPresent = options.setTextIfPresent;
    const setDisplayIfPresent = options.setDisplayIfPresent;

    levels.forEach(function(level) {
      const scores = getPositiveSceneScores(level.scenes, bestScores);
      setTextIfPresent(`clvPbAvg${level.level}`, formatAvgPb(scores));
      if (level.level === 1) return;
      const isUnlocked = hasUnlockedScene(level.scenes, unlockedSceneIds);
      setDisplayIfPresent(`clvLock${level.level}`, isUnlocked ? 'none' : '');
      setDisplayIfPresent(`clvCta${level.level}`, isUnlocked ? '' : 'none');
    });
  }

  function renderLevelPanelDisplay(options) {
    const level = options.level;
    const bestScores = options.bestScores || {};
    const unlockedSceneIds = options.unlockedSceneIds || [];
    const scenes = options.scenes || {};
    const refs = options.refs || {};
    const helpers = options.helpers || {};

    refs.badgeEl.textContent = level.label;
    refs.badgeEl.className = 'clv-panel-badge ' + level.cls;
    refs.titleEl.textContent = 'Level ' + level.level;
    refs.subEl.textContent = level.desc;
    helpers.setTextIfPresent('clvPanelNum', String(level.level).padStart(2, '0'));
    helpers.setHtmlIfPresent('clvPanelCount', helpers.buildLevelPanelCountHTML(level.scenes.length));

    const scores = helpers.getPositiveSceneScores(level.scenes, bestScores);
    helpers.setTextIfPresent('clvPanelAvg', helpers.formatAvgPb(scores));

    refs.listEl.innerHTML = level.scenes.map(function(sceneId) {
      const scene = scenes[sceneId] || {};
      const pb = bestScores[sceneId] ? Math.round(bestScores[sceneId]) : null;
      return helpers.buildLevelPanelSceneCardHTML({
        color: helpers.getSceneColor(sceneId),
        locked: !unlockedSceneIds.includes(sceneId),
        movie: scene.movie || sceneId,
        pb: pb,
        poster: helpers.getScenePoster(sceneId),
        quote: scene.quote ? scene.quote.slice(0, 55) + (scene.quote.length > 55 ? '\u2026' : '') : '',
        sid: sceneId,
        year: scene.year,
      });
    }).join('');

    return helpers.findFirstUnlockedSceneId(level.scenes, unlockedSceneIds);
  }

  window.MIRROR_LEVEL_PANEL_DOMAIN = {
    renderLevelPanelDisplay,
    updateLevelCardStatsDisplay,
  };
})();
