(function() {
  function openLevelPanelSurface(options) {
    const level = options.levels.find(function(item) { return item.level === options.level; });
    if (!level) return;

    const refs = options.refs || {};
    refs.badgeEl.textContent = level.label;
    refs.badgeEl.className = 'clv-panel-badge ' + level.cls;
    options.setText('clvPanelTitle', 'Level ' + level.level);
    options.setText('clvPanelSub', level.desc);
    options.setTextIfPresent('clvPanelNum', String(level.level).padStart(2, '0'));

    const best = options.getBestScores(options.userProgress);
    const unlocked = options.getUnlockedSceneIds(options.userProgress, options.getDefaultUnlockedScenes());
    const firstScene = options.renderLevelPanelDisplay({
      bestScores: best,
      helpers: options.helpers,
      level: level,
      refs: {
        badgeEl: refs.badgeEl,
        listEl: refs.listEl,
        subEl: refs.subEl,
        titleEl: refs.titleEl,
      },
      scenes: options.scenes,
      unlockedSceneIds: unlocked,
    });

    if (firstScene) {
      refs.playBtn.style.display = '';
      refs.playBtn.onclick = function() { options.onPlayFirstScene(firstScene); };
    } else {
      refs.playBtn.style.display = 'none';
    }

    options.setPanelOpen('clvPanel', 'clvPanelBackdrop', true);
  }

  window.MIRROR_APP_LEVEL_PANEL_ORCHESTRATION = {
    openLevelPanelSurface,
  };
})();
