(function() {
  function refreshSceneCardsSurface(options) {
    options.renderSceneCardsDisplay({
      createCardElement: options.createCardElement,
      grids: options.grids,
      scenes: options.scenes,
      setTextIfPresent: options.setTextIfPresent,
      userProgress: options.userProgress,
    });
  }

  function refreshLevelBarSurface(options) {
    const pct = options.renderLevelBarDisplay({
      levelNames: options.levelNames,
      refs: options.refs,
      userProgress: options.userProgress,
    });
    if (pct === null) return;
    options.requestAnimationFrameFn(function() {
      options.requestAnimationFrameFn(function() {
        const fill = options.getFillEl();
        if (fill) fill.style.width = `${pct}%`;
      });
    });
  }

  function refreshLevelCardStatsSurface(options) {
    const best = options.getBestScores(options.userProgress);
    const unlocked = options.getUnlockedSceneIds(options.userProgress, options.getDefaultUnlockedScenes());
    options.updateLevelCardStatsDisplay({
      bestScores: best,
      formatAvgPb: options.formatAvgPb,
      getPositiveSceneScores: options.getPositiveSceneScores,
      hasUnlockedScene: options.hasUnlockedScene,
      levels: options.levels,
      setDisplayIfPresent: options.setDisplayIfPresent,
      setTextIfPresent: options.setTextIfPresent,
      unlockedSceneIds: unlocked,
    });
  }

  window.MIRROR_APP_PROGRESS_ORCHESTRATION = {
    refreshLevelBarSurface,
    refreshLevelCardStatsSurface,
    refreshSceneCardsSurface,
  };
})();
