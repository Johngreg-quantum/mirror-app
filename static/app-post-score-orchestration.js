(function() {
  async function refreshPostScoreSurfaces(options) {
    options.setActiveLeaderboardTab(options.activeScene);
    await Promise.all([options.loadScores(), options.loadProgress()]);
    options.renderCards();
    const currentLevel = options.getCurrentLevel();
    if (currentLevel > options.previousLevel) options.showLevelUp(currentLevel);
  }

  window.MIRROR_APP_POST_SCORE_ORCHESTRATION = {
    refreshPostScoreSurfaces,
  };
})();
