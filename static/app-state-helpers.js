(function() {
  function getBestScores(progress) {
    return (progress && progress.best_scores) ? progress.best_scores : {};
  }

  function getUnlockedSceneIds(progress, defaultUnlockedSceneIds) {
    return (progress && progress.unlocked_scenes && progress.unlocked_scenes.length)
      ? progress.unlocked_scenes
      : defaultUnlockedSceneIds;
  }

  function getPositiveSceneScores(sceneIds, bestScores) {
    return sceneIds.map(function(sid) { return bestScores[sid]; }).filter(function(v) { return v > 0; });
  }

  function hasUnlockedScene(sceneIds, unlockedSceneIds) {
    return sceneIds.some(function(sid) { return unlockedSceneIds.includes(sid); });
  }

  function findFirstUnlockedSceneId(sceneIds, unlockedSceneIds) {
    return sceneIds.find(function(sid) { return unlockedSceneIds.includes(sid); });
  }

  function computeImprovedIds(history) {
    const improved = new Set();
    for (let i = 0; i < history.length; i++) {
      for (let j = i + 1; j < history.length; j++) {
        if (history[j].scene_id === history[i].scene_id) {
          if (history[i].sync_score > history[j].sync_score) improved.add(history[i].id);
          break;
        }
      }
    }
    return improved;
  }

  window.MIRROR_APP_STATE_HELPERS = {
    computeImprovedIds,
    findFirstUnlockedSceneId,
    getBestScores,
    getPositiveSceneScores,
    getUnlockedSceneIds,
    hasUnlockedScene,
  };
})();
