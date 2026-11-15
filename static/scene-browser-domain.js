(function() {
  function renderSceneCardsDisplay(options) {
    const scenes = options.scenes || {};
    const grids = options.grids || {};
    const createCardElement = options.createCardElement;
    const userProgress = options.userProgress || {};
    const setTextIfPresent = options.setTextIfPresent;

    Object.values(grids).forEach(function(grid) {
      if (grid) grid.innerHTML = '';
    });

    Object.entries(scenes).forEach(function(entry) {
      const sceneId = entry[0];
      const scene = entry[1];
      const target = grids[scene.difficulty] || grids.Beginner;
      if (target) target.appendChild(createCardElement(sceneId, scene));
    });

    setTextIfPresent('lockIntermediate', userProgress.level >= 2 ? '' : 'Unlock at 60%');
    setTextIfPresent('lockAdvanced', userProgress.level >= 3 ? '' : 'Unlock at 70%');
  }

  function renderLevelBarDisplay(options) {
    const userProgress = options.userProgress || {};
    const levelNames = options.levelNames || [];
    const refs = options.refs || {};

    refs.levelNumEl.textContent = userProgress.level;
    const nextLevel = userProgress.next_level;

    if (!nextLevel) {
      refs.detailsEl.innerHTML = `<span class="level-maxed">&#127916; Todas las escenas abiertas: llegaste al nivel m\u00e1s alto</span>`;
      return null;
    }

    const fillPercent = nextLevel.required_score > 0
      ? Math.min(100, (nextLevel.best_score / nextLevel.required_score) * 100)
      : 100;

    refs.detailsEl.innerHTML = `
    <div class="level-next-text">
      Saca <strong>${nextLevel.required_score}%</strong> en una escena del
      <strong>Nivel ${userProgress.level}</strong> para abrir el
      <strong>Nivel ${nextLevel.level}</strong>
    </div>
    <div class="level-track"><div class="level-fill" id="lvlFill"></div></div>
    <div class="level-score-text">
      Tu mejor: <strong>${nextLevel.best_score}%</strong> &nbsp;/&nbsp; meta ${nextLevel.required_score}%
    </div>`;

    return fillPercent;
  }

  window.MIRROR_SCENE_BROWSER_DOMAIN = {
    renderLevelBarDisplay,
    renderSceneCardsDisplay,
  };
})();
