(function() {
  function renderLeaderboardSurface(options) {
    const data = options.data;
    const scenes = options.scenes;
    const refs = options.refs;
    const sceneIds = Object.keys(scenes);

    refs.tabsEl.innerHTML = refs.panelsEl.innerHTML = '';
    if (!sceneIds.length) return;

    let activeTab = options.activeTab;
    if (!activeTab || !sceneIds.includes(activeTab)) {
      activeTab = sceneIds[0];
      options.setActiveTab(activeTab);
    }

    for (const sid of sceneIds) {
      const scene = scenes[sid] || {};
      const color = options.getSceneColor(sid);
      const rows = data[sid] || [];
      const active = sid === activeTab;

      const tab = options.createElement('button');
      tab.className = 'lb-tab' + (active ? ' active' : '');
      tab.textContent = scene.movie || sid;
      tab.style.setProperty('--tab-color', color);
      tab.addEventListener('click', function() { options.onTabSelected(sid); });
      refs.tabsEl.appendChild(tab);

      const panel = options.createElement('div');
      panel.className = 'lb-panel' + (active ? ' active' : '');
      panel.id = `lb-panel-${sid}`;
      panel.innerHTML = options.buildPanelHTML(rows);
      refs.panelsEl.appendChild(panel);
    }
  }

  function switchLeaderboardTabSurface(options) {
    options.setActiveTab(options.sceneId);
    const ids = Object.keys(options.scenes);
    options.tabs.forEach(function(tab, index) {
      tab.classList.toggle('active', ids[index] === options.sceneId);
    });
    options.panels.forEach(function(panel) {
      panel.classList.toggle('active', panel.id === `lb-panel-${options.sceneId}`);
    });
  }

  window.MIRROR_APP_LEADERBOARD_ORCHESTRATION = {
    renderLeaderboardSurface,
    switchLeaderboardTabSurface,
  };
})();
