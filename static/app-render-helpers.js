(function() {
  function buildHistoryItemHTML(item) {
    return `<div class="history-item${item.isImproved ? ' improved' : ''}">
      <span class="history-movie">${item.movie}</span>
      <span class="history-date">${item.date}</span>
      <span class="history-score" style="color:${item.color}">${item.score}<sup style="font-size:11px;opacity:.7">%</sup></span>
      ${item.isImproved ? '<span class="history-improved-badge">\u2191 Improved</span>' : ''}
    </div>`;
  }

  function buildPersonalBestItemHTML(item) {
    return `<div class="pb-item">
      <div class="pb-rank">#${item.rank}</div>
      <div class="pb-movie">${item.movie}</div>
      ${item.improved ? '<div class="pb-arrow">&#8593;</div>' : ''}
      <div class="pb-score" style="color:${item.color}">${item.score}<sup>%</sup></div>
    </div>`;
  }

  function buildLevelPanelCountHTML(sceneCount) {
    return `<strong>${sceneCount}</strong> scenes`;
  }

  function buildLevelPanelSceneCardHTML(card) {
    const posterHTML = card.poster
      ? `<img src="${card.poster}" alt="${card.movie}" loading="lazy">`
      : `<div class="sc-poster-ph" style="background:linear-gradient(150deg,${card.color},#111)"></div>`;

    let scoreHTML = '';
    if (card.locked) {
      scoreHTML = '<span class="sc-lock-badge"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg></span>';
    } else if (card.pb) {
      const scoreColor = card.pb >= 85 ? '#06d6a0' : card.pb >= 65 ? '#ffd166' : '#C9A84C';
      scoreHTML = `<span class="sc-score" style="color:${scoreColor};background:${scoreColor}18;border-color:${scoreColor}44">${card.pb}%</span>`;
    }

    return `<div class="sc-card${card.locked ? ' locked' : ''}"${card.locked ? '' : ` onclick="selectScene('${card.sid}')"`}>
      <div class="sc-poster">
        ${posterHTML}
        <div class="sc-poster-overlay"></div>
        ${card.year ? `<span class="sc-year">${card.year}</span>` : ''}
        ${scoreHTML}
        ${card.locked ? '' : '<div class="sc-play"><svg width="14" height="14" viewBox="0 0 24 24" fill="#fff"><path d="M8 5v14l11-7z"/></svg></div>'}
      </div>
      <div class="sc-info">
        <div class="sc-accent" style="background:${card.color}"></div>
        <div class="sc-movie">${card.movie}</div>
        <div class="sc-quote">&ldquo;${card.quote}&rdquo;</div>
      </div>
    </div>`;
  }

  window.MIRROR_APP_RENDER_HELPERS = {
    buildHistoryItemHTML,
    buildLevelPanelCountHTML,
    buildLevelPanelSceneCardHTML,
    buildPersonalBestItemHTML,
  };
})();
