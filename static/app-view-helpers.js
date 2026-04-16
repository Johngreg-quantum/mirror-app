(function() {
  const APP_HELPERS = window.MIRROR_APP_HELPERS || {};
  const { getDivision, timeAgo } = APP_HELPERS;

  function btnRecordHTML() {
    return `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="9"/></svg> Record`;
  }

  function btnPlayHTML() {
    return `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Playback`;
  }

  function btnStopPlayHTML() {
    return `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop`;
  }

  function buildPanelHTML(rows) {
    if (!rows.length) {
      return `<div style="color:var(--muted);text-align:center;padding:36px 14px;font-size:13px">No scores yet — be the first!</div>`;
    }

    const MEDAL = ['', '🥇', '🥈', '🥉'];
    const RCLS  = ['', 'gold', 'silver', 'bronze'];

    const trs = rows.map((s, i) => {
      const rank  = i + 1;
      const c     = s.sync_score >= 85 ? '#06d6a0' : s.sync_score >= 65 ? '#ffd166' : '#e63946';
      const div   = s.division || getDivision(s.user_points || 0);
      const badge = s.username
        ? `<span class="div-badge" style="background:${div.color}18;color:${div.color}">${div.name}</span>`
        : '';
      const streak = s.streak > 0
        ? ` <span class="streak-badge">&#128293;${s.streak}</span>`
        : '';
      const name  = s.username
        ? `<strong>${s.username}</strong> ${badge}${streak}`
        : `<span style="color:var(--muted)">—</span>`;
      const pts = s.user_points
        ? `<span style="color:var(--muted);font-size:11px">${s.user_points}pts</span>`
        : '—';
      return `<tr>
      <td class="rank-num ${rank <= 3 ? RCLS[rank] : ''}" style="width:52px">${rank <= 3 ? MEDAL[rank] : rank}</td>
      <td>${name}</td>
      <td style="width:90px"><span class="chip" style="background:${c}18;color:${c}">${s.sync_score}%</span></td>
      <td style="width:72px;text-align:right">${pts}</td>
      <td style="color:var(--muted);white-space:nowrap;width:80px">${timeAgo(s.created_at)}</td>
    </tr>`;
    }).join('');

    return `<table class="lb-table">
    <thead><tr>
      <th style="width:52px">Rank</th><th>Name</th>
      <th style="width:90px">Score</th><th style="width:72px;text-align:right">Points</th>
      <th style="width:80px">When</th>
    </tr></thead>
    <tbody>${trs}</tbody>
  </table>`;
  }

  window.MIRROR_APP_VIEW_HELPERS = {
    buildPanelHTML,
    btnPlayHTML,
    btnRecordHTML,
    btnStopPlayHTML,
  };
})();
