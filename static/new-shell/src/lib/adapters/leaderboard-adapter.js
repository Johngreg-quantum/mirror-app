// Maps `/api/leaderboard` scene-keyed rows into the leaderboard
// panel model. It does not fetch scores or own tab-selection state.
export function adaptLeaderboard(rawLeaderboard, scenes, preferredSceneId) {
  const sceneId = preferredSceneId
    ? preferredSceneId
    : scenes.find((scene) => rawLeaderboard?.[scene.id])?.id || scenes[0]?.id || '';
  const scene = scenes.find((item) => item.id === sceneId);
  const rows = (rawLeaderboard?.[sceneId] || []).map((row, index) => ({
    rank: index + 1,
    name: row.username || 'Anonymous',
    score: Math.round(Number(row.sync_score || 0)),
    note: row.division?.name || `${row.user_points || 0} points`,
  }));

  return {
    activeSceneId: sceneId,
    title: scene?.title || 'Leaderboard',
    tabs: ['Scene', 'Friends', 'Global'],
    rows,
  };
}
