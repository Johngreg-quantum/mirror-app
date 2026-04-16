// Maps authenticated read-only `/api/progress`, `/api/profile`, and
// `/api/history` responses into dashboard cards. Mutating score, points, PB,
// and streak behavior remains server-owned.
export function adaptProfile(rawProfile) {
  if (!rawProfile) {
    return null;
  }

  return {
    displayName: rawProfile.username || 'Performer',
    handle: rawProfile.username ? `@${rawProfile.username}` : '@performer',
    level: null,
    levelTitle: rawProfile.division?.name || 'Unranked',
    points: rawProfile.total_points || 0,
    nextLevelPoints: rawProfile.next_division?.min || rawProfile.total_points || 0,
    division: rawProfile.division?.name || 'Unranked',
    streakDays: rawProfile.streak || 0,
    dailyStatus: rawProfile.daily_done_today ? 'Completed today' : 'Ready',
    rank: null,
    source: rawProfile,
  };
}

export function adaptProgressSummary({ progress, profile, history }) {
  const stats = history?.stats || {};
  const sceneStats = profile?.scene_stats || {};

  return {
    scoreAverage: Math.round(stats.avg_score || 0),
    scenesCompleted: stats.unique_scenes || Object.keys(sceneStats).length,
    personalBests: Object.keys(progress?.best_scores || {}).length,
    unlockedScenes: progress?.unlocked_scenes?.length || 0,
    weeklyMinutes: Math.max(0, Math.round((stats.total_attempts || 0) * 0.75)),
    nextUnlockScore: progress?.next_level?.required_score || 0,
  };
}

export function adaptPersonalBests({ progress, scenes }) {
  return Object.entries(progress?.best_scores || {})
    .map(([sceneId, score]) => {
      const scene = scenes.find((item) => item.id === sceneId);
      return {
        sceneTitle: scene?.title || sceneId,
        film: scene?.film || sceneId,
        score: Math.round(score),
        date: 'Best',
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
}

export function adaptRecentHistory(history) {
  return (history?.history || []).slice(0, 6).map((item) => ({
    sceneTitle: item.movie || item.scene_id,
    score: Math.round(item.sync_score || 0),
    delta: '',
    result: item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Recent take',
  }));
}

export function adaptFocusAreas(history) {
  if (!history?.history?.length) {
    return ['Record a first take to unlock coaching signals.'];
  }

  return [
    'Compare recent takes against your current personal bests.',
    'Replay lower-scoring scenes before moving to the next level.',
    'Use the daily challenge to keep streak and points surfaces fresh.',
  ];
}
