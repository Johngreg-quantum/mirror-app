// Maps `/api/daily` into the daily card model.
export function adaptDailyChallenge(rawDaily, scenes = [], profile = null) {
  if (!rawDaily) {
    return null;
  }

  const scene = scenes.find((item) => item.id === rawDaily.scene_id) || null;
  const secs = Math.max(0, Number(rawDaily.secs_until_reset || 0));
  const hours = Math.floor(secs / 3600);
  const minutes = Math.floor((secs % 3600) / 60);

  return {
    id: `daily-${rawDaily.date}`,
    scene: scene || {
      id: rawDaily.scene_id,
      title: rawDaily.scene?.movie || rawDaily.scene_id,
      film: rawDaily.scene?.movie || rawDaily.scene_id,
      year: rawDaily.scene?.year || '',
      quote: rawDaily.scene?.quote || '',
      actor: rawDaily.scene?.actor || '',
      levelName: rawDaily.scene?.difficulty || 'Daily',
      difficulty: rawDaily.scene?.difficulty || 'Daily',
      runtime: 'Clip',
      targetScore: 70,
      personalBest: null,
      locked: false,
      isDaily: true,
      imageUrl: rawDaily.scene?.ui?.poster_image || '/static/beginner-card.jpg',
    },
    resetLabel: `Resets in ${hours}h ${minutes}m`,
    rewardPoints: 250,
    streakBonus: `${rawDaily.bonus_multiplier || 1}x daily bonus`,
    status: profile?.dailyStatus || 'Ready to record',
    source: rawDaily,
  };
}
