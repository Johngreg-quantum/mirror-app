// Maps `/api/daily` into the Daily Take card model.
//
// The daily is three short lines now. Which line is next and how many are done
// come from the server — this app has two frontends that share no JavaScript,
// so anything computed here would have to be computed again in static/app.js,
// and the two copies drifting is a real failure in this repo rather than a
// hypothetical one. The fallbacks below only cover the anonymous response,
// where there is no progress to report and line 1 is the only sensible answer.
function formatCountdown(secs) {
  const safe = Math.max(0, Number(secs || 0));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);

  return { hours, minutes };
}

function lineSceneIds(rawDaily) {
  if (Array.isArray(rawDaily.scene_ids) && rawDaily.scene_ids.length) {
    return rawDaily.scene_ids;
  }

  return rawDaily.scene_id ? [rawDaily.scene_id] : [];
}

export function adaptDailyChallenge(rawDaily, scenes = [], profile = null) {
  if (!rawDaily) {
    return null;
  }

  const sceneIds = lineSceneIds(rawDaily);
  const doneIds = Array.isArray(rawDaily.done_scene_ids) ? rawDaily.done_scene_ids : [];
  const sceneFor = (sceneId, index) => scenes.find((item) => item.id === sceneId) || {
    id: sceneId,
    title: rawDaily.scenes?.[index]?.movie || sceneId,
    film: rawDaily.scenes?.[index]?.movie || sceneId,
    year: rawDaily.scenes?.[index]?.year || '',
    quote: rawDaily.scenes?.[index]?.quote || '',
    actor: rawDaily.scenes?.[index]?.actor || '',
    levelName: rawDaily.scenes?.[index]?.difficulty || 'Daily',
    difficulty: rawDaily.scenes?.[index]?.difficulty || 'Daily',
    runtime: 'Clip',
    targetScore: 70,
    personalBest: null,
    locked: false,
    isDaily: true,
    imageUrl: rawDaily.scenes?.[index]?.ui?.poster_image || '/static/beginner-card.webp',
  };

  const lines = sceneIds.map((sceneId, index) => {
    const serverLine = (rawDaily.lines || [])[index] || null;
    const done = serverLine ? Boolean(serverLine.done) : doneIds.includes(sceneId);

    return {
      index: index + 1,
      sceneId,
      scene: sceneFor(sceneId, index),
      done,
      score: typeof serverLine?.score === 'number' ? Math.round(serverLine.score) : null,
    };
  });

  const linesDone = lines.filter((line) => line.done).length;
  const allDone = lines.length > 0 && linesDone === lines.length;
  const nextLine = lines.find((line) => !line.done) || null;
  const { hours, minutes } = formatCountdown(rawDaily.secs_until_reset);

  return {
    id: `daily-${rawDaily.date}`,
    // The line to record next, or the first one once the set is finished — the
    // card's button always needs a destination.
    scene: (nextLine || lines[0] || { scene: null }).scene,
    lines,
    lineTotal: lines.length,
    linesDone,
    allDone,
    nextSceneId: rawDaily.next_scene_id || nextLine?.sceneId || sceneIds[0] || '',
    progressLabel: lines.length > 1 ? `${linesDone} of ${lines.length} lines` : 'One line',
    resetLabel: `New lines in ${hours}h ${minutes}m`,
    // The countdown is to the user's own midnight, so this only ever fires late
    // at night -- when it is genuinely useful to know the set is about to be
    // replaced, rather than mid-evening because a server somewhere rolled over.
    resetWarning: hours < 1 && !allDone && linesDone > 0
      ? `These lines change in ${minutes}m — finish the set to keep your streak.`
      : '',
    rewardPoints: 250,
    streakBonus: `${rawDaily.bonus_multiplier || 1}x on the full set`,
    status: allDone
      ? 'Done today'
      : (profile?.dailyStatus || (linesDone > 0 ? 'Resume the set' : 'Ready to record')),
    source: rawDaily,
  };
}
