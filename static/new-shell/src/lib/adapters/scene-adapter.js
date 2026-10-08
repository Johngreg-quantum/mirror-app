// Maps `/api/scene-config` plus optional `/api/progress` and `/api/daily`
// into the scene and level view models. This is read-only display
// shaping; media, analyze, scoring, and unlock mutations remain out of scope.
const LEVEL_LABELS = {
  1: 'Beginner',
  2: 'Intermediate',
  3: 'Advanced',
};

function buildLevelMap(levels = []) {
  return levels.reduce((map, levelDef) => {
    (levelDef.scenes || []).forEach((sceneId) => {
      map[sceneId] = levelDef.level;
    });
    return map;
  }, {});
}

function formatRuntime(scene) {
  const start = Number(scene?.ui?.clip_start || 0);
  const end = Number(scene?.ui?.clip_end || 0);
  const seconds = Math.max(0, end - start);

  if (!seconds) {
    return 'Clip';
  }

  return `${seconds}s`;
}

// WebP variants; the .jpg originals remain on disk as a manual fallback.
// See optimize_cards.py for how both are generated.
function posterFallback(level) {
  if (level === 1) return '/static/beginner-card.webp';
  if (level === 2) return '/static/intermediate-card.webp';
  return '/static/advanced-card.webp';
}

// The Daily Take is three scenes. Testing `daily.scene_id` alone would mark two
// of them locked and not-daily while can_play_scene() accepts all three — the
// client refusing what the API allows.
function dailySceneIdSet(daily) {
  if (Array.isArray(daily?.scene_ids) && daily.scene_ids.length) {
    return new Set(daily.scene_ids);
  }

  return new Set(daily?.scene_id ? [daily.scene_id] : []);
}

export function adaptSceneConfig(rawConfig, { progress = null, daily = null } = {}) {
  const levelMap = buildLevelMap(rawConfig?.levels || []);
  const unlocked = new Set(progress?.unlocked_scenes || []);
  const hasProgress = !!progress;
  const dailyIds = dailySceneIdSet(daily);
  // Mirrors ENFORCE_ENTITLEMENTS. While it is false the server refuses nothing,
  // so no lock here may differ from plain score progression.
  const enforcing = !!rawConfig?.enforce_entitlements;

  const scenes = Object.entries(rawConfig?.scenes || {}).map(([id, scene]) => {
    const level = levelMap[id] || 1;
    const personalBest = progress?.best_scores?.[id];

    return {
      id,
      title: scene.movie,
      film: scene.movie,
      year: scene.year,
      quote: scene.quote,
      actor: scene.actor,
      level,
      levelName: LEVEL_LABELS[level] || `Level ${level}`,
      difficulty: scene.difficulty || LEVEL_LABELS[level] || 'Scene',
      runtime: formatRuntime(scene),
      targetScore: level > 1 ? 70 : 60,
      personalBest: personalBest ? Math.round(personalBest) : null,
      // No daily line is ever locked: can_play_scene() lets all three through
      // whatever the user owns, so locking one here would contradict the server
      // and disable analyze on a scene the API would have accepted. Only once
      // enforcement is on -- before that the daily follows ordinary score
      // progression, as it always has.
      locked: hasProgress
        ? (!unlocked.has(id) && !(enforcing && dailyIds.has(id)))
        : false,
      isDaily: dailyIds.has(id),
      tags: [scene.actor, scene.difficulty].filter(Boolean),
      imageUrl: scene?.ui?.poster_image || posterFallback(level),
      source: scene,
    };
  });

  const levels = (rawConfig?.levels || []).map((levelDef) => {
    const sceneIds = levelDef.scenes || [];
    const unlockedScenes = hasProgress
      ? sceneIds.filter((sceneId) => unlocked.has(sceneId)).length
      : sceneIds.length;

    return {
      id: `level-${levelDef.level}`,
      level: levelDef.level,
      title: LEVEL_LABELS[levelDef.level] || `Level ${levelDef.level}`,
      description: levelDef.level === 1
        ? 'Short, clear lines for building confidence.'
        : 'Unlocked by stronger scores on earlier scenes.',
      status: hasProgress && unlockedScenes === 0 ? 'locked' : hasProgress && unlockedScenes === sceneIds.length ? 'complete' : 'active',
      unlockedScenes,
      totalScenes: sceneIds.length,
      requiredScore: levelDef.unlock_score || 0,
      firstUnlockedSceneId: sceneIds.find((sceneId) => !hasProgress || unlocked.has(sceneId)) || sceneIds[0],
    };
  });

  return { scenes, levels };
}

export function findSceneById(scenes, sceneId) {
  return scenes.find((scene) => scene.id === sceneId) || null;
}
