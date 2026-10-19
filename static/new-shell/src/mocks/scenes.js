// Four of the real scenes from docs/original-scenes.json, in the shape
// scene-adapter.js produces. They were four films with their dialogue quoted
// verbatim; Mirror owns everything here instead.
//
// `imageUrl` points at the stage 3 art path. Until those files exist the mock
// renders the same broken-image fallback a real scene would, which is the
// honest preview.
export const mockScenes = [
  {
    id: 'last_train',
    title: 'Last Train',
    genre: 'Drama',
    emotion: 'Hurt but firm',
    line: "If you get on that train, don't come back.",
    level: 1,
    levelName: 'Beginner',
    difficulty: 'Beginner',
    runtime: 'Scene',
    targetScore: 60,
    personalBest: 91,
    locked: false,
    isDaily: false,
    tags: ['Drama', 'Beginner'],
    imageUrl: '/static/scenes/art/last_train.webp',
  },
  {
    id: 'countdown',
    title: 'Countdown',
    genre: 'Action',
    emotion: 'Urgent',
    line: 'We have ninety seconds. Grab the bag and run!',
    level: 1,
    levelName: 'Beginner',
    difficulty: 'Beginner',
    runtime: 'Scene',
    targetScore: 60,
    personalBest: 86,
    locked: false,
    isDaily: true,
    tags: ['Action', 'Beginner'],
    imageUrl: '/static/scenes/art/countdown.webp',
  },
  {
    id: 'night_shift',
    title: 'Night Shift',
    genre: 'Drama',
    emotion: 'Quiet intensity',
    line: 'I work two jobs so you can have the chances I never had.',
    level: 2,
    levelName: 'Intermediate',
    difficulty: 'Intermediate',
    runtime: 'Scene',
    targetScore: 70,
    personalBest: 82,
    locked: false,
    isDaily: false,
    tags: ['Drama', 'Intermediate'],
    imageUrl: '/static/scenes/art/night_shift.webp',
  },
  {
    id: 'the_bluff',
    title: 'The Bluff',
    genre: 'Crime',
    emotion: 'Smug',
    line: "You're bluffing. I can read you like a book, and it's a very short book.",
    level: 3,
    levelName: 'Advanced',
    difficulty: 'Advanced',
    runtime: 'Scene',
    targetScore: 70,
    personalBest: null,
    locked: true,
    isDaily: false,
    tags: ['Crime', 'Advanced'],
    imageUrl: '/static/scenes/art/the_bluff.webp',
  },
];

export const featuredScene = mockScenes[1];
