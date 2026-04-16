(function() {
  const FRONTEND_CONFIG = window.MIRROR_FRONTEND_CONFIG || {};
  const DIVISIONS = FRONTEND_CONFIG.DIVISIONS || [];

  const ES_DICT = {
    a:'un', an:'un', the:'el',
    i:'yo', you:'tú', he:'él', she:'ella', we:'nosotros', they:'ellos', it:'eso',
    me:'mí', him:'él', her:'ella', us:'nos', them:'ellos',
    my:'mi', your:'tu', his:'su', our:'nuestro', their:'su',
    is:'es', are:'son', was:'era', be:'ser', been:'sido',
    im:'soy',
    have:'tener', has:'tiene', had:'había',
    do:'hacer', does:'hace', did:'hizo', done:'hecho',
    will:'voy', would:'sería', can:'puedo', could:'podría', shall:'debo',
    get:'conseguir', got:'conseguí', go:'ir', going:'ir',
    know:'saber', knew:'sabía',
    see:'ver', saw:'vi', seen:'visto',
    find:'encontrar', found:'encontré',
    kill:'matar', killed:'maté',
    stop:'parar', talk:'hablar', talking:'hablando',
    fly:'volar', flying:'volando',
    need:'necesitar', needs:'necesita',
    smash:'aplastar',
    back:'volver',
    come:'venir',
    take:'tomar',
    give:'dar',
    want:'querer',
    think:'pensar',
    look:'mirar',
    tell:'decir',
    say:'decir',
    make:'hacer',
    box:'caja', road:'camino', roads:'caminos',
    people:'gente', person:'persona',
    duvet:'edredón',
    scene:'escena', movie:'película', time:'tiempo',
    man:'hombre', woman:'mujer', world:'mundo',
    life:'vida', day:'día', night:'noche',
    dead:'muerto', okay:'bien', ok:'bien',
    never:'nunca', always:'siempre', now:'ahora',
    here:'aquí', there:'allí', where:'dónde',
    not:'no', no:'no', yes:'sí',
    amateur:'aficionada', kung:'kung', fu:'fu',
    hulk:'hulk', jack:'jack', slick:'listo',
    dont:'no', doesnt:'no', wont:'no', cant:'no puedo',
    were:'íbamos', youre:'eres', ill:'voy a',
    whats:'qué es', gonna:'va a',
    and:'y', in:'en', on:'en', at:'en', of:'de',
    for:'para', to:'a', with:'con', from:'de', about:'sobre',
    but:'pero', or:'o', that:'eso', this:'esto', what:'qué',
  };

  function getDivision(points) {
    for (let i = DIVISIONS.length - 1; i >= 0; i--) {
      if (points >= DIVISIONS[i].min) return DIVISIONS[i];
    }
    return DIVISIONS[0];
  }

  function averageScore(values) {
    return values.length ? Math.round(values.reduce((a, b) => a + b, 0) / values.length) : 0;
  }

  function formatAvgPb(scores) {
    return scores.length ? `Avg PB: ${averageScore(scores)}%` : '';
  }

  function timeAgo(str) {
    if (!str) return '—';
    const diff = Math.floor((Date.now() - new Date(str)) / 1000);
    if (diff <    60) return 'just now';
    if (diff <  3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return new Date(str).toLocaleDateString();
  }

  function streakMessage(streak, doneToday) {
    if (streak === 0 && !doneToday) return 'Complete today’s challenge to start your streak!';
    if (!doneToday && streak > 0)   return `${streak}-day streak — complete today to keep it going →`;
    if (streak === 1)  return 'Day 1 done! Come back tomorrow to build your streak.';
    if (streak < 3)    return `${4 - streak} more day${streak === 3 ? '' : 's'} to your 3-day milestone!`;
    if (streak < 7)    return `${7 - streak} more day${streak === 6 ? '' : 's'} to your 1-week milestone 🔥`;
    if (streak === 7)  return 'One full week! Incredible consistency 🌟';
    if (streak < 14)   return `${14 - streak} days to your 2-week milestone!`;
    if (streak < 30)   return `${30 - streak} days to your 1-month milestone!`;
    return `🏆 Legendary ${streak}-day streak! You’re unstoppable.`;
  }

  function buildCircleSVG(pct, color) {
    const r = 48, cx = 60;
    const circ   = 2 * Math.PI * r;
    const offset = circ * (1 - Math.min(pct, 100) / 100);
    return `<svg width="140" height="140" viewBox="0 0 120 120">
    <circle cx="${cx}" cy="${cx}" r="${r}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="10"/>
    <circle cx="${cx}" cy="${cx}" r="${r}" fill="none" stroke="${color}" stroke-width="10"
      stroke-dasharray="${circ.toFixed(1)}" stroke-dashoffset="${offset.toFixed(1)}"
      stroke-linecap="round" transform="rotate(-90 ${cx} ${cx})"/>
    <text x="${cx}" y="${cx + 10}" text-anchor="middle" fill="${color}"
      font-family="'Bebas Neue',cursive" font-size="30" dominant-baseline="middle">${Math.round(pct)}</text>
  </svg>`;
  }

  function esTranslate(word) {
    const key = word.toLowerCase().replace(/[^a-z']/g, '').replace(/'/g, '');
    return ES_DICT[key] || 'traducir…';
  }

  function normalizeText(s) {
    return s.toLowerCase().replace(/[^\w'\s]/g, '').replace(/\s+/g, ' ').trim();
  }

  function lcsLength(a, b) {
    const prev = new Uint16Array(b.length + 1);
    let result = 0;
    for (let i = 0; i < a.length; i++) {
      const curr = new Uint16Array(b.length + 1);
      for (let j = 0; j < b.length; j++) {
        curr[j + 1] = a[i] === b[j] ? prev[j] + 1 : Math.max(curr[j], prev[j + 1]);
        if (curr[j + 1] > result) result = curr[j + 1];
      }
      prev.set(curr);
    }
    return result;
  }

  function charSeqRatio(a, b) {
    if (a === b) return 1;
    if (!a || !b) return 0;
    const m = lcsLength(a, b);
    return (2 * m) / (a.length + b.length);
  }

  function wordBreakdown(expected, transcribed) {
    const expTokens = expected.split(/\s+/).filter(Boolean);
    const trnWords  = normalizeText(transcribed).split(/\s+/).filter(Boolean);
    const used = new Array(trnWords.length).fill(false);

    return expTokens.map(token => {
      const norm = normalizeText(token);
      if (!norm) return { word: token, status: 'good' };

      let bestSim = 0;
      let bestIdx = -1;

      trnWords.forEach((tw, i) => {
        if (used[i]) return;
        const sim = charSeqRatio(norm, tw);
        if (sim > bestSim) {
          bestSim = sim;
          bestIdx = i;
        }
      });

      let status;
      if (bestIdx >= 0 && bestSim >= 0.9) {
        used[bestIdx] = true;
        status = 'good';
      } else if (bestIdx >= 0 && bestSim >= 0.55) {
        used[bestIdx] = true;
        status = 'close';
      } else {
        status = 'miss';
      }
      return { word: token, status };
    });
  }

  window.MIRROR_APP_HELPERS = {
    averageScore,
    buildCircleSVG,
    charSeqRatio,
    esTranslate,
    formatAvgPb,
    getDivision,
    lcsLength,
    normalizeText,
    streakMessage,
    timeAgo,
    wordBreakdown,
  };
})();
