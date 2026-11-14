/* 50 verbos que necesitas saber
 *
 * A study set inside "My Words": the 50 English verbs a Spanish speaker uses
 * most, each with its past and participle, a Spanish meaning, an example
 * (from a Mirror scene where one of the 47 lines uses it), browser text-to-
 * speech, and a 10-question practice round.
 *
 * Progress rides on the existing vocab mastery store: POST /api/vocab/mastery
 * with scene_id "verbs50" and word_en = the verb. Three correct answers make
 * a verb "dominado", the same rule as the flashcards.
 *
 * Reads the globals API and authToken from static/app.js.
 */
(function () {
  'use strict';

  var SET_ID = 'verbs50';
  var ROUND = 10;

  // [base, español, past, participle, irregular, example, ejemplo, scene title or '', note]
  var VERBS = [
    ['be', 'ser / estar', 'was / were', 'been', 1, 'You were there that night.', 'Estabas ahí esa noche.', 'Interrogation', ''],
    ['have', 'tener', 'had', 'had', 1, 'We have ninety seconds.', 'Tenemos noventa segundos.', 'Countdown', 'have to = tener que'],
    ['do', 'hacer', 'did', 'done', 1, 'Whoever did this never left the house.', 'Quien hizo esto nunca salió de la casa.', 'The Window', 'También ayuda a preguntar: Do you…?'],
    ['say', 'decir', 'said', 'said', 1, 'I almost said it a hundred times.', 'Casi te lo dije cien veces.', 'Rooftop', ''],
    ['go', 'ir', 'went', 'gone', 1, 'So my suitcase went to Paris, and I went to Chicago.', 'Así que mi maleta se fue a París, y yo a Chicago.', 'Lost Luggage', ''],
    ['get', 'conseguir / obtener', 'got', 'gotten', 1, 'If you get on that train, don’t come back.', 'Si te subes a ese tren, no regreses.', 'Last Train', 'get on = subirse (a un tren, bus)'],
    ['make', 'hacer (crear)', 'made', 'made', 1, 'I’m just reconsidering every decision I’ve ever made.', 'Solo estoy reconsiderando cada decisión que he tomado.', 'Cold Feet', 'make a decision = tomar una decisión'],
    ['know', 'saber / conocer', 'knew', 'known', 1, 'I don’t know how to dance.', 'No sé bailar.', 'First Dance', 'La k no se pronuncia: “nou”.'],
    ['think', 'pensar / creer', 'thought', 'thought', 1, 'I asked for one night to think.', 'Pedí una noche para pensarlo.', 'The Offer', ''],
    ['take', 'tomar / llevar', 'took', 'taken', 1, 'Take it or leave it.', 'Lo tomas o lo dejas.', 'The Deal', ''],
    ['see', 'ver', 'saw', 'seen', 1, 'Do you see that? It’s real.', '¿Ves eso? Es real.', 'Lost City', ''],
    ['come', 'venir', 'came', 'come', 1, 'This year, I came here ready for you.', 'Este año vine preparado para ti.', 'The Rival', 'come back = volver'],
    ['want', 'querer', 'wanted', 'wanted', 0, 'I want to speak English like in the movies.', 'Quiero hablar inglés como en las películas.', '', ''],
    ['look', 'mirar / verse', 'looked', 'looked', 0, 'Look at us now!', '¡Míranos ahora!', 'Champion', 'look at = mirar a'],
    ['use', 'usar', 'used', 'used', 0, 'Nobody uses real names.', 'Nadie usa nombres reales.', 'The Plan', ''],
    ['find', 'encontrar', 'found', 'found', 1, 'I found my keys in the car.', 'Encontré mis llaves en el carro.', '', ''],
    ['give', 'dar', 'gave', 'given', 1, 'Please, just give me one chance.', 'Por favor, solo dame una oportunidad.', 'The Audition', ''],
    ['tell', 'decir / contar', 'told', 'told', 1, 'Don’t tell anyone where you’re going.', 'No le digas a nadie a dónde vas.', 'Runaway', 'tell + persona: tell me, tell her'],
    ['work', 'trabajar / funcionar', 'worked', 'worked', 0, 'I work two jobs so you can have the chances I never had.', 'Trabajo en dos empleos para que tengas las oportunidades que yo nunca tuve.', 'Night Shift', ''],
    ['call', 'llamar', 'called', 'called', 0, 'Go ahead and call it luck.', 'Adelante, llámalo suerte.', 'Lucky', ''],
    ['try', 'intentar / probar', 'tried', 'tried', 0, 'Try it one more time.', 'Inténtalo una vez más.', '', ''],
    ['ask', 'preguntar / pedir', 'asked', 'asked', 0, 'Can I ask you something?', '¿Te puedo preguntar algo?', '', 'ask for = pedir algo'],
    ['need', 'necesitar', 'needed', 'needed', 0, 'I need more time.', 'Necesito más tiempo.', '', ''],
    ['feel', 'sentir(se)', 'felt', 'felt', 1, 'I feel ready.', 'Me siento listo.', '', ''],
    ['leave', 'irse / dejar', 'left', 'left', 1, 'Leave your phone, take only cash.', 'Deja tu teléfono, lleva solo efectivo.', 'Runaway', ''],
    ['put', 'poner', 'put', 'put', 1, 'Put your phone away.', 'Guarda tu teléfono.', '', 'Igual en presente, pasado y participio.'],
    ['let', 'dejar / permitir', 'let', 'let', 1, 'Let me be there now.', 'Déjame estar ahora.', 'Missed Games', 'let’s = vamos a…'],
    ['start', 'empezar', 'started', 'started', 0, 'The movie starts at nine.', 'La película empieza a las nueve.', '', ''],
    ['help', 'ayudar', 'helped', 'helped', 0, 'Can you help me?', '¿Me puedes ayudar?', '', ''],
    ['talk', 'hablar / platicar', 'talked', 'talked', 0, 'We need to talk.', 'Tenemos que hablar.', '', 'La l no se pronuncia: “tok”.'],
    ['speak', 'hablar (un idioma)', 'spoke', 'spoken', 1, 'Do you speak English?', '¿Hablas inglés?', '', ''],
    ['hear', 'oír', 'heard', 'heard', 1, 'Whatever you hear, don’t open it.', 'Oigas lo que oigas, no la abras.', 'Midnight Call', ''],
    ['listen', 'escuchar', 'listened', 'listened', 0, 'Listen to me.', 'Escúchame.', '', 'listen to + algo o alguien'],
    ['run', 'correr', 'ran', 'run', 1, 'Grab the bag and run!', '¡Agarra la bolsa y corre!', 'Countdown', ''],
    ['believe', 'creer', 'believed', 'believed', 0, 'Nobody believed in us.', 'Nadie creyó en nosotros.', 'Champion', 'believe in = creer en'],
    ['bring', 'traer', 'brought', 'brought', 1, 'Bring your friends.', 'Trae a tus amigos.', '', ''],
    ['write', 'escribir', 'wrote', 'written', 1, 'I wrote you a letter.', 'Te escribí una carta.', '', 'La w no se pronuncia: “rait”.'],
    ['read', 'leer', 'read', 'read', 1, 'I read your letter. Every word.', 'Leí tu carta. Cada palabra.', 'The Letter', 'En pasado se escribe igual pero suena “red”.'],
    ['lose', 'perder', 'lost', 'lost', 1, 'We lost engine two.', 'Perdimos el motor dos.', 'Mayday', ''],
    ['meet', 'conocer (a alguien)', 'met', 'met', 1, 'When they first met, she said he was annoying.', 'Cuando se conocieron, ella dijo que él era molesto.', 'The Toast', ''],
    ['pay', 'pagar', 'paid', 'paid', 1, 'I’ll pay for dinner.', 'Yo pago la cena.', '', ''],
    ['open', 'abrir', 'opened', 'opened', 0, 'Don’t open the door.', 'No abras la puerta.', 'Midnight Call', ''],
    ['close', 'cerrar', 'closed', 'closed', 0, 'Close your eyes.', 'Cierra los ojos.', 'Surprise', ''],
    ['stop', 'parar / dejar de', 'stopped', 'stopped', 0, 'Stop lying to me.', 'Deja de mentirme.', 'Interrogation', 'stop + -ing = dejar de hacer algo'],
    ['teach', 'enseñar', 'taught', 'taught', 1, 'She taught me to cook without a recipe.', 'Ella me enseñó a cocinar sin receta.', 'Grandma’s Kitchen', ''],
    ['learn', 'aprender', 'learned', 'learned', 0, 'I learn something new every day.', 'Aprendo algo nuevo cada día.', '', ''],
    ['understand', 'entender', 'understood', 'understood', 1, 'I don’t understand.', 'No entiendo.', '', ''],
    ['buy', 'comprar', 'bought', 'bought', 1, 'I bought two tickets.', 'Compré dos boletos.', '', ''],
    ['sleep', 'dormir', 'slept', 'slept', 1, 'Call it luck, if that helps you sleep at night.', 'Llámalo suerte, si eso te ayuda a dormir por las noches.', 'Lucky', ''],
    ['eat', 'comer', 'ate', 'eaten', 1, 'Did you eat?', '¿Ya comiste?', '', '']
  ].map(function (v, i) {
    return { n: i + 1, base: v[0], es: v[1], past: v[2], pp: v[3], irregular: !!v[4], ex: v[5], exEs: v[6], scene: v[7], note: v[8] };
  });

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function shuffle(a) {
    a = a.slice();
    for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)); var t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }
  function firstEs(v) { return v.es.split(' / ')[0]; }
  // First Spanish word of each meaning: say/tell both give "decir", so they
  // never appear as each other's wrong answer.
  function roots(v) {
    return v.es.split(' / ').map(function (p) { return p.replace(/\(.*?\)/g, '').trim().split(' ')[0]; });
  }
  function clash(a, b) {
    var rb = roots(b);
    return roots(a).some(function (r) { return rb.indexOf(r) >= 0; });
  }
  // The "-ed" a learner reaches for, used as a tempting wrong answer.
  function naivePast(base) {
    if (/e$/.test(base)) return base + 'd';
    if (/[^aeiou]y$/.test(base)) return base.slice(0, -1) + 'ied';
    return base + 'ed';
  }

  // ── Speech (browser TTS, no network) ──────────────────────────────────────
  var voice = null;
  function pickVoice() {
    if (!('speechSynthesis' in window)) return;
    var vs = window.speechSynthesis.getVoices() || [];
    voice = vs.find(function (v) { return /^en[-_]US/i.test(v.lang) && /Samantha|Google|Natural|Aria|Jenny/i.test(v.name); })
      || vs.find(function (v) { return /^en[-_]US/i.test(v.lang); })
      || vs.find(function (v) { return /^en/i.test(v.lang); }) || null;
  }
  if ('speechSynthesis' in window) {
    pickVoice();
    window.speechSynthesis.onvoiceschanged = pickVoice;
  }
  function say(text) {
    if (!('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(text);
      u.lang = 'en-US'; u.rate = 0.9;
      if (voice) u.voice = voice;
      window.speechSynthesis.speak(u);
    } catch (e) { /* no audio is not an error worth showing */ }
  }

  var ICON_PLAY = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l11-6.5z"/></svg>';

  var VerbsController = {
    mastery: {},
    filter: 'all',
    isOpen: false,
    quiz: null,
    _built: false,

    _el: function (id) { return document.getElementById(id); },

    _build: function () {
      if (this._built) return;
      this._built = true;
      var o = document.createElement('div');
      o.id = 'verbsOverlay';
      o.className = 'vb-overlay';
      o.setAttribute('role', 'dialog');
      o.setAttribute('aria-modal', 'true');
      o.setAttribute('aria-labelledby', 'vbTitle');
      o.setAttribute('lang', 'es');
      o.hidden = true;
      o.innerHTML =
        '<div class="vb-wrap">' +
          '<div class="vb-top">' +
            '<button type="button" class="vb-back" id="vbBack" aria-label="Volver">' +
              '<svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg></button>' +
            '<p class="vb-top-t">50 VERBOS</p>' +
            '<span class="vb-top-c" id="vbTopCount"></span>' +
          '</div>' +
          '<section id="vbListView">' +
            '<p class="vb-eyebrow">VOCABULARIO ESENCIAL</p>' +
            '<h2 class="vb-h" id="vbTitle">Los 50 verbos que <span>más vas a usar</span></h2>' +
            '<p class="vb-lead">Con su pasado, un ejemplo de las escenas y audio. Practica y cada verbo se marca como dominado al acertarlo 3 veces.</p>' +
            '<div class="vb-prog"><div class="vb-prog-bar"><i id="vbProgFill"></i></div><span id="vbProgTxt"></span></div>' +
            '<button type="button" class="vb-cta" id="vbPracticeBtn">Practicar 10 verbos</button>' +
            '<div class="vb-filters" role="group" aria-label="Filtrar verbos">' +
              '<button type="button" class="vb-fil" data-f="all" aria-pressed="true">Todos <b>50</b></button>' +
              '<button type="button" class="vb-fil" data-f="irr" aria-pressed="false">Irregulares <b id="vbIrrN"></b></button>' +
              '<button type="button" class="vb-fil" data-f="todo" aria-pressed="false">Por dominar <b id="vbTodoN"></b></button>' +
            '</div>' +
            '<ul class="vb-list" id="vbList"></ul>' +
          '</section>' +
          '<section id="vbQuizView" hidden></section>' +
        '</div>';
      document.body.appendChild(o);

      var self = this;
      this._el('vbBack').addEventListener('click', function () {
        if (self.quiz) { self.quiz = null; self._showList(); } else { self.close(); }
      });
      this._el('vbPracticeBtn').addEventListener('click', function () { self.startQuiz(); });
      o.querySelectorAll('.vb-fil').forEach(function (b) {
        b.addEventListener('click', function () {
          self.filter = b.getAttribute('data-f');
          o.querySelectorAll('.vb-fil').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
          self._renderList();
        });
      });
      this._el('vbList').addEventListener('click', function (e) {
        var sp = e.target.closest('[data-say]');
        if (sp) { say(sp.getAttribute('data-say')); return; }
        var head = e.target.closest('.vb-head');
        if (!head) return;
        var row = head.parentElement, det = row.querySelector('.vb-detail');
        var openNow = head.getAttribute('aria-expanded') !== 'true';
        head.setAttribute('aria-expanded', openNow ? 'true' : 'false');
        det.hidden = !openNow;
        if (openNow) say(row.getAttribute('data-base'));
      });
      this._el('vbQuizView').addEventListener('click', function (e) {
        var opt = e.target.closest('[data-opt]');
        if (opt && self.quiz && !self.quiz.answered) { self._answer(+opt.getAttribute('data-opt')); return; }
        if (e.target.closest('#vbNext')) { self._next(); return; }
        if (e.target.closest('#vbAgain')) { self.startQuiz(); return; }
        if (e.target.closest('#vbToList')) { self.quiz = null; self._showList(); return; }
        var sp = e.target.closest('[data-say]');
        if (sp) say(sp.getAttribute('data-say'));
      });
      document.addEventListener('keydown', function (e) {
        if (self.isOpen && e.key === 'Escape') self.close();
      });
    },

    count: function (base) { return Math.min(3, this.mastery[base] || 0); },
    masteredN: function () { var s = this; return VERBS.filter(function (v) { return s.count(v.base) >= 3; }).length; },

    // Loads progress for the entry card in My Words without opening anything.
    refreshEntry: function () {
      var self = this;
      return this._fetchMastery().then(function () { self._renderEntry(); });
    },

    _fetchMastery: function () {
      var self = this;
      if (typeof authToken === 'undefined' || !authToken) return Promise.resolve();
      return fetch(API + '/api/vocab/mastery?scene_id=' + SET_ID, { headers: { Authorization: 'Bearer ' + authToken } })
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (d) { self.mastery = d || {}; })
        .catch(function () { /* offline: keep what we have */ });
    },

    _renderEntry: function () {
      var c = this._el('verbsEntryCount');
      if (!c) return;
      var m = this.masteredN();
      c.textContent = m ? (m + '/50 dominados · sigue practicando') : 'Pasado, ejemplos, audio y práctica';
    },

    openSet: function () {
      this._build();
      var o = this._el('verbsOverlay');
      o.hidden = false;
      this.isOpen = true;
      this._prevOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      this.quiz = null;
      this._showList();
      var self = this;
      this._fetchMastery().then(function () { self._renderList(); self._renderEntry(); });
      setTimeout(function () { var b = self._el('vbBack'); if (b) b.focus(); }, 30);
    },

    close: function () {
      var o = this._el('verbsOverlay');
      if (o) o.hidden = true;
      this.isOpen = false;
      document.body.style.overflow = this._prevOverflow || '';
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
      this._renderEntry();
      var entry = this._el('verbsEntry');
      if (entry) entry.focus();
    },

    _showList: function () {
      this._el('vbListView').hidden = false;
      this._el('vbQuizView').hidden = true;
      this._renderList();
      var o = this._el('verbsOverlay');
      if (o) o.scrollTop = 0;
    },

    _stars: function (n) {
      var s = '';
      for (var i = 0; i < 3; i++) s += '<i class="' + (i < n ? 'on' : '') + '"></i>';
      return '<span class="vb-stars" aria-label="' + n + ' de 3 aciertos">' + s + '</span>';
    },

    _renderList: function () {
      var self = this;
      var m = this.masteredN();
      this._el('vbProgFill').style.width = (m / 50 * 100) + '%';
      this._el('vbProgTxt').textContent = m + '/50 dominados';
      this._el('vbTopCount').textContent = m + '/50';
      this._el('vbIrrN').textContent = VERBS.filter(function (v) { return v.irregular; }).length;
      this._el('vbTodoN').textContent = 50 - m;
      var rows = VERBS.filter(function (v) {
        if (self.filter === 'irr') return v.irregular;
        if (self.filter === 'todo') return self.count(v.base) < 3;
        return true;
      });
      if (!rows.length) {
        this._el('vbList').innerHTML = '<li class="vb-empty">¡Dominaste los 50! Practica cuando quieras para no olvidarlos.</li>';
        return;
      }
      this._el('vbList').innerHTML = rows.map(function (v) {
        var c = self.count(v.base);
        return '<li class="vb-row' + (c >= 3 ? ' is-done' : '') + '" data-base="' + esc(v.base) + '">' +
          '<button type="button" class="vb-head" aria-expanded="false">' +
            '<span class="vb-n">' + (v.n < 10 ? '0' : '') + v.n + '</span>' +
            '<span class="vb-main"><b lang="en">' + esc(v.base) + '</b><small>' + esc(v.es) + '</small></span>' +
            '<span class="vb-forms" lang="en">' + esc(v.past) + ' · ' + esc(v.pp) + '</span>' +
            self._stars(c) +
          '</button>' +
          '<div class="vb-detail" hidden>' +
            '<div class="vb-table">' +
              '<div><small>Presente</small><b lang="en">' + esc(v.base) + '</b></div>' +
              '<div><small>Pasado</small><b lang="en">' + esc(v.past) + '</b></div>' +
              '<div><small>Participio</small><b lang="en">' + esc(v.pp) + '</b></div>' +
            '</div>' +
            '<p class="vb-tag">' + (v.irregular ? 'IRREGULAR' : 'REGULAR · TERMINA EN -ED') + '</p>' +
            '<p class="vb-ex" lang="en">“' + esc(v.ex) + '”</p>' +
            '<p class="vb-ex-es">' + esc(v.exEs) + '</p>' +
            (v.scene ? '<p class="vb-scene">De la escena · ' + esc(v.scene.toUpperCase()) + '</p>' : '') +
            (v.note ? '<p class="vb-note">' + esc(v.note) + '</p>' : '') +
            '<div class="vb-acts">' +
              '<button type="button" class="vb-say" data-say="' + esc(v.base + ', ' + v.past.replace(' / ', ', ') + ', ' + v.pp) + '">' + ICON_PLAY + 'Formas</button>' +
              '<button type="button" class="vb-say" data-say="' + esc(v.ex.replace(/[“”]/g, '')) + '">' + ICON_PLAY + 'Ejemplo</button>' +
            '</div>' +
          '</div>' +
        '</li>';
      }).join('');
    },

    // ── Practice ─────────────────────────────────────────────────────────────
    _makeQuestion: function (v) {
      var others = shuffle(VERBS.filter(function (x) { return x.base !== v.base && !clash(x, v); }));
      var kinds = ['en', 'past'];
      if (v.past.split(' / ')[0] !== v.base) kinds.push('mean');
      var kind = kinds[Math.floor(Math.random() * kinds.length)];
      var q = { verb: v, kind: kind };
      var right, wrong = [];
      if (kind === 'en') {
        q.prompt = '¿Cómo se dice <b>“' + esc(v.es) + '”</b> en inglés?';
        right = v.base;
        wrong = others.slice(0, 3).map(function (x) { return x.base; });
      } else if (kind === 'past') {
        q.prompt = '¿Cuál es el pasado de <b lang="en">' + esc(v.base) + '</b>?';
        right = v.past;
        var pool = [];
        if (v.irregular && v.base !== 'be' && naivePast(v.base) !== v.past) pool.push(naivePast(v.base));
        if (v.pp !== v.past) pool.push(v.pp);
        others.forEach(function (x) { pool.push(x.past); });
        pool.forEach(function (w) { if (w !== right && wrong.indexOf(w) < 0 && wrong.length < 3) wrong.push(w); });
      } else {
        var form = v.past.split(' / ')[0];
        q.prompt = '¿Qué verbo es <b lang="en">' + esc(form) + '</b>?';
        right = firstEs(v);
        others.forEach(function (x) { var e = firstEs(x); if (e !== right && wrong.indexOf(e) < 0 && wrong.length < 3) wrong.push(e); });
      }
      q.options = shuffle([right].concat(wrong));
      q.right = q.options.indexOf(right);
      q.optLang = kind === 'mean' ? 'es' : 'en';
      return q;
    },

    startQuiz: function () {
      var self = this;
      var todo = shuffle(VERBS.filter(function (v) { return self.count(v.base) < 3; }));
      var done = shuffle(VERBS.filter(function (v) { return self.count(v.base) >= 3; }));
      var picks = todo.concat(done).slice(0, ROUND);
      this.quiz = { items: picks.map(function (v) { return self._makeQuestion(v); }), i: 0, score: 0, answered: false };
      this._el('vbListView').hidden = true;
      this._el('vbQuizView').hidden = false;
      this._renderQuestion();
    },

    _renderQuestion: function () {
      var q = this.quiz, it = q.items[q.i];
      var dots = q.items.map(function (x, k) {
        return '<i class="' + (k < q.i ? (x.ok ? 'ok' : 'no') : (k === q.i ? 'cur' : '')) + '"></i>';
      }).join('');
      this._el('vbQuizView').innerHTML =
        '<div class="vb-dots" aria-hidden="true">' + dots + '</div>' +
        '<p class="vb-qn">Pregunta ' + (q.i + 1) + ' de ' + q.items.length + '</p>' +
        '<h3 class="vb-q">' + it.prompt + '</h3>' +
        '<div class="vb-opts">' + it.options.map(function (o, k) {
          return '<button type="button" class="vb-opt" data-opt="' + k + '" lang="' + it.optLang + '">' + esc(o) + '</button>';
        }).join('') + '</div>' +
        '<div class="vb-fb" id="vbFb" aria-live="polite"></div>';
      var o = this._el('verbsOverlay'); if (o) o.scrollTop = 0;
    },

    _answer: function (k) {
      var q = this.quiz, it = q.items[q.i], v = it.verb;
      q.answered = true;
      it.ok = k === it.right;
      var btns = this._el('vbQuizView').querySelectorAll('.vb-opt');
      btns.forEach(function (b, idx) {
        b.disabled = true;
        if (idx === it.right) b.classList.add('is-right');
        else if (idx === k) b.classList.add('is-wrong');
      });
      if (it.ok) { q.score++; this._record(v.base); }
      if (navigator.vibrate) { try { navigator.vibrate(it.ok ? 18 : [30, 40, 30]); } catch (e) {} }
      var last = q.i === q.items.length - 1;
      this._el('vbFb').innerHTML =
        '<p class="vb-fb-t ' + (it.ok ? 'ok' : 'no') + '">' + (it.ok ? '¡Bien!' : 'Casi.') + '</p>' +
        '<p class="vb-fb-forms"><b lang="en">' + esc(v.base) + ' · ' + esc(v.past) + ' · ' + esc(v.pp) + '</b> = ' + esc(v.es) + '</p>' +
        '<p class="vb-fb-ex" lang="en">“' + esc(v.ex) + '”</p>' +
        '<div class="vb-fb-row">' +
          '<button type="button" class="vb-say" data-say="' + esc(v.ex.replace(/[“”]/g, '')) + '">' + ICON_PLAY + 'Escuchar</button>' +
          '<button type="button" class="vb-cta vb-cta--sm" id="vbNext">' + (last ? 'Ver resultado' : 'Siguiente →') + '</button>' +
        '</div>';
      var n = this._el('vbNext'); if (n) n.focus();
    },

    _next: function () {
      var q = this.quiz;
      if (q.i < q.items.length - 1) { q.i++; q.answered = false; this._renderQuestion(); return; }
      this._renderResult();
    },

    _renderResult: function () {
      var q = this.quiz, m = this.masteredN();
      var verdict = q.score >= 9 ? '¡Impecable!' : q.score >= 7 ? '¡Muy bien!' : q.score >= 5 ? 'Vas bien' : 'Sigue practicando';
      this._el('vbQuizView').innerHTML =
        '<div class="vb-res">' +
          '<p class="vb-eyebrow">RESULTADO</p>' +
          '<p class="vb-res-n">' + q.score + '<span>/' + q.items.length + '</span></p>' +
          '<p class="vb-res-v">' + verdict + '</p>' +
          '<p class="vb-lead">Llevas <b>' + m + '/50</b> verbos dominados.</p>' +
          '<ul class="vb-res-list">' + q.items.map(function (it) {
            return '<li class="' + (it.ok ? 'ok' : 'no') + '"><b lang="en">' + esc(it.verb.base) + '</b> <span lang="en">' + esc(it.verb.past) + ' · ' + esc(it.verb.pp) + '</span><small>' + esc(it.verb.es) + '</small></li>';
          }).join('') + '</ul>' +
          '<button type="button" class="vb-cta" id="vbAgain">Practicar otros 10</button>' +
          '<button type="button" class="vb-link" id="vbToList">Ver la lista</button>' +
        '</div>';
      this.quiz.answered = true;
      var o = this._el('verbsOverlay'); if (o) o.scrollTop = 0;
    },

    // Same store and rule as the flashcards: correct answers count, misses don't
    // undo progress. A failed save rolls the count back so the screen matches.
    _record: function (base) {
      var self = this, prev = this.mastery[base] || 0;
      this.mastery[base] = Math.min(3, prev + 1);
      if (typeof authToken === 'undefined' || !authToken) return;
      fetch(API + '/api/vocab/mastery', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + authToken },
        body: JSON.stringify({ scene_id: SET_ID, word_en: base, correct: true })
      }).then(function (r) {
        if (r.ok) return r.json().then(function (d) { if (d && typeof d.correct_count === 'number') self.mastery[base] = d.correct_count; });
        self.mastery[base] = prev;
      }).catch(function () { self.mastery[base] = prev; });
    }
  };

  // Public entry point (the My Words card calls VerbsController.open()).
  VerbsController.open = function () { VerbsController.openSet(); };
  VerbsController.VERBS = VERBS;
  window.VerbsController = VerbsController;
})();
