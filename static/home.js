/* Home cards under the cinematic hero (index.html .home-row):
 *  - Daily Take: today's three lines as posters (done / next / pending), the
 *    count, the time left (written by startDailyCountdown) and one button.
 *  - Level: the existing level bar, unchanged ids, rendered by renderLevelBar.
 *  - Shortcuts: the level exam, 50 verbos, Mis palabras.
 *
 * Reads app.js globals at call time: dailyChallenge, dailyLines, scenes,
 * userProgress, getSceneArtThumb, openModal, QuizController, VerbsController.
 * renderHeroFeatured() calls HomeEnhancer.afterDaily() after each daily load.
 */
(function () {
  'use strict';

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function sceneById(id) {
    try { return (typeof scenes !== 'undefined' && scenes && scenes[id]) || null; } catch (e) { return null; }
  }
  function open(id) {
    var s = sceneById(id);
    if (s && typeof openModal === 'function') openModal(id, s);
  }

  var CHECK = '<svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#0b0b0e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';

  var HomeEnhancer = {
    _verbsLoaded: false,

    afterDaily: function () {
      this.renderDaily();
      this.renderQuiz();
      this.renderVerbs();
    },

    renderDaily: function () {
      var wrap = document.getElementById('homeDailyPosters');
      if (!wrap || typeof dailyChallenge === 'undefined' || !dailyChallenge || typeof dailyLines !== 'function') return;
      var lines = dailyLines(dailyChallenge);
      if (!lines.ids.length) return;

      wrap.innerHTML = lines.ids.map(function (id) {
        var s = sceneById(id) || {};
        var done = lines.done.indexOf(id) >= 0;
        var next = !done && id === lines.nextId;
        var state = done ? 'is-done' : (next ? 'is-next' : 'is-todo');
        var art = typeof getSceneArtThumb === 'function' ? getSceneArtThumb(id) : '';
        var label = (s.title || id) + (done ? ', hecha' : (next ? ', siguiente' : ', pendiente'));
        return '<button type="button" class="home-daily-poster ' + state + '" data-sid="' + esc(id) + '" aria-label="' + esc(label) + '">' +
          (art ? '<img src="' + esc(art) + '" alt="" loading="lazy">' : '') +
          (done ? '<span class="hd-check">' + CHECK + '</span>' : '') +
          '<span class="hd-name">' + esc(s.title || id) + '</span>' +
        '</button>';
      }).join('');
      wrap.querySelectorAll('[data-sid]').forEach(function (b) {
        b.addEventListener('click', function () { open(b.getAttribute('data-sid')); });
      });

      var count = document.getElementById('homeDailyCount');
      if (count) count.textContent = lines.allDone ? '¡Completo!' : (lines.doneCount + ' de ' + lines.total);

      var btn = document.getElementById('homeDailyBtn');
      if (btn) {
        var nextScene = sceneById(lines.nextId) || {};
        btn.textContent = lines.allDone
          ? 'Repetir ' + (nextScene.title || 'una línea')
          : ((lines.doneCount ? 'Seguir con ' : 'Empezar con ') + (nextScene.title || 'la primera línea'));
        btn.onclick = function () { open(lines.nextId); };
      }
    },

    // The exam that opens the next level: L1 opens Level 2, L2 opens Level 3.
    renderQuiz: function () {
      var link = document.getElementById('homeQuizLink');
      if (!link) return;
      var level = (typeof userProgress !== 'undefined' && userProgress && userProgress.level) || 1;
      if (level >= 3) { link.hidden = true; return; }
      link.hidden = false;
      var title = document.getElementById('homeQuizTitle');
      var sub = document.getElementById('homeQuizSub');
      if (title) title.textContent = 'Examen del Nivel ' + level;
      if (sub) sub.textContent = 'Apruébalo y se abre el Nivel ' + (level + 1);
      link.onclick = function () {
        if (window.QuizController) window.QuizController.open('L' + level);
      };
    },

    renderVerbs: function () {
      if (this._verbsLoaded || !window.VerbsController) return;
      this._verbsLoaded = true;
      var V = window.VerbsController;
      V.refreshEntry().then(function () {
        var n = V.masteredN();
        var sub = document.getElementById('homeVerbsSub');
        if (sub && n) sub.textContent = n + ' de 50 dominados';
      });
    }
  };

  // The level can change after a take; keep the exam link pointing at the
  // right one.
  if (typeof loadProgress === 'function') {
    var _loadProgress = loadProgress;
    loadProgress = async function () {
      var r = await _loadProgress.apply(this, arguments);
      HomeEnhancer.renderQuiz();
      return r;
    };
  }

  window.HomeEnhancer = HomeEnhancer;
})();
