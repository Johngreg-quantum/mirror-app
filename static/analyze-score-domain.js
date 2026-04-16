(function() {
  function getScoreDisplayMeta(score) {
    if (score >= 85) return { color: '#06d6a0', msg: 'Outstanding! \uD83C\uDFAC' };
    if (score >= 65) return { color: '#ffd166', msg: 'Great take!' };
    if (score >= 40) return { color: '#f4a261', msg: 'Keep practicing' };
    return { color: '#e63946', msg: 'Try again' };
  }

  function renderScoreDisplay(options) {
    const data = options.data || {};
    const refs = options.refs || {};
    const helpers = options.helpers || {};
    const meta = getScoreDisplayMeta(data.sync_score || 0);

    if (refs.panelEl) refs.panelEl.style.setProperty('--score-color', meta.color);
    if (refs.msgEl) refs.msgEl.textContent = meta.msg;
    if (refs.cmpYouEl) refs.cmpYouEl.textContent = `\u201c${data.transcription}\u201d`;
    if (refs.cmpOrigEl) refs.cmpOrigEl.textContent = `\u201c${data.expected}\u201d`;
    if (refs.panelEl) refs.panelEl.classList.toggle('on', true);
    if (refs.pbCompareEl) refs.pbCompareEl.classList.toggle('on', true);
    if (refs.hearActorBtn) refs.hearActorBtn.disabled = !options.hasYt;

    if (helpers.animateNum && refs.scoreValEl) {
      helpers.animateNum(refs.scoreValEl, 0, data.sync_score || 0, 900);
    }
    if (refs.scoreBarEl) {
      requestAnimationFrame(function() {
        refs.scoreBarEl.style.background = meta.color;
        requestAnimationFrame(function() {
          refs.scoreBarEl.style.width = `${data.sync_score || 0}%`;
        });
      });
    }
  }

  function renderPointsEarnedDisplay(options) {
    const data = options.data || {};
    const refs = options.refs || {};

    if (data.is_perfect && refs.perfectBadgeEl) {
      refs.perfectBadgeEl.classList.toggle('on', true);
    }

    if (data.points_earned > 0 || data.total_points !== undefined) {
      if (refs.ptsAmountEl) refs.ptsAmountEl.textContent = data.points_earned || 0;
      if (refs.ptsTotalValEl) refs.ptsTotalValEl.textContent = data.total_points || 0;

      const ptsPanel = refs.ptsPanelEl;
      let extra = '';
      if (data.is_daily && data.daily_bonus > 0) {
        extra += `<div style="font-size:11px;color:var(--gold);margin-top:4px">&#9733; Daily 2&times; bonus +${data.daily_bonus}pts</div>`;
      }
      if (data.is_daily && data.streak > 0 && !data.daily_already_done) {
        extra += `<div style="font-size:11px;color:#fb923c;margin-top:2px">&#128293; ${data.streak}-day streak!</div>`;
      }
      if (extra && ptsPanel) {
        let extraEl = ptsPanel.querySelector('.pts-extra');
        if (!extraEl) {
          extraEl = document.createElement('div');
          extraEl.className = 'pts-extra';
          ptsPanel.appendChild(extraEl);
        }
        extraEl.innerHTML = extra;
      }
      if (ptsPanel) ptsPanel.classList.toggle('on', true);
    }

    if (data.translation_unlocked && data.translation) {
      if (refs.transTextEl) refs.transTextEl.textContent = data.translation;
      if (refs.transRevealEl) refs.transRevealEl.classList.toggle('on', true);
    }
  }

  function renderPhonemeBreakdownDisplay(options) {
    const expected = options.expected || '';
    const transcribed = options.transcribed || '';
    const refs = options.refs || {};
    const helpers = options.helpers || {};
    const tokens = helpers.wordBreakdown ? helpers.wordBreakdown(expected, transcribed) : [];

    if (!tokens.length) {
      if (refs.sectionEl) refs.sectionEl.style.display = 'none';
      return;
    }

    if (refs.wordsEl) {
      refs.wordsEl.innerHTML = tokens.map(function(token) {
        return `<span class="phon-word ${token.status}">
       <span class="phon-inner">
         <span class="phon-face phon-front">${token.word}</span>
         <span class="phon-face phon-back">&#127466;&#127480; ${helpers.esTranslate(token.word)}</span>
       </span>
     </span>`;
      }).join('');
    }
    if (refs.sectionEl) refs.sectionEl.style.display = '';
  }

  function renderChallengeResultDisplay(options) {
    const score = options.score;
    const scoreToBeat = options.scoreToBeat;
    const refs = options.refs || {};
    const won = score > scoreToBeat;

    if (!refs.resultEl) return;
    refs.resultEl.className = 'challenge-result ' + (won ? 'won' : 'lost');
    if (won) {
      refs.resultEl.innerHTML = `<div class="chlg-result-icon">\uD83C\uDFC6</div>
      <div class="chlg-result-title">YOU WON!</div>
      <div class="chlg-result-sub">You scored ${score}% vs ${scoreToBeat}% \u2014 challenge beaten!</div>`;
    } else {
      refs.resultEl.innerHTML = `<div class="chlg-result-icon">\uD83D\uDE24</div>
      <div class="chlg-result-title">So Close!</div>
      <div class="chlg-result-sub">You scored ${score}% — need ${scoreToBeat}% to win. Try again!</div>`;
    }
  }

  window.MIRROR_ANALYZE_SCORE_DOMAIN = {
    renderChallengeResultDisplay,
    renderPhonemeBreakdownDisplay,
    renderPointsEarnedDisplay,
    renderScoreDisplay,
  };
})();
