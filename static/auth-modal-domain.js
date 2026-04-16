(function() {
  function renderAuthTabDisplay(options) {
    const isLogin = options.isLogin;
    const refs = options.refs || {};

    if (refs.loginTabBtn) refs.loginTabBtn.classList.toggle('active', isLogin);
    if (refs.registerTabBtn) refs.registerTabBtn.classList.toggle('active', !isLogin);
    if (refs.loginForm) refs.loginForm.classList.toggle('hidden', !isLogin);
    if (refs.registerForm) refs.registerForm.classList.toggle('hidden', isLogin);
    if (refs.loginErrorEl) refs.loginErrorEl.textContent = '';
    if (refs.registerErrorEl) refs.registerErrorEl.textContent = '';
  }

  function renderAuthErrorDisplay(options) {
    const refs = options.refs || {};
    if (refs.errorEl) refs.errorEl.textContent = options.message || '';
  }

  function renderAuthSubmitDisplay(options) {
    const refs = options.refs || {};
    if (!refs.buttonEl) return;
    refs.buttonEl.disabled = !!options.disabled;
    refs.buttonEl.textContent = options.text || '';
  }

  window.MIRROR_AUTH_MODAL_DOMAIN = {
    renderAuthErrorDisplay,
    renderAuthSubmitDisplay,
    renderAuthTabDisplay,
  };
})();
