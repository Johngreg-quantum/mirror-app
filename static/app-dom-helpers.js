(function() {
  function setTextIfPresent(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
  }

  function setHtmlIfPresent(id, value) {
    const node = document.getElementById(id);
    if (node) node.innerHTML = value;
  }

  function setDisplayIfPresent(id, value) {
    const node = document.getElementById(id);
    if (node) node.style.display = value;
  }

  window.MIRROR_APP_DOM_HELPERS = {
    setDisplayIfPresent,
    setHtmlIfPresent,
    setTextIfPresent,
  };
})();
