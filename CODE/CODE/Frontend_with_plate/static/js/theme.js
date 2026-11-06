(function () {
  var storageKey = "violatracker-theme";
  var root = document.documentElement;

  function systemTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  function currentTheme() {
    var saved = localStorage.getItem(storageKey);
    return saved === "light" || saved === "dark" ? saved : systemTheme();
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    localStorage.setItem(storageKey, theme);
    var toggle = document.querySelector(".theme-toggle");
    if (toggle) {
      var next = theme === "dark" ? "light" : "dark";
      toggle.setAttribute("aria-label", "Switch to " + next + " mode");
      toggle.setAttribute("title", "Switch to " + next + " mode");
    }
  }

  function ensureThemeCss() {
    if (document.querySelector('link[href="/static/css/theme.css"]')) return;
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/static/css/theme.css";
    link.setAttribute("data-theme-css", "true");
    document.head.appendChild(link);
  }

  function ensureToggle() {
    if (document.querySelector(".theme-toggle")) return;

    var button = document.createElement("button");
    button.type = "button";
    button.className = "theme-toggle";
    button.innerHTML = '<span class="theme-toggle-icon" aria-hidden="true"></span>';
    button.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
    });
    document.body.appendChild(button);
    applyTheme(root.getAttribute("data-theme") || currentTheme());
  }

  ensureThemeCss();
  applyTheme(currentTheme());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureToggle);
  } else {
    ensureToggle();
  }
})();
