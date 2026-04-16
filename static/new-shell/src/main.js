import { createAppShell } from './app-shell.js';

const root = document.querySelector('#newShellRoot');

if (!root) {
  throw new Error('Mirror app root was not found.');
}

createAppShell({ root });
