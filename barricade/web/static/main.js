const N = 9, CELL = 72, SIZE = N * CELL;
const COLORS = ['#e76f51', '#4d7cfe'];
const COLORS_DARK = ['#c0563d', '#3457c9'];
const NAMES = ['P1', 'P2'];
const AI_DELAY = 400; // ms pause so the last move is visible before the AI replies

const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const hintEl = document.getElementById('hint');
const debugEl = document.getElementById('debug');
const clockEls = [document.getElementById('c0'), document.getElementById('c1')];

let state = null;
let display = [[8, 4], [0, 4]]; // animated pawn positions
let selected = null;            // key of selected own pawn
let hoverWall = null;           // {ori,r,c,d} or null
let hoverCell = null;           // [r,c] under the cursor
let flip = false;
let debugOn = false;
let aiBusy = false;
let pendingAiAt = null;         // timestamp to fire the delayed AI move
let clockAnchorAt = 0;          // performance.now() when current state was received
let engines = {};

const key = p => p[0] + ',' + p[1];

async function api(path, body) {
  const r = await fetch(path, body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {});
  return r.json();
}

function isHumanTurn() {
  if (!state || state.game_over) return false;
  if (state.ai_player === null) return true;
  if (state.ai_player === 'both') return false;
  return state.turn !== state.ai_player;
}

function isAiTurn() {
  if (!state || state.game_over || aiBusy) return false;
  if (state.ai_player === null) return false;
  return state.ai_player === 'both' || state.turn === state.ai_player;
}

function mousePos(e) {
  const rect = canvas.getBoundingClientRect();
  let mx = (e.clientX - rect.left) * SIZE / rect.width;
  let my = (e.clientY - rect.top) * SIZE / rect.height;
  if (flip) { mx = SIZE - mx; my = SIZE - my; }
  return [mx, my];
}

function cellAt(mx, my) {
  const r = Math.floor(my / CELL), c = Math.floor(mx / CELL);
  if (r < 0 || r >= N || c < 0 || c >= N) return null;
  return [r, c];
}

function wallFromMouse(mx, my) {
  const hr = Math.round(my / CELL) - 1, hc = Math.floor(mx / CELL);
  const vc = Math.round(mx / CELL) - 1, vr = Math.floor(my / CELL);
  const hs = { ori: 'H', r: hr, c: hc, d: Math.abs(my - (hr + 1) * CELL) };
  const vs = { ori: 'V', r: vr, c: vc, d: Math.abs(mx - (vc + 1) * CELL) };
  const opts = [hs, vs].filter(w => legalWall(w)).sort((a, b) => a.d - b.d);
  return opts[0] || null;
}

function legalWall(w) {
  return state.legal_walls.some(x => x[0] === w.ori && x[1] === w.r && x[2] === w.c);
}

function adopt(s) {
  state = s;
  selected = null;
  clockAnchorAt = performance.now();
  updateDebug();
  if (isAiTurn()) pendingAiAt = performance.now() + AI_DELAY;
}

// ----- interaction -----

canvas.addEventListener('mousemove', e => {
  hoverWall = null;
  hoverCell = null;
  if (!state || !isHumanTurn()) return;
  const [mx, my] = mousePos(e);
  const w = wallFromMouse(mx, my);
  hoverWall = (w && w.d <= 14) ? w : null;
  hoverCell = cellAt(mx, my);
});

canvas.addEventListener('mouseleave', () => { hoverWall = null; hoverCell = null; });

canvas.addEventListener('click', async e => {
  if (!state || state.game_over || !isHumanTurn()) return;
  const [mx, my] = mousePos(e);

  if (hoverWall) {
    const prev = state;
    const res = await api('/api/move', { move: ['wall', [hoverWall.ori, hoverWall.r, hoverWall.c]] });
    if (!res || res.error) { state = prev; render(); return; }
    adopt(res);
    return;
  }

  const cell = cellAt(mx, my);
  if (!cell) return;
  const k = key(cell);

  if (state.legal_moves.some(m => key(m) === k)) {
    const prev = state;
    const res = await api('/api/move', { move: ['move', [cell[0], cell[1]]] });
    if (!res || res.error) { state = prev; render(); return; }
    adopt(res);
  } else if (k === key(state.positions[state.turn])) {
    selected = selected === k ? null : k;   // toggle pawn highlight
  } else {
    selected = null;
  }
});

async function doAiMove() {
  aiBusy = true;
  const prev = state;
  const res = await api('/api/ai', {});
  aiBusy = false;
  if (!res || res.error) { state = prev; render(); return; }
  adopt(res);
}

// ----- menu -----

const menuEl = document.getElementById('menu');
const modeSel = document.getElementById('m-mode');
const engineSel = document.getElementById('m-engine');
const sideSel = document.getElementById('m-side');
const timeInp = document.getElementById('m-time');
const flipInp = document.getElementById('m-flip');
const engineF = document.getElementById('m-engine-f');
const sideF = document.getElementById('m-side-f');

function populateEngineSelect() {
  engineSel.innerHTML = '';
  const list = engines.list || ['greedy'];
  for (const name of list) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    engineSel.appendChild(opt);
  }
  if (engines.default) engineSel.value = engines.default;
}

function openMenu() {
  sideSel.value = String(state && state.ai_player === 0 ? 0 : 1);
  menuEl.classList.remove('hidden');
}

function closeMenu() {
  menuEl.classList.add('hidden');
}

function onModeChange() {
  const ai = modeSel.value !== 'pvp';
  engineF.classList.toggle('hidden', !ai);
  sideF.classList.toggle('hidden', modeSel.value !== 'ai');
}

document.getElementById('btn-new').addEventListener('click', openMenu);
document.getElementById('m-cancel').addEventListener('click', closeMenu);
modeSel.addEventListener('change', onModeChange);

document.getElementById('m-start').addEventListener('click', async () => {
  const mode2 = modeSel.value;
  const engine = engineSel.value;
  const aiPlayer = mode2 === 'ai' ? parseInt(sideSel.value, 10) : 1;
  const tc = Math.max(0, parseFloat(timeInp.value) || 0);
  closeMenu();
  const state2 = await startNew(mode2, aiPlayer, engine, tc || null);
  if (!state2 || state2.error) return;
  if (flipInp.checked) flip = true;
  display = state2.positions.map(p => [p[0], p[1]]);
  adopt(state2);
});

document.getElementById('btn-flip').addEventListener('click', () => {
  flip = !flip;
});

document.getElementById('chk-debug').addEventListener('change', e => {
  debugOn = e.target.checked;
  updateDebug();
});

function updateDebug() {
  if (!state) return;
  if (!debugOn) { debugEl.classList.add('hidden'); return; }
  debugEl.classList.remove('hidden');
  const lines = [];
  if (state.mode !== 'pvp') {
    const info = state.engine_info;
    if (info) {
      lines.push('Engine: ' + (info.engine || '?'));
      if (info.info && Object.keys(info.info).length) {
        lines.push(Object.entries(info.info).map(([k, v]) => k + ': ' + v).join('  ·  '));
      }
    }
  }
  if (state.last_action) lines.push('Last move: ' + JSON.stringify(state.last_action));
  debugEl.textContent = lines.join('\n') || 'No info yet.';
}

// ----- rendering -----

function applyTransform() {
  if (flip) ctx.setTransform(-1, 0, 0, -1, SIZE, SIZE);
  else ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function roundRectPath(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function render() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, SIZE, SIZE);
  applyTransform();

  // board background
  const bg = ctx.createLinearGradient(0, 0, 0, SIZE);
  bg.addColorStop(0, '#efe7d4');
  bg.addColorStop(1, '#e4d9c2');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, SIZE, SIZE);

  // goal zones
  const goalTop = ctx.createLinearGradient(0, 0, 0, CELL);
  goalTop.addColorStop(0, COLORS[0] + '55');
  goalTop.addColorStop(1, COLORS[0] + '00');
  ctx.fillStyle = goalTop;
  ctx.fillRect(0, 0, SIZE, CELL);
  const goalBot = ctx.createLinearGradient(0, SIZE - CELL, 0, SIZE);
  goalBot.addColorStop(0, COLORS[1] + '00');
  goalBot.addColorStop(1, COLORS[1] + '55');
  ctx.fillStyle = goalBot;
  ctx.fillRect(0, SIZE - CELL, SIZE, CELL);

  // grid
  ctx.strokeStyle = 'rgba(60,50,30,0.35)'; ctx.lineWidth = 1;
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(SIZE, i * CELL); ctx.stroke();
  }

  drawWallSet(state.h_walls, 'H');
  drawWallSet(state.v_walls, 'V');

  // always-visible hints for the player to move
  if (!state.game_over && isHumanTurn()) {
    for (const [ori, r, c] of state.legal_walls) {
      drawWall(ori, r, c, 'rgba(60,180,60,0.12)');
    }
    for (const [r, c] of state.legal_moves) {
      dot(c * CELL + CELL / 2, r * CELL + CELL / 2,
          selected && selected === key([r, c]) ? 'rgba(46,170,70,0.95)' : 'rgba(46,170,70,0.45)',
          selected && selected === key([r, c]) ? 9 : 7);
    }
  }

  // hover feedback
  if (hoverWall) drawWall(hoverWall.ori, hoverWall.r, hoverWall.c, 'rgba(46,170,70,0.85)');
  if (hoverCell && !hoverWall) {
    const [r, c] = hoverCell;
    if (state.legal_moves.some(m => key(m) === key([r, c]))) {
      ctx.fillStyle = 'rgba(46,170,70,0.18)';
      ctx.fillRect(c * CELL + 3, r * CELL + 3, CELL - 6, CELL - 6);
    }
  }

  // last move highlight
  if (state.last_action) {
    const [kind, , arg] = state.last_action;
    if (kind === 'move') {
      const [r, c] = arg;
      ctx.strokeStyle = 'rgba(255,180,0,0.9)'; ctx.lineWidth = 4;
      ctx.strokeRect(c * CELL + 5, r * CELL + 5, CELL - 10, CELL - 10);
    } else if (kind === 'wall') {
      drawWall(arg[0], arg[1], arg[2], 'rgba(255,180,0,0.85)');
    }
  }

  // pawns
  state.positions.forEach((p, i) => {
    const t = display[i];
    const x = t[1] * CELL + CELL / 2, y = t[0] * CELL + CELL / 2;
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.35)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetY = 3;
    const g = ctx.createRadialGradient(x - 7, y - 9, 4, x, y, 25);
    g.addColorStop(0, COLORS[i]);
    g.addColorStop(1, COLORS_DARK[i]);
    ctx.beginPath(); ctx.arc(x, y, 24, 0, Math.PI * 2);
    ctx.fillStyle = g; ctx.fill();
    ctx.restore();
    ctx.lineWidth = 3; ctx.strokeStyle = '#222'; ctx.stroke();
    if (selected === key(p)) {
      ctx.lineWidth = 4; ctx.strokeStyle = '#ffd54f'; ctx.stroke();
    }
    ctx.fillStyle = '#fff'; ctx.font = 'bold 15px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(NAMES[i], x, y + 1);
  });

  updateHud();
}

function drawWallSet(walls, ori) {
  for (const [r, c] of walls) drawWall(ori, r, c, '#43392b');
}

function drawWall(ori, r, c, color) {
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.25)';
  ctx.shadowBlur = 4;
  ctx.shadowOffsetY = 2;
  ctx.fillStyle = color;
  if (ori === 'H') {
    roundRectPath(c * CELL + 1, (r + 1) * CELL - 8, 2 * CELL - 2, 16, 6);
    ctx.fill();
  } else {
    roundRectPath((c + 1) * CELL - 8, r * CELL + 1, 16, 2 * CELL - 2, 6);
    ctx.fill();
  }
  ctx.restore();
}

function dot(x, y, color, r) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
}

function fmtClock(t) {
  if (t === null || t === undefined) return '';
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return m + ':' + String(s).padStart(2, '0');
}

function updateClocks(now) {
  if (!state || !state.time_left) {
    clockEls.forEach(el => { el.textContent = ''; });
    return;
  }
  const elapsed = (now - clockAnchorAt) / 1000;
  for (let i = 0; i < 2; i++) {
    let t = state.time_left[i];
    if (!state.game_over && state.turn === i) t -= elapsed;
    clockEls[i].textContent = fmtClock(Math.max(0, t));
    clockEls[i].classList.toggle('low', t < 10);
  }
}

function updateHud() {
  document.getElementById('w0').textContent = state.walls_left[0] + ' w';
  document.getElementById('w1').textContent = state.walls_left[1] + ' w';
  document.getElementById('p0').classList.toggle('active', !state.game_over && state.turn === 0);
  document.getElementById('p1').classList.toggle('active', !state.game_over && state.turn === 1);

  if (state.game_over) {
    const reason = state.game_over_reason === 'time' ? ' (time out)' : '';
    statusEl.textContent = state.winner === null ? 'Draw' : 'Winner: ' + NAMES[state.winner] + reason;
    hintEl.textContent = 'Click "New game" to play again.';
  } else {
    statusEl.textContent = 'Turn: ' + NAMES[state.turn];
    if (aiBusy) { statusEl.textContent = 'AI thinking…'; hintEl.textContent = ''; }
    else if (!isHumanTurn()) { statusEl.textContent = 'AI thinking…'; hintEl.textContent = ''; }
    else hintEl.textContent = 'Click a green dot to move, or a grid line to place a wall.';
  }
}

// ----- main loop -----

function animate() {
  if (!state) return;
  state.positions.forEach((p, i) => {
    const t = display[i];
    t[0] += (p[0] - t[0]) * 0.18;
    t[1] += (p[1] - t[1]) * 0.18;
    if (Math.abs(p[0] - t[0]) < 0.02) t[0] = p[0];
    if (Math.abs(p[1] - t[1]) < 0.02) t[1] = p[1];
  });
}

function tick(now) {
  animate();
  updateClocks(now);
  if (isAiTurn() && pendingAiAt !== null && now >= pendingAiAt) {
    pendingAiAt = null;
    doAiMove();
  }
  render();
  requestAnimationFrame(tick);
}

// ----- new game -----

function startNew(mode2, aiPlayer, engine, timeControl) {
  return api('/api/new', { mode: mode2, ai_player: aiPlayer, engine, time_control: timeControl });
}

// ----- init -----

(async () => {
  engines = await api('/api/meta');
  populateEngineSelect();
  onModeChange();
  state = await api('/api/state');
  display = state.positions.map(p => [p[0], p[1]]);
  adopt(state);
  requestAnimationFrame(tick);
})();