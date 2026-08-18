const N = 9, CELL = 72, SIZE = N * CELL;
const COLORS = ['#d64541', '#4169e1'];
const NAMES = ['P1', 'P2'];
const AI_DELAY = 400; // ms pause so the last move is visible before the AI replies

const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const hintEl = document.getElementById('hint');
const clockEls = [document.getElementById('c0'), document.getElementById('c1')];

let state = null;
let display = [[8, 4], [0, 4]]; // animated pawn positions
let selected = null;            // key of selected own pawn
let hoverWall = null;           // {ori,r,c,d} or null
let aiBusy = false;
let pendingAiAt = null;         // timestamp to fire the delayed AI move
let clockAnchorAt = 0;          // performance.now() when current state was received

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
  return [(e.clientX - rect.left) * SIZE / rect.width,
          (e.clientY - rect.top) * SIZE / rect.height];
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
  if (isAiTurn()) pendingAiAt = performance.now() + AI_DELAY;
}

// ----- interaction -----

canvas.addEventListener('mousemove', e => {
  if (!state || !isHumanTurn()) return;
  const [mx, my] = mousePos(e);
  const w = wallFromMouse(mx, my);
  hoverWall = (w && w.d <= 14) ? w : null;
});

canvas.addEventListener('mouseleave', () => { hoverWall = null; });

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

  if (selected) {
    if (state.legal_moves.some(m => key(m) === k)) {
      const prev = state;
      const res = await api('/api/move', { move: ['move', [cell[0], cell[1]]] });
      if (!res || res.error) { state = prev; render(); return; }
      adopt(res);
    } else {
      selected = null;
    }
  } else if (k === key(state.positions[state.turn])) {
    selected = k;
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

// ----- rendering -----

function render() {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = '#f5f0e1';
  ctx.fillRect(0, 0, SIZE, SIZE);

  ctx.fillStyle = 'rgba(0,0,0,0.06)';
  ctx.fillRect(0, 0, SIZE, CELL);           // P1 goal (top)
  ctx.fillRect(0, SIZE - CELL, SIZE, CELL); // P2 goal (bottom)

  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(SIZE, i * CELL); ctx.stroke();
  }

  drawWallSet(state.h_walls, 'H');
  drawWallSet(state.v_walls, 'V');

  // always-visible hints for the player to move
  if (!state.game_over) {
    for (const [ori, r, c] of state.legal_walls) {
      drawWall(ori, r, c, 'rgba(0,180,0,0.15)');
    }
    const dots = state.legal_moves.map(key);
    for (const [r, c] of state.legal_moves) {
      const active = selected && dots.includes(key([r, c]));
      dot(c * CELL + CELL / 2, r * CELL + CELL / 2,
          active ? 'rgba(0,160,0,1)' : 'rgba(0,160,0,0.5)', active ? 9 : 7);
    }
  }

  if (hoverWall) drawWall(hoverWall.ori, hoverWall.r, hoverWall.c, 'rgba(0,200,0,0.9)');

  // last move highlight
  if (state.last_action) {
    const [kind, , arg] = state.last_action;
    if (kind === 'move') {
      const [r, c] = arg;
      ctx.strokeStyle = 'rgba(255,180,0,0.9)'; ctx.lineWidth = 4;
      ctx.strokeRect(c * CELL + 5, r * CELL + 5, CELL - 10, CELL - 10);
    } else if (kind === 'wall') {
      drawWall(arg[0], arg[1], arg[2], 'rgba(255,180,0,0.9)');
    }
  }

  state.positions.forEach((p, i) => {
    const t = display[i];
    const x = t[1] * CELL + CELL / 2, y = t[0] * CELL + CELL / 2;
    ctx.beginPath(); ctx.arc(x, y, 25, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[i]; ctx.fill();
    ctx.lineWidth = 3; ctx.strokeStyle = '#222'; ctx.stroke();
    if (selected === key(p)) {
      ctx.lineWidth = 4; ctx.strokeStyle = '#ffd54f'; ctx.stroke();
    }
    ctx.fillStyle = '#fff'; ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(NAMES[i], x, y);
  });

  updateHud();
}

function drawWallSet(walls, ori) {
  for (const [r, c] of walls) drawWall(ori, r, c, '#3b3b3b');
}

function drawWall(ori, r, c, color) {
  ctx.fillStyle = color;
  if (ori === 'H') ctx.fillRect(c * CELL, (r + 1) * CELL - 7, 2 * CELL, 14);
  else ctx.fillRect((c + 1) * CELL - 7, r * CELL, 14, 2 * CELL);
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
    if (aiBusy) hintEl.textContent = 'AI thinking...';
    else if (!isHumanTurn()) hintEl.textContent = 'AI thinking...';
    else hintEl.textContent = 'Click your pawn, a green square to move, or a grid line to place a wall.';
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

document.getElementById('btn-new').addEventListener('click', async () => {
  const choice = prompt('Mode:\n1 = PvP\n2 = vs AI (kernel)\n3 = vs AI (alpha-beta)\n4 = vs AI (greedy)\n5 = vs AI (MCTS)\n6 = AI vs AI (kernel)\n\nTime per player in seconds (0 = none):');
  if (choice === null) return;
  const parts = choice.trim().split(/\s+/);
  const sel = parts[0];
  let tc = parseFloat(parts[1]);
  if (!Number.isFinite(tc) || tc < 0) tc = null;
  let state2;
  if (sel === '1') state2 = await startNew('pvp', 1, 'kernel', tc);
  else if (sel === '2') state2 = await startNew('ai', 1, 'kernel', tc);
  else if (sel === '3') state2 = await startNew('ai', 1, 'alphabeta', tc);
  else if (sel === '4') state2 = await startNew('ai', 1, 'greedy', tc);
  else if (sel === '5') state2 = await startNew('ai', 1, 'mcts', tc);
  else if (sel === '6') state2 = await startNew('ai2', 1, 'kernel', tc);
  else return;
  if (!state2 || state2.error) return;
  display = state2.positions.map(p => [p[0], p[1]]);
  adopt(state2);
});

// ----- init -----

(async () => {
  state = await api('/api/state');
  display = state.positions.map(p => [p[0], p[1]]);
  adopt(state);
  requestAnimationFrame(tick);
})();