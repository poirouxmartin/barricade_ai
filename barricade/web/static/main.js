const N = 9, CELL = 72, SIZE = N * CELL;
const COLORS = ['#d64541', '#4169e1'];
const NAMES = ['P1', 'P2'];

const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const modeBtn = document.getElementById('btn-mode');
const hintEl = document.getElementById('hint');

let state = null;
let mode = 'move';      // 'move' | 'wall'
let selected = null;    // key of selected own pawn, or null
let hover = null;       // wall preview {ori,r,c} or null

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

function mousePos(e) {
  const rect = canvas.getBoundingClientRect();
  return [(e.clientX - rect.left) * SIZE / rect.width,
          (e.clientY - rect.top) * SIZE / rect.height];
}

function wallFromMouse(mx, my) {
  const hr = Math.round(my / CELL) - 1, hc = Math.floor(mx / CELL);
  const vc = Math.round(mx / CELL) - 1, vr = Math.floor(my / CELL);
  const hs = { ori: 'H', r: hr, c: hc, d: Math.abs(my - (hr + 1) * CELL) };
  const vs = { ori: 'V', r: vr, c: vc, d: Math.abs(mx - (vc + 1) * CELL) };
  const opts = [hs, vs].filter(legalWall).sort((a, b) => a.d - b.d);
  return opts[0] || null;
}

function legalWall(w) {
  return state.legal_walls.some(x => x[0] === w.ori && x[1] === w.r && x[2] === w.c);
}

function legalMoveTargets() {
  if (!state || selected !== key(state.positions[state.turn])) return [];
  return state.legal_moves.map(key);
}

// ----- rendering -----

function render() {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = '#f5f0e1';
  ctx.fillRect(0, 0, SIZE, SIZE);

  ctx.fillStyle = 'rgba(0,0,0,0.06)';
  ctx.fillRect(0, 0, SIZE, CELL);          // P1 goal (top)
  ctx.fillRect(0, SIZE - CELL, SIZE, CELL); // P2 goal (bottom)

  // grid
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(SIZE, i * CELL); ctx.stroke();
  }

  drawWallSet(state.h_walls, 'H');
  drawWallSet(state.v_walls, 'V');

  // legal move hints
  if (mode === 'move' && selected) {
    for (const [r, c] of state.legal_moves) {
      dot(c * CELL + CELL / 2, r * CELL + CELL / 2, 'rgba(0,160,0,0.9)', 8);
    }
  }

  // wall preview
  if (mode === 'wall' && hover) drawWall(hover.ori, hover.r, hover.c, 'rgba(0,180,0,0.8)');

  // pawns
  state.positions.forEach((p, i) => {
    const x = p[1] * CELL + CELL / 2, y = p[0] * CELL + CELL / 2;
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
  ctx.fillStyle = '#3b3b3b';
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

function updateHud() {
  document.getElementById('w0').textContent = state.walls_left[0] + ' walls';
  document.getElementById('w1').textContent = state.walls_left[1] + ' walls';
  document.getElementById('p0').classList.toggle('active', !state.game_over && state.turn === 0);
  document.getElementById('p1').classList.toggle('active', !state.game_over && state.turn === 1);

  if (state.game_over) {
    statusEl.textContent = 'Winner: ' + NAMES[state.winner];
    hintEl.textContent = 'Click "New game" to play again.';
  } else {
    statusEl.textContent = 'Turn: ' + NAMES[state.turn] + '  (Mode: ' + mode + ')';
    if (!isHumanTurn()) hintEl.textContent = 'AI thinking...';
    else if (mode === 'wall') hintEl.textContent = 'Click a grid line to place a wall (green = legal).';
    else hintEl.textContent = 'Click your pawn, then a highlighted square to move.';
  }
}

// ----- interactions -----

function cellAt(mx, my) {
  const r = Math.floor(my / CELL), c = Math.floor(mx / CELL);
  if (r < 0 || r >= N || c < 0 || c >= N) return null;
  return [r, c];
}

canvas.addEventListener('mousemove', e => {
  if (mode !== 'wall' || !state || !isHumanTurn()) return;
  const [mx, my] = mousePos(e);
  hover = wallFromMouse(mx, my);
  render();
});

canvas.addEventListener('mouseleave', () => {
  hover = null;
  if (mode === 'wall') render();
});

canvas.addEventListener('click', async e => {
  if (!state || !isHumanTurn() || state.game_over) return;
  const [mx, my] = mousePos(e);

  if (mode === 'wall') {
    if (hover) {
      const move = ['wall', [hover.ori, hover.r, hover.c]];
      const prev = state;
      state = await api('/api/move', { move });
      if (!state || state.error) { state = prev; render(); return; }
      selected = null; hover = null;
    }
    render();
    return;
  }

  const cell = cellAt(mx, my);
  if (!cell) return;
  const k = key(cell);

  if (selected) {
    if (legalMoveTargets().includes(k)) {
      const prev = state;
      state = await api('/api/move', { move: ['move', [cell[0], cell[1]]] });
      if (!state || state.error) { state = prev; render(); return; }
      selected = null;
    } else {
      selected = null;
    }
  } else if (k === key(state.positions[state.turn])) {
    selected = k;
  }
  render();
});

modeBtn.addEventListener('click', () => {
  mode = mode === 'move' ? 'wall' : 'move';
  selected = null; hover = null;
  modeBtn.textContent = 'Mode: ' + (mode === 'move' ? 'Move' : 'Wall');
  render();
});

// ----- new game -----

function startNew(mode2, aiPlayer) {
  return api('/api/new', { mode: mode2, ai_player: aiPlayer });
}

document.getElementById('btn-new').addEventListener('click', async () => {
  const choice = prompt('Mode:\n1 = PvP\n2 = vs AI\n3 = AI vs AI');
  let state2;
  if (choice === '1') state2 = await startNew('pvp');
  else if (choice === '2') state2 = await startNew('ai', 1);
  else if (choice === '3') state2 = await startNew('ai2');
  else return;
  state = state2; selected = null; hover = null;
  render();
});

// ----- init -----

(async () => {
  state = await api('/api/state');
  render();
})();