const MARGIN = 24, CELL = 72, N = 9, BOARD = N * CELL, W = BOARD + 2 * MARGIN, H = W;
const COLORS = ['#e76f51', '#4d7cfe'];
const COLORS_DARK = ['#c0563d', '#3457c9'];
const NAMES = ['P1', 'P2'];
const FILES = 'abcdefghi';
const AI_DELAY = 400; // ms pause so the last move is visible before the AI replies

const canvas = document.getElementById('board');
canvas.width = W; canvas.height = H;
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const hintEl = document.getElementById('hint');
const debugEl = document.getElementById('debug');
const clockEls = [document.getElementById('c0'), document.getElementById('c1')];

const analEl = document.getElementById('analysis');
const analStatusEl = document.getElementById('anal-status');
const analStatsEl = document.getElementById('anal-stats');
const analPvEl = document.getElementById('anal-pv');
const analTopEl = document.getElementById('anal-top');
const analParamsEl = document.getElementById('anal-params');
const parDepthEl = document.getElementById('par-depth');
const parTickEl = document.getElementById('par-tick');
const parNodesEl = document.getElementById('par-nodes');
const heurBodyEl = document.getElementById('heur-body');
const evalbarEl = document.getElementById('evalbar');
const evalFillEl = document.getElementById('eval-fill');
const evalLEl = document.getElementById('eval-l');
const evalREl = document.getElementById('eval-r');
const evalDetailEl = document.getElementById('eval-detail');
const navEl = document.getElementById('nav');
const navPosEl = document.getElementById('nav-pos');
const navListEl = document.getElementById('nav-list');
const matchEl = document.getElementById('match');
const matchStatusEl = document.getElementById('match-status');
const matchResultsEl = document.getElementById('match-results');

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
let spent = [0, 0];             // accumulated seconds per player (count-up mode)
let turnStartedAt = 0;          // performance.now() when the current turn began
let engines = {};
let analysisTimer = null;
let analysisState = null;
let view = null;                // reviewed snapshot, null = live position
let matchView = null;           // live snapshot of a running bot match
let navIndex = null;
let historySnapshots = [];
let matchTimer = null;
let matchState = null;
let evalData = null;            // /api/eval of the live position

const key = p => p[0] + ',' + p[1];
const kof = (p) => p[0] + ',' + p[1];

async function api(path, body) {
  const r = await fetch(path, body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {});
  return r.json();
}

function cellName(r, c) { return FILES[c] + (9 - r); }   // a1 = bottom-left (P1), i9 = top-right (P2)
function fmtMove(m) {
  if (!m) return '—';
  const [kind, arg] = m;
  if (kind === 'move') return cellName(arg[0], arg[1]);
  return arg[0] + ' ' + cellName(arg[1], arg[2]);
}
function isLegal(m) {
  if (!state) return false;
  if (m[0] === 'move') return state.legal_moves.some(x => key(x) === key(m[1]));
  return state.legal_walls.some(x => x[0] === m[1][0] && x[1] === m[1][1] && x[2] === m[1][2]);
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
  let mx = (e.clientX - rect.left) * W / rect.width;
  let my = (e.clientY - rect.top) * W / rect.height;
  mx -= MARGIN; my -= MARGIN;
  if (flip) { mx = BOARD - mx; my = BOARD - my; }
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
  if (state && !state.time_left && !state.game_over && s.turn !== state.turn) {
    spent[state.turn] += (performance.now() - turnStartedAt) / 1000;
  }
  state = s;
  selected = null;
  clockAnchorAt = performance.now();
  if (!s.time_left) turnStartedAt = performance.now();
  stopAnalysisUI();
  resetView();
  updateDebug();
  fetchHistory();
  fetchEval();
  if (isAiTurn()) pendingAiAt = performance.now() + AI_DELAY;
}

// ----- navigation -----

function resetView() {
  view = null;
  navIndex = null;
}

function updateNav() {
  const count = historySnapshots.length;
  navEl.classList.toggle('hidden', matchView !== null || count === 0);
  if (count === 0) { navListEl.innerHTML = ''; return; }
  const idx = navIndex === null ? count - 1 : navIndex;
  navPosEl.textContent = (navIndex === null ? count - 1 : navIndex) + '/' + (count - 1);
  navListEl.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const b = document.createElement('button');
    b.textContent = i === 0 ? 'début' : i + '.' + fmtMove(historySnapshots[i].last_action);
    b.title = b.textContent;
    b.classList.toggle('cur', i === idx);
    b.addEventListener('click', () => goNav(i));
    navListEl.appendChild(b);
  }
  const cur = navListEl.children[idx];
  if (cur) cur.scrollIntoView({ inline: 'nearest', block: 'nearest' });
}

async function fetchHistory() {
  try {
    const res = await api('/api/history');
    if (res && Array.isArray(res.snapshots)) {
      historySnapshots = res.snapshots;
      updateNav();
    }
  } catch (_) { /* ignore */ }
}

function goNav(i) {
  if (i === historySnapshots.length - 1) {
    resetView();
    fetchEval();
  } else {
    view = historySnapshots[i];
    navIndex = i;
    display = view.positions.map(p => [p[0], p[1]]);
  }
  render();
  updateNav();
}

document.getElementById('nav-prev').addEventListener('click', () => {
  const idx = navIndex === null ? historySnapshots.length - 1 : navIndex;
  if (idx > 0) goNav(idx - 1);
});
document.getElementById('nav-next').addEventListener('click', () => {
  const idx = navIndex === null ? historySnapshots.length - 1 : navIndex;
  if (idx < historySnapshots.length - 1) goNav(idx + 1);
});
document.getElementById('nav-live').addEventListener('click', () => {
  goNav(historySnapshots.length - 1);
});

// arrow-key navigation across the game
document.addEventListener('keydown', e => {
  const tag = (e.target.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (matchView || historySnapshots.length === 0) return;
  const last = historySnapshots.length - 1;
  const idx = navIndex === null ? last : navIndex;
  if (e.key === 'ArrowLeft') { e.preventDefault(); if (idx > 0) goNav(idx - 1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); if (idx < last) goNav(idx + 1); }
  else if (e.key === 'Home') { e.preventDefault(); goNav(0); }
  else if (e.key === 'End') { e.preventDefault(); goNav(last); }
});

// ----- interaction -----

canvas.addEventListener('mousemove', e => {
  hoverWall = null;
  hoverCell = null;
  if (!state || view || matchView || !isHumanTurn()) return;
  const [mx, my] = mousePos(e);
  const w = wallFromMouse(mx, my);
  hoverWall = (w && w.d <= 14) ? w : null;
  hoverCell = cellAt(mx, my);
});

canvas.addEventListener('mouseleave', () => { hoverWall = null; hoverCell = null; });

canvas.addEventListener('click', async e => {
  if (!state || view || matchView || state.game_over || !isHumanTurn()) return;
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
  const poll = setInterval(async () => {
    try {
      const info = await api('/api/info');
      if (info && info.thinking) updateDebugFromInfo(info);
    } catch (_) { /* keep polling */ }
  }, 200);
  const prev = state;
  const res = await api('/api/ai', {});
  clearInterval(poll);
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
const matchF = document.getElementById('m-match-f');
const p1Sel = document.getElementById('m-p1');
const p2Sel = document.getElementById('m-p2');
const mTimeInp = document.getElementById('m-mtime');
const mDepthInp = document.getElementById('m-mdepth');
const mGamesInp = document.getElementById('m-mgames');

const ENGINE_LABELS = {
  kernel: 'kernel (compiled — fast)',
  alphabeta: 'alphabeta (compiled — fast)',
  mcts: 'mcts (experimental)',
  greedy: 'greedy',
  random: 'random',
};
const ENGINE_ORDER = ['kernel', 'alphabeta', 'greedy', 'mcts', 'random'];

function populateEngineSelect() {
  const avail = new Set(engines.engines || ['greedy']);
  for (const sel of [engineSel, p1Sel, p2Sel]) {
    sel.innerHTML = '';
    for (const name of ENGINE_ORDER) {
      if (!avail.has(name)) continue;
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = ENGINE_LABELS[name] || name;
      sel.appendChild(opt);
    }
  }
  if (engines.default && avail.has(engines.default)) engineSel.value = engines.default;
}

function openMenu() {
  sideSel.value = String(state && state.ai_player === 0 ? 0 : 1);
  menuEl.classList.remove('hidden');
}

function closeMenu() {
  menuEl.classList.add('hidden');
}

function onModeChange() {
  const mode = modeSel.value;
  engineF.classList.toggle('hidden', mode === 'pvp' || mode === 'match');
  sideF.classList.toggle('hidden', mode !== 'ai');
  matchF.classList.toggle('hidden', mode !== 'match');
}

document.getElementById('btn-new').addEventListener('click', openMenu);
document.getElementById('m-cancel').addEventListener('click', closeMenu);
modeSel.addEventListener('change', onModeChange);

document.getElementById('m-start').addEventListener('click', async () => {
  stopAnalysisUI();
  matchEl.classList.add('hidden');
  clearInterval(matchTimer); matchTimer = null;
  const mode2 = modeSel.value;
  if (mode2 === 'match') {
    closeMenu();
    await startMatch();
    return;
  }
  const engine = engineSel.value;
  const aiPlayer = mode2 === 'ai' ? parseInt(sideSel.value, 10) : 1;
  const tc = Math.max(0, parseFloat(timeInp.value) || 0);
  closeMenu();
  const state2 = await startNew(mode2, aiPlayer, engine, tc || null);
  if (!state2 || state2.error) return;
  if (flipInp.checked) flip = true;
  spent = [0, 0];
  turnStartedAt = performance.now();
  display = state2.positions.map(p => [p[0], p[1]]);
  adopt(state2);
});

document.getElementById('btn-flip').addEventListener('click', () => {
  flip = !flip;
});

document.getElementById('btn-analyze').addEventListener('click', () => {
  if (!state || view || state.game_over) return;
  startAnalysis();
});

const analEngineEl = document.getElementById('sel-anal-engine');
analEngineEl.addEventListener('change', () => {
  updateAnalParams();
  if (analysisTimer) startAnalysis();
});

function updateAnalParams() {
  const mcts = analEngineEl.value === 'mcts';
  parDepthEl.parentElement.style.display = mcts ? 'none' : '';
  parTickEl.parentElement.style.display = mcts ? '' : 'none';
  parNodesEl.parentElement.style.display = mcts ? '' : 'none';
}

document.getElementById('btn-anal-stop').addEventListener('click', () => {
  stopAnalysis(false);
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

function updateDebugFromInfo(info) {
  if (!debugOn) return;
  debugEl.classList.remove('hidden');
  const lines = ['Engine: ' + (info.engine || '?') + '  (thinking…)'];
  if (info.progress && Object.keys(info.progress).length) {
    lines.push(Object.entries(info.progress).map(([k, v]) => k + ': ' + v).join('  ·  '));
  }
  debugEl.textContent = lines.join('\n');
}

// ----- analysis -----

function gapWinPct(gap) {
  return 50 + 50 * Math.tanh(gap / 4);
}

function stopAnalysisUI() {
  clearInterval(analysisTimer);
  analysisTimer = null;
  analysisState = null;
  analEl.classList.add('hidden');
  analEngineEl.classList.add('hidden');
}

async function startAnalysis() {
  const engine = analEngineEl.value;
  const body = { engine };
  if (engine === 'mcts') {
    body.mcts_tick = Math.max(0.05, parseFloat(parTickEl.value) || 0.25);
    body.mcts_nodes = Math.max(16, (parseFloat(parNodesEl.value) || 262) * 1024);
  } else {
    body.max_depth = Math.max(1, Math.min(30, parseInt(parDepthEl.value, 10) || 30));
  }
  const res = await api('/api/analysis/start', body);
  if (res && res.error) {
    analStatusEl.textContent = 'Unavailable: ' + res.error;
    analEl.classList.remove('hidden');
    return;
  }
  analEl.classList.remove('hidden');
  analEngineEl.classList.remove('hidden');
  analParamsEl.classList.remove('hidden');
  updateAnalParams();
  clearInterval(analysisTimer);
  analysisTimer = setInterval(pollAnalysis, 500);
  pollAnalysis();
}

async function stopAnalysis(quiet) {
  clearInterval(analysisTimer);
  analysisTimer = null;
  await api('/api/analysis/stop', {});
  if (!quiet) pollAnalysis();
}

async function pollAnalysis() {
  const a = await api('/api/analysis');
  renderAnalysis(a);
}

function renderAnalysis(a) {
  if (!a || a.error) {
    analStatusEl.textContent = a && a.error ? 'Error: ' + a.error : '…';
    return;
  }
  analysisState = a;
  const frozen = state && state.turn !== a.turn ? ' · position figée' : '';
  analStatusEl.textContent = (a.running ? 'searching' : 'stopped') + frozen;
  if (a.running) analStatusEl.classList.add('pulse');
  else analStatusEl.classList.remove('pulse');

  if (a.depth === 0) {
    analStatsEl.textContent = a.running ? (a.engine === 'mcts' ? 'sampling…' : 'compiling…') : 'no data yet';
    analPvEl.textContent = '';
    analTopEl.innerHTML = '';
    return;
  }

  if (a.engine === 'mcts') {
    analStatsEl.textContent = 'MCTS  ·  ' + a.nodes.toLocaleString() + ' iters'
      + '  ·  ' + (a.nps / 1e6).toFixed(2) + 'M ips'
      + '  ·  Score ' + (a.score > 0 ? '+' : '') + a.score
      + '  ·  Win ' + Math.round(gapWinPct(a.dist_gap)) + '%';
  } else {
    analStatsEl.textContent = 'Depth ' + a.depth
      + '  ·  Score ' + (a.score > 0 ? '+' : '') + a.score
      + '  ·  Win ' + Math.round(gapWinPct(a.dist_gap)) + '%'
      + '  ·  ' + a.nodes.toLocaleString() + ' nodes'
      + '  ·  ' + (a.nps / 1e6).toFixed(2) + 'M nps';
  }

  if (a.pv && a.pv.length) {
    analPvEl.textContent = 'Best ' + fmtMove(a.pv[0]) + '  ·  PV: ' + a.pv.map(fmtMove).join(' ');
  } else {
    analPvEl.textContent = 'Best: ' + (a.best_move ? fmtMove(a.best_move) : '—');
  }

  const playable = !view && positionMatches(a);
  analTopEl.innerHTML = '';
  if (a.top_moves && a.top_moves.length) {
    const best = a.top_moves[0].score;
    const worst = a.top_moves[a.top_moves.length - 1].score;
    const span = Math.max(best - worst, 1);
    for (let i = 0; i < a.top_moves.length; i++) {
      const t = a.top_moves[i];
      const row = document.createElement('div');
      row.className = 'top-row';
      const ok = playable && isLegal(t.move);
      row.classList.toggle('clickable', ok);
      row.title = ok ? 'Jouer ce coup' : '';
      row.addEventListener('click', () => { if (ok) playAnalysisMove(t.move); });
      const label = document.createElement('span');
      label.className = 'top-label';
      label.textContent = (i + 1) + '. ' + fmtMove(t.move);
      const bar = document.createElement('div');
      bar.className = 'top-bar-wrap';
      const fill = document.createElement('div');
      fill.className = 'top-bar-fill';
      fill.style.width = Math.max(4, 100 * (best - t.score + span / 4) / (span * 1.25)) + '%';
      bar.appendChild(fill);
      const val = document.createElement('span');
      val.className = 'top-val';
      val.textContent = (t.score > 0 ? '+' : '') + t.score;
      row.appendChild(label);
      row.appendChild(bar);
      row.appendChild(val);
      analTopEl.appendChild(row);
    }
  }
}

async function playAnalysisMove(move) {
  if (!move || view) return;
  stopAnalysisUI();
  const res = await api('/api/move', { move });
  if (res && !res.error) {
    adopt(res);
  } else {
    render();
  }
}

// ----- evaluation bar + heuristics -----

function positionMatches(a) {
  if (!state || !a) return false;
  if (state.turn !== a.turn) return false;
  if (kof(state.positions[0]) !== kof(a.positions[0])) return false;
  if (kof(state.positions[1]) !== kof(a.positions[1])) return false;
  const s1 = state.h_walls.map(key).sort().join(), s2 = a.h_walls.map(key).sort().join();
  if (s1 !== s2) return false;
  const v1 = state.v_walls.map(key).sort().join(), v2 = a.v_walls.map(key).sort().join();
  if (v1 !== v2) return false;
  return true;
}

function evalSource() {
  if (view || !state) return null;
  if (analysisState && analysisState.depth > 0 && positionMatches(analysisState)) {
    return { turn: analysisState.turn, dist_gap: analysisState.dist_gap,
             confidence: analysisState.confidence, score: analysisState.score };
  }
  return evalData;
}

function updateEvalBar(data) {
  if (!data) {
    evalbarEl.classList.add('hidden');
    evalDetailEl.classList.add('hidden');
    return;
  }
  const p1win = data.turn === 0 ? gapWinPct(data.dist_gap) : 100 - gapWinPct(data.dist_gap);
  evalFillEl.style.width = p1win + '%';
  evalLEl.textContent = Math.round(p1win) + '%';
  evalREl.textContent = Math.round(100 - p1win) + '%';
  evalbarEl.classList.remove('hidden');
  const lead = data.dist_gap > 0 ? '+' + data.dist_gap + ' cases'
    : data.dist_gap < 0 ? '−' + (-data.dist_gap) + ' cases' : 'équilibré';
  evalDetailEl.textContent = NAMES[data.turn] + ' à jouer · ' + lead
    + ' · confiance ' + Math.round((data.confidence || 1) * 100) + '%'
    + ' · score ' + (data.score > 0 ? '+' : '') + data.score;
  evalDetailEl.classList.remove('hidden');
}

async function fetchEval() {
  try {
    const res = await api('/api/eval');
    if (res && !res.error) {
      evalData = res;
      renderHeuristics(res);
      render();
    }
  } catch (_) { /* ignore */ }
}

const MCTS_DESC =
  '<h4>MCTS</h4>'
  + '<p>UCT : à chaque itération on sélectionne le nœud qui maximise '
  + '<code>v + C·√(ln N / n)</code> (C = 1.4), on l\'étend (un enfant par itération), '
  + 'puis on joue une <b>simulation</b> : course BFS-greedy — on avance sur une case parmi les '
  + 'plus proches du but (tirage <b>pseudo-aléatoire</b>, PRNG déterministe de numba, seed 7), '
  + 'avec 40 % de chances de tenter un mur légal au hasard, 60 plies maximum ; si la partie '
  + 'n\'est pas finie, l\'évaluation statique <code>tanh(éval / 30)</code> clôture. '
  + 'Rétro-propagation négamax. Le score d\'un coup = moyenne des valeurs de ses simulations ; '
  + 'le meilleur coup = le plus visité.</p>';

const HEUR_DESC =
  '<h4>Heuristiques (évaluation statique)</h4>'
  + '<p>Pour chaque pion, BFS vers sa rangée but ; avantage = '
  + '(distance adverse − distance perso) × 4. En avance, la <b>confiance</b> = '
  + 'max(1 − 0.12 × murs_adverses, 0.3) pondère l\'avance (un adversaire sans mur ne peut plus '
  + 'allonger ton chemin). Un déficit n\'est jamais dévalué. La <b>parité de murs</b> '
  + 'ajoute +1 par mur restant de plus que l\'adversaire.</p>';

function renderHeuristics(ev) {
  let html = HEUR_DESC;
  if (ev && ev.components) {
    html += '<h4>Valeurs actuelles</h4>';
    for (const c of ev.components) {
      html += '<div class="heur-line"><span class="k">' + c.name + '</span>'
        + '<span class="v">' + c.formula + '</span></div>'
        + '<div class="heur-desc">' + c.note + '</div>';
    }
    html += '<div class="heur-line"><span class="k">Win % (cases)</span>'
      + '<span class="v">' + Math.round(gapWinPct(ev.dist_gap)) + '%</span></div>';
  }
  html += MCTS_DESC;
  heurBodyEl.innerHTML = html;
}

// ----- match -----

async function startMatch() {
  const body = {
    p1: p1Sel.value,
    p2: p2Sel.value,
    time: Math.max(0.05, parseFloat(mTimeInp.value) || 1),
    depth: Math.max(1, Math.min(30, parseInt(mDepthInp.value, 10) || 10)),
    games: Math.max(1, Math.min(20, parseInt(mGamesInp.value, 10) || 1)),
  };
  stopAnalysisUI();
  const res = await api('/api/match/start', body);
  if (res && res.error) {
    statusEl.textContent = 'Match error: ' + res.error;
    return;
  }
  matchEl.classList.remove('hidden');
  clearInterval(matchTimer);
  matchTimer = setInterval(pollMatch, 500);
  pollMatch();
}

async function pollMatch() {
  const m = await api('/api/match/state');
  if (m && m.running !== undefined) renderMatch(m);
  if (m && m.running && m.snapshot) {
    matchView = m.snapshot;
    display = matchView.positions.map(p => [p[0], p[1]]);
  }
  if (m && !m.running && (matchState && matchState.running || matchView)) {
    matchView = null;
    clearInterval(matchTimer);
    matchTimer = null;
    const st = await api('/api/state');
    if (st && !st.error) {
      display = st.positions.map(p => [p[0], p[1]]);
      adopt(st);
    }
  }
  matchState = m;
  render();
  updateNav();
}

function renderMatch(m) {
  if (!m || m.running === undefined) return;
  matchStatusEl.textContent = m.running
    ? m.status + ' · coup ' + m.ply
    : (m.status === 'done' ? 'terminé' : m.status);
  if (m.running) matchStatusEl.classList.add('pulse');
  else matchStatusEl.classList.remove('pulse');
  let html = '<div class="match-score">'
    + (m.p1 || '?') + ' ' + (m.score ? m.score[0] : 0)
    + ' — ' + (m.score ? m.score[1] : 0) + ' ' + (m.p2 || '?')
    + ' · ' + (m.draws || 0) + ' nulles</div>';
  if (m.results) {
    for (const r of m.results) {
      const w = r.winner === 0 ? 'P1' : r.winner === 1 ? 'P2' : 'nulle';
      html += '<div class="match-game ' + (r.winner !== null ? 'win' : '') + '">G' + r.game
        + ': ' + w + ' · ' + r.plies + ' plies · ' + (r.reason || '') + '</div>';
    }
  }
  matchResultsEl.innerHTML = html;
}

document.getElementById('btn-match-stop').addEventListener('click', async () => {
  await api('/api/match/stop', {});
  pollMatch();
});

// ----- rendering -----

function badge(x, y, text, bg, fg) {
  ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fillStyle = bg; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = '#111'; ctx.stroke();
  ctx.fillStyle = fg || '#111'; ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y + 0.5);
}

function arrow(x1, y1, x2, y2, color) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 4) return;
  const ux = dx / len, uy = dy / len;
  const a1 = Math.atan2(dy, dx);
  ctx.strokeStyle = color; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(x1 + ux * 26, y1 + uy * 26); ctx.lineTo(x2 - ux * 16, y2 - uy * 16); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2 - ux * 16, y2 - uy * 16);
  ctx.lineTo(x2 - ux * 16 - 8 * Math.cos(a1 - 0.45), y2 - uy * 16 - 8 * Math.sin(a1 - 0.45));
  ctx.lineTo(x2 - ux * 16 - 8 * Math.cos(a1 + 0.45), y2 - uy * 16 - 8 * Math.sin(a1 + 0.45));
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
}

function wallCenter(ori, r, c) {
  return [(c + 1) * CELL, (r + 1) * CELL];
}

function drawAnalysisOverlay() {
  const a = analysisState;
  if (!a || !a.pv || !a.pv.length) return;
  if (view || matchView || !positionMatches(a)) return;

  // PV: numbered arrows/badges + wall heads, following the analyzed position
  const pos = a.positions.map(p => [p[0], p[1]]);
  let turn = a.turn;
  for (let i = 0; i < a.pv.length; i++) {
    const mv = a.pv[i];
    const num = String(i + 1);
    if (mv[0] === 'move') {
      const [r, c] = mv[1];
      const from = pos[turn];
      arrow(from[1] * CELL + CELL / 2, from[0] * CELL + CELL / 2,
            c * CELL + CELL / 2, r * CELL + CELL / 2, 'rgba(255,255,255,0.8)');
      badge(c * CELL + CELL / 2, r * CELL + CELL / 2, num, 'rgba(255,255,255,0.95)');
      pos[turn] = [r, c];
    } else {
      const [ori, r, c] = mv[1];
      drawWall(ori, r, c, 'rgba(255,224,130,0.9)');
      const [x, y] = wallCenter(ori, r, c);
      badge(x, y, num, 'rgba(255,224,130,0.95)');
    }
    turn = 1 - turn;
  }

  // top moves: gold rank badges + wall heads (same position to move)
  if (a.top_moves && a.top_moves.length) {
    for (let i = 0; i < Math.min(a.top_moves.length, 5); i++) {
      const mv = a.top_moves[i].move;
      let x, y;
      if (mv[0] === 'move') {
        const [r, c] = mv[1];
        x = c * CELL + CELL / 2; y = r * CELL + CELL / 2;
      } else {
        const [ori, r, c] = mv[1];
        drawWall(ori, r, c, 'rgba(255,213,79,0.8)');
        [x, y] = wallCenter(ori, r, c);
      }
      badge(x, y, String(i + 1), 'rgba(255,213,79,0.95)');
    }
  }
}

function applyTransform() {
  if (flip) ctx.setTransform(-1, 0, 0, -1, W - MARGIN, W - MARGIN);
  else ctx.setTransform(1, 0, 0, 1, MARGIN, MARGIN);
}

function labelPos(bx, by) {
  return [flip ? W - MARGIN - bx : MARGIN + bx,
          flip ? W - MARGIN - by : MARGIN + by];
}

function drawCoordinates() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = 'rgba(240,244,252,0.9)';
  ctx.font = '600 13px ui-monospace, Consolas, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let c = 0; c < N; c++) {
    const [sx, sy] = labelPos(c * CELL + CELL / 2, BOARD);
    ctx.fillText(FILES[c], sx, sy);
  }
  for (let r = 0; r < N; r++) {
    const [sx, sy] = labelPos(0, r * CELL + CELL / 2);
    ctx.fillText(String(9 - r), sx, sy);
  }
  ctx.restore();
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
  const s = matchView || view || state;
  if (!s) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#1c212b';
  ctx.fillRect(0, 0, W, H);
  drawCoordinates();
  applyTransform();

  // board background (dark)
  const bg = ctx.createLinearGradient(0, 0, 0, BOARD);
  bg.addColorStop(0, '#2b3140');
  bg.addColorStop(1, '#1f2530');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, BOARD, BOARD);

  // goal zones
  const goalTop = ctx.createLinearGradient(0, 0, 0, CELL);
  goalTop.addColorStop(0, COLORS[0] + '99');
  goalTop.addColorStop(1, COLORS[0] + '00');
  ctx.fillStyle = goalTop;
  ctx.fillRect(0, 0, BOARD, CELL);
  const goalBot = ctx.createLinearGradient(0, BOARD - CELL, 0, BOARD);
  goalBot.addColorStop(0, COLORS[1] + '00');
  goalBot.addColorStop(1, COLORS[1] + '99');
  ctx.fillStyle = goalBot;
  ctx.fillRect(0, BOARD - CELL, BOARD, CELL);

  // grid
  ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.lineWidth = 1;
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, BOARD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(BOARD, i * CELL); ctx.stroke();
  }

  drawWallSet(s.h_walls, 'H');
  drawWallSet(s.v_walls, 'V');

  // always-visible hints for the player to move (legal moves only)
  if (!s.game_over && !view && !matchView && isHumanTurn()) {
    for (const [r, c] of s.legal_moves) {
      dot(c * CELL + CELL / 2, r * CELL + CELL / 2,
          selected && selected === key([r, c]) ? 'rgba(64,220,180,0.9)' : 'rgba(64,220,180,0.28)',
          selected && selected === key([r, c]) ? 9 : 6);
    }
  }

  // hover feedback
  if (!view && !matchView && hoverWall) drawWall(hoverWall.ori, hoverWall.r, hoverWall.c, 'rgba(64,220,180,0.5)');
  if (!view && !matchView && hoverCell && !hoverWall) {
    const [r, c] = hoverCell;
    if (s.legal_moves.some(m => key(m) === key([r, c]))) {
      ctx.fillStyle = 'rgba(64,220,180,0.12)';
      ctx.fillRect(c * CELL + 3, r * CELL + 3, CELL - 6, CELL - 6);
    }
  }

  // last move highlight
  if (s.last_action) {
    const [kind, , arg] = s.last_action;
    if (kind === 'move') {
      const [r, c] = arg;
      ctx.strokeStyle = 'rgba(255,180,0,0.9)'; ctx.lineWidth = 4;
      ctx.strokeRect(c * CELL + 5, r * CELL + 5, CELL - 10, CELL - 10);
    } else if (kind === 'wall') {
      drawWall(arg[0], arg[1], arg[2], 'rgba(255,180,0,0.85)');
    }
  }

  // pawns
  s.positions.forEach((p, i) => {
    const t = display[i];
    const x = t[1] * CELL + CELL / 2, y = t[0] * CELL + CELL / 2;
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.45)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetY = 3;
    const g = ctx.createRadialGradient(x - 7, y - 9, 4, x, y, 25);
    g.addColorStop(0, COLORS[i]);
    g.addColorStop(1, COLORS_DARK[i]);
    ctx.beginPath(); ctx.arc(x, y, 24, 0, Math.PI * 2);
    ctx.fillStyle = g; ctx.fill();
    ctx.restore();
    ctx.lineWidth = 3; ctx.strokeStyle = '#10151f'; ctx.stroke();
    if (selected === key(s.positions[i])) {
      ctx.lineWidth = 4; ctx.strokeStyle = '#ffd54f'; ctx.stroke();
    }
    ctx.fillStyle = '#fff'; ctx.font = 'bold 15px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(NAMES[i], x, y + 1);
  });

  drawAnalysisOverlay();
  updateEvalBar(evalSource());
  updateHud();
}

function drawWallSet(walls, ori) {
  for (const [r, c] of walls) drawWall(ori, r, c, '#9a8c74');
}

function drawWall(ori, r, c, color) {
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.35)';
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
  if (!state) return;
  if (!state.time_left) {
    const el = (now - turnStartedAt) / 1000;
    for (let i = 0; i < 2; i++) {
      const t = spent[i] + (!state.game_over && state.turn === i ? el : 0);
      clockEls[i].textContent = fmtClock(t);
      clockEls[i].classList.toggle('low', false);
    }
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
  const s = matchView || view || state;
  if (!s) return;
  document.getElementById('w0').textContent = s.walls_left[0] + ' w';
  document.getElementById('w1').textContent = s.walls_left[1] + ' w';
  document.getElementById('p0').classList.toggle('active', !s.game_over && s.turn === 0);
  document.getElementById('p1').classList.toggle('active', !s.game_over && s.turn === 1);

  if (matchView) {
    const m = matchState || {};
    statusEl.textContent = 'Match ' + (m.p1 || '') + ' vs ' + (m.p2 || '')
      + ' — partie ' + (m.current || 0) + '/' + (m.games || 0)
      + ' · coup ' + (m.ply || 0);
    hintEl.textContent = 'Partie de bots en cours…';
  } else if (view) {
    statusEl.textContent = 'Reviewing move ' + navIndex + ' / ' + (historySnapshots.length - 1);
    hintEl.textContent = 'Click "Live" to resume playing.';
  } else if (s.game_over) {
    const reason = s.game_over_reason === 'time' ? ' (time out)' : '';
    statusEl.textContent = s.winner === null ? 'Draw' : 'Winner: ' + NAMES[s.winner] + reason;
    hintEl.textContent = 'Click "New game" to play again.';
  } else {
    statusEl.textContent = 'Turn: ' + NAMES[s.turn];
    if (aiBusy) { statusEl.textContent = 'AI thinking…'; hintEl.textContent = ''; }
    else if (!isHumanTurn()) { statusEl.textContent = 'AI thinking…'; hintEl.textContent = ''; }
    else hintEl.textContent = 'Click a green dot to move, or a grid line to place a wall.';
  }
}

// ----- main loop -----

function animate() {
  if (!state || view || matchView) return;
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
  if (!view && !matchView && isAiTurn() && pendingAiAt !== null && now >= pendingAiAt) {
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
  openMenu();   // ask for the game type before playing
  requestAnimationFrame(tick);
})();