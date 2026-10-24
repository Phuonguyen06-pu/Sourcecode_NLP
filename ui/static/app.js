'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const hasLetter = s => /[\p{L}\p{N}_]/u.test(s);

// Dữ liệu chỉ nằm trong biến này (bộ nhớ trình duyệt), đóng tab là mất
const state = { meta: null, mode: 'text', file: null, run: null, model: null, tab: 'list', sel: null, metrics: null };
const explainCache = new Map();

const ICONS = {
  allow: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.2 4.2L19 7"/></svg>',
  review: '<svg viewBox="0 0 24 24"><path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="2.8"/></svg>',
  block: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M5.6 5.6l12.8 12.8"/></svg>',
};
const BAR_COLORS = ['#8fd8b5', '#f6c27f', '#f29bab'];
const actClass = a => ['allow', 'review', 'block'][state.meta.actions.indexOf(a)] || 'allow';

// tiện ích 
async function api(path, { method = 'GET', json, form } = {}) {
  const res = await fetch(path, {
    method,
    headers: json ? { 'Content-Type': 'application/json' } : undefined,
    body: json ? JSON.stringify(json) : form,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); if (typeof j.detail === 'string') msg = j.detail; } catch (_) { /* giữ statusText */ }
    throw new Error(msg);
  }
  return res.json();
}
function notice(msg) { const n = $('#notice'); n.textContent = msg; n.hidden = false; }
function hideNotice() { $('#notice').hidden = true; }
let toastTimer;
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 4200);
}

// khởi tạo
async function init() {
  try { state.meta = await api('/api/status'); }
  catch (e) { notice('Không kết nối được máy chủ: ' + e.message); return; }
  bind();
  renderPills();
}

function bind() {
  document.querySelectorAll('#modeSeg button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $('#runBtn').addEventListener('click', onRun);
  $('#textInput').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) onRun(); });
  $('#fileInput').addEventListener('change', e => { const f = e.target.files[0]; if (f) onFile(f); e.target.value = ''; });
  $('#colSelect').addEventListener('change', e => { if (state.file) state.file.col = e.target.value; });
  const drop = $('#drop');
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) onFile(f); });
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('#modeSeg button').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
  $('#textPane').hidden = mode !== 'text';
  $('#filePane').hidden = mode !== 'file';
  hideNotice();
}

async function onFile(file) {
  hideNotice();
  const fd = new FormData();
  fd.append('file', file);
  try {
    const d = await api('/api/upload', { method: 'POST', form: fd });
    state.file = { name: d.name || file.name, columns: d.columns, rows: d.rows, col: d.guess ?? d.columns[0] };
    $('#fileName').innerHTML = `${esc(state.file.name)}<small>${d.rows.length} dòng</small>`;
    $('#colSelect').innerHTML = d.columns.map(c => `<option${c === state.file.col ? ' selected' : ''}>${esc(c)}</option>`).join('');
    $('#fileInfo').hidden = false;
    if (d.guess === null) notice('Không tự nhận ra cột văn bản, hãy chọn cột đúng ở ô bên cạnh.');
  } catch (e) {
    state.file = null;
    $('#fileInfo').hidden = true;
    notice('Không đọc được file: ' + e.message);
  }
}

// kiểm tra dữ liệu và chạy
function collect() {
  if (state.mode === 'text') {
    const s = $('#textInput').value.trim();
    if (!s) { notice('Bạn chưa nhập văn bản. Hãy nhập một câu rồi bấm Chạy.'); return null; }
    if (!hasLetter(s)) { notice('Văn bản không có chữ nào (chỉ có ký hiệu hoặc khoảng trắng).'); return null; }
    return { mode: 'text', texts: [s], rows: [1], dropped: 0, source: 'Văn bản nhập' };
  }
  const f = state.file;
  if (!f) { notice('Bạn chưa tải file. Hãy chọn file .csv hoặc .xlsx rồi bấm Chạy.'); return null; }
  if (!f.rows.length) { notice('File không có dòng dữ liệu nào.'); return null; }
  const ci = f.columns.indexOf(f.col);
  const texts = [], rows = [];
  f.rows.forEach((r, i) => {
    const s = String(r[ci] ?? '').trim();
    if (hasLetter(s)) { texts.push(s); rows.push(i + 1); }
  });
  if (!texts.length) { notice(`Cột “${f.col}” không có dòng nào chứa chữ. Hãy chọn lại cột văn bản.`); return null; }
  return { mode: 'file', texts, rows, dropped: f.rows.length - texts.length, source: f.name };
}

async function onRun() {
  hideNotice();
  const data = collect();
  if (!data) return;
  const btn = $('#runBtn'), status = $('#runStatus');
  btn.disabled = true;
  const results = {}, errors = {};
  const models = state.meta.models;
  for (let i = 0; i < models.length; i++) {
    const m = models[i];
    if (!m.available) { errors[m.id] = 'chưa có mô hình đã train'; continue; }
    status.textContent = `Đang chạy ${m.title} (${i + 1}/${models.length})...`;
    try {
      const r = await api('/api/predict', { method: 'POST', json: { model: m.id, texts: data.texts } });
      results[m.id] = r.results;
    } catch (e) { errors[m.id] = e.message; }
  }
  btn.disabled = false;
  status.textContent = '';
  if (!Object.keys(results).length) {
    notice('Không có mô hình nào chạy được. Hãy train mô hình trước rồi thử lại.');
    return;
  }
  explainCache.clear();
  state.run = { ...data, results, errors };
  state.model = models.find(m => results[m.id]).id;
  state.sel = null;
  state.tab = 'list';
  renderPills();
  renderView();
  $('.results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// chọn mô hình
function renderPills() {
  const box = $('#pills');
  box.innerHTML = '';
  state.meta.models.forEach(m => {
    const has = !!(state.run && state.run.results[m.id]);
    const b = document.createElement('button');
    b.className = 'pill' + (has ? '' : ' off') + (has && state.model === m.id ? ' on' : '');
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', String(has && state.model === m.id));
    b.innerHTML = esc(m.title) + (m.available ? '' : '<small>chưa train</small>');
    b.addEventListener('click', () => pickModel(m));
    box.appendChild(b);
  });
}

function pickModel(m) {
  if (!state.run) { toast('Chưa có kết quả. Hãy nhập dữ liệu và bấm Chạy trước.'); return; }
  if (!state.run.results[m.id]) { toast(`${m.title} chưa có kết quả nên chưa thể hiển thị. Hãy chọn mô hình khác.`); return; }
  state.model = m.id;
  renderPills();
  renderView();
}

// vùng kết quả
function infosHtml(run) {
  const chips = [`<span class="info">${esc(run.source)} · ${run.texts.length} câu</span>`];
  state.meta.models.forEach(m => {
    if (run.errors[m.id]) chips.push(`<span class="info warn">${esc(m.title)}: ${esc(run.errors[m.id])}</span>`);
  });
  if (run.dropped) chips.push(`<span class="info warn">Đã bỏ qua ${run.dropped} dòng trống hoặc không có chữ</span>`);
  return `<div class="infos">${chips.join('')}</div>`;
}

function renderView() {
  const run = state.run, view = $('#view');
  $('#empty').hidden = !!run;
  view.hidden = !run;
  if (!run) return;
  view.innerHTML = infosHtml(run) + '<div id="body"></div>';
  const body = $('#body', view);

  if (run.mode === 'text') {
    body.innerHTML = '<div class="card" id="detail"></div><div class="card" id="metricsBox" style="margin-top:20px"></div>';
    renderDetail($('#detail'), 0);
    renderMetrics($('#metricsBox'));
    return;
  }
  const tabs = [['list', 'Danh sách và chi tiết'], ['dash', 'Dashboard'], ['metrics', 'Chỉ số các mô hình']];
  body.innerHTML = '<div class="nav" role="tablist">' +
    tabs.map(([k, t]) => `<button role="tab" data-tab="${k}" class="${state.tab === k ? 'on' : ''}">${t}</button>`).join('') +
    '</div><div id="tab"></div>';
  body.querySelectorAll('.nav button').forEach(b => b.addEventListener('click', () => { state.tab = b.dataset.tab; renderView(); }));
  const tab = $('#tab', body);
  if (state.tab === 'list') renderList(tab);
  else if (state.tab === 'dash') renderDashboard(tab);
  else { tab.innerHTML = '<div class="card" id="metricsBox"></div>'; renderMetrics($('#metricsBox', tab)); }
}

// chi tiết một câu 
function probBars(res) {
  return state.meta.labels.map((label, i) => {
    const p = res.probs[label] ?? 0;
    return `<div class="bar-row"><div class="top"><span>${esc(label)}</span><b>${(p * 100).toFixed(1)}%</b></div>` +
      `<div class="bar"><i class="b-${i}" style="width:${(p * 100).toFixed(1)}%"></i></div></div>`;
  }).join('');
}

function crossTable(idx) {
  const rows = state.meta.models.map(m => {
    const list = state.run.results[m.id];
    if (!list) return `<tr><td>${esc(m.title)}</td><td colspan="3" style="color:var(--muted)">chưa có</td></tr>`;
    const r = list[idx];
    return `<tr><td>${esc(m.title)}</td><td><span class="chip c-${r.label_id}">${esc(r.label)}</span></td>` +
      `<td><span class="chip ${actClass(r.action)}">${esc(r.action)}</span></td><td class="num">${r.toxic_score.toFixed(2)}</td></tr>`;
  }).join('');
  return `<table><thead><tr><th>Mô hình</th><th>Nhãn</th><th>Mức xử lý</th><th class="num">Điểm độc hại</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderDetail(el, idx) {
  const run = state.run, model = state.model;
  const res = run.results[model][idx], text = run.texts[idx];
  const cls = actClass(res.action), th = state.meta.thresholds;
  const long = res.clean_text.split(/\s+/).length > 100;
  const allowed = res.action === state.meta.actions[0];     
  el.dataset.key = model + '\u0000' + text;
  el.innerHTML =
     (run.mode === 'file' ? `<h3>Chi tiết dòng ${run.rows[idx]}</h3>` : '') +
    `<div class="banner ${cls}"><div class="ico">${ICONS[cls]}</div><div><div class="act">${esc(res.action)}</div>` +
    `<div class="sub">Dự đoán <b>${esc(res.label)}</b> · điểm độc hại ${(res.toxic_score * 100).toFixed(0)}%</div></div></div>` +
    `<div class="orig">${esc(text)}</div>` +
    `<div class="two"><div><h4>Xác suất từng nhãn</h4>${probBars(res)}` +
    `<div class="legend">Kiểm tra từ ${th.low.toFixed(2)}, chặn từ ${th.high.toFixed(2)}</div></div>` +
    (allowed
      ? `<div><h4>Từ ảnh hưởng đến kết quả</h4><div class="legend">Câu được cho phép nên không có từ nào bị đánh dấu.</div></div></div>`
      : `<div><h4>Từ ảnh hưởng đến kết quả</h4><div class="hl spin" id="hl">Đang phân tích...</div><div class="topw" id="topw"></div></div></div>`) +
    (long ? '<div class="note">Câu khá dài: PhoBERT và ViSoBERT chỉ đọc tối đa 128 token đầu, phần sau có thể bị cắt.</div>' : '') +
    `<div class="cross"><h4>So sánh giữa các mô hình</h4>${crossTable(idx)}</div>`;
  if (!allowed) loadHighlight(el, model, text);
}

async function loadHighlight(el, model, text) {
  const key = model + '\u0000' + text;
  let exp = explainCache.get(key);
  try {
    if (!exp) {
      exp = await api('/api/explain', { method: 'POST', json: { model, text } });
      explainCache.set(key, exp);
    }
  } catch (e) {
    if (el.dataset.key === key && $('#hl', el)) $('#hl', el).textContent = 'Không phân tích được: ' + e.message;
    return;
  }
  if (el.dataset.key !== key) return;          // người dùng đã chuyển sang câu hoặc mô hình khác
  const hl = $('#hl', el), topw = $('#topw', el);
  if (!hl) return;
  if (!exp.tokens.length) { hl.textContent = 'Không có nội dung để tô sáng.'; return; }
  const max = Math.max(...exp.scores, 1e-9);
  hl.className = 'hl';
  hl.innerHTML = exp.tokens.map((w, i) => {
    const s = exp.scores[i];
    if (s > 0.01) {
      const a = (0.18 + 0.55 * Math.min(s / max, 1)).toFixed(2);
      return `<mark style="--a:${a}" title="+${s.toFixed(2)}">${esc(w)}</mark>`;
    }
    return esc(w);
  }).join(' ');
  topw.innerHTML = exp.top.length
    ? exp.top.map(t => `<span class="chip c-2">${esc(t.word)} +${t.score.toFixed(1)}</span>`).join('')
    : '<span class="legend">Không có từ nào đẩy rõ về phía độc hại.</span>';
}

// danh sách (tải file)
function renderList(tab) {
  const run = state.run, res = run.results[state.model];
  const rows = res.map((r, i) =>
    `<tr data-i="${i}" tabindex="0" class="${state.sel === i ? 'sel' : ''}"><td class="num">${run.rows[i]}</td>` +
    `<td class="txt"><div>${esc(run.texts[i])}</div></td>` +
    `<td><span class="chip c-${r.label_id}">${esc(r.label)}</span></td>` +
    `<td><span class="chip ${actClass(r.action)}">${esc(r.action)}</span></td>` +
    `<td class="num"><span class="mini"><i style="width:${Math.round(r.toxic_score * 100)}%"></i></span>${r.toxic_score.toFixed(2)}</td></tr>`).join('');
  tab.innerHTML = '<div class="split"><div class="table-wrap"><table><thead><tr><th class="num">Dòng</th><th>Văn bản</th><th>Nhãn</th>' +
    `<th>Mức xử lý</th><th class="num">Điểm</th></tr></thead><tbody>${rows}</tbody></table></div><div class="card side" id="side"></div></div>`;
  const side = $('#side', tab);
  const show = i => {
    state.sel = i;
    tab.querySelectorAll('tbody tr').forEach(tr => tr.classList.toggle('sel', +tr.dataset.i === i));
    renderDetail(side, i);
  };
  const tbody = $('tbody', tab);
  tbody.addEventListener('click', e => { const tr = e.target.closest('tr'); if (tr) show(+tr.dataset.i); });
  tbody.addEventListener('keydown', e => { if (e.key === 'Enter') { const tr = e.target.closest('tr'); if (tr) show(+tr.dataset.i); } });
  if (state.sel !== null) show(state.sel);
  else side.innerHTML = '<div class="placeholder">Bấm vào một dòng để xem chi tiết.</div>';
}

// dashboard 
function chartHtml(title, labels, values, colors) {
  const max = Math.max(...values, 1);
  const cols = values.map((v, i) =>
    `<div class="col"><div class="v">${v}</div><div class="b" style="height:${(v / max * 80).toFixed(1)}%;background:${colors[i]}"></div></div>`).join('');
  const xl = labels.map(l => `<div>${esc(l)}</div>`).join('');
  return `<div class="card chart"><h4>${esc(title)}</h4><div class="cols">${cols}</div><div class="xlabels">${xl}</div></div>`;
}

function renderDashboard(tab) {
  const run = state.run, res = run.results[state.model], n = res.length, meta = state.meta;
  const labelCounts = meta.labels.map((_, i) => res.filter(r => r.label_id === i).length);
  const actionCounts = meta.actions.map(a => res.filter(r => r.action === a).length);
  const hist = Array(10).fill(0);
  res.forEach(r => { hist[Math.min(9, Math.floor(r.toxic_score * 10))]++; });
  const mean = res.reduce((s, r) => s + r.toxic_score, 0) / n;
  const histColors = hist.map((_, i) => {
    const c = (i + 0.5) / 10;
    return c >= meta.thresholds.high ? BAR_COLORS[2] : c >= meta.thresholds.low ? BAR_COLORS[1] : BAR_COLORS[0];
  });
  const histLabels = hist.map((_, i) => (i / 10).toFixed(1));
  const all = Object.values(run.results);
  let agree = '';
  if (all.length >= 2) {
    let same = 0;
    for (let i = 0; i < n; i++) if (all.every(list => list[i].label_id === all[0][i].label_id)) same++;
    agree = `<div class="agree">Cả ${all.length} mô hình cho cùng một nhãn ở ${(same / n * 100).toFixed(0)}% số câu.</div>`;
  }
  const stat = (cls, big, small) => `<div class="stat ${cls}"><b>${big}</b><span>${small}</span></div>`;
  tab.innerHTML =
    '<div class="stats">' + stat('', n, 'Số câu') +
    meta.actions.map((a, i) => stat(['allow', 'review', 'block'][i], `${actionCounts[i]} <small style="font-size:14px;font-weight:500">(${(actionCounts[i] / n * 100).toFixed(0)}%)</small>`, esc(a))).join('') +
    stat('', mean.toFixed(2), 'Điểm độc hại trung bình') + '</div>' +
    '<div class="charts">' +
    chartHtml('Phân bố nhãn', meta.labels, labelCounts, BAR_COLORS) +
    chartHtml('Phân bố mức xử lý', meta.actions, actionCounts, BAR_COLORS) +
    chartHtml('Phân bố điểm độc hại (từ)', histLabels, hist, histColors) +
    '</div>' + agree;
}

// chỉ số các mô hình 
async function renderMetrics(el) {
  el.innerHTML = '<h3>Chỉ số đánh giá của các mô hình</h3><div class="spin">Đang tải...</div>';
  try { if (!state.metrics) state.metrics = await api('/api/metrics'); }
  catch (e) { el.innerHTML = '<h3>Chỉ số đánh giá của các mô hình</h3><div class="note">Không tải được chỉ số: ' + esc(e.message) + '</div>'; return; }
  const { columns, rows } = state.metrics;
  const best = {};
  columns.forEach(c => {
    const vals = rows.map(r => r[c]).filter(v => typeof v === 'number');
    if (vals.length >= 2) best[c] = Math.max(...vals);
  });
  const head = columns.map((c, i) => `<th class="${i ? 'num' : ''}">${esc(c)}</th>`).join('');
  const body = rows.map(r => '<tr>' + columns.map((c, i) => {
    const v = r[c];
    if (i === 0) return `<td>${esc(v)}</td>`;
    if (typeof v !== 'number') return '<td class="num" style="color:var(--muted)">—</td>';
    return `<td class="num${best[c] === v ? ' best' : ''}">${v.toFixed(4)}</td>`;
  }).join('') + '</tr>').join('');
  el.innerHTML = `<h3>Chỉ số đánh giá của các mô hình</h3><div class="table-wrap"><table class="metrics"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

init();
