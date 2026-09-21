/* ==========================================================================
   NSW Crash Risk Console
   --------------------------------------------------------------------------
   data.js ships:
     AGG        population aggregates for the full 170,962 crashes
     POINTS     stratified sample of 9,413 crashes, compactly encoded
     MODEL      logistic-regression coefficients + held-out evaluation
     TU         aggregates from the traffic-unit spreadsheet
   Everything below runs client-side: scoring, clustering and retraining.
   ========================================================================== */
(function () {
  'use strict';

  const D = window.CRASH_DATA;
  const M = D.MODEL;

  // Two targets ship with the page. 'fatal' is whether the crash killed someone;
  // 'ksi' is killed OR seriously injured, the standard road-safety target, which
  // is 15x more common and points at a different set of roads. Same features and
  // same split for both; only the fitted coefficients and baseline differ.
  const TARGETS = {
    fatal: { key: 'fatal', label: 'Fatal',
             short: 'fatal', noun: 'a death',
             coef: M.coef, intercept: M.intercept, baseRate: M.baseRate,
             metrics: M.metrics, roc: M.roc, sweep: M.sweep },
    ksi:   { key: 'ksi', label: 'Killed or seriously injured',
             short: 'KSI', noun: 'death or serious injury',
             coef: M.ksi.coef, intercept: M.ksi.intercept, baseRate: M.ksi.baseRate,
             metrics: M.ksi.metrics, roc: M.ksi.roc, sweep: M.ksi.sweep }
  };
  let TGT = TARGETS.fatal;

  // Every colour is read from the stylesheet, so light and dark themes are one
  // set of CSS variables rather than two copies of the palette in two files.
  const C = {};
  const cssVar = (n) =>
    getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const toRGB = (hex) => {
    const h = hex.replace('#', '');
    const v = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
    return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16)).join(',');
  };

  function readTheme() {
    ['fatal', 'injury', 'minor', 'model', 'hivis', 'dim', 'dimmer', 'rule',
     'line', 'slab', 'faint'].forEach((k) => { C[k] = cssVar('--' + k); });
    ['fatal', 'injury', 'minor', 'model', 'hivis', 'dimmer'].forEach((k) => {
      C[k + 'RGB'] = toRGB(C[k]);
    });
    C.dark = document.documentElement.getAttribute('data-theme') === 'dark';
    // Sequential ramps run light-to-dark on a light basemap and the reverse on
    // a dark one, so the strongest values are always the most prominent.
    C.risk = C.dark ? [[62, 92, 140], [150, 197, 255]]
                    : [[190, 214, 238], [14, 58, 112]];
    C.density = C.dark ? [[96, 110, 140], [255, 210, 63], [255, 92, 77]]
                       : [[253, 226, 213], [223, 114, 88], [138, 20, 26]];
  }
  readTheme();

  // Chart.js defaults and the severity array are built further down, so they
  // are refreshed separately once they exist (on every theme switch).
  function refreshChartTheme() {
    TICK.color = C.dimmer;
    GRID.color = cssVar('--grid');
    TOOLTIP.backgroundColor = cssVar('--tooltip-bg');
    TOOLTIP.borderColor = C.rule;
    TOOLTIP.titleColor = C.line;
    TOOLTIP.bodyColor = C.dim;
    SEV_COLOR[0] = C.fatal; SEV_COLOR[1] = C.injury; SEV_COLOR[2] = C.minor;
  }

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const fmt = (n) => Math.round(n).toLocaleString('en-AU');
  const pct = (p, d = 2) => (p * 100).toFixed(d) + '%';
  const title = (s) => s.replace(/\b[a-z]/g, (m) => m.toUpperCase());

  /* ══ 1. FEATURE ENCODING — must match train_model.py exactly ══════════ */

  const VULNERABLE = new Set(M.vehGroups.vulnerable);
  const HEAVY = new Set(M.vehGroups.heavy);
  const LIGHT_COM = new Set(M.vehGroups.light_com);

  // Four groups for filtering and display, so "Lit" means lit and a blank field
  // is visible as a blank field. The model sees only three levels: 'nil' and
  // 'off' as dummies, everything else — including blanks — at the reference.
  function litGroup(l) {
    if (l === 'nil') return 'nil';
    if (l === 'off') return 'off';
    if (l === 'on') return 'on';
    return 'unknown';
  }

  function wxBucket(w) {
    if (w === 'raining') return 'rain';
    if (w === 'overcast') return 'overcast';
    if (w === 'fog or mist' || w === 'snowing' || w === 'other') return 'adverse';
    return 'clear';
  }

  function vehGroup(v) {
    if (v === 'motorcycle') return 'motorcycle';
    if (VULNERABLE.has(v)) return 'vulnerable';
    if (HEAVY.has(v)) return 'heavy';
    if (LIGHT_COM.has(v)) return 'light_com';
    if (v === '4 wheel drive / suv') return 'suv';
    return 'car';
  }

  // Raw design vector, in MODEL.features order
  function rawVec(o) {
    return [
      o.speed / 10,
      o.wxB === 'rain' ? 1 : 0,
      o.wxB === 'overcast' ? 1 : 0,
      o.wxB === 'adverse' ? 1 : 0,
      o.vehG === 'motorcycle' ? 1 : 0,
      o.vehG === 'vulnerable' ? 1 : 0,
      o.vehG === 'heavy' ? 1 : 0,
      o.vehG === 'light_com' ? 1 : 0,
      o.vehG === 'suv' ? 1 : 0,
      o.loc === 1 ? 1 : 0,
      o.loc === 2 ? 1 : 0,
      o.litG === 'nil' ? 1 : 0,
      o.litG === 'off' ? 1 : 0,
      (o.litG === 'nil' ? 1 : 0) * (o.speed - 60) / 10,
      o.year - 2022
    ];
  }

  const MU = M.features.map((f) => (M.mu[f] !== undefined ? M.mu[f] : 0));
  const SD = M.features.map((f) => (M.sd[f] !== undefined ? M.sd[f] : 1));

  const zVec = (raw) => raw.map((v, i) => (v - MU[i]) / SD[i]);
  const sigmoid = (t) => 1 / (1 + Math.exp(-t));

  function scoreZ(z, coef, intercept) {
    let t = intercept;
    for (let i = 0; i < z.length; i++) t += coef[i] * z[i];
    return sigmoid(t);
  }

  const predict = (o) => scoreZ(zVec(rawVec(o)), TGT.coef, TGT.intercept);

  /* ══ 2. DECODE THE CRASH SAMPLE ══════════════════════════════════════ */

  const SEV_NAME = ['Fatal', 'Injury', 'Non-casualty'];
  const SEV_COLOR = [C.fatal, C.injury, C.minor];
  const SEV_KEY = ['F', 'I', 'N'];

  const PTS = D.POINTS.map((a) => {
    const o = {
      lat: a[0], lon: a[1], sev: a[2], year: a[3], speed: a[4],
      wx: D.DICT.wx[a[5]], veh: D.DICT.veh[a[6]], lga: D.DICT.lga[a[7]], loc: a[8],
      lit: D.DICT.lit[a[9]], ksi: a[10]
    };
    o.litG = litGroup(o.lit);
    o.wxB = wxBucket(o.wx);
    o.vehG = vehGroup(o.veh);
    o.w = M.weights[SEV_KEY[o.sev]];      // inverse-sampling weight
    o.p = predict(o);
    return o;
  });

  /* ══ 3. STATE + FILTERING ════════════════════════════════════════════ */

  const state = {
    year: 'all', sev: 'all', wx: 'all', veh: 'all', loc: 'all',
    lit: 'all', mapMode: 'dot', k: 6, view: 'explore'
  };

  let filtered = PTS;

  function applyFilter() {
    filtered = PTS.filter((p) =>
      (state.year === 'all' || p.year === +state.year) &&
      (state.sev === 'all' || (state.sev === 'ksi' ? p.ksi === 1 : p.sev === +state.sev)) &&
      (state.wx === 'all' || p.wxB === state.wx) &&
      (state.veh === 'all' || p.vehG === state.veh) &&
      (state.loc === 'all' || p.loc === +state.loc) &&
      (state.lit === 'all' || p.litG === state.lit));
  }

  /* ══ 4. CHART.JS SHARED SETUP ════════════════════════════════════════ */

  Chart.defaults.font.family = "'IBM Plex Mono', monospace";
  Chart.defaults.font.size = 10;
  Chart.defaults.color = C.dim;
  Chart.defaults.animation.duration = 500;
  Chart.defaults.plugins.legend.display = false;

  const TICK = { color: C.dimmer, font: { size: 9.5 } };
  const GRID = { color: cssVar('--grid'), drawTicks: false };
  const NOGRID = { display: false };

  const TOOLTIP = {
    backgroundColor: cssVar('--tooltip-bg'),
    borderColor: C.rule, borderWidth: 1, padding: 9,
    titleColor: C.line, bodyColor: C.dim, displayColors: true, boxWidth: 9
  };

  const charts = {};
  function chart(id, cfg) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (charts[id]) charts[id].destroy();
    cfg.options = cfg.options || {};
    cfg.options.responsive = true;
    cfg.options.maintainAspectRatio = false;
    cfg.options.plugins = Object.assign({ tooltip: TOOLTIP }, cfg.options.plugins);
    charts[id] = new Chart(el, cfg);
    return charts[id];
  }

  /* ══ 5. MAP ══════════════════════════════════════════════════════════ */

  let map, layer, renderer;


  /* Basemap. Two keyless Esri services, chosen by theme and zoom:
       zoomed out  - a plain grey canvas, so the crash points carry the picture
       zoomed in   - the full street map, so a point can be tied to a street
     Labels ride in their own pane above the dots, so place names stay legible
     however dense the crashes get. */
  const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
  const DETAIL_FROM = 12;
  let baseLayers = [], baseKind = null, fellBack = false;

  function tile(path, opts) {
    return L.tileLayer(ESRI + path + '/MapServer/tile/{z}/{y}/{x}',
      Object.assign({ attribution: 'Tiles &copy; Esri | Crash data: Transport for NSW',
                      maxZoom: 19, maxNativeZoom: 16 }, opts || {}));
  }

  function setBasemap() {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    const detail = !dark && map.getZoom() >= DETAIL_FROM;
    const kind = dark ? 'dark' : (detail ? 'street' : 'gray');
    if (kind === baseKind || fellBack) return;
    baseKind = kind;

    baseLayers.forEach((l) => map.removeLayer(l));
    if (kind === 'street') {
      // World Street Map carries its own labels, so no reference layer.
      baseLayers = [tile('World_Street_Map', { maxNativeZoom: 19 })];
    } else {
      const shade = dark ? 'Canvas/World_Dark_Gray' : 'Canvas/World_Light_Gray';
      baseLayers = [tile(shade + '_Base'),
                    tile(shade + '_Reference', { pane: 'placenames' })];
    }

    // If Esri is unreachable, fall back once to OpenStreetMap, which carries
    // its own street names, rather than leaving an empty grid behind the dots.
    baseLayers[0].on('tileerror', function () {
      if (fellBack) return;
      fellBack = true;
      baseLayers.forEach((l) => map.removeLayer(l));
      baseLayers = [L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors | Crash data: Transport for NSW',
        maxZoom: 19, className: 'osm-fallback'
      })];
      baseLayers[0].addTo(map);
    });
    baseLayers.forEach((l) => l.addTo(map));
  }

  function initMap() {
    map = L.map('map', {
      center: [-33.4, 149.6], zoom: 6, zoomControl: false, maxZoom: 19,
      preferCanvas: true, worldCopyJump: false
    });
    map.createPane('placenames');
    map.getPane('placenames').style.zIndex = 450;
    map.getPane('placenames').style.pointerEvents = 'none';

    setBasemap();
    map.on('zoomend', setBasemap);

    L.control.zoom({ position: 'topright' }).addTo(map);
    renderer = L.canvas({ padding: 0.4 });
    layer = L.layerGroup().addTo(map);
  }

  // Blue ramp for model output, warm ramp for observed density
  function riskColor(p) {
    const t = Math.min(1, Math.sqrt(p / 0.12));
    const from = C.risk[0], to = C.risk[1];
    const c = from.map((v, i) => Math.round(v + (to[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function densityColor(t) {
    const stops = C.density;
    const x = Math.min(1, Math.max(0, t)) * 2;
    const i = Math.min(1, Math.floor(x));
    const f = x - i;
    const c = stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function drawMap() {
    if (!map) return;
    layer.clearLayers();

    if (state.mapMode === 'dot' || state.mapMode === 'risk') {
      const byRisk = state.mapMode === 'risk';
      const pts = filtered.slice().sort((a, b) => a.sev === 0 ? 1 : b.sev === 0 ? -1 : 0);
      pts.forEach((p) => {
        const col = byRisk ? riskColor(p.p) : SEV_COLOR[p.sev];
        L.circleMarker([p.lat, p.lon], {
          renderer, radius: p.sev === 0 ? 3.6 : 2.4,
          color: col, weight: 0, fillColor: col,
          fillOpacity: byRisk ? 0.72 : (p.sev === 0 ? 0.9 : 0.55)
        }).bindTooltip(pointTip(p), { direction: 'top', opacity: 0.96 }).addTo(layer);
      });
    }

    if (state.mapMode === 'grid') drawGrid();
    if (state.mapMode === 'cluster') drawClusters();
    drawLegend();
  }

  function pointTip(p) {
    return `<b>${SEV_NAME[p.sev]}</b> &middot; ${p.year}<br>` +
      `${title(p.lga)} &middot; ${p.speed} km/h<br>` +
      `${p.veh} &middot; ${p.wx}<br>` +
      `<span style="color:${C.model}">model risk ${pct(p.p)}</span>`;
  }

  function drawGrid() {
    const CELL = 0.09;
    const cells = new Map();
    filtered.forEach((p) => {
      const gy = Math.floor(p.lat / CELL), gx = Math.floor(p.lon / CELL);
      const key = gy + ':' + gx;
      let c = cells.get(key);
      if (!c) { c = { gy, gx, w: 0, f: 0 }; cells.set(key, c); }
      c.w += p.w;
      if (p.sev === 0) c.f += p.w;
    });
    const list = Array.from(cells.values()).sort((a, b) => b.w - a.w).slice(0, 900);
    if (!list.length) return;
    const max = Math.log1p(list[0].w);
    list.forEach((c) => {
      const t = Math.log1p(c.w) / max;
      L.rectangle([[c.gy * CELL, c.gx * CELL], [(c.gy + 1) * CELL, (c.gx + 1) * CELL]], {
        renderer, stroke: false, fillColor: densityColor(t), fillOpacity: 0.18 + 0.62 * t
      }).bindTooltip(
        `<b>${fmt(c.w)}</b> est. crashes<br>${fmt(c.f)} fatal (${pct(c.f / c.w, 1)})`,
        { direction: 'top' }).addTo(layer);
    });
  }

  /* k-means over crash coordinates, k-means++ seeding, longitude scaled
     by cos(latitude) so a degree east and a degree north are comparable. */
  function kmeans(points, k, iters) {
    if (points.length < k) return [];
    const latMean = points.reduce((s, p) => s + p.lat, 0) / points.length;
    const kx = Math.cos(latMean * Math.PI / 180);
    const X = points.map((p) => [p.lat, p.lon * kx]);

    let rnd = 20240825;                     // deterministic seed
    const rand = () => (rnd = (rnd * 1103515245 + 12345) % 2147483648) / 2147483648;

    const cent = [X[Math.floor(rand() * X.length)].slice()];
    while (cent.length < k) {
      const d2 = X.map((x) => Math.min.apply(null,
        cent.map((c) => (x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2)));
      const tot = d2.reduce((a, b) => a + b, 0);
      let r = rand() * tot, i = 0;
      while (i < d2.length - 1 && (r -= d2[i]) > 0) i++;
      cent.push(X[i].slice());
    }

    const assign = new Array(X.length).fill(0);
    for (let it = 0; it < iters; it++) {
      let moved = false;
      for (let i = 0; i < X.length; i++) {
        let best = 0, bd = Infinity;
        for (let c = 0; c < k; c++) {
          const d = (X[i][0] - cent[c][0]) ** 2 + (X[i][1] - cent[c][1]) ** 2;
          if (d < bd) { bd = d; best = c; }
        }
        if (assign[i] !== best) { assign[i] = best; moved = true; }
      }
      const sum = Array.from({ length: k }, () => [0, 0, 0]);
      for (let i = 0; i < X.length; i++) {
        const a = assign[i];
        sum[a][0] += X[i][0]; sum[a][1] += X[i][1]; sum[a][2]++;
      }
      for (let c = 0; c < k; c++) {
        if (sum[c][2]) { cent[c][0] = sum[c][0] / sum[c][2]; cent[c][1] = sum[c][1] / sum[c][2]; }
      }
      if (!moved && it > 0) break;
    }

    return Array.from({ length: k }, (_, c) => {
      const members = points.filter((_, i) => assign[i] === c);
      const w = members.reduce((s, p) => s + p.w, 0);
      const f = members.reduce((s, p) => s + (p.sev === 0 ? p.w : 0), 0);
      return {
        lat: cent[c][0], lon: cent[c][1] / kx, n: members.length, w, f,
        risk: members.reduce((s, p) => s + p.p * p.w, 0) / (w || 1),
        speed: members.reduce((s, p) => s + p.speed * p.w, 0) / (w || 1),
        radiusKm: Math.sqrt(members.reduce((s, p) =>
          s + ((p.lat - cent[c][0]) ** 2 + ((p.lon - cent[c][1] / kx) * kx) ** 2), 0) /
          (members.length || 1)) * 111
      };
    }).filter((c) => c.n > 0).sort((a, b) => b.w - a.w);
  }

  function drawClusters() {
    filtered.forEach((p) => {
      L.circleMarker([p.lat, p.lon], {
        renderer, radius: 1.6, weight: 0,
        fillColor: C.faint, fillOpacity: 0.3
      }).addTo(layer);
    });

    const cl = kmeans(filtered, state.k, 25);
    const maxW = Math.max.apply(null, cl.map((c) => c.w));
    cl.forEach((c, i) => {
      L.circleMarker([c.lat, c.lon], {
        renderer, radius: 9 + 26 * Math.sqrt(c.w / maxW),
        color: riskColor(c.risk), weight: 2,
        fillColor: riskColor(c.risk), fillOpacity: 0.28
      }).bindTooltip(
        `<b>Cluster ${i + 1}</b><br>${fmt(c.w)} est. crashes<br>` +
        `${pct(c.f / c.w, 1)} fatal &middot; ${Math.round(c.speed)} km/h avg zone<br>` +
        `<span style="color:${C.model}">mean model risk ${pct(c.risk)}</span><br>` +
        `spread ≈ ${Math.round(c.radiusKm)} km`,
        { direction: 'top', opacity: 0.96 }).addTo(layer);
    });
  }

  function drawLegend() {
    const el = $('#legend');
    const row = (col, label, sq) =>
      `<div class="legend-row"><div class="swatch ${sq ? 'sq' : ''}" style="background:${col}"></div>${label}</div>`;

    if (state.mapMode === 'dot') {
      el.innerHTML = '<h4>Crash severity</h4>' +
        row(C.fatal, 'Fatal') + row(C.injury, 'Injury') + row(C.minor, 'Non-casualty');
    } else if (state.mapMode === 'grid') {
      el.innerHTML = '<h4>Crash density (9 km cells)</h4>' +
        row(densityColor(0.05), 'Lower', true) +
        row(densityColor(0.5), 'Medium', true) +
        row(densityColor(1), 'Higher', true);
    } else if (state.mapMode === 'cluster') {
      el.innerHTML = '<h4>K-means hotspots</h4>' +
        '<div class="legend-row" style="color:var(--dim)">Size = crash volume</div>' +
        row(riskColor(0.01), 'Lower model risk') + row(riskColor(0.08), 'Higher model risk');
    } else {
      el.innerHTML = '<h4>Predicted fatality risk</h4>' +
        row(riskColor(0.005), 'Under 1%') + row(riskColor(0.03), '~3%') +
        row(riskColor(0.1), '10%+');
    }
  }

  /* ══ 6. EXPLORE PANELS ═══════════════════════════════════════════════ */

  function hbars(id, items, colorFn) {
    const el = document.getElementById(id);
    if (!el || !items.length) return;
    const max = Math.max.apply(null, items.map((i) => i.n));
    el.innerHTML = items.map((i) => `
      <div class="hbar">
        <div class="hbar-label" title="${i.label}">${i.label}</div>
        <div class="hbar-track">
          <div class="hbar-fill" style="width:${(i.n / max * 100).toFixed(1)}%;background:${colorFn(i, max)}"></div>
        </div>
        <div class="hbar-val">${i.n >= 1000 ? (i.n / 1000).toFixed(1) + 'k' : i.n}</div>
      </div>`).join('');
  }

  function updateSelection() {
    const n = filtered.length;
    $('#pt-count').textContent = fmt(n);

    const W = filtered.reduce((s, p) => s + p.w, 0);
    const Wf = filtered.reduce((s, p) => s + (p.sev === 0 ? p.w : 0), 0);
    const speed = filtered.reduce((s, p) => s + p.speed * p.w, 0) / (W || 1);
    const risk = filtered.reduce((s, p) => s + p.p * p.w, 0) / (W || 1);

    $('#sel-crashes').textContent = n ? fmt(W) : '—';
    $('#sel-fatal').textContent = n ? pct(Wf / W, 2) : '—';
    $('#sel-speed').textContent = n ? Math.round(speed) + ' km/h' : '—';
    $('#sel-risk').textContent = n ? pct(risk, 2) : '—';
    const Wk = filtered.reduce((s, p) => s + (p.ksi ? p.w : 0), 0);
    $('#sel-ksi').textContent = n ? pct(Wk / W, 1) : '—';

    const base = D.AGG.severity[0].n / D.AGG.totals.crashes;
    const note = $('#sel-note');
    if (!n) {
      note.textContent = 'No crashes match this combination of filters.';
    } else if (state.sev !== 'all') {
      // 'ksi' is not one of the three severity codes, so it needs its own label
      const sevLabel = state.sev === 'ksi'
        ? 'killed or seriously injured'
        : SEV_NAME[+state.sev].toLowerCase();
      note.innerHTML = `Filtered to <b>${sevLabel}</b> crashes, so the outcome mix is fixed by the ` +
        `filter. The model risk of <b>${pct(risk, 2)}</b> still reflects the conditions these ` +
        `crashes happened in.`;
    } else {
      const r = (Wf / W) / base;
      note.innerHTML = `Crashes matching these filters are fatal <b>${pct(Wf / W, 2)}</b> of the time — ` +
        `<b>${r.toFixed(2)}&times;</b> the NSW average of ${pct(base, 2)}.`;
    }

    const sevCounts = [0, 1, 2].map((s) => ({
      label: SEV_NAME[s],
      n: Math.round(filtered.reduce((t, p) => t + (p.sev === s ? p.w : 0), 0)),
      s
    }));
    hbars('sev-bars', sevCounts, (i) => SEV_COLOR[i.s]);
  }

  function buildStaticPanels() {
    hbars('lga-bars', D.AGG.by_lga.map((d) => ({ label: title(d.l), n: d.n })),
      (i, max) => `rgba(${C.modelRGB},${(0.35 + 0.6 * i.n / max).toFixed(2)})`);

    hbars('veh-bars', D.AGG.by_veh.map((d) => ({ label: title(d.v), n: d.n })),
      (i, max) => i.label.toLowerCase().includes('motorcycle')
        ? C.fatal : `rgba(${C.minorRGB},${(0.3 + 0.65 * i.n / max).toFixed(2)})`);

    hbars('man-bars', D.TU.manoeuvre.map((d) => ({ label: d.k, n: d.n })),
      (i, max) => `rgba(${C.hivisRGB},${(0.3 + 0.6 * i.n / max).toFixed(2)})`);

    hbars('obj-bars', D.TU.object_hit.map((d) => ({ label: d.k, n: d.n })),
      (i, max) => `rgba(${C.injuryRGB},${(0.32 + 0.6 * i.n / max).toFixed(2)})`);

    const share = D.TU.off_path / D.TU.total;
    $('#obj-note').innerHTML = `<b>${pct(share, 1)}</b> of crashes ran off the road before impact ` +
      `(RUM codes beginning "Off"). That is the same run-off-road pattern the model picks up as ` +
      `rural high-speed risk.`;
  }

  function buildExploreCharts() {
    // Fatality rate by speed zone
    chart('c-speed', {
      type: 'bar',
      data: {
        labels: D.SPEED_RATE.map((d) => d.speed),
        datasets: [{
          data: D.SPEED_RATE.map((d) => d.rate * 100),
          backgroundColor: D.SPEED_RATE.map((d) =>
            d.speed >= 100 ? C.fatal : d.speed >= 80 ? C.injury : `rgba(${C.injuryRGB},.45)`),
          borderRadius: 2, borderSkipped: false
        }]
      },
      options: {
        scales: {
          x: { grid: NOGRID, ticks: Object.assign({ callback: (v, i) => D.SPEED_RATE[i].speed }, TICK) },
          y: { grid: GRID, ticks: Object.assign({ callback: (v) => v + '%' }, TICK) }
        },
        plugins: {
          tooltip: Object.assign({}, TOOLTIP, {
            callbacks: {
              title: (c) => c[0].label + ' km/h zone',
              label: (c) => ` ${c.parsed.y.toFixed(2)}% of crashes fatal ` +
                `(≈${fmt(D.SPEED_RATE[c.dataIndex].n)} crashes)`
            }
          })
        }
      }
    });

    chart('c-year', {
      type: 'bar',
      data: {
        labels: D.AGG.by_year.map((d) => d.y),
        datasets: [{
          data: D.AGG.by_year.map((d) => d.n),
          backgroundColor: C.model, borderRadius: 2, borderSkipped: false
        }]
      },
      options: {
        scales: {
          x: { grid: NOGRID, ticks: TICK },
          y: { grid: GRID, ticks: Object.assign({ callback: (v) => (v / 1000) + 'k' }, TICK) }
        },
        plugins: { tooltip: Object.assign({}, TOOLTIP, { callbacks: { label: (c) => ' ' + fmt(c.parsed.y) + ' crashes' } }) }
      }
    });

    const hours = D.AGG.by_hour.filter((d) => d.h !== 'unknown');
    chart('c-hour', {
      type: 'bar',
      data: {
        labels: hours.map((d) => d.h.split(' ')[0].slice(0, 2)),
        datasets: [{
          data: hours.map((d) => d.n),
          backgroundColor: hours.map((d) => d.n > 20000 ? C.fatal : d.n > 13000 ? C.injury : `rgba(${C.dimmerRGB},.45)`),
          borderRadius: 2, borderSkipped: false
        }]
      },
      options: {
        scales: {
          x: { grid: NOGRID, ticks: TICK },
          y: { grid: GRID, ticks: Object.assign({ callback: (v) => (v / 1000) + 'k' }, TICK) }
        },
        plugins: {
          tooltip: Object.assign({}, TOOLTIP, {
            callbacks: { title: (c) => hours[c[0].dataIndex].h, label: (c) => ' ' + fmt(c.parsed.y) + ' crashes' }
          })
        }
      }
    });

    chart('c-day', {
      type: 'bar',
      data: {
        labels: D.AGG.by_day.map((d) => title(d.d).slice(0, 3)),
        datasets: [{
          data: D.AGG.by_day.map((d) => d.n),
          backgroundColor: D.AGG.by_day.map((d) => d.d === 'friday' ? C.hivis : `rgba(${C.modelRGB},.55)`),
          borderRadius: 2, borderSkipped: false
        }]
      },
      options: {
        scales: {
          x: { grid: NOGRID, ticks: TICK },
          y: { grid: GRID, ticks: Object.assign({ callback: (v) => (v / 1000) + 'k' }, TICK) }
        },
        plugins: { tooltip: Object.assign({}, TOOLTIP, { callbacks: { label: (c) => ' ' + fmt(c.parsed.y) + ' crashes' } }) }
      }
    });

    chart('c-month', {
      type: 'line',
      data: {
        labels: D.AGG.by_month.map((d) => title(d.m).slice(0, 3)),
        datasets: [{
          data: D.AGG.by_month.map((d) => d.n),
          borderColor: C.hivis, backgroundColor: `rgba(${C.hivisRGB},.15)`,
          borderWidth: 2, fill: true, tension: 0.35,
          pointRadius: 2.5, pointBackgroundColor: C.hivis
        }]
      },
      options: {
        scales: {
          x: { grid: NOGRID, ticks: TICK },
          y: { grid: GRID, ticks: Object.assign({ callback: (v) => (v / 1000) + 'k' }, TICK) }
        },
        plugins: { tooltip: Object.assign({}, TOOLTIP, { callbacks: { label: (c) => ' ' + fmt(c.parsed.y) + ' crashes' } }) }
      }
    });

    /* Counts alone hide the point (fine weather dominates everything), so the
       bars carry volume and the line carries the fatal share. */
    const wxNames = Array.from(new Set(D.AGG.by_weather.map((d) => d.weather)))
      .filter((w) => w !== 'unknown' && w !== 'other');
    const pick = (w, sev) => {
      const r = D.AGG.by_weather.find((d) => d.weather === w && d.degree_of_crash === sev);
      return r ? r.n : 0;
    };
    const wxTotal = wxNames.map((w) =>
      pick(w, 'fatal') + pick(w, 'injury') + pick(w, 'non-casualty (towaway)'));
    const wxShare = wxNames.map((w, i) => pick(w, 'fatal') / wxTotal[i] * 100);

    chart('c-weather', {
      type: 'bar',
      data: {
        labels: wxNames.map((w) => title(w).replace(' Or Mist', '/mist')),
        datasets: [
          {
            label: 'Crashes', data: wxTotal, yAxisID: 'y',
            backgroundColor: `rgba(${C.dimmerRGB},.40)`, borderRadius: 2, borderSkipped: false, order: 2
          },
          {
            label: '% fatal', data: wxShare, yAxisID: 'y2', type: 'line',
            borderColor: C.fatal, backgroundColor: C.fatal, borderWidth: 2,
            pointRadius: 3, pointBackgroundColor: C.fatal, tension: 0.25, order: 1
          }
        ]
      },
      options: {
        scales: {
          x: { grid: NOGRID, ticks: TICK },
          y: {
            type: 'logarithmic', grid: GRID, position: 'left',
            ticks: Object.assign({ callback: (v) => v >= 1000 ? (v / 1000) + 'k' : v }, TICK)
          },
          y2: {
            position: 'right', grid: NOGRID, beginAtZero: true,
            ticks: Object.assign({ callback: (v) => v + '%', color: C.fatal }, TICK)
          }
        },
        plugins: {
          legend: { display: true, position: 'bottom', labels: { boxWidth: 9, padding: 7, font: { size: 9.5 } } },
          tooltip: Object.assign({}, TOOLTIP, {
            callbacks: {
              label: (c) => c.datasetIndex === 0
                ? ' ' + fmt(c.parsed.y) + ' crashes'
                : ' ' + c.parsed.y.toFixed(2) + '% ended in a death'
            }
          })
        }
      }
    });
  }

  /* ══ 7. MODEL VIEW ═══════════════════════════════════════════════════ */

  let modelBuilt = false;

  function buildModelView() {
    if (modelBuilt) return;
    modelBuilt = true;

    // Odds multipliers, sorted by strength
    const rows = M.features.map((f, i) => ({
      label: M.labels[i], odds: Math.exp(TGT.coef[i]), coef: TGT.coef[i]
    })).sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));

    chart('c-coef', {
      type: 'bar',
      data: {
        labels: rows.map((r) => r.label),
        datasets: [{
          data: rows.map((r) => [1, r.odds]),
          backgroundColor: rows.map((r) => r.coef > 0 ? `rgba(${C.fatalRGB},.85)` : `rgba(${C.minorRGB},.85)`),
          borderRadius: 2, borderSkipped: false
        }]
      },
      options: {
        indexAxis: 'y',
        scales: {
          x: {
            min: 0.7, max: 3.1, grid: GRID,
            ticks: Object.assign({ callback: (v) => v + '\u00d7' }, TICK)
          },
          y: { grid: NOGRID, ticks: Object.assign({ font: { size: 10 } }, TICK) }
        },
        plugins: {
          tooltip: Object.assign({}, TOOLTIP, {
            callbacks: {
              label: (c) => {
                const r = rows[c.dataIndex];
                return ` odds \u00d7${r.odds.toFixed(2)}  (\u03b2 = ${r.coef >= 0 ? '+' : ''}${r.coef.toFixed(3)})`;
              }
            }
          })
        }
      }
    });

    // ROC
    chart('c-roc', {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Model', data: TGT.roc.map((p) => ({ x: p[0], y: p[1] })),
            borderColor: C.model, backgroundColor: `rgba(${C.modelRGB},.14)`,
            borderWidth: 2, fill: true, pointRadius: 0, tension: 0.05
          },
          {
            label: 'Coin flip', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: C.dimmer, borderWidth: 1, borderDash: [4, 4], pointRadius: 0
          }
        ]
      },
      options: {
        scales: {
          x: {
            type: 'linear', min: 0, max: 1, grid: GRID,
            title: { display: true, text: 'False positive rate', color: C.dimmer, font: { size: 10 } },
            ticks: TICK
          },
          y: {
            min: 0, max: 1, grid: GRID,
            title: { display: true, text: 'True positive rate', color: C.dimmer, font: { size: 10 } },
            ticks: TICK
          }
        },
        plugins: {
          legend: { display: true, position: 'bottom', labels: { boxWidth: 9, padding: 7, font: { size: 9.5 } } },
          tooltip: Object.assign({}, TOOLTIP, {
            callbacks: { label: (c) => ` FPR ${c.parsed.x.toFixed(2)} · TPR ${c.parsed.y.toFixed(2)}` }
          })
        }
      }
    });
    $('#roc-auc').textContent = TGT.metrics.auc.toFixed(3);

    // Comparison table
    const best = M.compare.reduce((a, b) => (b.auc > a.auc ? b : a));
    $('#compare-table tbody').innerHTML = M.compare.map((m) => `
      <tr class="${m.name === best.name ? 'is-best' : ''}">
        <td>${m.name}</td><td>${m.auc.toFixed(3)}</td>
        <td>${pct(m.recall, 1)}</td><td>${pct(m.precision, 1)}</td>
        <td>${pct(m.accuracy, 1)}</td>
      </tr>`).join('');

    // Threshold + confusion matrix
    $('#thresh').addEventListener('input', (e) => renderConfusion(+e.target.value));
    renderConfusion(+$('#thresh').value);

    renderTree();
    buildTrainerCharts();
  }

  function renderConfusion(idx) {
    const s = TGT.sweep[idx];
    const [tn, fp, fn, tp] = s.confusion;
    $('#thresh-val').textContent = pct(s.t, 2);
    $('#confusion').innerHTML = `
      <div class="cell head"></div>
      <div class="cell head">Predicted<br>not fatal</div>
      <div class="cell head">Predicted<br>fatal</div>
      <div class="cell head">Actually<br>not fatal</div>
      <div class="cell"><div class="cv">${fmt(tn)}</div><div class="cl">correct pass</div></div>
      <div class="cell"><div class="cv" style="color:${C.injury}">${fmt(fp)}</div><div class="cl">false alarm</div></div>
      <div class="cell head">Actually<br>fatal</div>
      <div class="cell miss"><div class="cv">${fmt(fn)}</div><div class="cl">missed</div></div>
      <div class="cell hit"><div class="cv">${fmt(tp)}</div><div class="cl">caught</div></div>`;
    $('#thresh-note').innerHTML =
      `At a cut-off of <b>${pct(s.t, 2)}</b> the model flags <b>${fmt(tp + fp)}</b> crashes as high risk ` +
      `and catches <b>${pct(s.recall, 1)}</b> of the fatal ones, but only <b>${pct(s.precision, 1)}</b> of ` +
      `what it flags actually is fatal. Because fatal crashes are 1.41% of the total, high recall and ` +
      `high precision cannot both happen — screening tools live on the left of this slider, ` +
      `enforcement targeting on the right.`;
  }

  function renderTree() {
    const node = (n, cond) => {
      if (n.leaf) {
        const hot = n.p > M.baseRate * 2;  // the tree is fitted on the fatal target only
        return `<li><span class="yes">${cond}</span> &rarr; ` +
          `<span class="leafv ${hot ? 'hot' : ''}">${pct(n.p, 2)} fatal</span>` +
          ` <span style="color:var(--dimmer)">(n=${fmt(n.n)})</span></li>`;
      }
      const raw = n.name === 'speed_10' ? `${(n.rawThr * 10).toFixed(0)} km/h`
        : n.name === 'year_c' ? `${(2022 + n.rawThr).toFixed(1)}` : null;
      const test = raw
        ? `<span class="cond">${n.label.replace(' (per 10 km/h)', '')} &le; ${raw}</span>`
        : `<span class="cond">${n.label}</span>`;
      const yes = raw ? 'yes' : 'no';
      const no = raw ? 'no' : 'yes';
      return `<li>${cond ? `<span class="yes">${cond}</span> &rarr; ` : ''}${test}
        <ul>${node(n.left, yes)}${node(n.right, no)}</ul></li>`;
    };
    $('#tree').innerHTML = `<ul>${node(M.tree, '')}</ul>`;
  }

  /* ══ 8. IN-BROWSER TRAINING ══════════════════════════════════════════ */

  const NF = M.features.length;
  let Xz = null, Yv = null, Wv = null;

  function buildMatrix() {
    if (Xz) return;
    Xz = new Float64Array(PTS.length * NF);
    Yv = new Float64Array(PTS.length);
    Wv = new Float64Array(PTS.length);
    let wsum = 0;
    PTS.forEach((p, i) => {
      const z = zVec(rawVec(p));
      for (let j = 0; j < NF; j++) Xz[i * NF + j] = z[j];
      Yv[i] = p.sev === 0 ? 1 : 0;
      Wv[i] = p.w;
      wsum += p.w;
    });
    const scale = PTS.length / wsum;           // keep gradients on a sane scale
    for (let i = 0; i < PTS.length; i++) Wv[i] *= scale;
  }

  /* Weighted AUC by the Mann-Whitney identity: walk the scores in ascending
     order and accumulate the negative weight sitting below each positive. */
  function weightedAUC(scores) {
    const idx = Array.from({ length: scores.length }, (_, i) => i)
      .sort((a, b) => scores[a] - scores[b]);
    let negBelow = 0, acc = 0, Wp = 0, Wn = 0;
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k];
      if (Yv[i] === 1) { acc += negBelow * Wv[i]; Wp += Wv[i]; }
      else { negBelow += Wv[i]; Wn += Wv[i]; }
    }
    return Wp && Wn ? acc / (Wp * Wn) : 0.5;
  }

  let trainerCharts = false;
  function buildTrainerCharts() {
    if (trainerCharts) return;
    trainerCharts = true;

    chart('c-loss', {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Weighted log-loss', data: [],
          borderColor: C.hivis, backgroundColor: `rgba(${C.hivisRGB},.12)`,
          borderWidth: 2, fill: true, pointRadius: 0, tension: 0.25
        }]
      },
      options: {
        animation: false,
        scales: {
          x: { grid: NOGRID, ticks: Object.assign({ maxTicksLimit: 8 }, TICK), title: { display: true, text: 'Epoch', color: C.dimmer, font: { size: 10 } } },
          y: { grid: GRID, ticks: Object.assign({ callback: (v) => v.toFixed(3) }, TICK) }
        },
        plugins: { tooltip: Object.assign({}, TOOLTIP, { callbacks: { label: (c) => ' loss ' + c.parsed.y.toFixed(5) } }) }
      }
    });

    const order = M.features.map((f, i) => i).sort((a, b) => Math.abs(TGT.coef[b]) - Math.abs(TGT.coef[a]));
    chart('c-weights', {
      type: 'bar',
      data: {
        labels: order.map((i) => M.labels[i]),
        datasets: [
          { label: 'scikit-learn', data: order.map((i) => TGT.coef[i]), backgroundColor: `rgba(${C.modelRGB},.85)`, borderRadius: 2 },
          { label: 'trained in browser', data: order.map(() => 0), backgroundColor: `rgba(${C.hivisRGB},.9)`, borderRadius: 2 }
        ]
      },
      options: {
        indexAxis: 'y',
        animation: false,
        scales: {
          x: { grid: GRID, ticks: TICK, title: { display: true, text: 'Coefficient (log-odds)', color: C.dimmer, font: { size: 10 } } },
          y: { grid: NOGRID, ticks: Object.assign({ font: { size: 9 } }, TICK) }
        },
        plugins: {
          legend: { display: true, position: 'bottom', labels: { boxWidth: 9, padding: 7, font: { size: 9.5 } } },
          tooltip: Object.assign({}, TOOLTIP, { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.x.toFixed(3)}` } })
        }
      }
    });

    $('#train-btn').addEventListener('click', runTraining);
  }

  function log(msg, cls) {
    const el = $('#console');
    el.innerHTML += `\n${cls ? `<span class="${cls}">${msg}</span>` : msg}`;
    el.scrollTop = el.scrollHeight;
  }

  let training = false;
  function runTraining() {
    if (training) return;
    training = true;
    const btn = $('#train-btn');
    btn.disabled = true;
    btn.textContent = 'Training…';

    buildMatrix();
    const n = PTS.length;
    const b = new Float64Array(NF);
    let b0 = Math.log(TGT.baseRate / (1 - TGT.baseRate));    // start at the base rate
    const LR = 0.05, EPOCHS = 600, CHUNK = 20;
    const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
    const lossChart = charts['c-loss'];
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];

    $('#console').innerHTML = `Fitting ${NF} coefficients + intercept on ${fmt(n)} weighted rows…`;
    log(`optimiser: Adam · learning rate ${LR} · ${EPOCHS} full-batch epochs`);

    let epoch = 0;
    const grad = new Float64Array(NF);
    const mAvg = new Float64Array(NF + 1);
    const vAvg = new Float64Array(NF + 1);
    const scores = new Float64Array(n);

    function step() {
      for (let c = 0; c < CHUNK && epoch < EPOCHS; c++) {
        epoch++;
        grad.fill(0);
        let g0 = 0, loss = 0, wsum = 0;
        for (let i = 0; i < n; i++) {
          let t = b0;
          const off = i * NF;
          for (let j = 0; j < NF; j++) t += b[j] * Xz[off + j];
          const p = 1 / (1 + Math.exp(-t));
          scores[i] = p;
          const w = Wv[i], e = (p - Yv[i]) * w;
          g0 += e;
          for (let j = 0; j < NF; j++) grad[j] += e * Xz[off + j];
          loss -= w * Math.log(Math.max(Yv[i] === 1 ? p : 1 - p, 1e-12));
          wsum += w;
        }
        g0 /= n;
        for (let j = 0; j < NF; j++) grad[j] /= n;

        const c1 = 1 - Math.pow(B1, epoch), c2 = 1 - Math.pow(B2, epoch);
        mAvg[NF] = B1 * mAvg[NF] + (1 - B1) * g0;
        vAvg[NF] = B2 * vAvg[NF] + (1 - B2) * g0 * g0;
        b0 -= LR * (mAvg[NF] / c1) / (Math.sqrt(vAvg[NF] / c2) + EPS);
        for (let j = 0; j < NF; j++) {
          mAvg[j] = B1 * mAvg[j] + (1 - B1) * grad[j];
          vAvg[j] = B2 * vAvg[j] + (1 - B2) * grad[j] * grad[j];
          b[j] -= LR * (mAvg[j] / c1) / (Math.sqrt(vAvg[j] / c2) + EPS);
        }

        if (epoch % 5 === 0 || epoch === EPOCHS) {
          lossChart.data.labels.push(epoch);
          lossChart.data.datasets[0].data.push(loss / wsum);
        }
        if (epoch % 150 === 0) log(`epoch ${String(epoch).padStart(3)}  loss ${(loss / wsum).toFixed(5)}`);
      }

      lossChart.update('none');
      charts['c-weights'].data.datasets[1].data =
        charts['c-weights'].data.labels.map((lab) => b[M.labels.indexOf(lab)]);
      charts['c-weights'].update('none');

      if (epoch < EPOCHS) {
        requestAnimationFrame(step);
      } else {
        const auc = weightedAUC(scores);
        const diff = Math.max.apply(null, TGT.coef.map((c, i) => Math.abs(c - b[i])));
        log(`converged — in-sample weighted AUC ${auc.toFixed(4)}`, 'ok');
        log(`intercept ${b0.toFixed(3)} here vs ${M.intercept.toFixed(3)} from scikit-learn`, 'ok');
        log(`largest coefficient gap: ${diff.toFixed(3)} log-odds — the offline fit used only the ` +
          `75% training split, this one used all ${fmt(n)} rows`, 'ok');
        btn.disabled = false;
        btn.textContent = 'Run again';
        training = false;
      }
    }
    requestAnimationFrame(step);
  }

  /* ══ 9. PREDICT VIEW ═════════════════════════════════════════════════ */

  const scenario = { speed: 100, wxB: 'clear', vehG: 'car', loc: 2, year: 2024, litG: 'nil' };
  const BASELINE = { speed: 60, wxB: 'clear', vehG: 'car', loc: 0, year: 2022 };
  let dialMax = 0.2, predictBuilt = false;

  function buildPredictView() {
    if (predictBuilt) return;
    predictBuilt = true;

    // Dial ceiling = the worst scenario the model can produce, rounded up
    let mx = 0;
    [40, 50, 60, 70, 80, 90, 100, 110].forEach((sp) =>
      ['clear', 'rain', 'overcast', 'adverse'].forEach((wxB) =>
        ['car', 'suv', 'light_com', 'motorcycle', 'vulnerable', 'heavy'].forEach((vehG) =>
          [0, 1, 2].forEach((loc) =>
            [2020, 2024].forEach((year) => {
              mx = Math.max(mx, predict({ speed: sp, wxB, vehG, loc, year }));
            })))));
    dialMax = Math.ceil((mx * 100 + 2) / 5) * 5 / 100;   // headroom past the worst case

    drawDialFace();

    $('#p-speed').addEventListener('input', (e) => {
      scenario.speed = +e.target.value;
      $('#p-speed-val').textContent = scenario.speed + ' km/h';
      updatePredict();
    });
    $('#p-wx').addEventListener('change', (e) => { scenario.wxB = e.target.value; updatePredict(); });
    $('#p-veh').addEventListener('change', (e) => { scenario.vehG = e.target.value; updatePredict(); });
    $('#p-loc').addEventListener('change', (e) => { scenario.loc = +e.target.value; updatePredict(); });
    $('#p-year').addEventListener('change', (e) => { scenario.year = +e.target.value; updatePredict(); });
    $('#p-lit').addEventListener('change', (e) => { scenario.litG = e.target.value; updatePredict(); });

    $$('[data-preset]').forEach((b) => b.addEventListener('click', () => {
      const p = b.dataset.preset;
      if (p === 'worst') Object.assign(scenario, { speed: 110, wxB: 'overcast', vehG: 'motorcycle', loc: 2, year: 2024, litG: 'nil' });
      if (p === 'safest') Object.assign(scenario, { speed: 40, wxB: 'rain', vehG: 'suv', loc: 0, year: 2020, litG: 'on' });
      if (p === 'typical') Object.assign(scenario, { speed: 60, wxB: 'clear', vehG: 'car', loc: 0, year: 2024, litG: 'on' });
      syncScenarioInputs();
      updatePredict();
    }));

    $('#lga-risk-table tbody').innerHTML = D.LGA_RISK.map((r) => `
      <tr><td>${r.lga}</td><td>${r.speed} km/h</td><td>${r.n}</td>
      <td style="color:${C.model}">${pct(r.risk, 2)}</td></tr>`).join('');

    syncScenarioInputs();
    updatePredict();
  }

  function syncScenarioInputs() {
    $('#p-speed').value = scenario.speed;
    $('#p-speed-val').textContent = scenario.speed + ' km/h';
    $('#p-wx').value = scenario.wxB;
    $('#p-veh').value = scenario.vehG;
    $('#p-loc').value = String(scenario.loc);
    $('#p-year').value = String(scenario.year);
    $('#p-lit').value = scenario.litG;
  }

  /* --- the dial ------------------------------------------------------- */
  const DIAL = { cx: 210, cy: 200, r: 148, sw: 17 };

  // sqrt spacing keeps the low end of the scale readable
  const dialT = (p) => Math.min(1, Math.sqrt(Math.max(p, 0) / dialMax));
  const dialAngle = (p) => 180 + dialT(p) * 180;

  function polar(cx, cy, r, deg) {
    const a = deg * Math.PI / 180;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  }

  function drawDialFace() {
    const svg = $('#dial');
    const { cx, cy, r } = DIAL;
    const [x0, y0] = polar(cx, cy, r, 180);
    const [x1, y1] = polar(cx, cy, r, 360);
    let h = `<path class="dial-arc-bg" d="M${x0} ${y0} A${r} ${r} 0 0 1 ${x1} ${y1}"/>`;
    h += `<path class="dial-arc-fg" id="dial-arc" d="M${x0} ${y0} A${r} ${r} 0 0 1 ${x1} ${y1}"/>`;

    [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25].filter((v) => v <= dialMax).forEach((v) => {
      const a = dialAngle(v);
      const [ax, ay] = polar(cx, cy, r - 13, a);
      const [bx, by] = polar(cx, cy, r + 13, a);
      const [lx, ly] = polar(cx, cy, r + 26, a);
      h += `<line class="dial-tick" x1="${ax.toFixed(1)}" y1="${ay.toFixed(1)}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}"/>`;
      h += `<text class="dial-ticklabel" x="${lx.toFixed(1)}" y="${(ly + 3).toFixed(1)}">${(v * 100).toFixed(v < 0.01 ? 1 : 0)}%</text>`;
    });

    // NSW average marker
    const ab = dialAngle(TGT.baseRate);
    const [bx0, by0] = polar(cx, cy, r - 22, ab);
    const [bx1, by1] = polar(cx, cy, r + 4, ab);
    h += `<line class="dial-base" x1="${bx0.toFixed(1)}" y1="${by0.toFixed(1)}" x2="${bx1.toFixed(1)}" y2="${by1.toFixed(1)}"/>`;
    const [tx, ty] = polar(cx, cy, r - 42, ab);
    h += `<text class="dial-ticklabel" style="fill:var(--hivis-ink)" x="${tx.toFixed(1)}" y="${ty.toFixed(1)}">NSW avg</text>`;

    h += `<g id="dial-needle-g" transform="rotate(180 ${cx} ${cy})">
            <line class="dial-needle" x1="${cx}" y1="${cy}" x2="${cx - r + 26}" y2="${cy}"/>
          </g>`;
    h += `<circle class="dial-hub" cx="${cx}" cy="${cy}" r="7"/>`;
    svg.innerHTML =
      '<title id="dial-title">Predicted probability that this crash is fatal</title>' + h;

    const len = Math.PI * r;
    const arc = $('#dial-arc');
    arc.style.strokeDasharray = `${len} ${len}`;
    arc.style.strokeDashoffset = len;
  }

  function updatePredict() {
    const p = predict(scenario);
    const ratio = p / TGT.baseRate;

    $('#dial-value').textContent = pct(p, 2);
    const arc = $('#dial-arc');
    const len = Math.PI * DIAL.r;
    arc.style.strokeDashoffset = len * (1 - dialT(p));
    arc.style.stroke = ratio > 3 ? C.fatal : ratio > 1.4 ? C.injury : C.model;
    $('#dial-value').style.color = ratio > 3 ? C.fatal : ratio > 1.4 ? C.injury : C.model;
    $('#dial-needle-g').setAttribute('transform',
      `rotate(${dialAngle(p) - 180} ${DIAL.cx} ${DIAL.cy})`);

    const word = ratio > 4 ? 'far above' : ratio > 1.3 ? 'above' : ratio < 0.75 ? 'below' : 'close to';
    $('#dial-compare').innerHTML =
      `That is <span class="x">${ratio.toFixed(1)}\u00d7</span> the model's baseline of ` +
      `<b>${pct(TGT.baseRate, 2)}</b> — ${word} typical. Put another way, about ` +
      `<b>1 in ${fmt(1 / p)}</b> crashes under these conditions ends in a death.`;

    renderContributions(p);
  }

  function renderContributions(p) {
    const zs = zVec(rawVec(scenario));
    const zb = zVec(rawVec(BASELINE));
    const rows = [];
    for (let i = 0; i < NF; i++) {
      const d = TGT.coef[i] * (zs[i] - zb[i]);
      if (Math.abs(d) > 1e-9) rows.push({ label: M.labels[i], d });
    }
    rows.sort((a, b) => Math.abs(b.d) - Math.abs(a.d));

    const el = $('#contrib');
    if (!rows.length) {
      el.innerHTML = '<div class="contrib-row"><div class="contrib-label" style="grid-column:1/-1">' +
        'These are exactly the baseline conditions — nothing shifts the score.</div></div>';
    } else {
      const max = Math.max.apply(null, rows.map((r) => Math.abs(r.d)));
      el.innerHTML = rows.map((r) => {
        const w = (Math.abs(r.d) / max) * 49;
        const side = r.d > 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
        return `<div class="contrib-row">
          <div class="contrib-label">${r.label.replace(' (per 10 km/h)', '')}</div>
          <div class="contrib-track"><div class="contrib-zero"></div>
            <div class="contrib-fill ${r.d > 0 ? 'up' : 'down'}" style="${side}"></div></div>
          <div class="contrib-val" style="color:${r.d > 0 ? C.fatal : C.minor}">
            ${r.d > 0 ? '+' : ''}${r.d.toFixed(2)}</div></div>`;
      }).join('');
    }

    const pb = predict(BASELINE);
    const top = rows[0];
    $('#contrib-note').innerHTML =
      `Baseline crash: <b>${pct(pb, 2)}</b>. These conditions move it to <b>${pct(p, 2)}</b>` +
      (top ? `, and <b>${top.label.replace(' (per 10 km/h)', '').toLowerCase()}</b> does the most work ` +
        `(${top.d > 0 ? '+' : ''}${top.d.toFixed(2)} log-odds, odds \u00d7${Math.exp(top.d).toFixed(2)}).` : '.');
  }

  /* ══ 10. WIRING ══════════════════════════════════════════════════════ */

  function fillSelects() {
    const yearSel = $('#f-year');
    D.AGG.by_year.forEach((d) => {
      yearSel.insertAdjacentHTML('beforeend', `<option value="${d.y}">${d.y}</option>`);
    });
    const wxSel = $('#f-wx');
    [['clear', 'Fine'], ['rain', 'Raining'], ['overcast', 'Overcast'], ['adverse', 'Fog / snow / other']]
      .forEach(([v, l]) => wxSel.insertAdjacentHTML('beforeend', `<option value="${v}">${l}</option>`));
    const vehSel = $('#f-veh');
    [['car', 'Car / wagon'], ['suv', '4WD / SUV'], ['light_com', 'Ute / light truck'],
    ['motorcycle', 'Motorcycle'], ['vulnerable', 'Bicycle / scooter'], ['heavy', 'Heavy vehicle']]
      .forEach(([v, l]) => vehSel.insertAdjacentHTML('beforeend', `<option value="${v}">${l}</option>`));
  }

  // Every point's score depends on the active target, so switching re-scores.
  function rescorePoints() {
    PTS.forEach((o) => { o.p = predict(o); });
  }

  function refresh() {
    applyFilter();
    updateSelection();
    drawMap();
  }

  function wireFilters() {
    const map3 = { '#f-year': 'year', '#f-sev': 'sev', '#f-wx': 'wx', '#f-veh': 'veh',
                   '#f-loc': 'loc', '#f-lit': 'lit' };
    Object.entries(map3).forEach(([sel, key]) => {
      $(sel).addEventListener('change', (e) => { state[key] = e.target.value; refresh(); });
    });
    const FILTER_KEYS = ['year', 'sev', 'wx', 'veh', 'loc', 'lit'];
    function updateFilterBadge() {
      const n = FILTER_KEYS.filter((k) => state[k] !== 'all').length;
      const badge = $('#filter-badge');
      badge.textContent = n;
      badge.classList.toggle('is-on', n > 0);
      badge.title = n ? `${n} filter${n > 1 ? 's' : ''} applied` : '';
    }

    $('#filter-toggle').addEventListener('click', () => {
      const bar = $('#filterbar');
      const open = bar.classList.toggle('is-collapsed') === false;
      $('#filter-toggle').setAttribute('aria-expanded', String(open));
      // the map shares the row's space, so it has to re-measure
      if (map) setTimeout(() => map.invalidateSize(), 180);
    });

    Object.entries(map3).forEach(([sel]) => {
      $(sel).addEventListener('change', updateFilterBadge);
    });
    updateFilterBadge();

    $('#f-reset').addEventListener('click', () => {
      Object.assign(state, { year: 'all', sev: 'all', wx: 'all', veh: 'all', loc: 'all', lit: 'all' });
      Object.keys(map3).forEach((s) => { $(s).value = 'all'; });
      updateFilterBadge();
      refresh();
    });

    $$('[data-mapmode]').forEach((b) => b.addEventListener('click', () => {
      $$('[data-mapmode]').forEach((o) => o.classList.remove('is-active'));
      b.classList.add('is-active');
      state.mapMode = b.dataset.mapmode;
      $('#k-control').classList.toggle('is-on', state.mapMode === 'cluster');
      drawMap();
    }));

    $('#k-slider').addEventListener('input', (e) => {
      state.k = +e.target.value;
      $('#k-val').textContent = state.k;
      if (state.mapMode === 'cluster') drawMap();
    });
  }

  function wireTabs() {
    $$('.tab').forEach((t) => t.addEventListener('click', () => {
      $$('.tab').forEach((o) => o.setAttribute('aria-selected', 'false'));
      t.setAttribute('aria-selected', 'true');
      const id = t.id.replace('tab-', '');
      state.view = id;
      $$('.view').forEach((v) => v.classList.toggle('is-on', v.id === 'view-' + id));
      window.scrollTo({ top: 0, behavior: 'auto' });
      if (id === 'explore' && map) setTimeout(() => map.invalidateSize(), 60);
      if (id === 'model') buildModelView();
      if (id === 'predict') buildPredictView();
    }));
  }


  /* ══ THEME ═══════════════════════════════════════════════════════════ */

  function applyTheme(dark) {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
    const btn = $('#theme-toggle');
    if (btn) {
      btn.textContent = dark ? 'Light mode' : 'Dark mode';
      btn.setAttribute('aria-pressed', String(dark));
    }
    readTheme();
    refreshChartTheme();

    // Charts hold their colours from when they were drawn, so redraw them.
    buildStaticPanels();
    buildExploreCharts();
    modelBuilt = false;
    predictBuilt = false;
    if (state.view === 'model') buildModelView();
    if (state.view === 'predict') buildPredictView();

    if (map) { setBasemap(); drawMap(); }
  }

  function setTarget(key) {
    TGT = TARGETS[key];
    $$('[data-target]').forEach((b) => b.classList.toggle('is-active', b.dataset.target === key));
    document.querySelectorAll('.tgt-label').forEach((el) => { el.textContent = TGT.label.toLowerCase(); });
    document.querySelectorAll('.tgt-short').forEach((el) => { el.textContent = TGT.short; });

    rescorePoints();
    $('#kpi-auc').textContent = TGT.metrics.auc.toFixed(3);
    $('#stamp-auc').textContent = TGT.metrics.auc.toFixed(3);
    $('#kpi-model-note').textContent =
      `${TGT.label.toLowerCase()} · logistic regression · AUC ${TGT.metrics.auc.toFixed(3)}`;

    // the model and predict views hold fitted numbers, so rebuild them
    modelBuilt = false; predictBuilt = false;
    if (state.view === 'model') buildModelView();
    if (state.view === 'predict') buildPredictView();
    refresh();
  }

  function wireTarget() {
    $$('[data-target]').forEach((b) =>
      b.addEventListener('click', () => setTarget(b.dataset.target)));
    setTarget('fatal');
  }

  function wireTheme() {
    // The <head> script already set data-theme, so only the label needs syncing.
    const btn = $('#theme-toggle');
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.textContent = dark ? 'Light mode' : 'Dark mode';
    btn.setAttribute('aria-pressed', String(dark));
    btn.addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') !== 'dark';
      applyTheme(next);
      try { localStorage.setItem('crash-theme', next ? 'dark' : 'light'); } catch (e) { /* ignore */ }
    });
  }

  function init() {
    // Headline counts come straight from the data file, so the strip can never
    // drift away from what the build script actually produced.
    const T = D.AGG.totals;
    $('#kpi-total').textContent = fmt(T.crashes);
    $('#kpi-fatal').textContent = fmt(T.fatal);
    $('#kpi-fatal-note').textContent = pct(T.fatal / T.crashes, 2) + ' of all crashes';
    $('#kpi-killed').textContent = fmt(T.killed);
    $('#kpi-killed-note').textContent = (T.killed / T.fatal).toFixed(2) + ' per fatal crash';
    $('#kpi-injured').textContent = fmt(T.injured);
    $('#kpi-injured-note').textContent = fmt(T.injured / 1826) + ' per day';
    $('#kpi-auc').textContent = TGT.metrics.auc.toFixed(3);
    // Captions that quote a number are written from the data, not typed in,
    // so they cannot go stale when the build is re-run.
    const sr = Object.fromEntries(D.SPEED_RATE.map((r) => [r.speed, r.rate]));
    const spKilled = Object.fromEntries(D.AGG.sp_fatal.map((r) => [r.speed_limit, r.killed]));
    $('#speed-note').innerHTML =
      `A crash in a <b>100 km/h</b> zone is <b>${(sr[100] / sr[50]).toFixed(1)}&times;</b> more ` +
      `likely to be fatal than one in a 50 km/h zone, and 100 km/h zones account for ` +
      `<b>${fmt(spKilled['100 km/h'])}</b> deaths — more than any other speed limit.`;

    const pk = D.AGG.by_hour.reduce((a, b) => (b.n > a.n ? b : a));
    $('#hour-note').innerHTML =
      `Peak risk is <b>${pk.h}</b> (${fmt(pk.n)} crashes), with a second rise across the morning ` +
      `commute — the working day drives the shape of the day.`;

    const mi = M.features.indexOf('veh_motorcycle');
    $('#moto-odds').textContent = Math.exp(TGT.coef[mi]).toFixed(1) + '\u00d7';
    const W = M.weights;
    $('#wt-note').innerHTML =
      `&times;${W.F.toFixed(1)} (fatal), &times;${W.I.toFixed(1)} (injury) and ` +
      `&times;${W.N.toFixed(1)} (non-casualty)`;

    $('#stamp-auc').textContent = TGT.metrics.auc.toFixed(3);
    $('#kpi-model-note').textContent =
      `logistic regression · ${fmt(M.trainN)} train / ${fmt(M.testN)} test`;

    fillSelects();
    wireFilters();
    wireTabs();
    wireTheme();
    wireTarget();
    initMap();
    buildStaticPanels();
    buildExploreCharts();
    refresh();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
