"use strict";
(() => {
  const T = window.Trader, F = T.fmt, el = T.svg;
  const $ = (id) => document.getElementById(id);
  const REFRESH_MS = 30000;
  const simId = decodeURIComponent(location.pathname.replace(/^\/s\//, "").replace(/\/$/, ""));

  let range = "7d";
  try { range = localStorage.getItem("range") || range; } catch (e) { /* storage blocked: default */ }
  if (!["1d", "7d", "30d", "all"].includes(range)) range = "7d";
  let state = null;
  let failedSince = null;
  let hoverIndex = null;
  let geometry = null;
  let described = false;

  const sim = () => (state && state.simulation) || { base: "BTC", quote: "USDT", interval: "1m" };

  function renderStatus() {
    const box = $("status");
    const s = T.statusLine(state && state.status, failedSince);
    box.classList.remove("live", "stale");
    if (s.tone) box.classList.add(s.tone);
    $("status-text").textContent = s.text;
  }

  function renderDescription() {
    if (described || !state) return;
    described = true;
    const s = state.simulation;
    document.title = s.name + " · ai-trader";
    $("sim-name").textContent = s.name;
    $("sim-sub").textContent = (s.description ? s.description + " " : "") + s.strategy_title + " on " + s.base + "/" + s.quote
      + ", " + T.candles(s.interval) + ".";
    $("chart-title").textContent = "Portfolio value, " + s.quote;
    $("th-qty").textContent = s.base;
    $("explain").textContent = s.explain;
    const rows = [["Strategy", s.strategy_title + " (" + s.strategy + ")"], ["Pair", s.base + "/" + s.quote + " (Binance spot)"],
      ["Candles", T.candles(s.interval)], ["Starting cash", F.money.format(s.start_cash) + " " + s.quote],
      ["Fee", F.money.format(s.fee * 100) + "% per trade"]];
    for (const p of s.params) rows.push([p.name, String(p.value), p.help]);
    const dl = $("settings");
    dl.replaceChildren();
    for (const [k, v, help] of rows) {
      const wrap = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      if (help) {
        const small = document.createElement("span");
        small.className = "help";
        small.textContent = help;
        dd.append(" ", small);
      }
      wrap.append(dt, dd);
      dl.appendChild(wrap);
    }
  }

  function renderKpis() {
    const p = state && state.portfolio;
    if (!p) return;
    const s = sim();
    const since = F.day.format(Date.parse(p.started_at));
    const simPct = (p.value / p.start_cash - 1) * 100;
    const bhPct = (p.benchmark / p.start_cash - 1) * 100;
    $("k-value").textContent = F.money.format(p.value);
    $("k-value-note").textContent = F.pct.format(simPct) + "% since " + since;
    T.setTone($("k-value-note"), simPct);
    $("k-bench").textContent = F.money.format(p.benchmark);
    $("k-bench-note").textContent = F.pct.format(bhPct) + "% since " + since;
    T.setTone($("k-bench-note"), bhPct);
    const diff = p.value - p.benchmark;
    $("k-diff").textContent = F.signed.format(diff);
    T.setTone($("k-diff"), diff);
    $("k-diff-note").textContent = F.signed.format(simPct - bhPct) + " percentage points";
    $("k-position").textContent = p.position;
    const parts = [];
    if (p.crypto > 0) parts.push(F.coin.format(p.crypto) + " " + s.base);
    if (p.cash > 0.005 || p.crypto <= 0) parts.push(F.money.format(p.cash) + " " + s.quote);
    $("k-position-note").textContent = parts.join(" + ") + (p.last_action ? " · last call: " + p.last_action : "");
    const st = state.stats;
    if (st) {
      const m = st.last_24h;
      $("decision-mix").textContent = "Calls in the last 24 h: " + m.buy + " buy, " + m.sell + " sell, " + m.hold
        + " hold (a buy with no cash left, or a sell with nothing to sell, changes nothing). "
        + st.decisions + (st.decisions === 1 ? " decision" : " decisions") + " since " + F.full.format(Date.parse(p.started_at)) + ", " + s.base + " at "
        + F.money.format(p.price) + " " + s.quote + " now.";
    }
    $("last-reason").textContent = p.last_reason ? "Last signal: " + p.last_reason + "." : "";
  }

  function renderTrades() {
    const body = $("trades-body");
    body.replaceChildren();
    const trades = (state && state.trades) || [];
    $("trades-empty").hidden = trades.length > 0;
    const s = state && state.stats;
    $("trades-count").textContent = s && s.trades
      ? s.trades + " in total, " + F.money.format(s.fees) + " " + sim().quote + " paid in fees" + (s.trades > trades.length ? " (latest " + trades.length + " shown)" : "")
      : "";
    for (const t of trades) {
      const tr = document.createElement("tr");
      const cells = [
        [F.full.format(Date.parse(t.ts)), ""],
        [t.side === "buy" ? "Buy" : "Sell", "side side-" + t.side],
        [F.money0.format(t.price), "num"],
        [F.coin.format(t.quantity), "num wide"],
        [F.money.format(t.fee_usdt), "num wide"],
        [F.money.format(t.value_after), "num"],
      ];
      for (const [v, cls] of cells) {
        const td = document.createElement("td");
        td.textContent = v;
        if (cls) td.className = cls;
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
  }

  function niceTicks(lo, hi, count) {
    const span = hi - lo || 1;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return { ticks: out, step };
  }

  const MIN = 60000, HOUR = 60 * MIN, DAY = 24 * HOUR;
  function timeTicks(t0, t1, maxTicks) {
    const steps = [15 * MIN, 30 * MIN, HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, DAY, 2 * DAY, 7 * DAY, 14 * DAY, 30 * DAY];
    const step = steps.find((s) => (t1 - t0) / s <= maxTicks) || 60 * DAY;
    const start = new Date(t0);
    start.setHours(0, 0, 0, 0); // align to local midnight, then walk
    const out = [];
    if (step < DAY) {
      for (let t = start.getTime(); t <= t1; t += step) if (t >= t0) out.push(t);
    } else {
      const d = new Date(start);
      const days = Math.round(step / DAY);
      while (d.getTime() <= t1) {
        if (d.getTime() >= t0) out.push(d.getTime());
        d.setDate(d.getDate() + days);
      }
    }
    return { ticks: out, daily: step >= DAY };
  }

  function renderChart() {
    const svg = $("chart-svg");
    svg.replaceChildren();
    const series = (state && state.series) || [];
    $("chart-empty").hidden = series.length > 1 || !state;
    $("chart-empty").textContent = state && !state.portfolio && state.status.error
      ? "No data to show right now: " + state.status.error + "."
      : "No decisions yet: the first one comes when the next " + T.candles(sim().interval).replace(/s$/, "") + " closes.";
    $("tooltip").hidden = true;
    geometry = null;
    if (series.length < 2) return;
    const box = $("chart");
    const W = box.clientWidth, H = box.clientHeight;
    const narrow = W < 520;
    const m = { l: narrow ? 46 : 56, r: narrow ? 42 : 92, t: 10, b: 24 };
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    const xs = series.map((r) => r[0]);
    let lo = Infinity, hi = -Infinity;
    for (const r of series) { lo = Math.min(lo, r[1], r[2]); hi = Math.max(hi, r[1], r[2]); }
    const pad = Math.max((hi - lo) * 0.08, 0.5);
    const yt = niceTicks(lo - pad, hi + pad, narrow ? 4 : 5);
    const y0 = Math.min(lo - pad, yt.ticks[0]), y1 = Math.max(hi + pad, yt.ticks[yt.ticks.length - 1]);
    const t0 = xs[0], t1 = xs[xs.length - 1] === t0 ? t0 + MIN : xs[xs.length - 1];
    const X = (t) => m.l + (t - t0) / (t1 - t0) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const decimals = yt.step < 1 ? 2 : 0;
    const yFmt = new Intl.NumberFormat("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });

    for (const v of yt.ticks) {
      svg.appendChild(el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      svg.appendChild(el("text", { class: "tick", x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, yFmt.format(v)));
    }
    svg.appendChild(el("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    const tt = timeTicks(t0, t1, Math.max(2, Math.floor((W - m.l - m.r) / (narrow ? 70 : 90))));
    for (const t of tt.ticks) {
      const x = X(t);
      if (x < m.l + 16 || x > W - m.r - 16) continue;
      svg.appendChild(el("text", { class: "tick", x, y: H - 6, "text-anchor": "middle" },
        tt.daily ? F.day.format(t) : (new Date(t).getHours() === 0 ? F.day.format(t) : F.time.format(t))));
    }
    const path = (k) => series.map((r, i) => (i ? "L" : "M") + X(r[0]).toFixed(1) + " " + Y(r[k]).toFixed(1)).join("");
    svg.appendChild(el("path", { class: "line line-bh", d: path(2) }));
    svg.appendChild(el("path", { class: "line line-ai", d: path(1) }));

    // direct labels at the line ends, pushed apart when they would overlap
    const last = series[series.length - 1];
    let ya = Y(last[1]), yb = Y(last[2]);
    if (Math.abs(ya - yb) < 14) {
      const mid = (ya + yb) / 2, up = ya <= yb ? -7 : 7;
      ya = mid + up; yb = mid - up;
    }
    svg.appendChild(el("text", { class: "end-label", x: W - m.r + 8, y: ya + 4 }, narrow ? "Sim" : "Simulation"));
    svg.appendChild(el("text", { class: "end-label", x: W - m.r + 8, y: yb + 4 }, narrow ? "B&H" : "Buy & hold"));

    const hover = el("g", { visibility: "hidden" });
    const cross = el("line", { class: "cross", y1: m.t, y2: H - m.b });
    const da = el("circle", { class: "dot-ai", r: 4 }), db = el("circle", { class: "dot-bh", r: 4 });
    hover.append(cross, db, da);
    svg.appendChild(hover);
    geometry = { X, Y, xs, series, hover, cross, da, db, m, W, H };
    if (hoverIndex !== null) showHover(Math.min(hoverIndex, series.length - 1));
  }

  function tipRow(cls, value, name) {
    const row = document.createElement("div");
    row.className = "row";
    const key = document.createElement("span");
    key.className = "key " + cls;
    const strong = document.createElement("strong");
    strong.textContent = value;
    const label = document.createElement("span");
    label.textContent = name;
    row.append(key, strong, label);
    return row;
  }

  function showHover(i) {
    const g = geometry;
    if (!g) return;
    hoverIndex = i;
    const r = g.series[i];
    const x = g.X(r[0]);
    g.cross.setAttribute("x1", x); g.cross.setAttribute("x2", x);
    g.da.setAttribute("cx", x); g.da.setAttribute("cy", g.Y(r[1]));
    g.db.setAttribute("cx", x); g.db.setAttribute("cy", g.Y(r[2]));
    g.hover.setAttribute("visibility", "visible");
    const tip = $("tooltip");
    const head = document.createElement("div");
    head.className = "t";
    head.textContent = F.full.format(r[0]) + " · " + sim().base + " " + F.money0.format(r[3]);
    tip.replaceChildren(head, tipRow("key-ai", F.money.format(r[1]), "This simulation"), tipRow("key-bh", F.money.format(r[2]), "Buy and hold"));
    tip.hidden = false;
    const w = tip.offsetWidth;
    const left = x + 12 + w > g.W ? x - 12 - w : x + 12;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = g.m.t + "px";
  }

  function hideHover() {
    hoverIndex = null;
    if (geometry) geometry.hover.setAttribute("visibility", "hidden");
    $("tooltip").hidden = true;
  }

  function nearest(clientX) {
    const g = geometry;
    const rect = $("chart-svg").getBoundingClientRect();
    const px = clientX - rect.left;
    let lo = 0, hi = g.xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (g.X(g.xs[mid]) < px) lo = mid; else hi = mid;
    }
    return Math.abs(g.X(g.xs[lo]) - px) <= Math.abs(g.X(g.xs[hi]) - px) ? lo : hi;
  }

  function render() {
    renderDescription();
    renderStatus();
    renderKpis();
    renderChart();
    renderTrades();
  }

  async function load() {
    $("chart").classList.add("loading");
    try {
      const r = await fetch("/api/simulations/" + encodeURIComponent(simId) + "?range=" + encodeURIComponent(range), { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const next = await r.json();
      // database down: keep showing the last data received, with the new status
      state = !next.portfolio && state && state.portfolio && next.status.error ? Object.assign({}, state, { status: next.status }) : next;
      failedSince = null;
    } catch (e) {
      failedSince = failedSince || Date.now(); // keep the last render, say it is old
    } finally {
      $("chart").classList.remove("loading");
    }
    render();
  }

  function setRange(next) {
    range = next;
    try { localStorage.setItem("range", range); } catch (e) { /* not essential */ }
    for (const b of document.querySelectorAll(".ranges button")) b.setAttribute("aria-pressed", String(b.dataset.range === range));
    hoverIndex = null;
    load();
  }

  document.addEventListener("DOMContentLoaded", () => {
    for (const b of document.querySelectorAll(".ranges button")) {
      b.setAttribute("aria-pressed", String(b.dataset.range === range));
      b.addEventListener("click", () => setRange(b.dataset.range));
    }
    const svg = $("chart-svg");
    svg.setAttribute("tabindex", "0");
    svg.addEventListener("pointermove", (e) => { if (geometry) showHover(nearest(e.clientX)); });
    svg.addEventListener("pointerleave", hideHover);
    svg.addEventListener("blur", hideHover);
    svg.addEventListener("keydown", (e) => {
      if (!geometry) return;
      const n = geometry.series.length;
      let i = hoverIndex === null ? n - 1 : hoverIndex;
      if (e.key === "ArrowLeft") i = Math.max(0, i - 1);
      else if (e.key === "ArrowRight") i = Math.min(n - 1, i + 1);
      else if (e.key === "Escape") return hideHover();
      else return;
      e.preventDefault();
      showHover(i);
    });
    let resizeTimer;
    window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderChart, 100); });
    load();
    setInterval(load, REFRESH_MS);
    setInterval(renderStatus, 5000);
  });
})();
