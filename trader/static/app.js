"use strict";
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const REFRESH_MS = 30000;
  const $ = (id) => document.getElementById(id);
  const usd = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const usd0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const btc = new Intl.NumberFormat("en-US", { minimumFractionDigits: 6, maximumFractionDigits: 6 });
  const pct = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero" });
  const signed = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero" });
  const timeFmt = new Intl.DateTimeFormat("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  const dayFmt = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" });
  const fullFmt = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });

  let range = "7d";
  try { range = localStorage.getItem("range") || range; } catch (e) { /* storage blocked: default */ }
  if (!["1d", "7d", "30d", "all"].includes(range)) range = "7d";
  let state = null;
  let failedSince = null;
  let hoverIndex = null;
  let geometry = null;

  function el(tag, attrs, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function ago(ms) {
    const s = Math.max(0, Math.round(ms / 1000));
    if (s < 60) return s + " s ago";
    const m = Math.round(s / 60);
    if (m < 60) return m + " min ago";
    const h = Math.round(m / 60);
    return h < 48 ? h + " h ago" : Math.round(h / 24) + " days ago";
  }

  function setTone(node, x) {
    node.classList.remove("up", "down");
    if (x > 0) node.classList.add("up");
    if (x < 0) node.classList.add("down");
  }

  function renderStatus() {
    const box = $("status"), text = $("status-text");
    box.classList.remove("live", "stale");
    if (failedSince) {
      box.classList.add("stale");
      text.textContent = "Data not updated: the service does not answer";
      return;
    }
    if (!state) return;
    const st = state.status;
    const last = st.last_candle ? Date.parse(st.last_candle) : null;
    if (st.stale) {
      box.classList.add("stale");
      const why = st.error ? " · " + st.error : "";
      text.textContent = last ? "Data not updated since " + fullFmt.format(last) + why
                              : "Waiting for the first decision" + why;
    } else {
      box.classList.add("live");
      text.textContent = "Live · last decision " + ago(Date.now() - last);
    }
  }

  function renderKpis() {
    const p = state && state.portfolio;
    if (!p) return;
    const since = dayFmt.format(Date.parse(p.started_at));
    const aiPct = (p.value / p.start_cash - 1) * 100;
    const bhPct = (p.benchmark / p.start_cash - 1) * 100;
    $("k-value").textContent = usd.format(p.value);
    $("k-value-note").textContent = pct.format(aiPct) + "% since " + since;
    setTone($("k-value-note"), aiPct);
    $("k-bench").textContent = usd.format(p.benchmark);
    $("k-bench-note").textContent = pct.format(bhPct) + "% since " + since;
    setTone($("k-bench-note"), bhPct);
    const diff = p.value - p.benchmark;
    $("k-diff").textContent = signed.format(diff);
    setTone($("k-diff"), diff);
    $("k-diff-note").textContent = signed.format(aiPct - bhPct) + " percentage points";
    $("k-position").textContent = p.position;
    $("k-position-note").textContent = (p.position === "BTC" ? btc.format(p.crypto) + " BTC" : usd.format(p.cash) + " USDT")
      + (p.last_action ? " · last call: " + p.last_action : "");
    const s = state.stats;
    if (s) {
      const m = s.last_24h;
      $("decision-mix").textContent = "Model calls in the last 24 h: " + m.buy + " buy, " + m.sell + " sell, " + m.hold
        + " hold (a buy while already in BTC, or a sell while in USDT, changes nothing). "
        + s.decisions + " decisions since " + fullFmt.format(Date.parse(p.started_at)) + ", BTC at " + usd.format(p.price) + " USDT now.";
    }
  }

  function renderTrades() {
    const body = $("trades-body");
    body.replaceChildren();
    const trades = (state && state.trades) || [];
    $("trades-empty").hidden = trades.length > 0;
    const s = state && state.stats;
    $("trades-count").textContent = s && s.trades
      ? s.trades + " in total, " + usd.format(s.fees) + " USDT paid in fees" + (s.trades > trades.length ? " (latest " + trades.length + " shown)" : "")
      : "";
    for (const t of trades) {
      const tr = document.createElement("tr");
      const cells = [
        [fullFmt.format(Date.parse(t.ts)), ""],
        [t.side === "buy" ? "Buy" : "Sell", "side side-" + t.side],
        [usd0.format(t.price), "num"],
        [btc.format(t.quantity), "num wide"],
        [usd.format(t.fee_usdt), "num wide"],
        [usd.format(t.value_after), "num"],
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
    $("tooltip").hidden = true;
    geometry = null;
    if (series.length < 2) return;
    const box = $("chart");
    const W = box.clientWidth, H = box.clientHeight;
    const narrow = W < 520;
    const m = { l: narrow ? 46 : 56, r: narrow ? 70 : 92, t: 10, b: 24 };
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
        tt.daily ? dayFmt.format(t) : (new Date(t).getHours() === 0 ? dayFmt.format(t) : timeFmt.format(t))));
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
    svg.appendChild(el("text", { class: "end-label", x: W - m.r + 8, y: ya + 4 }, "AI"));
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
    head.textContent = fullFmt.format(r[0]) + " · BTC " + usd0.format(r[3]);
    tip.replaceChildren(head, tipRow("key-ai", usd.format(r[1]), "AI portfolio"), tipRow("key-bh", usd.format(r[2]), "Buy and hold"));
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
    renderStatus();
    renderKpis();
    renderChart();
    renderTrades();
  }

  async function load() {
    $("chart").classList.add("loading");
    try {
      const r = await fetch("/api/state?range=" + encodeURIComponent(range), { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      state = await r.json();
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
