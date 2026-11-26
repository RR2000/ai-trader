"use strict";
(() => {
  const T = window.Trader, F = T.fmt, R = T.risk;
  const $ = (id) => document.getElementById(id);
  const REFRESH_MS = 30000;
  let data = null;
  let failedSince = null;

  function cell(row, label, cls, content) {
    const td = document.createElement("td");
    td.dataset.label = label;
    if (cls) td.className = cls;
    if (content instanceof Node) td.appendChild(content); else td.textContent = content;
    row.appendChild(td);
    return td;
  }

  function spark(points) {
    const W = 120, H = 34, P = 2;
    const svg = T.svg("svg", { class: "spark", viewBox: "0 0 " + W + " " + H, "aria-hidden": "true", focusable: "false" });
    if (!points || points.length < 2) return svg;
    let lo = Infinity, hi = -Infinity;
    for (const p of points) { lo = Math.min(lo, p[1], p[2]); hi = Math.max(hi, p[1], p[2]); }
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const t0 = points[0][0], t1 = points[points.length - 1][0] || t0 + 1;
    const X = (t) => P + (t - t0) / (t1 - t0 || 1) * (W - 2 * P);
    const Y = (v) => P + (1 - (v - lo) / (hi - lo)) * (H - 2 * P);
    const line = (k) => points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[k]).toFixed(1)).join("");
    svg.appendChild(T.svg("path", { class: "line line-bh", d: line(2) }));
    svg.appendChild(T.svg("path", { class: "line line-ai", d: line(1) }));
    return svg;
  }

  function withNote(text, note) {
    const wrap = document.createElement("span");
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = note;
    wrap.append(text, meta);
    return wrap;
  }

  // volatility, max drawdown and Sharpe, with buy and hold's below; one cell while there is too little history
  function riskCells(tr, risk, rules) {
    const m = risk && risk.strategy, b = risk && risk.buy_and_hold;
    if (!m || !m.enough) {
      const td = cell(tr, "Risk", "risk-wait", m ? "not enough data yet" : "–");
      td.colSpan = 3;
      if (m && rules) td.title = R.rule(rules);
      return;
    }
    const bh = (f) => (b && b.enough ? "B&H " + f(b) : "");
    cell(tr, "Volatility", "num", withNote(R.volatility(m), bh(R.volatility)));
    const dd = cell(tr, "Max drawdown", "num", withNote(R.drawdown(m), R.low(m)));
    if (b && b.enough) dd.title = "Buy and hold: " + R.drawdown(b) + ", " + R.low(b);
    cell(tr, "Sharpe", "num", withNote(R.sharpe(m), bh(R.sharpe)));
  }

  function updated(status) {
    const span = document.createElement("span");
    if (!status.last_candle) {
      span.textContent = status.error ? "⚠ " + status.error : "waiting for the first candle";
      if (status.error) span.className = "warn";
      return span;
    }
    span.textContent = (status.stale ? "⚠ " : "") + T.ago(Date.now() - Date.parse(status.last_candle));
    if (status.stale) {
      span.className = "warn";
      span.title = "Data not updated since " + F.full.format(Date.parse(status.last_candle)) + (status.error ? ": " + status.error : "");
    }
    return span;
  }

  function render() {
    const body = $("sims-body");
    const box = $("status");
    box.classList.remove("live", "stale");
    if (failedSince || !data) {
      if (failedSince) {
        box.classList.add("stale");
        $("status-text").textContent = "Data not updated: the service does not answer";
      }
      if (!data) { $("sims-empty").hidden = false; $("sims-empty").textContent = failedSince ? "The service does not answer right now." : "Loading…"; }
      if (!data) return;
    }
    const sims = data.simulations;
    const stale = sims.filter((s) => s.status.stale).length;
    if (!failedSince) {
      if (data.error) { box.classList.add("stale"); $("status-text").textContent = "Data not updated · " + data.error; }
      else if (stale) { box.classList.add("stale"); $("status-text").textContent = stale + " of " + sims.length + " simulations not updated"; }
      else { box.classList.add("live"); $("status-text").textContent = "Live · " + sims.length + " simulations"; }
    }
    $("sims-empty").hidden = sims.length > 0;
    body.replaceChildren();
    for (const s of sims) {
      const tr = document.createElement("tr");
      const name = document.createElement("div");
      const a = document.createElement("a");
      a.href = "/s/" + encodeURIComponent(s.id);
      a.textContent = s.name;
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = s.strategy_title + " · " + s.base + "/" + s.quote + " · " + s.interval;
      name.append(a, meta);
      cell(tr, "Simulation", "name", name);
      const p = s.portfolio;
      if (p) {
        const ret = (p.value / p.start_cash - 1) * 100;
        const vs = ret - (p.benchmark / p.start_cash - 1) * 100;
        cell(tr, "Value", "num", F.money.format(p.value) + " " + s.quote);
        T.setTone(cell(tr, "Return", "num", F.pct.format(ret) + "%"), ret);
        T.setTone(cell(tr, "vs buy and hold", "num", F.signed.format(vs) + " pp"), vs);
        riskCells(tr, s.risk, data.risk_rules);
        cell(tr, "Trades", "num", String(p.trades));
      } else {
        cell(tr, "Value", "num", "–");
        cell(tr, "Return", "num", "–");
        cell(tr, "vs buy and hold", "num", "–");
        riskCells(tr, null, data.risk_rules);
        cell(tr, "Trades", "num", "–");
      }
      cell(tr, "Updated", "updated", updated(s.status));
      cell(tr, "", "spark-col", spark(p && p.spark));
      body.appendChild(tr);
    }
  }

  async function load() {
    try {
      const r = await fetch("/api/simulations", { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const next = await r.json();
      // database down: keep the last numbers received, with the new status
      if (next.error && data) {
        for (const s of next.simulations) {
          const old = data.simulations.find((o) => o.id === s.id);
          if (old && old.portfolio && !s.portfolio) s.portfolio = old.portfolio;
        }
      }
      data = next;
      failedSince = null;
    } catch (e) {
      failedSince = failedSince || Date.now();
    }
    render();
  }

  document.addEventListener("DOMContentLoaded", () => {
    load();
    setInterval(load, REFRESH_MS);
    setInterval(render, 15000);
  });
})();
