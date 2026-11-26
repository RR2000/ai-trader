"use strict";
// Helpers shared by the home page and the simulation page.
window.Trader = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const money = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const money0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const coin = new Intl.NumberFormat("en-US", { minimumFractionDigits: 6, maximumFractionDigits: 6 });
  const pct = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero" });
  const signed = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero" });
  const time = new Intl.DateTimeFormat("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  const day = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" });
  const full = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
  const INTERVAL_WORDS = { m: "minute", h: "hour", d: "day" };

  function svg(tag, attrs, text) {
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

  // green or red, but not for a number shown as 0.00
  function setTone(node, x) {
    node.classList.remove("up", "down");
    if (x >= 0.005) node.classList.add("up");
    if (x <= -0.005) node.classList.add("down");
  }

  // "15m" -> "15-minute candles", "1h" -> "hourly candles"
  function candles(interval) {
    const n = parseInt(interval, 10), unit = INTERVAL_WORDS[interval.slice(-1)];
    if (n === 1) return (unit === "day" ? "daily" : unit === "hour" ? "hourly" : "one-minute") + " candles";
    return n + "-" + unit + " candles";
  }

  // the status pill: live, stale with the reason, or the service not answering
  function statusLine(status, failed) {
    if (failed) return { tone: "stale", text: "Data not updated: the service does not answer" };
    if (!status) return { tone: "", text: "Loading…" };
    const last = status.last_candle ? Date.parse(status.last_candle) : null;
    if (status.stale) {
      const why = status.error ? " · " + status.error : "";
      return { tone: "stale", text: last ? "Data not updated since " + full.format(last) + why
                                         : status.error ? "Data not updated · " + status.error : "Waiting for the first candle" };
    }
    return { tone: "live", text: "Live · last decision " + ago(Date.now() - last) };
  }

  return { svg, ago, setTone, candles, statusLine, fmt: { money, money0, coin, pct, signed, time, day, full } };
})();
