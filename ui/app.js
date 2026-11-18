/* PathSnap Evaluator Console — frontend logic.
 *
 * Design notes (for the viva):
 *  - All backend communication goes through apiGet/apiPost below; the rest
 *    of the file only sees normalised JSON. If the backend team changes
 *    endpoint paths or payload shapes, only the "API CONTRACT" section
 *    and normalize*() helpers need updating.
 *  - The API base URL is resolved at runtime (query param > localStorage >
 *    config.js) because Laptop 2 talks to Laptop 1 over the LAN; nothing
 *    is hardcoded to localhost.
 *  - Ground-truth labels are never fetched or rendered. The map shows only
 *    GPS observations and whatever route the API returns as its prediction.
 *  - Mock mode (?mock=1) loads mocks/dev_mock.js, which intercepts fetch()
 *    for the evaluator endpoints only. Without the flag the mock module is
 *    never loaded, so real responses are the only thing that can appear.
 */
"use strict";

/* ============================ API CONTRACT (provisional) ============================
 *
 * GET  {base}/api/evaluator/trips
 *   -> { "trips": [ { "trip_id": "00000015", "observations": 6854 }, ... ] }
 *
 * POST {base}/api/evaluator/trips/{trip_id}/predict        (JSON body: {})
 *   -> {
 *        "trip_id": "00000015",
 *        "status": "success",
 *        "predicted_route":    [ [lon, lat], ... ]  |  [ {"longitude":..,"latitude":..} ],
 *        "gps_observations":   [ [lon, lat, t?], ... ] | [ {"longitude":..,"latitude":..,"timestamp":..} ],
 *        "metrics":   { "accuracy": 0.87, "precision": .., "recall": .., "f1": ..,
 *                      "route_continuity": .., ... },   // whatever API returns
 *        "timings":   { "inference_s": 1.2, "total_s": 1.4 }
 *      }
 *
 * PROVISIONAL: the backend team's evaluator routes are still in progress.
 * normalize*() below tolerate the common shape variants; tighten them once
 * the real schema is merged.
 * ============================================================================== */

const $ = (id) => document.getElementById(id);

/* ------------------------- configuration ------------------------- */

function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("api");
  if (fromQuery) {
    localStorage.setItem("pathsnap_api_base", fromQuery);
    return fromQuery.replace(/\/+$/, "");
  }
  const fromStorage = localStorage.getItem("pathsnap_api_base");
  if (fromStorage) return fromStorage.replace(/\/+$/, "");
  const fromConfig = (window.APISNAP_CONFIG || {}).apiBaseUrl || "";
  return fromConfig.replace(/\/+$/, "");
}

let API_BASE = resolveApiBase();
const EP = (window.APISNAP_CONFIG || {}).endpoints || {};

/* ------------------------- mock mode (dev only) ------------------------- */

const MOCK_MODE = new URLSearchParams(window.location.search).has("mock");
if (MOCK_MODE) {
  // Dynamically inject the dev-only mock module. It is never referenced
  // elsewhere, so production builds/pages cannot accidentally use it.
  const s = document.createElement("script");
  s.src = "mocks/dev_mock.js";
  document.head.appendChild(s);
  $("mock-banner").classList.remove("hidden");
}

/* ------------------------- HTTP helpers ------------------------- */

async function apiGet(path) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 10000);
  try {
    const res = await fetch(API_BASE + path, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function apiPost(path, body) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 300000); // large trips can take minutes
  try {
    const res = await fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
      signal: ctrl.signal,
    });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    if (!res.ok) {
      const detail = data && data.detail ? data.detail : `HTTP ${res.status}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

/* ------------------------- normalisers ------------------------- */
// Each accepts the documented shape plus common variants and always returns
// a canonical structure, so the rendering code stays schema-agnostic.

function normTrips(payload) {
  const list = Array.isArray(payload) ? payload
    : Array.isArray(payload.trips) ? payload.trips
    : Array.isArray(payload.evaluator_trips) ? payload.evaluator_trips
    : [];
  return list.map((t) => ({
    trip_id: String(t.trip_id ?? t.id ?? t.tripId ?? ""),
    observations: t.observations ?? t.num_observations ?? t.obs_count ?? null,
  })).filter((t) => t.trip_id);
}

// -> array of [lon, lat]
function normRoute(coords) {
  if (!Array.isArray(coords)) return [];
  return coords
    .map((p) => {
      if (Array.isArray(p) && p.length >= 2) return [Number(p[0]), Number(p[1])];
      if (p && typeof p === "object") {
        const lon = p.longitude ?? p.lon ?? p.x;
        const lat = p.latitude ?? p.lat ?? p.y;
        if (lon != null && lat != null) return [Number(lon), Number(lat)];
      }
      return null;
    })
    .filter((p) => p && Number.isFinite(p[0]) && Number.isFinite(p[1]));
}

function normObsList(list) {
  if (!Array.isArray(list)) return [];
  return list
    .map((p) => {
      if (Array.isArray(p) && p.length >= 2) return [Number(p[0]), Number(p[1])];
      if (p && typeof p === "object") {
        const lon = p.longitude ?? p.lon ?? p.x;
        const lat = p.latitude ?? p.lat ?? p.y;
        if (lon != null && lat != null) return [Number(lon), Number(lat)];
      }
      return null;
    })
    .filter((p) => p && Number.isFinite(p[0]) && Number.isFinite(p[1]));
}

function normPredict(payload) {
  const route = normRoute(
    payload.predicted_route ?? payload.prediction ?? payload.route ?? payload.predicted_path
  );
  const obs = normObsList(
    payload.gps_observations ?? payload.observations ?? payload.track ?? payload.gps
  );
  const metrics = payload.metrics ?? payload.evaluation ?? {};
  const timings = payload.timings ?? payload.timing ?? {};
  return { route, obs, metrics, timings, raw: payload };
}

/* ------------------------- UI state ------------------------- */

const state = { trips: [], selected: null, layers: null, map: null };

function setStatus(kind, text) {
  const el = $("api-status");
  el.className = "status status-" + kind;
  el.textContent = "API: " + text;
}

function showError(msg) {
  const el = $("error-box");
  el.textContent = msg;
  el.classList.remove("hidden");
}
function clearError() { $("error-box").classList.add("hidden"); }

/* ------------------------- map ------------------------- */

function initMap() {
  // OSM tiles require internet on Laptop 2; swap for a local tile layer
  // if the demo venue has no connectivity.
  state.map = L.map("map").setView([20, 0], 2);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  state.layers = { gps: L.layerGroup(), pred: L.layerGroup() };
  state.layers.gps.addTo(state.map);
  state.layers.pred.addTo(state.map);
}

function renderMap({ obs, route }) {
  state.layers.gps.clearLayers();
  state.layers.pred.clearLayers();

  obs.forEach(([lon, lat]) => {
    L.circleMarker([lat, lon], {
      radius: 3, color: "#8250df", weight: 1, fillOpacity: 0.55,
    }).addTo(state.layers.gps);
  });

  if (route.length > 1) {
    // Leaflet wants [lat, lon]; data files store (lon, lat).
    L.polyline(route.map(([lon, lat]) => [lat, lon]), {
      color: "#d1242f", weight: 4, opacity: 0.9,
    }).addTo(state.layers.pred);
  }

  const all = obs.concat(route);
  if (all.length) {
    state.map.fitBounds(L.latLngBounds(all.map(([lon, lat]) => [lat, lon])), {
      padding: [30, 30],
    });
  }
  $("map-legend").classList.remove("hidden");
}

/* ------------------------- metrics ------------------------- */

function renderMetrics(metrics, timings) {
  const cards = $("metric-cards");
  cards.innerHTML = "";
  // Iterate whatever metrics the API returned — no hardcoded field list,
  // so new metrics appear automatically. Values that look like fractions
  // in [0,1] are shown as percentages.
  for (const [key, val] of Object.entries(metrics)) {
    if (val == null || typeof val === "object") continue;
    const card = document.createElement("div");
    card.className = "card";
    const num = Number(val);
    const display = Number.isFinite(num)
      ? (num >= 0 && num <= 1 ? (num * 100).toFixed(1) + "%" : String(val))
      : String(val);
    card.innerHTML = `<div class="value"></div><div class="label"></div>`;
    card.querySelector(".value").textContent = display;
    card.querySelector(".label").textContent = key.replace(/_/g, " ");
    cards.appendChild(card);
  }

  const dl = $("timings");
  dl.innerHTML = "";
  for (const [key, val] of Object.entries(timings)) {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key.replace(/_/g, " ");
    dd.textContent = typeof val === "number" ? val.toFixed(3) + " s" : String(val);
    row.append(dt, dd);
    dl.appendChild(row);
  }

  $("metrics-empty").classList.add("hidden");
  $("metrics").classList.remove("hidden");
}

/* ------------------------- actions ------------------------- */

async function loadTrips() {
  clearError();
  try {
    const path = (EP.trips || "/api/evaluator/trips");
    const payload = await apiGet(path);
    state.trips = normTrips(payload);
    const sel = $("trip-select");
    sel.innerHTML = "";
    if (!state.trips.length) {
      sel.innerHTML = '<option value="">— no trips returned —</option>';
      showError("API responded but returned no evaluator trips.");
      return;
    }
    state.trips.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t.trip_id;
      opt.textContent = t.observations != null
        ? `${t.trip_id}  (${t.observations} obs)`
        : t.trip_id;
      sel.appendChild(opt);
    });
    $("predict-btn").disabled = false;
    setStatus("ok", "connected");
  } catch (e) {
    setStatus("err", "unreachable");
    $("trip-select").innerHTML = '<option value="">— API unreachable —</option>';
    showError(`Could not load trips: ${e.message}\n\nCheck the API base URL and that the backend is running on Laptop 1.`);
  }
}

async function runPrediction() {
  const tripId = $("trip-select").value;
  if (!tripId) return;
  clearError();
  $("predict-btn").disabled = true;
  $("loading").classList.remove("hidden");
  $("metrics").classList.add("hidden");
  $("metrics-empty").classList.remove("hidden");

  try {
    const tmpl = EP.predict || "/api/evaluator/trips/{trip_id}/predict";
    const path = tmpl.replace("{trip_id}", encodeURIComponent(tripId));
    const payload = await apiPost(path, { trip_id: tripId });
    const { route, obs, metrics, timings } = normPredict(payload);

    renderMap({ obs, route });
    renderMetrics(metrics, timings);

    $("trip-meta").innerHTML =
      `<div>Trip: <strong>${tripId}</strong></div>` +
      `<div>Route points: ${route.length}</div>` +
      `<div>GPS points: ${obs.length}</div>`;
    setStatus("ok", "connected");
  } catch (e) {
    setStatus("err", "error");
    showError(`Prediction failed for trip ${tripId}:\n${e.message}`);
  } finally {
    $("loading").classList.add("hidden");
    $("predict-btn").disabled = false;
  }
}

async function checkHealth() {
  if (!API_BASE) { setStatus("unknown", "not configured"); return; }
  try {
    await apiGet(EP.health || "/health");
    setStatus("ok", "connected");
  } catch {
    setStatus("err", "unreachable");
  }
}

/* ------------------------- boot ------------------------- */

function boot() {
  initMap();

  if (!API_BASE) {
    $("setup-banner").classList.remove("hidden");
    $("api-url-save").addEventListener("click", () => {
      const v = $("api-url-input").value.trim();
      if (!v) return;
      localStorage.setItem("pathsnap_api_base", v);
      location.reload();
    });
    return;
  }

  $("trip-select").addEventListener("change", () => { $("predict-btn").disabled = false; });
  $("predict-btn").addEventListener("click", runPrediction);

  checkHealth();
  loadTrips();
}

document.addEventListener("DOMContentLoaded", boot);
