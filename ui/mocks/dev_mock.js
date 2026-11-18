/* DEV-ONLY MOCK — PathSnap evaluator API.
 *
 * This file is loaded ONLY when the page URL contains ?mock=1
 * (see app.js). It monkey-patches window.fetch so the two provisional
 * evaluator endpoints return synthetic data, letting the UI be developed
 * and demoed before the backend team's routes exist.
 *
 * The data below is FABRICATED: a tiny square "route" near Frankfurt.
 * It is never presented as a real prediction — the UI shows a red
 * "MOCK DATA" banner whenever this module is active.
 *
 * DELETE or leave unused for the real demo: without ?mock=1 this file
 * is never requested, so the final UI cannot display mock results.
 */
(function () {
  "use strict";

  const MOCK_TRIPS = [
    { trip_id: "00000015", observations: 120 },
    { trip_id: "00000017", observations: 98 },
    { trip_id: "00000023", observations: 140 },
    { trip_id: "00000030", observations: 75 },
    { trip_id: "00000049", observations: 160 },
    { trip_id: "00000059", observations: 88 },
    { trip_id: "00000067", observations: 110 },
    { trip_id: "00000073", observations: 132 },
    { trip_id: "00000087", observations: 95 },
    { trip_id: "00000094", observations: 150 },
  ];

  const CENTER = [8.6821, 50.1109]; // [lon, lat] — arbitrary demo location

  function fakePredict(tripId) {
    // Deterministic pseudo-random jitter so different trips differ slightly.
    let seed = 0;
    for (const c of tripId) seed = (seed * 31 + c.charCodeAt(0)) >>> 0;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };

    const n = 40;
    const route = [];
    const obs = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      // Lissajous-ish squiggle so it looks like a road route, not a straight line.
      const lon = CENTER[0] + 0.02 * Math.sin(t * Math.PI * 4) + (rand() - 0.5) * 0.002;
      const lat = CENTER[1] + 0.015 * Math.cos(t * Math.PI * 2) + (rand() - 0.5) * 0.002;
      route.push([lon, lat]);
      obs.push([lon + (rand() - 0.5) * 0.0008, lat + (rand() - 0.5) * 0.0008, i]);
    }

    return {
      trip_id: tripId,
      status: "success",
      predicted_route: route,
      gps_observations: obs,
      metrics: {
        accuracy: 0.5 + rand() * 0.4,       // clearly synthetic
        precision: 0.6 + rand() * 0.3,
        recall: 0.55 + rand() * 0.35,
        f1: 0.58 + rand() * 0.32,
        route_continuity: 0.9 + rand() * 0.09,
      },
      timings: { inference_s: 0.05 + rand() * 0.1, total_s: 0.08 + rand() * 0.1 },
    };
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : input.url;
    const method = (init && init.method) || "GET";

    // Only intercept the provisional evaluator endpoints; everything
    // else (e.g. /health) passes through to the real backend.
    if (url.includes("/api/evaluator/trips") && method === "GET") {
      return Promise.resolve(
        new Response(JSON.stringify({ trips: MOCK_TRIPS }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }
    const m = url.match(/\/api\/evaluator\/trips\/([^/?]+)\/predict/);
    if (m && method === "POST") {
      return new Promise((resolve) =>
        setTimeout(
          () =>
            resolve(
              new Response(JSON.stringify(fakePredict(decodeURIComponent(m[1]))), {
                status: 200,
                headers: { "Content-Type": "application/json" },
              })
            ),
          400 // simulate inference latency
        )
      );
    }
    return realFetch(input, init);
  };
})();
