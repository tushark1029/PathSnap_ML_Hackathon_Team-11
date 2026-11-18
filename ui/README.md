# PathSnap Evaluator UI

Browser-based evaluator console. Runs on **Laptop 2**; calls the FastAPI
backend on **Laptop 1** over the LAN. Static files only — no build step,
no Node.js, no Python required on the UI machine (just a modern browser).

## Run it

Any static file server works. From the repo root:

```bash
# option 1 — Python
python -m http.server 8080 --directory ui

# option 2 — any static host (nginx, VS Code Live Server, etc.)
```

Then open, pointing at the API laptop's IP:

```
http://localhost:8080/?api=http://<laptop1-ip>:8000
```

The `?api=` value is saved to `localStorage`, so it only needs to be
typed once per browser. Alternatives:

- edit `apiBaseUrl` in `ui/config.js` for a permanent default, or
- use the setup banner on first load to type the URL.

`localhost` is never assumed — if no URL is configured the UI shows a
setup banner instead of guessing.

## API contract (PROVISIONAL)

The backend team's evaluator routes are still in progress. The UI is
coded against this proposed contract; all calls live in `ui/app.js`
(API CONTRACT comment + `norm*()` normalisers tolerate shape variants).

```
GET  /api/evaluator/trips
  -> { "trips": [ { "trip_id": "00000015", "observations": 6854 }, ... ] }

POST /api/evaluator/trips/{trip_id}/predict     body: {}
  -> {
       "trip_id": "...",
       "status": "success",
       "predicted_route":  [[lon, lat], ...],
       "gps_observations": [{"longitude":..,"latitude":..,"timestamp":..}, ...],
       "metrics":  { "accuracy": .., "precision": .., "recall": .., "f1": .., ... },
       "timings":  { "inference_s": .., "total_s": .. }
     }
```

When the real schema lands, update `ui/config.js` `endpoints` and the
normalisers in `ui/app.js` — nothing else should need to change.

Ground-truth labels (`.route` files) are never requested or displayed;
the UI renders only what the API returns.

## Mock mode (development only)

```
http://localhost:8080/?api=http://<anything>/?mock=1
```

Appending `?mock=1` loads `ui/mocks/dev_mock.js`, which intercepts only
the two evaluator endpoints and returns clearly-fake data plus a red
MOCK DATA banner. Without the flag the mock file is never loaded, so the
shipped UI cannot show mock results.

## CORS

The backend already sets `allow_origins=["*"]`, so no backend change is
needed for Laptop 2 access. If the backend team later restricts origins,
add the UI origin (e.g. `http://<laptop2-ip>:8080`) to the allow-list.

## Files

| File | Purpose |
|---|---|
| `index.html` | page skeleton |
| `app.js` | API client, normalisers, map + metrics rendering |
| `config.js` | base-URL resolution + endpoint paths |
| `styles.css` | layout |
| `mocks/dev_mock.js` | dev-only fake API (never loaded without `?mock=1`) |
