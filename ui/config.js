/* PathSnap evaluator UI — API configuration.
 *
 * The UI runs on a SECOND laptop (Laptop 2) and calls the FastAPI
 * backend on Laptop 1 over the LAN, so the base URL is NEVER
 * hardcoded to localhost in app logic.
 *
 * Resolution order (first match wins):
 *   1. ?api=http://<laptop1-ip>:8000   URL query parameter
 *   2. localStorage["pathsnap_api_base"]  (set once via the UI banner)
 *   3. window.APISNAP_CONFIG.apiBaseUrl below (edit for permanent default)
 *
 * Example:  open  http://localhost:8080/ui/?api=http://192.168.1.20:8000
 */
window.APISNAP_CONFIG = {
  // Leave empty to force the user to supply the URL at runtime.
  apiBaseUrl: "",
  // Provisional endpoint paths — confirmed/adjusted once the backend
  // team's evaluator routes are merged. See ui/README.md.
  endpoints: {
    trips: "/api/evaluator/trips",
    predict: "/api/evaluator/trips/{trip_id}/predict",
    health: "/health",
  },
};
