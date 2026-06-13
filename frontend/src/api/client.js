const API_BASE = '';

async function parseJsonResponse(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

function errorMessage(res, body, fallback) {
  if (body && typeof body.detail === 'string') return body.detail;
  if (body && Array.isArray(body.detail)) {
    return body.detail.map((d) => d.msg || String(d)).join('; ');
  }
  if (body && body.message) return body.message;
  return fallback || `Request failed (${res.status})`;
}

/**
 * @returns {Promise<{ ok: true, data: object } | { ok: false, error: string, status?: number }>}
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      return {
        ok: false,
        error: `Health check failed (${res.status})`,
        status: res.status,
      };
    }
    const data = await res.json();
    return { ok: true, data };
  } catch (err) {
    return {
      ok: false,
      error: err.message || 'Network error — is the backend running?',
    };
  }
}

/**
 * @returns {Promise<{ ok: true, data: object } | { ok: false, error: string, status?: number }>}
 */
export async function evaluateSite(lat, lng, numPorts, locationType) {
  try {
    const res = await fetch(`${API_BASE}/site/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lat: parseFloat(lat),
        lng: parseFloat(lng),
        num_ports: parseInt(numPorts, 10),
        location_type: locationType,
      }),
    });
    const body = await parseJsonResponse(res);
    if (!res.ok) {
      return {
        ok: false,
        error: errorMessage(res, body, `Site evaluation failed (${res.status})`),
        status: res.status,
      };
    }
    return { ok: true, data: body };
  } catch (err) {
    return {
      ok: false,
      error: err.message || 'Request failed — check that the backend is running',
    };
  }
}

/**
 * @returns {Promise<{ ok: true, data: object } | { ok: false, error: string, status?: number }>}
 */
export async function coldStartForecast(stationId, site, lat, lng, numPorts, sessions) {
  try {
    const payload = {
      station_id: stationId,
      site,
      lat: parseFloat(lat),
      lng: parseFloat(lng),
      num_ports: parseInt(numPorts, 10),
      sessions: sessions ?? [],
    };

    const res = await fetch(`${API_BASE}/forecast/cold-start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await parseJsonResponse(res);
    if (!res.ok) {
      return {
        ok: false,
        error: errorMessage(res, body, `Cold-start forecast failed (${res.status})`),
        status: res.status,
      };
    }
    return { ok: true, data: body };
  } catch (err) {
    return {
      ok: false,
      error: err.message || 'Request failed — check that the backend is running',
    };
  }
}

/**
 * @returns {Promise<{ ok: true, data: object } | { ok: false, error: string, status?: number }>}
 */
export async function getKnownStationForecast(stationId) {
  try {
    const res = await fetch(`${API_BASE}/forecast/stations/${encodeURIComponent(stationId)}`);
    const body = await parseJsonResponse(res);
    if (!res.ok) {
      const fallback =
        res.status === 404
          ? `Station '${stationId}' not found — check the station ID`
          : `Forecast failed (${res.status})`;
      return {
        ok: false,
        error: errorMessage(res, body, fallback),
        status: res.status,
      };
    }
    return { ok: true, data: body };
  } catch (err) {
    return {
      ok: false,
      error: err.message || 'Request failed — check that the backend is running',
    };
  }
}

export { API_BASE };
