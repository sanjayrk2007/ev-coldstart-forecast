import React, { useState, useEffect } from 'react';
import { 
  Zap, 
  TrendingUp, 
  Activity, 
  Sun, 
  Moon, 
  RefreshCw,
  AlertTriangle,
  Play,
  FileText,
  Info
} from 'lucide-react';
import GeospatialMap from './components/GeospatialMap';
import ForecastChart from './components/ForecastChart';
import {
  checkHealth,
  evaluateSite,
  coldStartForecast,
  getKnownStationForecast,
} from './api/client';

function getDemandTierChipClass(tier) {
  switch (tier) {
    case 'Very High':
      return 'chip-success';
    case 'High':
      return 'chip-amber';
    case 'Moderate':
      return 'chip-yellow';
    case 'Low':
      return 'chip-muted';
    default:
      return 'chip-info';
  }
}

function getRecommendationChipClass(recommendation) {
  if (!recommendation) return 'chip-muted';
  if (recommendation.includes('Strong')) return 'chip-success';
  if (recommendation.includes('Moderate')) return 'chip-yellow';
  return 'chip-warning';
}

export default function App() {
  const [theme, setTheme] = useState('dark');
  const [activeTab, setActiveTab] = useState('evaluator');
  const [apiStatus, setApiStatus] = useState('connecting');
  const [modelVersion, setModelVersion] = useState('—');
  const [healthStatus, setHealthStatus] = useState(null);
  const [loadingHealth, setLoadingHealth] = useState(false);

  const isOnline = apiStatus === 'online';

  const [coldMode, setColdMode] = useState('existing');
  const [historicalStationId, setHistoricalStationId] = useState('');
  const [forecast, setForecast] = useState([]);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [errorForecast, setErrorForecast] = useState('');
  const [forecastLoaded, setForecastLoaded] = useState(false);

  const [coldStationId, setColdStationId] = useState('');
  const [coldSite, setColdSite] = useState('office');
  const [coldLat, setColdLat] = useState('');
  const [coldLng, setColdLng] = useState('');
  const [coldPorts, setColdPorts] = useState('');
  const [sessionsText, setSessionsText] = useState('');
  const [fineTunedStatus, setFineTunedStatus] = useState(false);

  const [evalLat, setEvalLat] = useState('');
  const [evalLng, setEvalLng] = useState('');
  const [evalPorts, setEvalPorts] = useState('');
  const [evalClassification, setEvalClassification] = useState('retail');
  const [evalResult, setEvalResult] = useState(null);
  const [loadingEval, setLoadingEval] = useState(false);
  const [errorEval, setErrorEval] = useState('');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const refreshHealth = async () => {
    setLoadingHealth(true);
    setApiStatus('connecting');
    const result = await checkHealth();
    if (result.ok && result.data.global_model_loaded) {
      setHealthStatus(result.data);
      setModelVersion(result.data.model_version || '—');
      setApiStatus('online');
    } else {
      setHealthStatus(null);
      setApiStatus('offline');
    }
    setLoadingHealth(false);
  };

  useEffect(() => {
    refreshHealth();
  }, []);

  const fetchHistoricalForecast = async () => {
    const stationId = historicalStationId.trim();
    if (!stationId) {
      setErrorForecast('Enter a station ID (e.g. caltech_2-39-123-23 or jpl_1-1-178-817)');
      return;
    }

    setLoadingForecast(true);
    setErrorForecast('');
    setForecast([]);
    setForecastLoaded(false);
    setFineTunedStatus(false);

    const result = await getKnownStationForecast(stationId);
    if (result.ok) {
      setForecast(result.data.forecast);
      setFineTunedStatus(result.data.fine_tuned);
      setForecastLoaded(true);
    } else {
      setErrorForecast(result.error);
    }
    setLoadingForecast(false);
  };

  const submitColdStart = async () => {
  const stationId = coldStationId.trim();
  if (!stationId) {
    setErrorForecast('Enter a candidate station identifier');
    return;
  }

  // ADD THIS BLOCK
  const parsedColdLat = parseFloat(coldLat);
  const parsedColdLng = parseFloat(coldLng);
  const parsedColdPorts = parseInt(coldPorts, 10);
  if (isNaN(parsedColdLat) || parsedColdLat < -90 || parsedColdLat > 90) {
    setErrorForecast('Enter a valid latitude between -90 and 90');
    return;
  }
  if (isNaN(parsedColdLng) || parsedColdLng < -180 || parsedColdLng > 180) {
    setErrorForecast('Enter a valid longitude between -180 and 180');
    return;
  }
  if (isNaN(parsedColdPorts) || parsedColdPorts < 1) {
    setErrorForecast('Enter a valid number of ports (minimum 1)');
    return;
  }
  setLoadingForecast(true);
  setErrorForecast('');
  setForecast([]);
  setForecastLoaded(false);
  setFineTunedStatus(false);

  let parsedSessions = [];
  if (sessionsText.trim()) {
    try {
      parsedSessions = JSON.parse(sessionsText);
      if (!Array.isArray(parsedSessions)) {
        throw new Error('Sessions must be a valid JSON array');
      }
    } catch (err) {
      setErrorForecast(`Invalid JSON format in session history: ${err.message}`);
      setLoadingForecast(false);
      return;
    }
  }

  const result = await coldStartForecast(
    stationId,
    coldSite,
    parsedColdLat,
    parsedColdLng,
    parsedColdPorts,
    parsedSessions
  );

  if (result.ok) {
    setForecast(result.data.forecast);
    setFineTunedStatus(result.data.fine_tuned);
    setForecastLoaded(true);
  } else {
    setErrorForecast(result.error);
  }
  setLoadingForecast(false);
};
  const triggerSiteEvaluation = async () => {
    const parsedLat = parseFloat(evalLat);
    const parsedLng = parseFloat(evalLng);
    const parsedPorts = parseInt(evalPorts, 10);
    if (isNaN(parsedLat) || parsedLat < -90 || parsedLat > 90) {
      setErrorEval('Enter a valid latitude between -90 and 90');
      return;
    }
    if (isNaN(parsedLng) || parsedLng < -180 || parsedLng > 180) {
      setErrorEval('Enter a valid longitude between -180 and 180');
      return;
    }
    if (isNaN(parsedPorts) || parsedPorts < 1) {
      setErrorEval('Enter a valid number of ports (minimum 1)');
      return;
    }
    setLoadingEval(true);
    setErrorEval('');
    setEvalResult(null);
    const result = await evaluateSite(parsedLat, parsedLng, parsedPorts, evalClassification);
    if (result.ok) {
      setEvalResult(result.data);
    } else {
      setErrorEval(result.error);
    }
    setLoadingEval(false);
  };

  const handleMapClick = (lat, lng) => {
    if (activeTab === 'evaluator') {
      setEvalLat(lat.toFixed(4));
      setEvalLng(lng.toFixed(4));
    } else if (coldMode === 'new') {
      setColdLat(lat.toFixed(4));
      setColdLng(lng.toFixed(4));
    }
  };

  const prefillSampleSessions = () => {
    const sampleSessions = [
      { start_time: '2026-06-01T09:00:00Z', end_time: '2026-06-01T11:30:00Z', energy_kwh: 18.5 },
      { start_time: '2026-06-02T10:15:00Z', end_time: '2026-06-02T13:00:00Z', energy_kwh: 22.0 },
      { start_time: '2026-06-03T08:00:00Z', end_time: '2026-06-03T11:00:00Z', energy_kwh: 15.6 },
    ];
    setSessionsText(JSON.stringify(sampleSessions, null, 2));
  };

  const statusDotColor =
    apiStatus === 'online'
      ? 'var(--color-primary)'
      : apiStatus === 'connecting'
        ? 'var(--color-outline)'
        : '#ffb6a3';

  const statusLabel =
    apiStatus === 'online'
      ? 'API Online'
      : apiStatus === 'connecting'
        ? 'Connecting...'
        : 'API Offline';

  const weeklyBarMax = evalResult
    ? Math.max(evalResult.confidence_interval.high, evalResult.predicted_weekly_sessions, 1)
    : 300;

  return (
    <div className="app-container">
      <header className="top-bar">
        <div className="brand-section">
          <Zap size={18} className="brand-logo" style={{ color: 'var(--color-primary)' }} />
          <span className="brand-logo" style={{ color: 'var(--color-on-surface)' }}>ChargePulse</span>
          <span className="data-mono" style={{ fontSize: '10px', background: 'var(--color-surface-container-highest)', padding: '2px 6px', borderRadius: '4px', marginLeft: '6px', color: 'var(--color-outline)' }}>
            {modelVersion}
          </span>
        </div>

        <nav className="navigation-tabs">
          <button 
            className={`nav-tab-btn ${activeTab === 'evaluator' ? 'active' : ''}`}
            onClick={() => setActiveTab('evaluator')}
          >
            Site Evaluator
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            Forecast Studio
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'nodes' ? 'active' : ''}`}
            onClick={() => setActiveTab('nodes')}
          >
            Diagnostics & Nodes
          </button>
        </nav>

        <div className="action-controls">
          <span className="flex-row" style={{ gap: '6px' }}>
            <span style={{ 
              width: '6px', 
              height: '6px', 
              borderRadius: '50%', 
              backgroundColor: statusDotColor,
              boxShadow: apiStatus === 'online' ? '0 0 8px var(--color-primary)' : 'none'
            }} />
            <span className="label-caps" style={{ fontSize: '9px', color: apiStatus === 'online' ? 'var(--color-primary)' : 'var(--color-on-surface-variant)' }}>
              {statusLabel}
            </span>
          </span>

          <button 
            className="action-btn"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title="Toggle Light/Dark Theme"
          >
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          </button>
          
          <button className="action-btn" onClick={refreshHealth} title="Refresh connection">
            <RefreshCw size={14} className={loadingHealth ? 'spin' : ''} style={{ animation: loadingHealth ? 'spin 1.5s linear infinite' : 'none' }} />
          </button>
        </div>
      </header>

      <main className="dashboard-workspace">

        {!isOnline && apiStatus !== 'connecting' && (
          <div style={{
            display: 'flex',
            gap: '8px',
            alignItems: 'center',
            color: 'var(--color-error)',
            backgroundColor: 'rgba(255,180,171,0.08)',
            padding: '12px 16px',
            borderRadius: 'var(--radius-default)',
            border: '1px solid rgba(255,180,171,0.2)',
            marginBottom: 'var(--spacing-gutter)',
          }}>
            <AlertTriangle size={16} />
            <span className="body-base" style={{ fontSize: '13px' }}>
              Backend offline — start FastAPI server at localhost:8000 to use live inference.
            </span>
          </div>
        )}

        {activeTab === 'dashboard' && (
          <div className="grid-2col">
            <div className="cp-card">
              <div className="cp-card-header">
                <span className="label-caps cp-card-title">Inference Workspace</span>
                <span className="chip chip-info">Transfer Learning</span>
              </div>

              <div className="form-group">
                <label className="label-caps form-label">Station Category</label>
                <div className="segmented-control" style={{ width: '100%' }}>
                  <button 
                    className={`segment-btn ${coldMode === 'existing' ? 'active' : ''}`}
                    onClick={() => setColdMode('existing')}
                    style={{ flex: 1 }}
                  >
                    Historical Node
                  </button>
                  <button 
                    className={`segment-btn ${coldMode === 'new' ? 'active' : ''}`}
                    onClick={() => setColdMode('new')}
                    style={{ flex: 1 }}
                  >
                    Cold-Start Candidate
                  </button>
                </div>
              </div>

              {coldMode === 'existing' ? (
                <div className="flex-col" style={{ gap: '12px' }}>
                  <div className="form-group">
                    <label className="label-caps form-label" htmlFor="station-id-input">Station ID</label>
                    <input
                      id="station-id-input"
                      type="text"
                      className="input-field data-mono"
                      value={historicalStationId}
                      onChange={(e) => setHistoricalStationId(e.target.value)}
                      placeholder="e.g. caltech_2-39-123-23 or jpl_1-1-178-817"
                    />
                    <span className="data-mono" style={{ fontSize: '10px', color: 'var(--color-on-surface-variant)', marginTop: '4px' }}>
                      107 training stations across Caltech and JPL — enter any valid station_id
                    </span>
                  </div>
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={fetchHistoricalForecast}
                    disabled={loadingForecast || !isOnline}
                  >
                    <Play size={12} fill="currentColor" /> {loadingForecast ? 'Running inference...' : 'Generate Forecast'}
                  </button>
                </div>
              ) : (
                <div className="flex-col" style={{ gap: '12px' }}>
                  <div className="form-group">
                    <label className="label-caps form-label">Candidate Identifier</label>
                    <input 
                      type="text" 
                      className="input-field" 
                      value={coldStationId} 
                      onChange={(e) => setColdStationId(e.target.value)}
                      placeholder="my-new-station-001"
                    />
                  </div>

                  <div className="grid-2col" style={{ gap: '10px', gridTemplateColumns: '1fr 1fr' }}>
                    <div className="form-group">
                      <label className="label-caps form-label">Latitude</label>
                      <input 
                        type="number" 
                        step="0.0001" 
                        className="input-field data-mono" 
                        value={coldLat} 
                        onChange={(e) => setColdLat(e.target.value)} 
                      />
                    </div>
                    <div className="form-group">
                      <label className="label-caps form-label">Longitude</label>
                      <input 
                        type="number" 
                        step="0.0001" 
                        className="input-field data-mono" 
                        value={coldLng} 
                        onChange={(e) => setColdLng(e.target.value)} 
                      />
                    </div>
                  </div>

                  <div className="grid-2col" style={{ gap: '10px', gridTemplateColumns: '1fr 1fr' }}>
                    <div className="form-group">
                      <label className="label-caps form-label">Ports Planned</label>
                      <input 
                        type="number" 
                        className="input-field data-mono" 
                        value={coldPorts} 
                        onChange={(e) => setColdPorts(e.target.value)} 
                      />
                    </div>
                    <div className="form-group">
                      <label className="label-caps form-label">Site Classification</label>
                      <select 
                        className="input-field" 
                        value={coldSite} 
                        onChange={(e) => setColdSite(e.target.value)}
                        style={{ backgroundColor: 'var(--color-surface-container-lowest)', border: '1px solid var(--border-opacity-10)' }}
                      >
                        <option value="office">Office / Workplace</option>
                        <option value="campus">Campus</option>
                        <option value="public">Public Parking</option>
                        <option value="unknown">Unknown</option>
                      </select>
                    </div>
                  </div>

                  <div className="form-group">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label className="label-caps form-label">Sparse Session History (JSON)</label>
                      <button 
                        type="button" 
                        className="label-caps" 
                        onClick={prefillSampleSessions}
                        style={{ background: 'none', border: 'none', color: 'var(--color-primary)', cursor: 'pointer', fontSize: '9px' }}
                      >
                        [Prefill Samples]
                      </button>
                    </div>
                    <textarea 
                      className="input-field data-mono" 
                      rows={5} 
                      value={sessionsText} 
                      onChange={(e) => setSessionsText(e.target.value)} 
                      placeholder='Optional: [{"start_time":"2026-06-01T09:00:00Z","end_time":"2026-06-01T11:00:00Z","energy_kwh":15.5}]'
                      style={{ resize: 'vertical', fontSize: '11px', backgroundColor: 'var(--color-surface-container-lowest)' }}
                    />
                  </div>

                  <button 
                    type="button" 
                    className="btn-primary" 
                    onClick={submitColdStart}
                    disabled={loadingForecast || !isOnline}
                  >
                    <Play size={12} fill="currentColor" /> {loadingForecast ? 'Running inference...' : 'Incorporate & Predict'}
                  </button>
                </div>
              )}
              {coldMode === 'new' && (
  <div className="form-group" style={{ marginTop: 'auto' }}>
    <label className="label-caps form-label">Geospatial Target</label>
    {!isNaN(parseFloat(coldLat)) && !isNaN(parseFloat(coldLng)) ? (
      <GeospatialMap
        lat={parseFloat(coldLat)}
        lng={parseFloat(coldLng)}
        onMapClick={handleMapClick}
        theme={theme}
      />
    ) : (
      <div style={{
        height: '200px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px dashed var(--color-outline)',
        borderRadius: 'var(--radius-default)',
        color: 'var(--color-on-surface-variant)',
        fontSize: '12px',
        fontFamily: 'var(--font-family-sans)',
      }}>
        Enter coordinates or click map to set location
      </div>
    )}
    <span className="data-mono" style={{ fontSize: '10px', color: 'var(--color-on-surface-variant)', textAlign: 'center', marginTop: '4px' }}>
      Click map to relocate coordinates
    </span>
  </div>
)}
            </div>

            <div className="flex-col" style={{ gap: 'var(--spacing-gutter)' }}>
              <div className="cp-card" style={{ gap: '4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <span className="label-caps" style={{ color: 'var(--color-primary)' }}>
                      {coldMode === 'existing' ? 'HISTORICAL FORECAST' : 'COLD-START GENERATIVE FORECAST'}
                    </span>
                    <h2 className="headline-md" style={{ marginTop: '4px' }}>
                      {coldMode === 'existing'
                        ? (historicalStationId.trim() || 'Enter station ID')
                        : (coldStationId.trim() || 'New candidate')}
                    </h2>
                  </div>
                  <div className="flex-row">
                    {forecastLoaded && (
                      fineTunedStatus
                        ? <span className="chip chip-success">Fine-tuned</span>
                        : <span className="chip chip-muted">Cold-start only</span>
                    )}
                    <span className="chip chip-info" style={{ fontFamily: 'var(--font-family-mono)' }}>
                      7D Horizon (168h)
                    </span>
                  </div>
                </div>
              </div>

              <div className="cp-card" style={{ flex: 1 }}>
                <div className="cp-card-header">
                  <span className="label-caps cp-card-title">Hourly Session Demand Profile</span>
                  <div className="flex-row" style={{ gap: '16px' }}>
                    <span className="flex-row" style={{ gap: '6px', fontSize: '11px', color: 'var(--color-on-surface-variant)' }}>
                      <span style={{ width: '12px', height: '6px', backgroundColor: 'var(--color-primary)', opacity: 0.12 }} /> 80% Bounds
                    </span>
                    <span className="flex-row" style={{ gap: '6px', fontSize: '11px', color: 'var(--color-on-surface-variant)' }}>
                      <span style={{ width: '12px', height: '6px', backgroundColor: 'var(--color-primary)', opacity: 0.06 }} /> 90% Bounds
                    </span>
                  </div>
                </div>

                {errorForecast && (
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center', color: 'var(--color-error)', backgroundColor: 'rgba(255,180,171,0.08)', padding: '10px', borderRadius: '4px', border: '1px solid rgba(255,180,171,0.2)' }}>
                    <AlertTriangle size={16} />
                    <span className="body-base" style={{ fontSize: '12px' }}>{errorForecast}</span>
                  </div>
                )}

                <div style={{ flex: 1, position: 'relative' }}>
                  {loadingForecast ? (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(16,20,23,0.5)', zIndex: 10 }}>
                      <div className="flex-col" style={{ alignItems: 'center', gap: '8px' }}>
                        <RefreshCw className="spin" style={{ animation: 'spin 1s linear infinite', color: 'var(--color-primary)' }} />
                        <span className="label-caps" style={{ fontSize: '10px' }}>Running inference...</span>
                      </div>
                    </div>
                  ) : null}

                  <ForecastChart forecast={forecast} />
                </div>
              </div>

              <div className="cp-card" style={{ gridColumn: 'span 1', flexDirection: 'row', gap: '16px', alignItems: 'center', backgroundColor: 'var(--color-surface-container-lowest)' }}>
                <Info size={24} style={{ color: 'var(--color-primary)', flexShrink: 0 }} />
                <p className="body-base" style={{ fontSize: '12px', color: 'var(--color-on-surface-variant)' }}>
                  <strong>Conformal Quantile Interpretation:</strong> In quiet intervals (typically 00:00–06:00), the model switches to a zero-demand conformal regime where intervals narrow considerably, mirroring the high-certainty zero activity. During daytime surges, the bounds dynamically expand to capture the larger variances in peak charging activities.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'evaluator' && (
          <div className="grid-2col">
            <div className="cp-card">
              <div className="cp-card-header">
                <span className="headline-md cp-card-title" style={{ fontSize: '18px', fontWeight: 600 }}>Candidate Location</span>
              </div>

              <div className="form-group" style={{ marginTop: '8px' }}>
                <label className="label-caps form-label" style={{ fontSize: '10px', color: 'var(--color-on-surface-variant)' }}>Coordinates</label>
                <div className="grid-2col" style={{ gap: '10px', gridTemplateColumns: '1fr 1fr' }}>
                  <input 
                    type="number" 
                    step="0.0001" 
                    className="input-field data-mono" 
                    value={evalLat} 
                    onChange={(e) => setEvalLat(e.target.value)}
                    placeholder="Latitude"
                  />
                  <input 
                    type="number" 
                    step="0.0001" 
                    className="input-field data-mono" 
                    value={evalLng} 
                    onChange={(e) => setEvalLng(e.target.value)} 
                    placeholder="Longitude"
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="label-caps form-label" style={{ fontSize: '10px', color: 'var(--color-on-surface-variant)' }}>Planned Infrastructure</label>
                <div style={{ position: 'relative' }}>
                  <input 
                    type="number" 
                    className="input-field data-mono" 
                    value={evalPorts} 
                    onChange={(e) => setEvalPorts(e.target.value)}
                    style={{ width: '100%', paddingRight: '60px' }}
                  />
                  <span className="label-caps" style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--color-outline)', pointerEvents: 'none' }}>
                    Ports
                  </span>
                </div>
              </div>

              <div className="form-group">
                <label className="label-caps form-label" style={{ fontSize: '10px', color: 'var(--color-on-surface-variant)' }}>Location Classification</label>
                <select 
                  className="input-field" 
                  value={evalClassification} 
                  onChange={(e) => setEvalClassification(e.target.value)}
                  style={{ backgroundColor: 'var(--color-surface-container-lowest)', border: '1px solid var(--border-opacity-10)', cursor: 'pointer' }}
                >
                  <option value="retail">Mixed-Use Commercial</option>
                  <option value="workplace">Office / Corporate Campus</option>
                  <option value="public">Public Parking / Municipal Hub</option>
                </select>
              </div>

              <button 
                type="button" 
                className="btn-primary" 
                onClick={triggerSiteEvaluation}
                disabled={loadingEval || !isOnline}
                style={{ width: '100%', height: '40px', marginTop: '12px', fontSize: '12px' }}
              >
                {loadingEval ? 'Running inference...' : 'EVALUATE SITE'}
              </button>

        <div className="form-group" style={{ marginTop: 'auto' }}>
          <label className="label-caps form-label" style={{ fontSize: '10px', color: 'var(--color-on-surface-variant)' }}>Geospatial Preview: Active</label>
          {!isNaN(parseFloat(evalLat)) && !isNaN(parseFloat(evalLng)) ? (
            <GeospatialMap 
      lat={parseFloat(evalLat)} 
      lng={parseFloat(evalLng)} 
      onMapClick={handleMapClick}
      theme={theme}
    />
  ) : (
    <div style={{
      height: '200px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      border: '1px dashed var(--color-outline)',
      borderRadius: 'var(--radius-default)',
      color: 'var(--color-on-surface-variant)',
      fontSize: '12px',
      fontFamily: 'var(--font-family-sans)',
    }}>
      Enter coordinates to preview map
    </div>
  )}
</div>
  
            </div>

            <div className="flex-col" style={{ gap: 'var(--spacing-gutter)' }}>

              {errorEval && (
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', color: 'var(--color-error)', backgroundColor: 'rgba(255,180,171,0.08)', padding: '10px', borderRadius: '4px', border: '1px solid rgba(255,180,171,0.2)' }}>
                  <AlertTriangle size={16} />
                  <span className="body-base" style={{ fontSize: '12px' }}>{errorEval}</span>
                </div>
              )}

              <div className="grid-2col" style={{ gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-gutter)' }}>
                <div className="cp-card" style={{ justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="label-caps" style={{ color: 'var(--color-on-surface-variant)' }}>Demand Tier</span>
                    {evalResult && (
                      <span className={`chip ${getDemandTierChipClass(evalResult.demand_tier)}`} style={{ padding: '2px 6px', fontSize: '10px' }}>
                        {evalResult.demand_tier}
                      </span>
                    )}
                  </div>
                  <div>
                    <h1 className="display-lg" style={{ fontSize: '42px', fontWeight: 600, color: 'var(--color-on-surface)' }}>
                      {loadingEval ? '—' : evalResult ? evalResult.demand_tier : '—'}
                    </h1>
                    {evalResult && (
                      <p className="data-mono" style={{ fontSize: '12px', color: 'var(--color-on-surface-variant)', marginTop: '4px' }}>
                        80% interval: {evalResult.confidence_interval.low.toFixed(1)} – {evalResult.confidence_interval.high.toFixed(1)} sessions/week
                      </p>
                    )}
                  </div>
                  <div>
                    <span className={`chip ${getRecommendationChipClass(evalResult?.recommendation)}`} style={{ padding: '4px 10px', fontSize: '10px', display: 'inline-flex', gap: '6px' }}>
                      <span style={{ width: '4px', height: '4px', borderRadius: '50%', backgroundColor: 'currentColor' }} />
                      {loadingEval ? 'PENDING' : evalResult ? evalResult.recommendation.toUpperCase() : 'PENDING'}
                    </span>
                  </div>
                </div>

                <div className="cp-card" style={{ borderLeft: '3px solid var(--color-primary)' }}>
                  <span className="label-caps" style={{ color: 'var(--color-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Activity size={12} /> Demand Signal
                  </span>
                  <div className="body-base" style={{ color: 'var(--color-on-surface)', fontSize: '13px', lineHeight: '20px', marginTop: '6px' }}>
                    {loadingEval ? (
                      <span style={{ color: 'var(--color-on-surface-variant)' }}>Running inference...</span>
                    ) : evalResult ? (
                      evalResult.roi_signal
                    ) : (
                      'Submit coordinates to analyze location demand.'
                    )}
                  </div>
                </div>
              </div>

              <div className="cp-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="label-caps" style={{ color: 'var(--color-on-surface-variant)' }}>Projected Weekly Sessions</span>
                  <span className="data-mono-lg" style={{ color: 'var(--color-primary)' }}>
                    {loadingEval ? '—' : evalResult
                      ? `${evalResult.predicted_weekly_sessions.toFixed(1)} sessions/week`
                      : '—'}
                  </span>
                </div>
                {evalResult && !loadingEval && (
                  <>
                    <div className="progress-bar-container" style={{ margin: '4px 0' }}>
                      <div 
                        className="progress-bar-fill" 
                        style={{ width: `${Math.min(100, (evalResult.predicted_weekly_sessions / weeklyBarMax) * 100)}%` }}
                      />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--color-outline)', fontFamily: 'var(--font-family-mono)' }}>
                      <span>{evalResult.confidence_interval.low.toFixed(0)}</span>
                      <span>{evalResult.predicted_weekly_sessions.toFixed(0)}</span>
                      <span>{evalResult.confidence_interval.high.toFixed(0)}</span>
                    </div>
                  </>
                )}
              </div>

              <div className="cp-card" style={{ flex: 1 }}>
                <div className="cp-card-header">
                  <span className="label-caps cp-card-title">Similar Reference Stations</span>
                </div>

                <div style={{ flex: 1, overflowX: 'auto' }}>
                  {loadingEval ? (
                    <div style={{ padding: '24px', textAlign: 'center', color: 'var(--color-on-surface-variant)' }}>
                      Running inference...
                    </div>
                  ) : (
                    <table className="cp-table">
                      <thead>
                        <tr>
                          <th className="label-caps" style={{ fontSize: '10px' }}>Station ID</th>
                          <th className="label-caps" style={{ fontSize: '10px' }}>Site</th>
                          <th className="label-caps" style={{ fontSize: '10px' }}>Weekly Sessions</th>
                          <th className="label-caps" style={{ fontSize: '10px' }}>Similarity</th>
                        </tr>
                      </thead>
                      <tbody className="data-mono">
                        {evalResult && evalResult.similar_stations.length > 0 ? (
                          evalResult.similar_stations.map((st) => (
                            <tr key={st.station_id}>
                              <td style={{ fontWeight: 'bold' }}>{st.station_id}</td>
                              <td style={{ fontFamily: 'var(--font-family-sans)', fontSize: '13px' }}>{st.site}</td>
                              <td>{st.weekly_mean_sessions.toFixed(2)}</td>
                              <td>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                  <div className="progress-bar-container" style={{ width: '60px', height: '4px' }}>
                                    <div className="progress-bar-fill" style={{ width: `${st.similarity_score * 100}%` }} />
                                  </div>
                                  <span>{(st.similarity_score * 100).toFixed(1)}%</span>
                                </div>
                              </td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan="4" style={{ textAlign: 'center', color: 'var(--color-on-surface-variant)', padding: '24px' }}>
                              {evalResult ? 'No similar stations returned.' : 'Run evaluation to load similar stations.'}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>

              {evalResult?.note && (
                <div className="cp-card" style={{ flexDirection: 'row', gap: '12px', alignItems: 'flex-start', backgroundColor: 'var(--color-surface-container-lowest)' }}>
                  <Info size={18} style={{ color: 'var(--color-on-surface-variant)', flexShrink: 0, marginTop: '2px' }} />
                  <p className="body-base" style={{ fontSize: '12px', color: 'var(--color-on-surface-variant)', lineHeight: '18px' }}>
                    {evalResult.note}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'nodes' && (
          <div className="flex-col" style={{ gap: 'var(--spacing-gutter)' }}>
            <div className="grid-2col" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--spacing-gutter)' }}>
              
              <div className="cp-card">
                <span className="label-caps" style={{ color: 'var(--color-on-surface-variant)' }}>Model State</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '6px' }}>
                  <Activity size={24} style={{ color: isOnline ? 'var(--color-primary)' : 'var(--color-error)' }} />
                  <div>
                    <h3 className="headline-md" style={{ fontSize: '18px' }}>
                      {apiStatus === 'connecting'
                        ? 'CONNECTING'
                        : isOnline
                          ? (healthStatus?.status || 'OK').toUpperCase()
                          : 'OFFLINE'}
                    </h3>
                    <span className="data-mono" style={{ fontSize: '12px', color: 'var(--color-outline)' }}>
                      {isOnline ? 'Inference Engine Active' : 'Backend unreachable at localhost:8000'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="cp-card">
                <span className="label-caps" style={{ color: 'var(--color-on-surface-variant)' }}>Conformal Calibration</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '6px' }}>
                  <FileText size={24} style={{ color: 'var(--color-primary)' }} />
                  <div>
                    <h3 className="headline-md" style={{ fontSize: '18px' }}>
                      {healthStatus && healthStatus.calibration_loaded ? 'CALIBRATED' : isOnline ? 'NOT LOADED' : 'OFFLINE'}
                    </h3>
                    <span className="data-mono" style={{ fontSize: '12px', color: 'var(--color-outline)' }}>
                      Calibration snapshot (office001 held-out set)
                    </span>
                  </div>
                </div>
              </div>

              <div className="cp-card">
                <span className="label-caps" style={{ color: 'var(--color-on-surface-variant)' }}>Active Training Catalog</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '6px' }}>
                  <TrendingUp size={24} style={{ color: 'var(--color-primary)' }} />
                  <div>
                    <h3 className="headline-md" style={{ fontSize: '18px' }}>
                      107 Stations
                    </h3>
                    <span className="data-mono" style={{ fontSize: '12px', color: 'var(--color-outline)' }}>
                      Across Caltech & JPL
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="cp-card">
              <div className="cp-card-header">
                <span className="label-caps cp-card-title">Conformal Calibrations State & Residuals</span>
                <span className="chip chip-info" style={{ fontSize: '9px' }}>Calibration snapshot (office001 held-out set)</span>
              </div>
              <div className="grid-2col" style={{ gridTemplateColumns: '4fr 8fr', gap: 'var(--spacing-gutter)' }}>
                <div className="flex-col" style={{ gap: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-opacity-10)', paddingBottom: '6px' }}>
                    <span className="body-base" style={{ color: 'var(--color-on-surface-variant)' }}>Calibration Fraction</span>
                    <span className="data-mono">70% Chrono</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-opacity-10)', paddingBottom: '6px' }}>
                    <span className="body-base" style={{ color: 'var(--color-on-surface-variant)' }}>q80 Single</span>
                    <span className="data-mono">0.082 sessions</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-opacity-10)', paddingBottom: '6px' }}>
                    <span className="body-base" style={{ color: 'var(--color-on-surface-variant)' }}>q80 Zero-Regime</span>
                    <span className="data-mono">0.000 sessions</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-opacity-10)', paddingBottom: '6px' }}>
                    <span className="body-base" style={{ color: 'var(--color-on-surface-variant)' }}>q80 Non-Zero-Regime</span>
                    <span className="data-mono">0.125 sessions</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px' }}>
                    <span className="body-base" style={{ color: 'var(--color-on-surface-variant)' }}>Target Alpha Level</span>
                    <span className="data-mono">80% / 90%</span>
                  </div>
                </div>

                <div className="flex-col" style={{ gap: '10px', justifyContent: 'center' }}>
                  <h4 className="headline-md" style={{ fontSize: '15px' }}>Conformal Confidence Bands Explanation</h4>
                  <p className="body-base" style={{ color: 'var(--color-on-surface-variant)', fontSize: '13px', lineHeight: '20px' }}>
                    To avoid generic margin indicators which oversimplify predictive confidence, ChargePulse implements <strong>Split Conformal Prediction</strong>. By partitioning residuals into dual regimes (zero and non-zero predictions), the system mathematically guarantees that actual charging station occupancy will fall within the shaded intervals exactly 80% and 90% of the time, respectively.
                  </p>
                  <p className="body-base" style={{ color: 'var(--color-on-surface-variant)', fontSize: '13px', lineHeight: '20px' }}>
                    Model updates and calibration parquets are recompiled automatically during night runs to capture grid-load trends.
                  </p>
                </div>
              </div>
            </div>

            <div className="cp-card" style={{ flex: 1 }}>
              <div className="cp-card-header">
                <span className="label-caps cp-card-title">System Node Catalog (Caltech / JPL)</span>
              </div>
              <p className="body-base" style={{ color: 'var(--color-on-surface-variant)', fontSize: '13px', lineHeight: '20px' }}>
                107 training stations across Caltech and JPL sites. Use Forecast Studio to query any station by ID.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
