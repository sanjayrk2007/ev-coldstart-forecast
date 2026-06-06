import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

export default function ForecastChart({ forecast }) {
  if (!forecast || forecast.length === 0) {
    return (
      <div style={{
        height: '320px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--color-on-surface-variant)',
        fontSize: '13px',
      }}>
        No forecast data loaded. Click Evaluate or select a station.
      </div>
    );
  }

  // Pre-process data for range rendering and custom labels
  const data = forecast.map((d) => {
    const date = new Date(d.timestamp);
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const dayName = days[date.getUTCDay()];
    const hour = date.getUTCHours().toString().padStart(2, '0');
    const formattedTime = `${dayName} ${hour}:00`;

    return {
      timestamp: d.timestamp,
      formattedTime,
      predicted: Math.max(0, d.predicted),
      range_80: [Math.max(0, d.lower_80), d.upper_80],
      range_90: [Math.max(0, d.lower_90), d.upper_90],
    };
  });

  // Keep XAxis clean by showing daily labels (every 24 hours)
  const dailyTicks = data
    .filter((_, index) => index % 24 === 0)
    .map((d) => d.formattedTime);

  // Custom tooltips in Geist Mono
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const predVal = payload.find(p => p.dataKey === 'predicted')?.value ?? 0;
      const range80 = payload.find(p => p.dataKey === 'range_80')?.value ?? [0, 0];
      const range90 = payload.find(p => p.dataKey === 'range_90')?.value ?? [0, 0];

      return (
        <div style={{
          backgroundColor: 'var(--color-surface-container-high)',
          border: '1px solid var(--border-opacity-10)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 12px',
          boxShadow: 'var(--shadow-premium)',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
        }}>
          <div className="label-caps" style={{ color: 'var(--color-on-surface-variant)', fontSize: '10px', marginBottom: '4px' }}>
            {label}
          </div>
          <div className="data-mono" style={{ color: 'var(--color-primary)', display: 'flex', justifyContent: 'space-between', gap: '16px' }}>
            <span>Forecast:</span>
            <strong>{predVal.toFixed(2)} sessions/h</strong>
          </div>
          <div className="data-mono" style={{ color: 'var(--color-on-surface)', display: 'flex', justifyContent: 'space-between', gap: '16px', fontSize: '12px' }}>
            <span>80% Range:</span>
            <span>[{range80[0].toFixed(2)}, {range80[1].toFixed(2)}]</span>
          </div>
          <div className="data-mono" style={{ color: 'var(--color-on-surface-variant)', display: 'flex', justifyContent: 'space-between', gap: '16px', fontSize: '12px' }}>
            <span>90% Range:</span>
            <span>[{range90[0].toFixed(2)}, {range90[1].toFixed(2)}]</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ width: '100%', height: '340px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
        >
          <defs>
            <linearGradient id="primaryGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--color-primary)" stopOpacity={0.25} />
              <stop offset="95%" stopColor="var(--color-primary)" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-outline-variant)"
            opacity={0.3}
          />
          <XAxis
            dataKey="formattedTime"
            ticks={dailyTicks}
            stroke="var(--color-on-surface-variant)"
            tickLine={false}
            axisLine={false}
            style={{
              fontSize: '10px',
              fontFamily: 'var(--font-family-mono)',
              textTransform: 'uppercase',
            }}
          />
          <YAxis
            stroke="var(--color-on-surface-variant)"
            tickLine={false}
            axisLine={false}
            style={{
              fontSize: '11px',
              fontFamily: 'var(--font-family-mono)',
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            verticalAlign="top"
            align="right"
            height={36}
            iconType="circle"
            iconSize={8}
            wrapperStyle={{
              fontSize: '11px',
              fontFamily: 'var(--font-family-sans)',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              fontWeight: 600,
            }}
          />
          {/* Conformal Bounds (90% Interval) */}
          <Area
            name="90% Conformal Range"
            type="monotone"
            dataKey="range_90"
            stroke="none"
            fill="var(--color-primary)"
            fillOpacity={0.06}
            activeDot={false}
          />
          {/* Conformal Bounds (80% Interval) */}
          <Area
            name="80% Conformal Range"
            type="monotone"
            dataKey="range_80"
            stroke="none"
            fill="var(--color-primary)"
            fillOpacity={0.12}
            activeDot={false}
          />
          {/* Point Forecast Line */}
          <Area
            name="Point Forecast"
            type="monotone"
            dataKey="predicted"
            stroke="var(--color-primary)"
            strokeWidth={2}
            fill="url(#primaryGlow)"
            activeDot={{
              r: 4,
              stroke: 'var(--color-surface)',
              strokeWidth: 2,
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
