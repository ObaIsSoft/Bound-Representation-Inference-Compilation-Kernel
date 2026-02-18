/**
 * Simulation Overlay - CFD/Stress Analysis HUD
 * 
 * Displays real-time simulation results as overlays on the 3D view.
 * Includes:
 * - Pressure/velocity heatmaps
 * - Stress concentration indicators
 * - Safety factor visualization
 * - Force vector arrows
 */

import React from 'react';

/**
 * Simulation Overlay Component
 */
export default function SimulationOverlay({ results, theme }) {
  if (!results) return null;
  
  const { type, metrics, timestamp } = results;
  
  return (
    <div 
      className="absolute top-20 left-4 w-80 rounded-xl p-4 z-50"
      style={{
        backgroundColor: theme.colors.bg.secondary + 'EE',
        backdropFilter: 'blur(12px)',
        border: `1px solid ${theme.colors.border.primary}`,
        color: theme.colors.text.primary
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div 
            className="w-2 h-2 rounded-full animate-pulse"
            style={{ backgroundColor: theme.colors.status.info }}
          />
          <span className="font-semibold">
            {type === 'CFD' ? 'CFD Analysis' : 'Structural Analysis'}
          </span>
        </div>
        <span style={{ color: theme.colors.text.muted }} className="text-xs">
          {timestamp ? new Date(timestamp).toLocaleTimeString() : '--:--:--'}
        </span>
      </div>
      
      {/* Metrics */}
      <div className="space-y-3">
        {type === 'CFD' && <CFDMetrics metrics={metrics} theme={theme} />}
        {type === 'STRESS' && <StressMetrics metrics={metrics} theme={theme} />}
      </div>
      
      {/* Legend */}
      <div 
        className="mt-4 pt-4"
        style={{ borderTop: `1px solid ${theme.colors.border.secondary}` }}
      >
        <div 
          className="text-xs mb-2"
          style={{ color: theme.colors.text.muted }}
        >
          Heatmap Legend
        </div>
        <HeatmapLegend type={type} theme={theme} />
      </div>
    </div>
  );
}

/**
 * CFD Metrics Display
 */
function CFDMetrics({ metrics, theme }) {
  return (
    <>
      <MetricRow
        label="Max Velocity"
        value={metrics?.maxVelocity}
        unit="m/s"
        color={theme.colors.status.info}
        theme={theme}
      />
      <MetricRow
        label="Pressure Drop"
        value={metrics?.pressureDrop}
        unit="Pa"
        color={theme.colors.status.success}
        theme={theme}
      />
      <MetricRow
        label="Drag Force"
        value={metrics?.dragForce}
        unit="N"
        color={theme.colors.accent.primary}
        theme={theme}
      />
      <MetricRow
        label="Lift Force"
        value={metrics?.liftForce}
        unit="N"
        color={theme.colors.status.success}
        theme={theme}
      />
      <MetricRow
        label="Flow Separation"
        value={metrics?.flowSeparation}
        unit=""
        color={metrics?.flowSeparation ? theme.colors.status.error : theme.colors.status.success}
        format={v => v ? 'Detected' : 'None'}
        theme={theme}
      />
    </>
  );
}

/**
 * Stress Metrics Display
 */
function StressMetrics({ metrics, theme }) {
  const safetyStatus = metrics?.safetyFactor > 2 ? 'safe' : 
                       metrics?.safetyFactor > 1.5 ? 'caution' : 'danger';
  
  const getStatusColors = () => {
    switch (safetyStatus) {
      case 'safe': return {
        bg: theme.colors.status.success + '20',
        text: theme.colors.status.success
      };
      case 'caution': return {
        bg: theme.colors.status.warning + '20',
        text: theme.colors.status.warning
      };
      default: return {
        bg: theme.colors.status.error + '20',
        text: theme.colors.status.error
      };
    }
  };
  
  const statusColors = getStatusColors();
  
  return (
    <>
      <MetricRow
        label="Max Stress"
        value={metrics?.maxStress}
        unit="MPa"
        color={theme.colors.status.error}
        format={v => v ? (v / 1e6).toFixed(2) : '-'}
        theme={theme}
      />
      <MetricRow
        label="Yield Stress"
        value={metrics?.yieldStress}
        unit="MPa"
        color={theme.colors.status.warning}
        format={v => v ? (v / 1e6).toFixed(2) : '-'}
        theme={theme}
      />
      <MetricRow
        label="Safety Factor"
        value={metrics?.safetyFactor}
        unit=""
        color={statusColors.text}
        format={v => v?.toFixed(2)}
        theme={theme}
      />
      <MetricRow
        label="Max Displacement"
        value={metrics?.maxDisplacement}
        unit="mm"
        color={theme.colors.status.info}
        format={v => v ? (v * 1000).toFixed(3) : '-'}
        theme={theme}
      />
      
      {/* Safety Status */}
      <div 
        className="mt-3 p-2 rounded-lg text-center font-medium"
        style={{
          backgroundColor: statusColors.bg,
          color: statusColors.text
        }}
      >
        {safetyStatus === 'safe' && '✓ Design Safe'}
        {safetyStatus === 'caution' && '⚠ Review Recommended'}
        {safetyStatus === 'danger' && '✗ Design Fails'}
      </div>
    </>
  );
}

/**
 * Metric Row Component
 */
function MetricRow({ label, value, unit, color, format, theme }) {
  const formattedValue = format ? format(value) : 
                         value !== undefined ? (value.toFixed ? value.toFixed(2) : value) : '-';
  
  return (
    <div className="flex justify-between items-center">
      <span className="text-sm" style={{ color: theme.colors.text.muted }}>{label}</span>
      <span 
        className="text-sm font-mono font-medium"
        style={{ color: color }}
      >
        {formattedValue} {unit}
      </span>
    </div>
  );
}

/**
 * Heatmap Legend
 */
function HeatmapLegend({ type, theme }) {
  if (type === 'CFD') {
    return (
      <div className="flex items-center gap-2 relative">
        <div 
          className="flex-1 h-2 rounded-full"
          style={{
            background: `linear-gradient(to right, ${theme.colors.status.info}, ${theme.colors.status.success}, ${theme.colors.status.warning})`
          }}
        />
        <div 
          className="flex justify-between text-xs w-full absolute mt-3"
          style={{ color: theme.colors.text.muted }}
        >
          <span>Low P</span>
          <span>High P</span>
        </div>
      </div>
    );
  }
  
  return (
    <div className="flex items-center gap-2 relative">
      <div 
        className="flex-1 h-2 rounded-full"
        style={{
          background: `linear-gradient(to right, ${theme.colors.status.info}, ${theme.colors.status.success}, ${theme.colors.status.warning}, ${theme.colors.status.error})`
        }}
      />
      <div 
        className="flex justify-between text-xs w-full absolute mt-3"
        style={{ color: theme.colors.text.muted }}
      >
        <span>Low σ</span>
        <span>High σ</span>
      </div>
    </div>
  );
}
