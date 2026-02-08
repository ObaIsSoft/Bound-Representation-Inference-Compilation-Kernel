/**
 * PerformanceDashboard Component
 * 
 * Real-time performance monitoring dashboard for BRICK OS.
 * Displays agent execution timing, pipeline progress, bottlenecks,
 * and system-wide performance metrics.
 */
import React, { useState } from 'react';
import usePerformance from '../../hooks/usePerformance';
import './PerformanceDashboard.css';

const PerformanceDashboard = ({ projectId = null, autoRefresh = true }) => {
  const {
    overview,
    activePipelines,
    recentPipelines,
    slowestAgents,
    websocketStatus,
    pipelineDetails,
    isLoading,
    error,
    refresh,
    formatDuration,
  } = usePerformance({ projectId, autoRefresh, refreshInterval: 5000 });

  const [selectedPipeline, setSelectedPipeline] = useState(null);
  const [showDetails, setShowDetails] = useState(false);

  // Status indicator component
  const StatusIndicator = ({ status }) => {
    const statusClass = status?.toLowerCase() || 'unknown';
    return <span className={`status-indicator status-${statusClass}`}>{status}</span>;
  };

  // Overview cards
  const OverviewCards = () => {
    if (!overview) return null;

    return (
      <div className="overview-cards">
        <div className="card">
          <h3>Active Pipelines</h3>
          <div className="metric">{overview.active_pipelines}</div>
        </div>
        <div className="card">
          <h3>Completed</h3>
          <div className="metric">{overview.completed_pipelines}</div>
        </div>
        <div className="card">
          <h3>Total Executions</h3>
          <div className="metric">{overview.total_agent_executions.toLocaleString()}</div>
        </div>
        <div className="card">
          <h3>Failure Rate</h3>
          <div className={`metric ${overview.failure_rate > 5 ? 'warning' : ''}`}>
            {overview.failure_rate.toFixed(2)}%
          </div>
        </div>
        <div className="card">
          <h3>WebSocket Connections</h3>
          <div className="metric">{websocketStatus?.total_connections || 0}</div>
        </div>
        <div className="card">
          <h3>Active Projects</h3>
          <div className="metric">{websocketStatus?.active_projects?.length || 0}</div>
        </div>
      </div>
    );
  };

  // Active pipelines table
  const ActivePipelinesTable = () => {
    if (!activePipelines || activePipelines.length === 0) {
      return (
        <div className="section">
          <h2>Active Pipelines</h2>
          <p className="empty-state">No active pipelines</p>
        </div>
      );
    }

    return (
      <div className="section">
        <h2>Active Pipelines ({activePipelines.length})</h2>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Project ID</th>
                <th>Duration</th>
                <th>Progress</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {activePipelines.map((pipeline) => (
                <tr 
                  key={pipeline.project_id}
                  className={selectedPipeline === pipeline.project_id ? 'selected' : ''}
                  onClick={() => {
                    setSelectedPipeline(pipeline.project_id);
                    setShowDetails(true);
                  }}
                >
                  <td className="project-id">{pipeline.project_id}</td>
                  <td>{formatDuration(pipeline.duration_ms)}</td>
                  <td>
                    <div className="progress-bar">
                      <div 
                        className="progress-fill"
                        style={{ 
                          width: `${(pipeline.completed_agents / Math.max(pipeline.total_agents, 1)) * 100}%` 
                        }}
                      />
                      <span className="progress-text">
                        {pipeline.completed_agents}/{pipeline.total_agents}
                      </span>
                    </div>
                  </td>
                  <td><StatusIndicator status={pipeline.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Slowest agents table
  const SlowestAgentsTable = () => {
    if (!slowestAgents || slowestAgents.length === 0) {
      return (
        <div className="section">
          <h2>Slowest Agents</h2>
          <p className="empty-state">No agent data available</p>
        </div>
      );
    }

    return (
      <div className="section">
        <h2>Slowest Agents</h2>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Agent</th>
                <th>Avg Duration</th>
                <th>Max Duration</th>
                <th>Executions</th>
              </tr>
            </thead>
            <tbody>
              {slowestAgents.map((agent) => (
                <tr key={agent.agent_name}>
                  <td>{agent.agent_name}</td>
                  <td>{formatDuration(agent.avg_duration_ms)}</td>
                  <td>{formatDuration(agent.max_duration_ms)}</td>
                  <td>{agent.total_executions.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Recent pipelines table
  const RecentPipelinesTable = () => {
    if (!recentPipelines || recentPipelines.length === 0) {
      return (
        <div className="section">
          <h2>Recent Pipelines</h2>
          <p className="empty-state">No completed pipelines</p>
        </div>
      );
    }

    return (
      <div className="section">
        <h2>Recent Pipelines</h2>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Project ID</th>
                <th>Duration</th>
                <th>Agents</th>
                <th>Status</th>
                <th>Bottlenecks</th>
              </tr>
            </thead>
            <tbody>
              {recentPipelines.map((pipeline) => (
                <tr key={pipeline.project_id}>
                  <td className="project-id">{pipeline.project_id}</td>
                  <td>{formatDuration(pipeline.duration_ms)}</td>
                  <td>{pipeline.total_agents}</td>
                  <td><StatusIndicator status={pipeline.status} /></td>
                  <td>
                    {pipeline.bottlenecks?.length > 0 ? (
                      <span className="bottleneck-badge">
                        {pipeline.bottlenecks.length} found
                      </span>
                    ) : (
                      <span className="no-bottleneck">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Pipeline details panel
  const PipelineDetailsPanel = () => {
    if (!showDetails || !pipelineDetails) return null;

    return (
      <div className="details-panel">
        <div className="details-header">
          <h2>Pipeline Details: {pipelineDetails.project_id}</h2>
          <button 
            className="close-button"
            onClick={() => setShowDetails(false)}
          >
            ×
          </button>
        </div>
        
        <div className="details-content">
          <div className="detail-row">
            <span className="label">Status:</span>
            <StatusIndicator status={pipelineDetails.status} />
          </div>
          <div className="detail-row">
            <span className="label">Duration:</span>
            <span>{formatDuration(pipelineDetails.duration_ms)}</span>
          </div>
          <div className="detail-row">
            <span className="label">Agents:</span>
            <span>{pipelineDetails.completed_agents} / {pipelineDetails.total_agents} completed</span>
          </div>
          
          {pipelineDetails.bottlenecks?.length > 0 && (
            <div className="bottlenecks-section">
              <h3>Bottlenecks</h3>
              <ul>
                {pipelineDetails.bottlenecks.map((b, i) => (
                  <li key={i} className={`bottleneck-${b.severity}`}>
                    <strong>{b.agent}</strong>: {formatDuration(b.avg_duration_ms)}
                    <span className="percentage">({b.percentage_of_total.toFixed(1)}%)</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {pipelineDetails.agent_timings && Object.keys(pipelineDetails.agent_timings).length > 0 && (
            <div className="agent-timings-section">
              <h3>Agent Timings</h3>
              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Agent</th>
                      <th>Executions</th>
                      <th>Timings</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(pipelineDetails.agent_timings).map(([agent, timings]) => (
                      <tr key={agent}>
                        <td>{agent}</td>
                        <td>{timings.length}</td>
                        <td>
                          {timings.slice(-3).map((t, i) => (
                            <span key={i} className={`timing-tag status-${t.status}`}>
                              {formatDuration(t.duration_ms)}
                            </span>
                          ))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="performance-dashboard">
      <header className="dashboard-header">
        <h1>Performance Monitor</h1>
        <div className="header-actions">
          <button 
            className="refresh-button"
            onClick={refresh}
            disabled={isLoading}
          >
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          Error: {error}
          <button onClick={refresh}>Retry</button>
        </div>
      )}

      <div className="dashboard-content">
        <OverviewCards />
        
        <div className="dashboard-grid">
          <div className="grid-left">
            <ActivePipelinesTable />
            <RecentPipelinesTable />
          </div>
          <div className="grid-right">
            <SlowestAgentsTable />
          </div>
        </div>
      </div>

      <PipelineDetailsPanel />
      
      {isLoading && <div className="loading-overlay">Loading...</div>}
    </div>
  );
};

export default PerformanceDashboard;
