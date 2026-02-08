/**
 * usePerformance Hook
 * 
 * React hook for fetching and managing performance monitoring data
 * from the BRICK OS backend.
 * 
 * @example
 * const { 
 *   overview, 
 *   activePipelines, 
 *   agentStats,
 *   refresh 
 * } = usePerformance({ autoRefresh: true, refreshInterval: 5000 });
 */
import { useState, useEffect, useCallback, useRef } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const usePerformance = (options = {}) => {
  const {
    autoRefresh = false,
    refreshInterval = 5000,
    projectId = null,
  } = options;

  // Data state
  const [overview, setOverview] = useState(null);
  const [activePipelines, setActivePipelines] = useState([]);
  const [recentPipelines, setRecentPipelines] = useState([]);
  const [agentStats, setAgentStats] = useState({});
  const [slowestAgents, setSlowestAgents] = useState([]);
  const [websocketStatus, setWebsocketStatus] = useState(null);
  const [pipelineDetails, setPipelineDetails] = useState(null);
  
  // Loading and error state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const refreshTimeoutRef = useRef(null);

  // Fetch helper
  const fetchData = useCallback(async (endpoint) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }, []);

  // Refresh overview
  const refreshOverview = useCallback(async () => {
    try {
      const data = await fetchData('/api/performance/overview');
      setOverview(data);
      return data;
    } catch (err) {
      console.error('[usePerformance] Failed to fetch overview:', err);
      throw err;
    }
  }, [fetchData]);

  // Refresh active pipelines
  const refreshActivePipelines = useCallback(async () => {
    try {
      const data = await fetchData('/api/performance/pipelines/active');
      setActivePipelines(data.pipelines || []);
      return data;
    } catch (err) {
      console.error('[usePerformance] Failed to fetch active pipelines:', err);
      throw err;
    }
  }, [fetchData]);

  // Refresh recent pipelines
  const refreshRecentPipelines = useCallback(async (limit = 10) => {
    try {
      const data = await fetchData(`/api/performance/pipelines/recent?limit=${limit}`);
      setRecentPipelines(data.pipelines || []);
      return data;
    } catch (err) {
      console.error('[usePerformance] Failed to fetch recent pipelines:', err);
      throw err;
    }
  }, [fetchData]);

  // Refresh agent stats
  const refreshAgentStats = useCallback(async (agentName = null) => {
    try {
      const endpoint = agentName 
        ? `/api/performance/agents?agent_name=${agentName}`
        : '/api/performance/agents';
      const data = await fetchData(endpoint);
      setAgentStats(data.agents || {});
      return data;
    } catch (err) {
      console.error('[usePerformance] Failed to fetch agent stats:', err);
      throw err;
    }
  }, [fetchData]);

  // Refresh slowest agents
  const refreshSlowestAgents = useCallback(async (limit = 5) => {
    try {
      const data = await fetchData(`/api/performance/agents/slowest?limit=${limit}`);
      setSlowestAgents(data.slowest_agents || []);
      return data;
    } catch (err) {
      console.error('[usePerformance] Failed to fetch slowest agents:', err);
      throw err;
    }
  }, [fetchData]);

  // Refresh WebSocket status
  const refreshWebsocketStatus = useCallback(async () => {
    try {
      const data = await fetchData('/api/performance/websocket/status');
      setWebsocketStatus(data);
      return data;
    } catch (err) {
      console.error('[usePerformance] Failed to fetch WebSocket status:', err);
      throw err;
    }
  }, [fetchData]);

  // Refresh specific pipeline details
  const refreshPipelineDetails = useCallback(async (pid) => {
    if (!pid) return;
    try {
      const data = await fetchData(`/api/performance/pipeline/${pid}`);
      setPipelineDetails(data);
      return data;
    } catch (err) {
      console.error('[usePerformance] Failed to fetch pipeline details:', err);
      throw err;
    }
  }, [fetchData]);

  // Refresh all data
  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await Promise.all([
        refreshOverview(),
        refreshActivePipelines(),
        refreshRecentPipelines(),
        refreshAgentStats(),
        refreshSlowestAgents(),
        refreshWebsocketStatus(),
      ]);

      if (projectId) {
        await refreshPipelineDetails(projectId);
      }
    } catch (err) {
      setError(err.message || 'Failed to refresh performance data');
    } finally {
      setIsLoading(false);
    }
  }, [
    refreshOverview,
    refreshActivePipelines,
    refreshRecentPipelines,
    refreshAgentStats,
    refreshSlowestAgents,
    refreshWebsocketStatus,
    refreshPipelineDetails,
    projectId
  ]);

  // Auto-refresh
  useEffect(() => {
    // Initial load
    refresh();

    if (autoRefresh) {
      const intervalId = setInterval(() => {
        refresh();
      }, refreshInterval);

      return () => {
        clearInterval(intervalId);
        if (refreshTimeoutRef.current) {
          clearTimeout(refreshTimeoutRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, refresh]);

  // Format helpers
  const formatDuration = useCallback((ms) => {
    if (!ms && ms !== 0) return 'N/A';
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }, []);

  const formatBytes = useCallback((bytes) => {
    if (!bytes && bytes !== 0) return 'N/A';
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
  }, []);

  const formatPercentage = useCallback((value) => {
    if (!value && value !== 0) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  }, []);

  return {
    // Data
    overview,
    activePipelines,
    recentPipelines,
    agentStats,
    slowestAgents,
    websocketStatus,
    pipelineDetails,
    
    // State
    isLoading,
    error,
    
    // Actions
    refresh,
    refreshOverview,
    refreshActivePipelines,
    refreshRecentPipelines,
    refreshAgentStats,
    refreshSlowestAgents,
    refreshWebsocketStatus,
    refreshPipelineDetails,
    
    // Helpers
    formatDuration,
    formatBytes,
    formatPercentage,
  };
};

export default usePerformance;
