/**
 * useWebSocketStream - Real-time communication with backend orchestrator
 * 
 * Manages WebSocket connection for:
 * - Streaming geometry updates
 * - Receiving AI suggestions (Ghost Mode)
 * - Simulation results
 * - Agent thoughts/progress
 * - User command responses
 */

import { useState, useEffect, useRef, useCallback } from 'react';

// WebSocket message types
const WS_MESSAGE_TYPES = {
  // Incoming from backend
  GEOMETRY_DELTA: 'GEOMETRY_DELTA',
  GEOMETRY_FULL: 'GEOMETRY_FULL',
  GHOST_GEOMETRY: 'GHOST_GEOMETRY',
  SIM_FRAME: 'SIM_FRAME',
  AGENT_THOUGHT: 'AGENT_THOUGHT',
  AGENT_PROGRESS: 'AGENT_PROGRESS',
  AGENT_COMPLETE: 'AGENT_COMPLETE',
  AGENT_ERROR: 'AGENT_ERROR',
  SUGGESTION_ACCEPTED: 'SUGGESTION_ACCEPTED',
  SUGGESTION_REJECTED: 'SUGGESTION_REJECTED',
  
  // Outgoing to backend
  INTENT: 'INTENT',
  ACCEPT_SUGGESTION: 'ACCEPT_SUGGESTION',
  REJECT_SUGGESTION: 'REJECT_SUGGESTION',
  ANNOTATION_CREATE: 'ANNOTATION_CREATE',
  ANNOTATION_UPDATE: 'ANNOTATION_UPDATE',
  MODE_CHANGE: 'MODE_CHANGE',
  CAMERA_POSITION: 'CAMERA_POSITION'
};

/**
 * useWebSocketStream Hook
 */
export function useWebSocketStream(projectId) {
  const wsRef = useRef(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [geometryStream, setGeometryStream] = useState(null);
  const [simulationResults, setSimulationResults] = useState(null);
  const [agentThoughts, setAgentThoughts] = useState([]);
  const [ghostSuggestions, setGhostSuggestions] = useState([]);
  const [activeAgents, setActiveAgents] = useState([]);
  const reconnectTimeoutRef = useRef(null);
  
  // Connect to WebSocket
  useEffect(() => {
    if (!projectId) return;
    
    const connect = () => {
      setConnectionStatus('connecting');
      
      // Determine WebSocket URL
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/api/ws/${projectId}`;
      
      console.log('[WebSocket] Connecting to:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log('[WebSocket] Connected');
        setConnectionStatus('connected');
        
        // Subscribe to project updates
        sendMessage({
          type: 'SUBSCRIBE',
          payload: { project_id: projectId }
        });
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleMessage(message);
        } catch (err) {
          console.error('[WebSocket] Failed to parse message:', err);
        }
      };
      
      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        setConnectionStatus('error');
      };
      
      ws.onclose = () => {
        console.log('[WebSocket] Disconnected');
        setConnectionStatus('disconnected');
        
        // Attempt to reconnect
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('[WebSocket] Attempting to reconnect...');
          connect();
        }, 3000);
      };
    };
    
    connect();
    
    // Cleanup
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [projectId]);
  
  // Handle incoming messages
  const handleMessage = useCallback((message) => {
    const { type, payload } = message;
    
    switch (type) {
      case WS_MESSAGE_TYPES.GEOMETRY_DELTA:
        // Incremental geometry update
        setGeometryStream(prev => ({
          ...prev,
          ...payload,
          lastUpdate: Date.now()
        }));
        break;
        
      case WS_MESSAGE_TYPES.GEOMETRY_FULL:
        // Full geometry replacement
        setGeometryStream({
          ...payload,
          lastUpdate: Date.now()
        });
        break;
        
      case WS_MESSAGE_TYPES.GHOST_GEOMETRY:
        // AI suggestion (Ghost Mode)
        setGhostSuggestions(prev => {
          // Remove existing suggestion with same ID
          const filtered = prev.filter(g => g.id !== payload.id);
          return [...filtered, payload];
        });
        break;
        
      case WS_MESSAGE_TYPES.SIM_FRAME:
        // Simulation results (CFD/Stress)
        setSimulationResults({
          ...payload,
          timestamp: Date.now()
        });
        break;
        
      case WS_MESSAGE_TYPES.AGENT_THOUGHT:
        // Agent reasoning/thought process
        setAgentThoughts(prev => [...prev.slice(-9), payload]);
        break;
        
      case WS_MESSAGE_TYPES.AGENT_PROGRESS:
        // Agent progress update
        setActiveAgents(prev => {
          const existing = prev.findIndex(a => a.id === payload.agent_id);
          if (existing >= 0) {
            const updated = [...prev];
            updated[existing] = {
              ...updated[existing],
              ...payload,
              lastUpdate: Date.now()
            };
            return updated;
          }
          return [...prev, { ...payload, lastUpdate: Date.now() }];
        });
        break;
        
      case WS_MESSAGE_TYPES.AGENT_COMPLETE:
        // Agent finished work
        setActiveAgents(prev => 
          prev.filter(a => a.id !== payload.agent_id)
        );
        break;
        
      case WS_MESSAGE_TYPES.AGENT_ERROR:
        // Agent encountered error
        console.error('[Agent Error]', payload);
        setActiveAgents(prev => 
          prev.map(a => a.id === payload.agent_id 
            ? { ...a, status: 'error', error: payload.error }
            : a
          )
        );
        break;
        
      case WS_MESSAGE_TYPES.SUGGESTION_ACCEPTED:
        // Suggestion was accepted, remove from ghosts
        setGhostSuggestions(prev => 
          prev.filter(g => g.id !== payload.suggestion_id)
        );
        break;
        
      case WS_MESSAGE_TYPES.SUGGESTION_REJECTED:
        // Suggestion was rejected, remove from ghosts
        setGhostSuggestions(prev => 
          prev.filter(g => g.id !== payload.suggestion_id)
        );
        break;
        
      default:
        console.log('[WebSocket] Unknown message type:', type);
    }
  }, []);
  
  // Send message to backend
  const sendMessage = useCallback((message) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('[WebSocket] Not connected, message not sent:', message);
    }
  }, []);
  
  // Send user intent/command
  const sendCommand = useCallback((command) => {
    sendMessage({
      type: WS_MESSAGE_TYPES.INTENT,
      payload: command
    });
  }, [sendMessage]);
  
  // Accept ghost suggestion
  const acceptSuggestion = useCallback((suggestionId) => {
    sendMessage({
      type: WS_MESSAGE_TYPES.ACCEPT_SUGGESTION,
      payload: { suggestion_id: suggestionId }
    });
    
    // Optimistically remove from UI
    setGhostSuggestions(prev => 
      prev.filter(g => g.id !== suggestionId)
    );
  }, [sendMessage]);
  
  // Reject ghost suggestion
  const rejectSuggestion = useCallback((suggestionId) => {
    sendMessage({
      type: WS_MESSAGE_TYPES.REJECT_SUGGESTION,
      payload: { suggestion_id: suggestionId }
    });
    
    // Optimistically remove from UI
    setGhostSuggestions(prev => 
      prev.filter(g => g.id !== suggestionId)
    );
  }, [sendMessage]);
  
  // Send annotation
  const sendAnnotation = useCallback((annotation) => {
    sendMessage({
      type: WS_MESSAGE_TYPES.ANNOTATION_CREATE,
      payload: annotation
    });
  }, [sendMessage]);
  
  // Send camera position (for collaborative features)
  const sendCameraPosition = useCallback((position, target) => {
    sendMessage({
      type: WS_MESSAGE_TYPES.CAMERA_POSITION,
      payload: { position, target }
    });
  }, [sendMessage]);
  
  return {
    // State
    connectionStatus,
    geometryStream,
    simulationResults,
    agentThoughts,
    ghostSuggestions,
    activeAgents,
    
    // Actions
    sendCommand,
    acceptSuggestion,
    rejectSuggestion,
    sendAnnotation,
    sendCameraPosition,
    
    // Raw message send (for advanced use)
    sendMessage
  };
}

export default useWebSocketStream;
