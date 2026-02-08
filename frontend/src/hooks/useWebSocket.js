/**
 * useWebSocket Hook
 * 
 * React hook for WebSocket communication with the BRICK OS backend.
 * Provides real-time updates for orchestrator progress, agent thoughts,
 * state changes, and performance metrics.
 * 
 * @example
 * const { 
 *   isConnected, 
 *   messages, 
 *   thoughts,
 *   progress,
 *   error 
 * } = useWebSocket({ projectId: 'proj_123' });
 */
import { useState, useEffect, useRef, useCallback } from 'react';

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

const useWebSocket = (options = {}) => {
  const {
    projectId,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    autoReconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
  } = options;

  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState(null);
  
  // Message state
  const [messages, setMessages] = useState([]);
  const [thoughts, setThoughts] = useState([]);
  const [progress, setProgress] = useState({});
  const [lastMessage, setLastMessage] = useState(null);
  
  // Refs for WebSocket and reconnection
  const wsRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef(null);
  const messageQueueRef = useRef([]);

  // Get WebSocket URL
  const getWsUrl = useCallback(() => {
    if (!projectId) return null;
    return `${WS_BASE_URL}/ws/orchestrator/${projectId}`;
  }, [projectId]);

  // Send message helper
  const sendMessage = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      // Queue message for when connection is established
      messageQueueRef.current.push(data);
    }
  }, []);

  // Send ping to keep connection alive
  const sendPing = useCallback(() => {
    sendMessage({ type: 'ping' });
  }, [sendMessage]);

  // Request metrics
  const requestMetrics = useCallback(() => {
    sendMessage({ type: 'get_metrics' });
  }, [sendMessage]);

  // Subscribe to specific channels
  const subscribe = useCallback((channels) => {
    sendMessage({ 
      type: 'subscribe', 
      channels: Array.isArray(channels) ? channels : [channels] 
    });
  }, [sendMessage]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    const url = getWsUrl();
    if (!url) {
      setError(new Error('No projectId provided'));
      return;
    }

    // Don't connect if already connecting or connected
    if (isConnecting || isConnected) return;

    setIsConnecting(true);
    setError(null);

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`[WebSocket] Connected to project ${projectId}`);
        setIsConnected(true);
        setIsConnecting(false);
        reconnectAttemptsRef.current = 0;

        // Send any queued messages
        while (messageQueueRef.current.length > 0) {
          const msg = messageQueueRef.current.shift();
          ws.send(JSON.stringify(msg));
        }

        onConnect?.();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          
          // Handle different message types
          switch (data.type) {
            case 'agent_progress':
              setProgress(prev => ({
                ...prev,
                [data.agent]: data.progress
              }));
              break;
              
            case 'thought':
              setThoughts(prev => [...prev, {
                agent: data.agent,
                thought: data.thought,
                timestamp: data.timestamp || new Date().toISOString()
              }]);
              break;
              
            case 'state_update':
              // State updates are handled via lastMessage
              break;
              
            case 'completed':
              console.log('[WebSocket] Pipeline completed', data.result);
              break;
              
            case 'error':
              console.error('[WebSocket] Server error:', data.error);
              setError(new Error(data.error));
              break;
              
            case 'pong':
              // Ping-pong keepalive
              break;
              
            case 'metrics':
              // Performance metrics received
              break;
              
            default:
              // Unknown message type
              break;
          }

          // Add to messages list (limit to last 100)
          setMessages(prev => {
            const newMessages = [...prev, data];
            return newMessages.slice(-100);
          });

          // Call external handler
          onMessage?.(data);
        } catch (err) {
          console.error('[WebSocket] Failed to parse message:', err);
        }
      };

      ws.onclose = (event) => {
        console.log(`[WebSocket] Disconnected from project ${projectId}`, event.code);
        setIsConnected(false);
        setIsConnecting(false);
        wsRef.current = null;

        onDisconnect?.(event);

        // Attempt reconnection if enabled
        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          console.log(`[WebSocket] Reconnecting... (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setError(new Error('Max reconnection attempts reached'));
        }
      };

      ws.onerror = (err) => {
        console.error('[WebSocket] Error:', err);
        setError(new Error('WebSocket error'));
        setIsConnecting(false);
        onError?.(err);
      };

    } catch (err) {
      console.error('[WebSocket] Failed to connect:', err);
      setError(err);
      setIsConnecting(false);
    }
  }, [
    getWsUrl, 
    projectId, 
    isConnecting, 
    isConnected, 
    autoReconnect, 
    reconnectInterval, 
    maxReconnectAttempts,
    onConnect,
    onDisconnect,
    onError,
    onMessage
  ]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      // Prevent reconnection on intentional close
      reconnectAttemptsRef.current = maxReconnectAttempts;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [maxReconnectAttempts]);

  // Clear thoughts
  const clearThoughts = useCallback(() => {
    setThoughts([]);
  }, []);

  // Clear messages
  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  // Reset progress
  const clearProgress = useCallback(() => {
    setProgress({});
  }, []);

  // Connect on mount / projectId change
  useEffect(() => {
    if (projectId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [projectId, connect, disconnect]);

  // Keepalive ping every 30 seconds
  useEffect(() => {
    if (!isConnected) return;

    const pingInterval = setInterval(() => {
      sendPing();
    }, 30000);

    return () => clearInterval(pingInterval);
  }, [isConnected, sendPing]);

  return {
    // Connection state
    isConnected,
    isConnecting,
    error,
    
    // Data
    messages,
    thoughts,
    progress,
    lastMessage,
    
    // Actions
    connect,
    disconnect,
    sendMessage,
    sendPing,
    requestMetrics,
    subscribe,
    clearThoughts,
    clearMessages,
    clearProgress,
  };
};

export default useWebSocket;
