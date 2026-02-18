/**
 * Agent Presence Overlay - AI Activity Visualization
 * 
 * Shows real-time activity of AI agents working in the background.
 * Displays agent thoughts, progress, and status indicators.
 */

import React from 'react';

/**
 * Agent Presence Overlay Component
 */
export default function AgentPresenceOverlay({ thoughts, activeAgents, theme }) {
  const hasContent = (thoughts?.length > 0) || (activeAgents?.length > 0);
  
  if (!hasContent) return null;
  
  return (
    <div 
      className="absolute bottom-32 left-4 w-80 max-h-96 overflow-y-auto z-40 rounded-xl"
      style={{
        backgroundColor: theme.colors.bg.secondary + '80',
        backdropFilter: 'blur(12px)',
        border: `1px solid ${theme.colors.border.primary}`
      }}
    >
      {/* Active Agents */}
      {activeAgents.length > 0 && (
        <div className="p-3 space-y-2">
          {activeAgents.map((agent, i) => (
            <AgentCard key={i} agent={agent} theme={theme} />
          ))}
        </div>
      )}
      
      {/* Recent Thoughts */}
      {thoughts.length > 0 && (
        <div className="p-3 pt-0 space-y-2">
          {thoughts.slice(-3).map((thought, i) => (
            <ThoughtBubble key={i} thought={thought} theme={theme} />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Agent Card Component
 */
function AgentCard({ agent, theme }) {
  const getStatusColor = () => {
    switch (agent.status) {
      case 'idle': return theme.colors.text.muted;
      case 'thinking': return theme.colors.status.warning;
      case 'working': return theme.colors.status.info;
      case 'complete': return theme.colors.status.success;
      case 'error': return theme.colors.status.error;
      default: return theme.colors.text.muted;
    }
  };
  
  return (
    <div 
      className="rounded-lg p-3"
      style={{
        backgroundColor: theme.colors.bg.tertiary + '80',
        border: `1px solid ${theme.colors.border.secondary}`
      }}
    >
      <div className="flex items-center gap-3">
        <div 
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: getStatusColor() }}
        />
        <div className="flex-1">
          <div className="flex justify-between items-center">
            <span 
              className="text-sm font-medium"
              style={{ color: theme.colors.text.primary }}
            >
              {agent.name || agent}
            </span>
            <span 
              className="text-xs capitalize"
              style={{ color: theme.colors.text.muted }}
            >
              {agent.status || 'active'}
            </span>
          </div>
          {agent.progress !== undefined && (
            <div className="mt-2">
              <div 
                className="h-1.5 rounded-full overflow-hidden"
                style={{ backgroundColor: theme.colors.bg.elevated }}
              >
                <div 
                  className="h-full transition-all duration-500"
                  style={{ 
                    width: `${agent.progress}%`,
                    backgroundColor: theme.colors.accent.primary 
                  }}
                />
              </div>
              <div 
                className="text-right text-xs mt-1"
                style={{ color: theme.colors.text.muted }}
              >
                {agent.progress.toFixed(0)}%
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Thought Bubble Component
 */
function ThoughtBubble({ thought, theme }) {
  return (
    <div 
      className="rounded-lg p-3 border-l-2"
      style={{
        backgroundColor: theme.colors.bg.tertiary + '60',
        borderLeftColor: theme.colors.accent.primary
      }}
    >
      <div className="flex items-start gap-2">
        <span className="text-lg">💭</span>
        <div>
          <div 
            className="text-xs font-medium mb-1"
            style={{ color: theme.colors.accent.primary }}
          >
            {thought.agent || 'Agent'}
          </div>
          <div 
            className="text-sm"
            style={{ color: theme.colors.text.secondary }}
          >
            {thought.message || thought.text || thought}
          </div>
        </div>
      </div>
    </div>
  );
}
