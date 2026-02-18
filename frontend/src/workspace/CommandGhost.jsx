/**
 * Command Ghost - Spatial Command Input
 * 
 * A text input interface that appears at the bottom of the 3D viewport,
 * allowing users to type natural language commands that are interpreted
 * by the AI orchestrator.
 * 
 * Features:
 * - Natural language command parsing
 * - Context-aware suggestions
 * - Active agent visualization
 * - Voice input support (optional)
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';

/**
 * Command Ghost Component
 */
export function CommandGhost({ onSubmit, activeAgents, suggestions = [] }) {
  const [input, setInput] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [commandHistory, setCommandHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const inputRef = useRef(null);
  
  // Focus input on mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);
  
  // Handle form submission
  const handleSubmit = useCallback((e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    // Add to history
    setCommandHistory(prev => [...prev, input]);
    setHistoryIndex(-1);
    
    // Send command
    onSubmit({
      text: input,
      timestamp: new Date().toISOString(),
      context: {
        activeAgents: activeAgents.map(a => a.name)
      }
    });
    
    // Clear input
    setInput('');
    setShowSuggestions(false);
  }, [input, onSubmit, activeAgents]);
  
  // Handle keyboard navigation
  const handleKeyDown = useCallback((e) => {
    // Up arrow - previous history
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[commandHistory.length - 1 - newIndex]);
      }
    }
    
    // Down arrow - next history
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[commandHistory.length - 1 - newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setInput('');
      }
    }
    
    // Escape - close suggestions
    if (e.key === 'Escape') {
      setShowSuggestions(false);
      inputRef.current?.blur();
    }
    
    // Tab - accept first suggestion
    if (e.key === 'Tab' && suggestions.length > 0) {
      e.preventDefault();
      setInput(suggestions[0]);
      setShowSuggestions(false);
    }
  }, [historyIndex, commandHistory, suggestions]);
  
  // Filter suggestions based on input
  const filteredSuggestions = suggestions.filter(s => 
    s.toLowerCase().includes(input.toLowerCase()) && s !== input
  );
  
  // Quick action buttons
  const quickActions = [
    { label: 'Make stronger', icon: '💪' },
    { label: 'Make lighter', icon: '⚡' },
    { label: 'Check safety', icon: '🛡️' },
    { label: 'Run CFD', icon: '💨' }
  ];
  
  return (
    <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 w-full max-w-2xl px-4 z-50">
      {/* Active Agent Indicators */}
      <div className="flex gap-2 mb-3 justify-center flex-wrap">
        {activeAgents.map(agent => (
          <AgentPulse key={agent.id} agent={agent} />
        ))}
      </div>
      
      {/* Quick Actions (shown when input empty) */}
      {!input && !isExpanded && (
        <div className="flex gap-2 mb-3 justify-center">
          {quickActions.map((action, i) => (
            <button
              key={i}
              onClick={() => setInput(action.label)}
              className="bg-black/60 hover:bg-black/80 backdrop-blur text-white/80 text-xs px-3 py-1.5 rounded-full transition-all flex items-center gap-1 border border-white/10"
            >
              <span>{action.icon}</span>
              <span>{action.label}</span>
            </button>
          ))}
        </div>
      )}
      
      {/* Main Input Form */}
      <form onSubmit={handleSubmit} className="relative">
        <div 
          className={`
            relative bg-black/70 backdrop-blur-xl rounded-2xl border border-white/20
            transition-all duration-300 ease-out
            ${isExpanded ? 'shadow-2xl shadow-blue-500/20' : 'shadow-lg'}
            ${showSuggestions && filteredSuggestions.length > 0 ? 'rounded-b-lg' : ''}
          `}
        >
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              setShowSuggestions(e.target.value.length > 0);
            }}
            onFocus={() => setIsExpanded(true)}
            onBlur={() => {
              setTimeout(() => {
                setIsExpanded(false);
                setShowSuggestions(false);
              }, 200);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Describe what you want to change..."
            className="w-full bg-transparent text-white placeholder-white/40 px-6 py-4 pr-14 focus:outline-none text-base"
            autoComplete="off"
            spellCheck={false}
          />
          
          {/* Submit Button */}
          <button
            type="submit"
            disabled={!input.trim()}
            className={`
              absolute right-2 top-1/2 transform -translate-y-1/2
              p-2.5 rounded-xl transition-all duration-200
              ${input.trim() 
                ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/30' 
                : 'bg-white/10 text-white/30 cursor-not-allowed'}
            `}
          >
            <SendIcon className="w-5 h-5" />
          </button>
        </div>
        
        {/* Suggestions Dropdown */}
        {showSuggestions && filteredSuggestions.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-black/80 backdrop-blur-xl rounded-b-2xl border border-white/20 border-t-0 overflow-hidden shadow-xl">
            {filteredSuggestions.slice(0, 5).map((suggestion, i) => (
              <button
                key={i}
                type="button"
                onClick={() => {
                  setInput(suggestion);
                  setShowSuggestions(false);
                  inputRef.current?.focus();
                }}
                className="w-full text-left px-6 py-3 text-sm text-white/80 hover:bg-white/10 transition-colors flex items-center gap-3"
              >
                <SearchIcon className="w-4 h-4 text-white/40" />
                <span>{suggestion}</span>
              </button>
            ))}
          </div>
        )}
      </form>
      
      {/* Hint Text */}
      <div className="text-center mt-2 text-white/30 text-xs">
        Press <kbd className="bg-white/10 px-1.5 py-0.5 rounded">Tab</kbd> for suggestions, 
        <kbd className="bg-white/10 px-1.5 py-0.5 rounded">↑</kbd>
        <kbd className="bg-white/10 px-1.5 py-0.5 rounded">↓</kbd> for history
      </div>
    </div>
  );
}

/**
 * Agent Pulse - Shows active AI agent activity
 */
function AgentPulse({ agent }) {
  const statusColors = {
    idle: 'bg-gray-500',
    thinking: 'bg-yellow-500 animate-pulse',
    working: 'bg-blue-500 animate-pulse',
    complete: 'bg-green-500',
    error: 'bg-red-500'
  };
  
  return (
    <div 
      className="flex items-center gap-2 bg-black/60 backdrop-blur px-3 py-1.5 rounded-full border border-white/10"
      title={agent.status}
    >
      <div className={`w-2 h-2 rounded-full ${statusColors[agent.status] || statusColors.idle}`} />
      <span className="text-xs text-white/80 font-medium">{agent.name}</span>
      {agent.progress !== undefined && (
        <div className="w-16 h-1 bg-white/20 rounded-full overflow-hidden">
          <div 
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${agent.progress}%` }}
          />
        </div>
      )}
    </div>
  );
}

/**
 * Send Icon
 */
function SendIcon({ className }) {
  return (
    <svg 
      className={className} 
      fill="none" 
      stroke="currentColor" 
      viewBox="0 0 24 24"
    >
      <path 
        strokeLinecap="round" 
        strokeLinejoin="round" 
        strokeWidth={2} 
        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" 
      />
    </svg>
  );
}

/**
 * Search Icon
 */
function SearchIcon({ className }) {
  return (
    <svg 
      className={className} 
      fill="none" 
      stroke="currentColor" 
      viewBox="0 0 24 24"
    >
      <path 
        strokeLinecap="round" 
        strokeLinejoin="round" 
        strokeWidth={2} 
        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" 
      />
    </svg>
  );
}

export default CommandGhost;
