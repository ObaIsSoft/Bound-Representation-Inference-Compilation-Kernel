import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';

/**
 * RegionSelector UI Overlay
 * Appears when entering Micro-Machining mode
 * Prompts user to select a region for Neural SDF training
 */
export const RegionSelector = ({ onRegionSelected, onCancel }) => {
    const { theme } = useTheme();

    return (
        <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 1000
        }}>
            {/* Instruction Overlay */}
            <div style={{
                position: 'absolute',
                top: '20px',
                left: '50%',
                transform: 'translateX(-50%)',
                background: theme.colors.bg.secondary || 'rgba(0, 0, 0, 0.8)',
                color: theme.colors.text.primary || 'white',
                padding: '16px 24px',
                borderRadius: '8px',
                pointerEvents: 'auto',
                border: `1px solid ${theme.colors.border.accent || 'rgba(255, 193, 7, 0.5)'}`,
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                maxWidth: '400px'
            }}>
                <div style={{
                    marginBottom: '8px',
                    fontWeight: 'bold',
                    fontSize: '14px',
                    color: theme.colors.accent.primary || '#ffc107'
                }}>
                    🔬 Micro-Machining Mode
                </div>
                <div style={{
                    fontSize: '12px',
                    opacity: 0.9,
                    marginBottom: '8px'
                }}>
                    The neural network will train on your current design.
                </div>
                <div style={{
                    fontSize: '11px',
                    opacity: 0.7,
                    marginBottom: '12px',
                    fontStyle: 'italic'
                }}>
                    Training takes 20-40 seconds. This view is for high-precision inspection.
                </div>
                <button
                    onClick={onCancel}
                    style={{
                        width: '100%',
                        padding: '8px 16px',
                        background: theme.colors.bg.tertiary || 'rgba(255, 255, 255, 0.1)',
                        color: theme.colors.text.primary || 'white',
                        border: `1px solid ${theme.colors.border.primary || 'rgba(255, 255, 255, 0.2)'}`,
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '12px',
                        fontWeight: '500',
                        transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                        e.target.style.background = theme.colors.bg.primary || 'rgba(255, 255, 255, 0.2)';
                        e.target.style.borderColor = theme.colors.accent.primary || '#ffc107';
                    }}
                    onMouseLeave={(e) => {
                        e.target.style.background = theme.colors.bg.tertiary || 'rgba(255, 255, 255, 0.1)';
                        e.target.style.borderColor = theme.colors.border.primary || 'rgba(255, 255, 255, 0.2)';
                    }}
                >
                    Cancel
                </button>
            </div>
        </div>
    );
};
