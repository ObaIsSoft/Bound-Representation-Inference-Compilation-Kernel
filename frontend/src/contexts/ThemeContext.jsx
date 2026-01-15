import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export const themes = {
    blue: {
        name: 'Blue',
        colors: {
            // Backgrounds
            bg: {
                primary: '#0f172a',
                secondary: '#1e293b',
                tertiary: '#334155',
                elevated: '#475569'
            },
            // Text
            text: {
                primary: '#f1f5f9',
                secondary: '#cbd5e1',
                tertiary: '#94a3b8',
                muted: '#64748b'
            },
            // Accents
            accent: {
                primary: '#fbbf24',
                secondary: '#f59e0b',
                hover: '#d97706',
                glow: 'rgba(251, 191, 36, 0.4)'
            },
            // Borders
            border: {
                primary: '#334155',
                secondary: '#475569',
                accent: 'rgba(251, 191, 36, 0.3)'
            },
            // Status
            status: {
                success: '#10b981',
                error: '#ef4444',
                warning: '#f59e0b',
                info: '#3b82f6'
            }
        }
    },
    'white-gold': {
        name: 'White and Gold',
        colors: {
            bg: {
                primary: '#ffffff',
                secondary: '#fafaf9',
                tertiary: '#f5f5f4',
                elevated: '#e7e5e4'
            },
            text: {
                primary: '#0f172a',
                secondary: '#1e293b',
                tertiary: '#475569',
                muted: '#64748b'
            },
            accent: {
                primary: '#f59e0b',
                secondary: '#d97706',
                hover: '#b45309',
                glow: 'rgba(245, 158, 11, 0.3)'
            },
            border: {
                primary: '#e7e5e4',
                secondary: '#d6d3d1',
                accent: 'rgba(245, 158, 11, 0.4)'
            },
            status: {
                success: '#059669',
                error: '#dc2626',
                warning: '#d97706',
                info: '#2563eb'
            }
        }
    },
    'high-contrast': {
        name: 'High Contrast',
        colors: {
            bg: {
                primary: '#000000',
                secondary: '#0a0a0a',
                tertiary: '#1a1a1a',
                elevated: '#262626'
            },
            text: {
                primary: '#ffffff',
                secondary: '#f5f5f5',
                tertiary: '#e5e5e5',
                muted: '#a3a3a3'
            },
            accent: {
                primary: '#fbbf24',
                secondary: '#fcd34d',
                hover: '#fde047',
                glow: 'rgba(251, 191, 36, 0.6)'
            },
            border: {
                primary: '#404040',
                secondary: '#525252',
                accent: 'rgba(251, 191, 36, 0.5)'
            },
            status: {
                success: '#22c55e',
                error: '#f87171',
                warning: '#fbbf24',
                info: '#60a5fa'
            }
        }
    },
    'black-gold': {
        name: 'Black and Gold',
        colors: {
            bg: {
                primary: '#15202b',
                secondary: '#1a2332',
                tertiary: '#1f2937',
                elevated: '#243040'
            },
            text: {
                primary: '#ffffff',
                secondary: '#f5f5f5',
                tertiary: '#d4d4d4',
                muted: '#a8a29e'
            },
            accent: {
                primary: '#d97706',
                secondary: '#b45309',
                hover: '#92400e',
                glow: 'rgba(217, 119, 6, 0.4)'
            },
            border: {
                primary: '#2d3748',
                secondary: '#374151',
                accent: 'rgba(217, 119, 6, 0.3)'
            },
            status: {
                success: '#16a34a',
                error: '#dc2626',
                warning: '#d97706',
                info: '#3b82f6'
            }
        }
    }
};

export const ThemeProvider = ({ children }) => {
    const [currentTheme, setCurrentTheme] = useState('blue');

    useEffect(() => {
        // Load theme from localStorage
        const savedTheme = localStorage.getItem('brick-theme');
        if (savedTheme && themes[savedTheme]) {
            setCurrentTheme(savedTheme);
        }
    }, []);

    const changeTheme = (themeName) => {
        if (themes[themeName]) {
            setCurrentTheme(themeName);
            localStorage.setItem('brick-theme', themeName);
        }
    };

    const theme = themes[currentTheme];

    return (
        <ThemeContext.Provider value={{ theme, currentTheme, changeTheme, themes }}>
            {children}
        </ThemeContext.Provider>
    );
};

export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within ThemeProvider');
    }
    return context;
};
