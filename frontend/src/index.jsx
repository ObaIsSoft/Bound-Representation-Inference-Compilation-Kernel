import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { ThemeProvider } from './contexts/ThemeContext';
import { SettingsProvider } from './contexts/SettingsContext';
import { DesignProvider } from './contexts/DesignContext';
import { SimulationProvider } from './contexts/SimulationContext';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <ThemeProvider>
            <SettingsProvider>
                <DesignProvider>
                    <SimulationProvider>
                        <App />
                    </SimulationProvider>
                </DesignProvider>
            </SettingsProvider>
        </ThemeProvider>
    </React.StrictMode>
);
