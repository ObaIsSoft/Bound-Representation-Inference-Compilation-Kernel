import React, { useState } from 'react';
import Header from './components/layout/Header';
import ActivityBar from './components/layout/ActivityBar';
import BootSequence from './components/ui/BootSequence';

import SettingsPage from './components/settings/SettingsPage';
import AccountPage from './components/settings/AccountPage';
import SearchPanel from './components/panels/SearchPanel';
import ComponentPanel from './components/panels/ComponentPanel';
import RunDebugPanel from './components/panels/RunDebugPanel';
import AgentPodsPanel from './components/panels/AgentPodsPanel';
import CompilePanel from './components/panels/CompilePanel';
import ManufacturingPanel from './components/panels/ManufacturingPanel';
import ForkPanel from './components/panels/ForkPanel';
import ExportPanel from './components/panels/ExportPanel';
import CompliancePanel from './components/panels/CompliancePanel';
import ISABrowserPanel from './components/panels/ISABrowserPanel';

import { ACTIVITY_BAR_WIDTH, DEFAULT_PANEL_SIZES } from './utils/constants';
import { useTheme } from './contexts/ThemeContext';

export default function App() {
    const { theme } = useTheme();
    const [activeActivity, setActiveActivity] = useState('search');
    const [leftWidth] = useState(DEFAULT_PANEL_SIZES.left);

    // Render left panel based on active activity
    const renderLeftPanel = () => {
        switch (activeActivity) {
            case 'search':
                return <SearchPanel width={leftWidth} />;
            case 'components':
                return <ComponentPanel width={leftWidth} />;
            case 'run':
                return <RunDebugPanel width={leftWidth} />;
            case 'agents':
                return <AgentPodsPanel width={leftWidth} />;
            case 'compile':
                return <CompilePanel width={leftWidth} />;
            case 'mfg':
                return <ManufacturingPanel width={leftWidth} />;
            case 'compliance':
                return <CompliancePanel width={leftWidth} />;
            case 'fork':
                return <ForkPanel width={leftWidth} />;
            case 'export':
                return <ExportPanel width={leftWidth} />;
            case 'isa':
                return <ISABrowserPanel width={leftWidth} />;
            default:
                return <SearchPanel width={leftWidth} />;
        }
    };

    return (
        <div
            className="flex flex-col h-screen w-full overflow-hidden select-none font-sans"
            style={{
                backgroundColor: theme.colors.bg.primary,
                color: theme.colors.text.primary
            }}
        >
            <BootSequence />
            <Header />

            {/* Main Interaction Matrix */}
            <div className="flex flex-1 overflow-hidden min-h-0 relative">
                <ActivityBar activeTab={activeActivity} setActiveTab={setActiveActivity} />

                {activeActivity === 'settings' || activeActivity === 'account' ? (
                    activeActivity === 'settings' ? <SettingsPage /> : <AccountPage />
                ) : (
                    <>
                        {renderLeftPanel()}

                        {/* Center Placeholder */}
                        <div
                            className="flex-1 flex items-center justify-center"
                            style={{ backgroundColor: theme.colors.bg.secondary }}
                        >
                            <div className="text-center space-y-4">
                                <div
                                    className="text-6xl font-black tracking-widest"
                                    style={{ color: theme.colors.accent.primary + '40' }}
                                >
                                    BRICK
                                </div>
                                <div
                                    className="text-sm font-mono tracking-wider"
                                    style={{ color: theme.colors.text.muted }}
                                >
                                    Core navigation preserved. Workspace TBD.
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

/* 
DEFERRED PLAN: SystemHealth Dashboard (Phase 11.3)

Goal: Real-time visualization of backend telemetry.
Architecture:
1. Backend WebSocket (TODO in main.py):
   - Endpoint: /ws/telemetry (pushes JSON every 2s)
   - Payload: { system: {cpu, mem}, latency: {avg, p95}, agents_active, status }
   
2. Frontend Component (SystemHealth.tsx):
   - Path: src/components/SystemHealth.tsx
   - Visuals:
     - [ ] Top Bar: Status Badge (Green/Red)
     - [ ] Left Panel: CPU/RAM animated bars
     - [ ] Right Panel: Latency Line Chart (recharts/chart.js)
     - [ ] Agent Grid: 64 dots indicating active/idle status

Integration:
- Add 'health' to activeActivity state.
- Add HealthPanel.tsx wrapping SystemHealth.
- Connect via useWebSocket hook.
*/
