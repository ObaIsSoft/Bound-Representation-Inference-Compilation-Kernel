import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import BootSequence from './components/ui/BootSequence';
import Landing from './pages/Landing';
import RequirementsGatheringPage from './pages/RequirementsGatheringPage';
import PlanningPage from './pages/PlanningPage';
import Workspace from './pages/Workspace';
import { useTheme } from './contexts/ThemeContext';
import { SidebarProvider } from './contexts/SidebarContext';

const BootWrapper = ({ onComplete }) => {
    return <BootSequence onComplete={onComplete} />;
};

import { PanelProvider } from './contexts/PanelContext';
import GlobalOverlay from './components/layout/GlobalOverlay';

// ... imports ...

const AppContent = () => {
    const { theme } = useTheme();
    // Check if boot is already complete in session storage
    const [bootComplete, setBootComplete] = useState(() => {
        return sessionStorage.getItem('brick_boot_v12_stable') === 'true';
    });

    const handleBootComplete = () => {
        sessionStorage.setItem('brick_boot_v12_stable', 'true');
        setBootComplete(true);
    };

    if (!bootComplete) {
        return <BootWrapper onComplete={handleBootComplete} />;
    }

    return (
        <PanelProvider>
            <SidebarProvider>
                <Router>
                    <GlobalOverlay />
                    <Routes>
                        <Route path="/" element={<Navigate to="/landing" replace />} />
                        <Route path="/landing" element={<Landing />} />
                        <Route path="/requirements" element={<RequirementsGatheringPage />} />
                        <Route path="/planning" element={<PlanningPage />} />
                        <Route path="/workspace" element={<Workspace />} />
                    </Routes>
                </Router>
            </SidebarProvider>
        </PanelProvider>
    );
};

export default function App() {
    return <AppContent />;
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