import React from 'react';
import { useLocation } from 'react-router-dom';
import GlobalTabs from './GlobalTabs';
import GlobalInputConsole from '../panels/GlobalInputConsole';
import MainPanel from '../panels/MainPanel';

const GlobalOverlay = () => {
    const location = useLocation();
    // Hide overlay on Landing and Requirements pages
    const hiddenPaths = ['/', '/landing', '/requirements'];

    if (hiddenPaths.includes(location.pathname)) {
        return null;
    }

    return (
        <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
            <GlobalTabs />
            <MainPanel />
            <GlobalInputConsole />
        </div>
    );
};

export default GlobalOverlay;
