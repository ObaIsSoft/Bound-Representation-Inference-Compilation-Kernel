import React from 'react';
import { usePanel, PANEL_IDS } from '../../contexts/PanelContext';
import { useTheme } from '../../contexts/ThemeContext';
import DraggablePanel from '../shared/DraggablePanel';
import InputConsole from '../panels/InputConsole';
import MainPanel from '../panels/MainPanel';

const GlobalOverlay = () => {
    const { panels } = usePanel();
    const { theme } = useTheme();

    return (
        <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
            {/* Input Console */}
            <DraggablePanel
                id={PANEL_IDS.INPUT}
                headerContent={<span className="text-xs font-mono opacity-70">INPUT CONSOLE</span>}
                className="pointer-events-auto shadow-xl"
            >
                <InputConsole />
            </DraggablePanel>

            {/* Main Panel (Intent Stream) */}
            <DraggablePanel
                id={PANEL_IDS.MAIN}
                headerContent={<span className="text-xs font-mono opacity-70">INTENT STREAM</span>}
                className="pointer-events-auto shadow-xl"
            >
                <MainPanel />
            </DraggablePanel>
        </div>
    );
};

export default GlobalOverlay;
