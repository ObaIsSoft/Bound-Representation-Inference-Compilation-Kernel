import React from 'react';
import { Terminal, Activity, FileText, Zap, Code2, Play, Square, ClipboardCheck } from 'lucide-react';
import TabButton from '../shared/TabButton';
import TerminalTab from './TerminalTab';
import DiagnosticsTab from './DiagnosticsTab';
import ConsoleTab from './ConsoleTab';
import OutputTab from './OutputTab';
import KCLTab from './KCLTab';
import TestResultsTab from './TestResultsTab';
import { useTheme } from '../../contexts/ThemeContext';
import { useSettings } from '../../contexts/SettingsContext';
import { useSimulation } from '../../contexts/SimulationContext';

const LowerPanel = ({ height, activeTab, setActiveTab, commandHistory, commandInput, setCommandInput, handleCommand }) => {
    const { theme } = useTheme();
    const { fontSize } = useSettings();
    const { isRunning, setIsRunning } = useSimulation();

    const tabs = [
        { id: 'terminal', label: 'Terminal', icon: Terminal },
        { id: 'console', label: 'Console', icon: Activity },
        { id: 'output', label: 'Output', icon: FileText },
        { id: 'diagnostics', label: 'Diagnostics', icon: Zap },
        { id: 'results', label: 'Test Results', icon: ClipboardCheck },
        { id: 'kcl', label: 'KCL', icon: Code2 },
    ];

    return (
        <div
            className="flex flex-col font-mono shrink-0 overflow-hidden"
            style={{
                height,
                backgroundColor: theme.colors.bg.secondary,
                borderTop: `1px solid ${theme.colors.border.primary}`
            }}
        >
            <div
                className="flex items-center justify-between shrink-0"
                style={{
                    backgroundColor: theme.colors.bg.primary,
                    borderBottom: `1px solid ${theme.colors.border.primary}`
                }}
            >
                <div className="flex overflow-x-auto no-scrollbar">
                    {tabs.map(tab => (
                        <TabButton
                            key={tab.id}
                            id={tab.id}
                            label={tab.label}
                            icon={tab.icon}
                            active={activeTab === tab.id}
                            onClick={setActiveTab}
                        />
                    ))}
                </div>
                <div className="flex items-center gap-2 px-3">
                    <button
                        onClick={() => setIsRunning(!isRunning)}
                        className="flex items-center gap-1.5 px-2 py-1 rounded text-[9px] font-bold uppercase tracking-wider transition-all"
                        style={{
                            backgroundColor: isRunning ? theme.colors.status.error + '1A' : theme.colors.status.success + '1A',
                            border: `1px solid ${isRunning ? theme.colors.status.error : theme.colors.status.success}33`,
                            color: isRunning ? theme.colors.status.error : theme.colors.status.success
                        }}
                    >
                        {isRunning ? <><Square size={10} /> Halt</> : <><Play size={10} /> Run</>}
                    </button>
                </div>
            </div>
            <div
                className="flex-1 overflow-y-auto p-3"
                style={{
                    backgroundColor: theme.colors.bg.primary + '33',
                    fontSize: `${fontSize}px`
                }}
            >
                {activeTab === 'terminal' && <TerminalTab commandHistory={commandHistory} commandInput={commandInput} setCommandInput={setCommandInput} handleCommand={handleCommand} />}
                {activeTab === 'diagnostics' && <DiagnosticsTab />}
                {activeTab === 'console' && <ConsoleTab isRunning={isRunning} />}
                {activeTab === 'output' && <OutputTab />}
                {activeTab === 'results' && <TestResultsTab />}
                {activeTab === 'kcl' && <KCLTab />}
            </div>
        </div>
    );
};

export default LowerPanel;
