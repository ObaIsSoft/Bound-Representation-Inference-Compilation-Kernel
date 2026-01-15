import React from 'react';
import { Server, Wifi, BarChart } from 'lucide-react';
import { useSimulation } from '../../contexts/SimulationContext';

const DiagnosticsTab = () => {
    const { isRunning, metrics, kernelLogs: logs } = useSimulation();
    const logEndRef = React.useRef(null);

    React.useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-2 animate-in fade-in duration-500 h-full">
            {/* Kernel Diagnostics */}
            <div className="space-y-3 bg-slate-900/50 p-3 rounded border border-slate-800">
                <div className="flex items-center gap-2 text-amber-500 text-[10px] font-bold uppercase tracking-wider">
                    <Server size={12} /> Kernel Analytics
                </div>
                <div className="space-y-2">
                    <div className="flex justify-between text-[10px]">
                        <span className="text-slate-500">CPU Load:</span>
                        <span className="text-slate-200">{metrics.cpu.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-950 h-1 rounded-full overflow-hidden">
                        <div
                            className="bg-amber-500 h-full transition-all duration-500"
                            style={{ width: `${metrics.cpu}%` }}
                        />
                    </div>
                    <div className="flex justify-between text-[10px] mt-2">
                        <span className="text-slate-500">Memory Pressure:</span>
                        <span className="text-slate-200">{metrics.memory.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-950 h-1 rounded-full overflow-hidden">
                        <div
                            className="bg-emerald-500 h-full transition-all duration-500"
                            style={{ width: `${metrics.memory}%` }}
                        />
                    </div>
                    <div className="flex justify-between text-[10px] mt-2">
                        <span className="text-slate-500">Node Cluster:</span>
                        <span className={isRunning ? "text-emerald-400 font-bold" : "text-amber-500 font-bold"}>
                            {isRunning ? "ACTIVE" : "STANDBY"}
                        </span>
                    </div>
                </div>
            </div>

            {/* Agent Swarm Diagnostics */}
            <div className="space-y-3 bg-slate-900/50 p-3 rounded border border-slate-800">
                <div className="flex items-center gap-2 text-blue-500 text-[10px] font-bold uppercase tracking-wider">
                    <Wifi size={12} /> Federated Agents
                </div>
                <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-[10px]">
                        <span className="text-slate-400">AERO_ENGINE_V4</span>
                        <span className="text-emerald-500">{metrics.network.aero.toFixed(0)}ms</span>
                    </div>
                    <div className="flex items-center justify-between text-[10px]">
                        <span className="text-slate-400">THERMAL_GRID_Z</span>
                        <span className="text-emerald-500">{metrics.network.thermal.toFixed(0)}ms</span>
                    </div>
                    <div className="flex items-center justify-between text-[10px]">
                        <span className="text-slate-400">GEO_MATER_PRO</span>
                        <span className="text-amber-500">{metrics.network.geo.toFixed(0)}ms</span>
                    </div>
                </div>
            </div>

            {/* Real-time Log Stream (New) */}
            <div className="space-y-3 bg-slate-900/50 p-3 rounded border border-slate-800 flex flex-col h-full overflow-hidden">
                <div className="flex items-center gap-2 text-purple-500 text-[10px] font-bold uppercase tracking-wider shrink-0">
                    <BarChart size={12} /> Live Event Stream
                </div>
                <div className="flex-1 overflow-y-auto font-mono text-[9px] space-y-1 text-slate-400">
                    {logs.map((log, i) => (
                        <div key={i} className="truncate">{log}</div>
                    ))}
                    <div ref={logEndRef} />
                </div>
            </div>
        </div>
    );
};

export default DiagnosticsTab;
