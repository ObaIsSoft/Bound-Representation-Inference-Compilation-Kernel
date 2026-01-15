import React from 'react';
import { CheckCircle2 } from 'lucide-react';

import { useSimulation } from '../../contexts/SimulationContext';

const OutputTab = () => {
    const { kernelLogs } = useSimulation();
    const logEndRef = React.useRef(null);

    React.useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [kernelLogs]);

    return (
        <div className="space-y-1 text-slate-400 not-italic font-mono h-full overflow-y-auto">
            {/* Historical Static Logs (simulated boot sequence) */}
            <div>[08:12:01] ISA Initialization complete.</div>
            <div className="text-emerald-400 font-bold flex items-center gap-1">
                <CheckCircle2 size={10} /> PASS: Physics feasibility verified for Type-C.
            </div>
            <div>[08:12:02] Compiling KCL {"->"} B-Rep Architecture...</div>
            <div className="text-blue-400">STATUS: All Digital Twins Matched.</div>

            {/* Live Kernel Logs */}
            {kernelLogs.map((log, i) => (
                <div key={i} className="truncate">{log}</div>
            ))}
            <div ref={logEndRef} />
        </div>
    );
};

export default OutputTab;
