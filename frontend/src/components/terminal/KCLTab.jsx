import React from 'react';
import { useSimulation } from '../../contexts/SimulationContext';

const KCLTab = () => {
    const { kclCode } = useSimulation();

    return (
        <div className="text-purple-300 not-italic whitespace-pre-wrap font-mono p-4 text-xs">
            {kclCode}
        </div>
    );
};

export default KCLTab;
