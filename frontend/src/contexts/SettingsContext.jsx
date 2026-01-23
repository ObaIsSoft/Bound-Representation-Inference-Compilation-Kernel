import React, { createContext, useContext, useState, useEffect } from 'react';

const SettingsContext = createContext();

export const SettingsProvider = ({ children }) => {
    const [fontSize, setFontSizeState] = useState('12');
    const [autoSave, setAutoSave] = useState(true);
    const [formatOnSave, setFormatOnSave] = useState(true);
    const [notifications, setNotifications] = useState(true);
    const [physicsKernel, setPhysicsKernel] = useState('EARTH_AERO');
    const [compilerOptimization, setCompilerOptimization] = useState('balanced');
    const [showTemperatureSensor, setShowTemperatureSensor] = useState(false);
    const [show3DThermometer, setShow3DThermometer] = useState(false);
    const [aiModel, setAiModel] = useState('mock'); // 'mock', 'openai', 'gemini-robotics', 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-flash', 'gemini-2.5-pro'
    const [visualizationQuality, setVisualizationQualityState] = useState('HIGH'); // 'ULTRA', 'HIGH', 'MEDIUM', 'LOW'
    const [meshRenderingMode, setMeshRenderingMode] = useState('sdf'); // 'sdf' (boolean ops), 'preview' (fast mesh)

    // New settings for previously broken buttons
    const [simulationFrequency, setSimulationFrequency] = useState('1000'); // Hz
    const [incrementalCompilation, setIncrementalCompilation] = useState(true);
    const [secureBoot, setSecureBoot] = useState(true);
    const [agentSandboxing, setAgentSandboxing] = useState(true);
    const [agentProposals, setAgentProposals] = useState(true);

    // Load settings from localStorage on mount
    useEffect(() => {
        const savedSettings = localStorage.getItem('brick-settings');
        if (savedSettings) {
            try {
                const settings = JSON.parse(savedSettings);
                if (settings.fontSize) setFontSizeState(settings.fontSize);
                if (settings.autoSave !== undefined) setAutoSave(settings.autoSave);
                if (settings.formatOnSave !== undefined) setFormatOnSave(settings.formatOnSave);
                if (settings.notifications !== undefined) setNotifications(settings.notifications);
                if (settings.physicsKernel) setPhysicsKernel(settings.physicsKernel);
                if (settings.compilerOptimization) setCompilerOptimization(settings.compilerOptimization);
                if (settings.showTemperatureSensor !== undefined) setShowTemperatureSensor(settings.showTemperatureSensor);
                if (settings.show3DThermometer !== undefined) setShow3DThermometer(settings.show3DThermometer);
                if (settings.aiModel) setAiModel(settings.aiModel);
                if (settings.visualizationQuality) setVisualizationQualityState(settings.visualizationQuality);
                if (settings.meshRenderingMode) setMeshRenderingMode(settings.meshRenderingMode);
                if (settings.simulationFrequency) setSimulationFrequency(settings.simulationFrequency);
                if (settings.incrementalCompilation !== undefined) setIncrementalCompilation(settings.incrementalCompilation);
                if (settings.secureBoot !== undefined) setSecureBoot(settings.secureBoot);
                if (settings.agentSandboxing !== undefined) setAgentSandboxing(settings.agentSandboxing);
                if (settings.agentProposals !== undefined) setAgentProposals(settings.agentProposals);
            } catch (e) {
                console.error('Failed to load settings:', e);
            }
        }

        // Also fetch from backend to sync
        fetch('http://localhost:8000/api/settings')
            .then(res => res.json())
            .then(data => {
                if (data.settings) {
                    const backendSettings = data.settings;
                    // Backend settings take precedence for runtime configuration
                    if (backendSettings.simulation_frequency) setSimulationFrequency(backendSettings.simulation_frequency.toString());
                    if (backendSettings.physics_kernel) setPhysicsKernel(backendSettings.physics_kernel);
                    if (backendSettings.compiler_optimization) setCompilerOptimization(backendSettings.compiler_optimization);
                    if (backendSettings.visualization_quality) setVisualizationQualityState(backendSettings.visualization_quality);
                    if (backendSettings.incremental_compilation !== undefined) setIncrementalCompilation(backendSettings.incremental_compilation);
                    if (backendSettings.secure_boot !== undefined) setSecureBoot(backendSettings.secure_boot);
                    if (backendSettings.agent_sandboxing !== undefined) setAgentSandboxing(backendSettings.agent_sandboxing);
                    if (backendSettings.agent_proposals !== undefined) setAgentProposals(backendSettings.agent_proposals);
                }
            })
            .catch(err => console.warn('Backend settings fetch failed:', err));
    }, []);

    // Save settings to localStorage whenever they change
    const saveSettings = (newSettings) => {
        const currentSettings = {
            fontSize,
            autoSave,
            formatOnSave,
            notifications,
            physicsKernel,
            compilerOptimization,
            showTemperatureSensor,
            show3DThermometer,
            aiModel,
            visualizationQuality,
            meshRenderingMode,
            simulationFrequency,
            incrementalCompilation,
            secureBoot,
            agentSandboxing,
            agentProposals,
            ...newSettings
        };
        localStorage.setItem('brick-settings', JSON.stringify(currentSettings));

        // Sync critical backend settings (runtime configuration)
        const backendSettings = {
            simulation_frequency: currentSettings.simulationFrequency ? parseInt(currentSettings.simulationFrequency) : undefined,
            physics_kernel: currentSettings.physicsKernel,
            compiler_optimization: currentSettings.compilerOptimization,
            visualization_quality: currentSettings.visualizationQuality,
            incremental_compilation: currentSettings.incrementalCompilation,
            secure_boot: currentSettings.secureBoot,
            agent_sandboxing: currentSettings.agentSandboxing,
            agent_proposals: currentSettings.agentProposals
        };

        // Send to backend (fire and forget, don't block UI)
        fetch('http://localhost:8000/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(backendSettings)
        }).catch(err => console.warn('Backend settings sync failed:', err));
    };

    const setFontSize = (size) => {
        setFontSizeState(size);
        saveSettings({ fontSize: size });
    };

    const updateAutoSave = (value) => {
        setAutoSave(value);
        saveSettings({ autoSave: value });
    };

    const updateFormatOnSave = (value) => {
        setFormatOnSave(value);
        saveSettings({ formatOnSave: value });
    };

    const updateNotifications = (value) => {
        setNotifications(value);
        saveSettings({ notifications: value });
    };

    const updatePhysicsKernel = (value) => {
        setPhysicsKernel(value);
        saveSettings({ physicsKernel: value });
    };

    const updateCompilerOptimization = (value) => {
        setCompilerOptimization(value);
        saveSettings({ compilerOptimization: value });
    };

    const updateShowTemperatureSensor = (value) => {
        setShowTemperatureSensor(value);
        saveSettings({ showTemperatureSensor: value });
    };

    const updateShow3DThermometer = (value) => {
        setShow3DThermometer(value);
        saveSettings({ show3DThermometer: value });
    };

    const updateAiModel = (value) => {
        setAiModel(value);
        saveSettings({ aiModel: value });
    };

    const updateVisualizationQuality = (value) => {
        setVisualizationQualityState(value);
        saveSettings({ visualizationQuality: value });
    };

    const updateSimulationFrequency = (value) => {
        setSimulationFrequency(value);
        saveSettings({ simulationFrequency: value });
    };

    const updateIncrementalCompilation = (value) => {
        setIncrementalCompilation(value);
        saveSettings({ incrementalCompilation: value });
    };

    const updateSecureBoot = (value) => {
        setSecureBoot(value);
        saveSettings({ secureBoot: value });
    };

    const updateAgentSandboxing = (value) => {
        setAgentSandboxing(value);
        saveSettings({ agentSandboxing: value });
    };

    const updateAgentProposals = (value) => {
        setAgentProposals(value);
        saveSettings({ agentProposals: value });
    };

    const resetToDefaults = () => {
        setFontSizeState('12');
        setAutoSave(true);
        setFormatOnSave(true);
        setNotifications(true);
        setPhysicsKernel('EARTH_AERO');
        setCompilerOptimization('balanced');
        setShowTemperatureSensor(false);
        setShow3DThermometer(false);
        setVisualizationQualityState('HIGH');
        setMeshRenderingMode('sdf');
        setSimulationFrequency('1000');
        setIncrementalCompilation(true);
        setSecureBoot(true);
        setAgentSandboxing(true);
        setAgentProposals(true);
        localStorage.removeItem('brick-settings');
    };

    return (
        <SettingsContext.Provider
            value={{
                fontSize,
                setFontSize,
                autoSave,
                setAutoSave: updateAutoSave,
                formatOnSave,
                setFormatOnSave: updateFormatOnSave,
                notifications,
                setNotifications: updateNotifications,
                physicsKernel,
                setPhysicsKernel: updatePhysicsKernel,
                compilerOptimization,
                setCompilerOptimization: updateCompilerOptimization,
                showTemperatureSensor,
                setShowTemperatureSensor: updateShowTemperatureSensor,
                show3DThermometer,
                setShow3DThermometer: updateShow3DThermometer,

                aiModel,
                setAiModel: updateAiModel,

                visualizationQuality,
                setVisualizationQuality: updateVisualizationQuality,

                meshRenderingMode,
                setMeshRenderingMode: (mode) => {
                    setMeshRenderingMode(mode);
                    saveSettings({ meshRenderingMode: mode });
                },

                simulationFrequency,
                setSimulationFrequency: updateSimulationFrequency,

                incrementalCompilation,
                setIncrementalCompilation: updateIncrementalCompilation,

                secureBoot,
                setSecureBoot: updateSecureBoot,

                agentSandboxing,
                setAgentSandboxing: updateAgentSandboxing,

                agentProposals,
                setAgentProposals: updateAgentProposals,

                resetToDefaults
            }}
        >
            {children}
        </SettingsContext.Provider>
    );
};

export const useSettings = () => {
    const context = useContext(SettingsContext);
    if (!context) {
        throw new Error('useSettings must be used within SettingsProvider');
    }
    return context;
};
