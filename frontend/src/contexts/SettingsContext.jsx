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
    const [showGrid, setShowGrid] = useState(true);
    const [showControlsHelp, setShowControlsHelp] = useState(false);
    const [showEditor, setShowEditor] = useState(true);

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
                if (settings.showGrid !== undefined) setShowGrid(settings.showGrid);
                if (settings.showControlsHelp !== undefined) setShowControlsHelp(settings.showControlsHelp);
                if (settings.showEditor !== undefined) setShowEditor(settings.showEditor);
            } catch (e) {
                console.error('Failed to load settings:', e);
            }
        }
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
            ...newSettings
        };
        localStorage.setItem('brick-settings', JSON.stringify(currentSettings));
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
        setShowGrid(true);
        setShowControlsHelp(false);
        setShowEditor(true);
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

                showGrid,
                setShowGrid: (value) => {
                    setShowGrid(value);
                    saveSettings({ showGrid: value });
                },

                showControlsHelp,
                setShowControlsHelp: (value) => {
                    setShowControlsHelp(value);
                    saveSettings({ showControlsHelp: value });
                },

                showEditor,
                setShowEditor: (value) => {
                    setShowEditor(value);
                    saveSettings({ showEditor: value });
                },

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
