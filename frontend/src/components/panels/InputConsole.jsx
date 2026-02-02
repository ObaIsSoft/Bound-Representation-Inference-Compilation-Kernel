import React, { useState } from 'react';
import { usePanel } from '../../contexts/PanelContext';
import { useTheme } from '../../contexts/ThemeContext';
import { GitBranch, Send, Mic, Plus } from 'lucide-react';

const InputConsole = () => {
    const { activeSession, setActiveSession } = usePanel();
    const { theme } = useTheme();
    const [inputValue, setInputValue] = useState('');
    const [showBranchMenu, setShowBranchMenu] = useState(false);

    const handleSend = () => {
        if (!inputValue.trim()) return;
        // In real app, this sends to backend
        console.log(`Sending to ${activeSession.branchName}: ${inputValue}`);
        setInputValue('');
    };

    const handleNewBranch = () => {
        const newBranchId = `branch-${Date.now()}`;
        setActiveSession({
            id: newBranchId,
            branchName: `Feature-${newBranchId.slice(-4)}`,
            parentId: activeSession.id,
            history: []
        });
        setShowBranchMenu(false);
    };

    return (
        <div className="flex flex-col h-full bg-white/5 backdrop-blur-md">
            {/* Context Bar */}
            <div className="flex items-center justify-between px-3 py-1.5 border-b text-xs select-none"
                style={{ borderColor: theme.colors.border.secondary, color: theme.colors.text.secondary }}>

                <div
                    className="flex items-center gap-2 cursor-pointer hover:text-white transition-colors"
                    onClick={() => setShowBranchMenu(!showBranchMenu)}
                >
                    <GitBranch size={12} />
                    <span>{activeSession.branchName}</span>
                    <span className="opacity-50 text-[10px]">{activeSession.parentId ? '(Child)' : '(Main)'}</span>
                </div>

                {/* Branch Menu (Simple Dropdown Mock) */}
                {showBranchMenu && (
                    <div className="absolute bottom-full left-0 mb-2 w-48 bg-[#1e1e1e] border border-gray-700 rounded-lg shadow-xl overflow-hidden z-50">
                        <div className="p-2 text-[10px] uppercase text-gray-500 font-bold">Switch Branch</div>
                        <button
                            className="w-full text-left px-3 py-2 text-xs hover:bg-white/10 text-white truncate"
                            onClick={() => {
                                setActiveSession({ ...activeSession, id: 'session-main', branchName: 'Main', parentId: null });
                                setShowBranchMenu(false);
                            }}
                        >
                            <GitBranch size={10} className="inline mr-2" /> Main
                        </button>
                        <div className="border-t border-gray-700 my-1"></div>
                        <button
                            className="w-full text-left px-3 py-2 text-xs hover:bg-white/10 text-blue-400 flex items-center"
                            onClick={handleNewBranch}
                        >
                            <Plus size={10} className="mr-2" /> New Branch
                        </button>
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="flex-1 flex flex-col p-2">
                <textarea
                    className="flex-1 bg-transparent border-none outline-none resize-none text-sm p-1 text-white placeholder-gray-500 font-mono"
                    placeholder={`Message ${activeSession.branchName}...`}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSend();
                        }
                    }}
                />

                <div className="flex justify-between items-center mt-2 px-1">
                    <button className="p-1.5 rounded-full hover:bg-white/10 text-gray-400 transition-colors">
                        <Mic size={16} />
                    </button>
                    <button
                        onClick={handleSend}
                        className="p-1.5 rounded-full bg-blue-600 hover:bg-blue-500 text-white shadow-lg transition-colors"
                    >
                        <Send size={14} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default InputConsole;
