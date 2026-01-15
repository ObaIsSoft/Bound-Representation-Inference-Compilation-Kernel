import React, { useRef, useState, useEffect } from 'react';
import { X, ChevronLeft, ChevronRight, Code } from 'lucide-react';
import { useDesign } from '../../contexts/DesignContext';
import { useTheme } from '../../contexts/ThemeContext';

const TabBar = () => {
    const { theme } = useTheme();
    const { tabs, activeTabId, setActiveTabId, closeTab, isEditorVisible, toggleEditor } = useDesign();
    const scrollContainerRef = useRef(null);
    const [showLeftArrow, setShowLeftArrow] = useState(false);
    const [showRightArrow, setShowRightArrow] = useState(false);

    const checkScrollButtons = () => {
        if (scrollContainerRef.current) {
            const { scrollLeft, scrollWidth, clientWidth } = scrollContainerRef.current;
            setShowLeftArrow(scrollLeft > 0);
            setShowRightArrow(scrollLeft < scrollWidth - clientWidth - 1);
        }
    };

    useEffect(() => {
        checkScrollButtons();
        const container = scrollContainerRef.current;
        if (container) {
            container.addEventListener('scroll', checkScrollButtons);
            window.addEventListener('resize', checkScrollButtons);
            return () => {
                container.removeEventListener('scroll', checkScrollButtons);
                window.removeEventListener('resize', checkScrollButtons);
            };
        }
    }, [tabs]);

    const scroll = (direction) => {
        if (scrollContainerRef.current) {
            const scrollAmount = 200;
            scrollContainerRef.current.scrollBy({
                left: direction === 'left' ? -scrollAmount : scrollAmount,
                behavior: 'smooth'
            });
        }
    };

    const handleCloseTab = (e, tabId) => {
        e.stopPropagation();
        closeTab(tabId);
    };

    if (tabs.length === 0) return null;

    return (
        <div
            className="flex items-center shrink-0 relative"
            style={{
                height: '32px',
                backgroundColor: theme.colors.bg.secondary,
                borderBottom: `1px solid ${theme.colors.border.primary}`
            }}
        >
            {/* Left scroll button */}
            {showLeftArrow && (
                <button
                    onClick={() => scroll('left')}
                    className="absolute left-0 z-10 h-full px-2 transition-opacity"
                    style={{
                        backgroundColor: theme.colors.bg.secondary,
                        borderRight: `1px solid ${theme.colors.border.primary}`,
                        color: theme.colors.text.muted
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.color = theme.colors.accent.primary}
                    onMouseLeave={(e) => e.currentTarget.style.color = theme.colors.text.muted}
                >
                    <ChevronLeft size={14} />
                </button>
            )}

            {/* Tabs container */}
            <div
                ref={scrollContainerRef}
                className="flex-1 flex overflow-x-auto scrollbar-hide"
                style={{
                    scrollbarWidth: 'none',
                    msOverflowStyle: 'none',
                    paddingLeft: showLeftArrow ? '32px' : '0',
                    paddingRight: showRightArrow ? '32px' : '0'
                }}
            >
                {tabs.map((tab) => {
                    const isActive = tab.id === activeTabId;
                    return (
                        <div
                            key={tab.id}
                            onClick={() => setActiveTabId(tab.id)}
                            className="flex items-center gap-2 px-3 cursor-pointer group shrink-0 relative"
                            style={{
                                height: '32px',
                                backgroundColor: isActive ? theme.colors.bg.primary : 'transparent',
                                borderRight: `1px solid ${theme.colors.border.primary}`,
                                borderTop: isActive ? `2px solid ${theme.colors.accent.primary}` : '2px solid transparent',
                                color: isActive ? theme.colors.text.primary : theme.colors.text.tertiary
                            }}
                            onMouseEnter={(e) => {
                                if (!isActive) {
                                    e.currentTarget.style.backgroundColor = theme.colors.bg.primary + '40';
                                }
                            }}
                            onMouseLeave={(e) => {
                                if (!isActive) {
                                    e.currentTarget.style.backgroundColor = 'transparent';
                                }
                            }}
                        >
                            <span className="text-[10px] font-mono whitespace-nowrap">
                                {tab.name}
                            </span>
                            {tab.modified && (
                                <div
                                    className="w-1.5 h-1.5 rounded-full"
                                    style={{ backgroundColor: theme.colors.accent.primary }}
                                />
                            )}
                            <button
                                onClick={(e) => handleCloseTab(e, tab.id)}
                                className="opacity-0 group-hover:opacity-100 transition-opacity ml-1"
                                style={{ color: theme.colors.text.muted }}
                                onMouseEnter={(e) => e.currentTarget.style.color = theme.colors.status.error}
                                onMouseLeave={(e) => e.currentTarget.style.color = theme.colors.text.muted}
                            >
                                <X size={12} />
                            </button>
                        </div>
                    );
                })}
            </div>

            {/* Right scroll button */}
            {showRightArrow && (
                <button
                    onClick={() => scroll('right')}
                    className="absolute right-0 z-10 h-full px-2 transition-opacity"
                    style={{
                        backgroundColor: theme.colors.bg.secondary,
                        borderLeft: `1px solid ${theme.colors.border.primary}`,
                        color: theme.colors.text.muted
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.color = theme.colors.accent.primary}
                    onMouseLeave={(e) => e.currentTarget.style.color = theme.colors.text.muted}
                >
                    <ChevronRight size={14} />
                </button>
            )}

            {/* Toggle Editor Button */}
            <div className="h-full flex items-center border-l z-20 relative" style={{ borderColor: theme.colors.border.primary }}>
                <button
                    onClick={toggleEditor}
                    className="h-full px-3 flex items-center justify-center transition-colors"
                    style={{
                        backgroundColor: isEditorVisible ? theme.colors.bg.primary : 'transparent',
                        color: isEditorVisible ? theme.colors.accent.primary : theme.colors.text.muted
                    }}
                    title="Toggle Code Editor"
                    onMouseEnter={(e) => !isEditorVisible && (e.currentTarget.style.color = theme.colors.text.primary)}
                    onMouseLeave={(e) => !isEditorVisible && (e.currentTarget.style.color = theme.colors.text.muted)}
                >
                    <Code size={14} />
                </button>
            </div>
        </div>
    );
};

export default TabBar;
