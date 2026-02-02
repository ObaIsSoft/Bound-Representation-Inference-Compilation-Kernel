import React, { useRef, useState, useEffect } from 'react';
import { motion, useDragControls } from 'framer-motion';
import { usePanel } from '../../contexts/PanelContext';
import { GripVertical } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const DraggablePanel = ({ id, children, className = '', headerContent }) => {
    const { panels, setPosition, setSize } = usePanel();
    const { theme } = useTheme();
    const panelState = panels[id];
    const resizeRef = useRef(null);
    const dragControls = useDragControls();

    // Local state for resizing to avoid context thrashing
    const [isResizing, setIsResizing] = useState(false);

    if (!panelState?.isOpen) return null;

    const handleDragEnd = (event, info) => {
        // Update persistent state with the delta
        // We add the drag delta to the original position
        let newX = panelState.position.x + info.offset.x;
        let newY = panelState.position.y + info.offset.y;

        // Visual viewports bounds
        // Allow panel to go off screen but keep 50px visible
        const minX = -panelState.size.width + 50;
        const maxX = window.innerWidth - 50;
        const minY = 0; // Don't allow going above top
        const maxY = window.innerHeight - 50;

        // Constrain
        newX = Math.max(minX, Math.min(newX, maxX));
        newY = Math.max(minY, Math.min(newY, maxY));

        setPosition(id, { x: newX, y: newY });
    };

    // Resize Logic
    const handleResizeStart = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsResizing(true);

        const startX = e.clientX;
        const startY = e.clientY;
        const startWidth = panelState.size.width;
        const startHeight = panelState.size.height;

        const handleResizeMove = (moveEvent) => {
            const deltaX = moveEvent.clientX - startX;
            const deltaY = moveEvent.clientY - startY;

            // Constrain min size
            const newWidth = Math.max(300, startWidth + deltaX);
            const newHeight = Math.max(60, startHeight + deltaY);

            // Directly update DOM for performance during resize (state update is laggy)
            if (resizeRef.current) {
                resizeRef.current.style.width = `${newWidth}px`;
                resizeRef.current.style.height = `${newHeight}px`;
            }
        };

        const handleResizeEnd = (endEvent) => {
            const deltaX = endEvent.clientX - startX;
            const deltaY = endEvent.clientY - startY;

            const newWidth = Math.max(300, startWidth + deltaX);
            const newHeight = Math.max(60, startHeight + deltaY);

            setSize(id, { width: newWidth, height: newHeight });
            setIsResizing(false);

            document.removeEventListener('pointermove', handleResizeMove);
            document.removeEventListener('pointerup', handleResizeEnd);
        };

        document.addEventListener('pointermove', handleResizeMove);
        document.addEventListener('pointerup', handleResizeEnd);
    };

    return (
        <motion.div
            ref={resizeRef}
            drag
            dragMomentum={false}
            dragElastic={0}
            dragListener={false} // Only drag using handle
            dragControls={dragControls}
            onDragEnd={handleDragEnd}
            initial={false}
            style={{
                position: 'absolute',
                left: panelState.position.x,
                top: panelState.position.y,
                width: panelState.size.width,
                height: panelState.size.height,
                zIndex: 50,
                touchAction: 'none',
                backgroundColor: theme.colors.bg.secondary + 'E6', // High opacity
                borderColor: theme.colors.border.secondary,
                borderWidth: '1px',
            }}
            // Reset transform after drag allows the style.left/top to take over again
            animate={{ x: 0, y: 0 }}
            transition={{ duration: 0 }}
            className={`flex flex-col rounded-xl shadow-2xl backdrop-blur-xl overflow-hidden ${className}`}
        >
            {/* Drag Handle / Header */}
            <div
                onPointerDown={(e) => dragControls.start(e)}
                className="flex items-center justify-between px-3 py-2 cursor-grab active:cursor-grabbing select-none border-b"
                style={{
                    backgroundColor: theme.colors.bg.tertiary,
                    borderColor: theme.colors.border.secondary
                }}
            >
                {/* Visual texture for grip */}
                <div className="opacity-50"><GripVertical size={14} color={theme.colors.text.tertiary} /></div>
                <div className="flex-1 px-2">{headerContent}</div>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden relative">
                {children}
            </div>

            {/* Resize Handle (Bottom Right) */}
            <div
                onPointerDown={handleResizeStart}
                className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize z-50 flex items-center justify-center group"
            >
                {/* Visual corner marker */}
                <div className="w-1.5 h-1.5 rounded-full bg-gray-400/50 group-hover:bg-white/80 transition-colors" />
            </div>
        </motion.div>
    );
};

export default DraggablePanel;
