import React, { useRef, useState, useEffect } from 'react';
import { motion, useDragControls } from 'framer-motion';
import { usePanel } from '../../contexts/PanelContext';
import { GripVertical } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const DraggablePanel = ({ id, children, className = '', headerContent }) => {
    const { panels, setPosition, setSize } = usePanel();
    const { theme } = useTheme();
    const panelState = panels[id];
    const constraintsRef = useRef(null);
    const resizeRef = useRef(null);

    // Local state for resizing to avoid context thrashing
    const [isResizing, setIsResizing] = useState(false);

    if (!panelState.isOpen) return null;

    const handleDragEnd = (event, info) => {
        // Update persistent state with the delta
        // We add the drag delta to the original position
        const newX = panelState.position.x + info.offset.x;
        const newY = panelState.position.y + info.offset.y;

        // Ensure we don't drag off screen (basic bounds)
        const boundedX = Math.max(0, Math.min(newX, window.innerWidth - panelState.size.width));
        const boundedY = Math.max(0, Math.min(newY, window.innerHeight - panelState.size.height));

        setPosition(id, { x: boundedX, y: boundedY });
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
            onDragEnd={handleDragEnd}
            initial={false}
            // We use the persistent positions for the 'initial' layout,
            // but while dragging, framer uses transforms.
            // On drag end, we commit to state, causing a re-render.
            // We must force the visual style to match the state.
            style={{
                position: 'absolute',
                left: panelState.position.x,
                top: panelState.position.y,
                width: panelState.size.width,
                height: panelState.size.height,
                zIndex: 50, // Could be dynamic based on focus
                touchAction: 'none' // Important for drag
            }}
            // Reset transform after drag allows the style.left/top to take over again
            animate={{ x: 0, y: 0 }}
            className={`flex flex-col rounded-xl shadow-2xl backdrop-blur-xl border overflow-hidden ${className}`}
            // dynamic border color based on theme
            borderColor={theme.colors.border.secondary}
        >
            {/* Drag Handle / Header */}
            <div
                className="flex items-center justify-between px-3 py-2 cursor-grab active:cursor-grabbing select-none"
                style={{ backgroundColor: theme.colors.bg.secondary + '90' }}
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
