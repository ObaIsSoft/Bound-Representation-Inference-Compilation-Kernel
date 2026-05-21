import React, { useRef, useState, useEffect } from 'react';
import { motion, useDragControls } from 'framer-motion';
import { usePanel } from '../../contexts/PanelContext';
import { GripVertical } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

// Minimum pixels of the panel that must remain visible after any drag or resize
const VISIBLE_MIN = 200;

const DraggablePanel = ({ id, children, className = '', headerContent, zIndex = 50 }) => {
    const { panels, setPosition, setSize } = usePanel();
    const { theme } = useTheme();
    const panelState = panels[id];
    const resizeRef = useRef(null);
    const dragControls = useDragControls();
    const panelStateRef = useRef(panelState);
    panelStateRef.current = panelState;

    // Local state for resizing to avoid context thrashing
    const [isResizing, setIsResizing] = useState(false);

    // Clamp panel into viewport on mount and on window resize.
    // Fixes the "console disappears after drag" issue — any position that slipped
    // off-screen is corrected the next time the panel becomes visible.
    useEffect(() => {
        const clamp = () => {
            const ps = panelStateRef.current;
            if (!ps?.isOpen || !ps.position || !ps.size) return;
            const cx = Math.max(
                -(ps.size.width - VISIBLE_MIN),
                Math.min(ps.position.x, window.innerWidth - VISIBLE_MIN)
            );
            const cy = Math.max(
                0,
                Math.min(ps.position.y, window.innerHeight - VISIBLE_MIN)
            );
            if (cx !== ps.position.x || cy !== ps.position.y) {
                setPosition(id, { x: cx, y: cy });
            }
        };
        clamp();
        window.addEventListener('resize', clamp);
        return () => window.removeEventListener('resize', clamp);
    }, [id, setPosition]);

    if (!panelState?.isOpen) return null;

    const handleDragEnd = (event, info) => {
        let newX = panelState.position.x + info.offset.x;
        let newY = panelState.position.y + info.offset.y;

        // Keep at least VISIBLE_MIN px of the panel within the viewport
        const minX = -(panelState.size.width - VISIBLE_MIN);
        const maxX = window.innerWidth - VISIBLE_MIN;
        const minY = 0;
        const maxY = window.innerHeight - VISIBLE_MIN;

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
                zIndex: zIndex,
                touchAction: 'none',
                backgroundColor: theme.colors.bg.secondary + 'E6', // High opacity
                borderColor: theme.colors.border.secondary,
                borderWidth: '1px',
            }}
            // Reset transform after drag allows the style.left/top to take over again
            animate={{ x: 0, y: 0 }}
            transition={{ duration: 0 }}
            className={`flex flex-col rounded-xl shadow-2xl backdrop-blur-xl ${className}`}
        >
            {/* Drag Handle / Header */}
            <div
                onPointerDown={(e) => dragControls.start(e)}
                className="flex items-center justify-between px-3 py-2 cursor-grab active:cursor-grabbing select-none rounded-t-xl"
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
            <div className="flex-1 relative min-h-0">
                {children}
            </div>

            {/* Resize Handle (Bottom Right) */}
            <div
                onPointerDown={handleResizeStart}
                className="absolute bottom-0 right-0 w-8 h-8 cursor-nwse-resize z-[100] flex items-end justify-end p-1.5 group"
            >
                {/* Visual corner marker */}
                <div className="w-2.5 h-2.5 rounded-full bg-white/20 group-hover:bg-white/60 transition-colors shadow-sm" />
            </div>
        </motion.div>
    );
};

export default DraggablePanel;
