/**
 * useSpatialState - 3D State Management for Omniviewport
 * 
 * Manages spatial state including:
 * - Camera position/target
 * - Selected/hovered objects
 * - Annotations
 * - Selection history
 * - Spatial bookmarks
 */

import { useState, useCallback, useRef } from 'react';

/**
 * useSpatialState Hook
 */
export function useSpatialState() {
  // Camera state
  const [cameraPosition, setCameraPosition] = useState([5, 5, 5]);
  const [cameraTarget, setCameraTarget] = useState([0, 0, 0]);
  
  // Selection state
  const [selectedObject, setSelectedObject] = useState(null);
  const [hoveredObject, setHoveredObject] = useState(null);
  const [selectionHistory, setSelectionHistory] = useState([]);
  
  // Annotations
  const [annotations, setAnnotations] = useState([]);
  const [activeAnnotation, setActiveAnnotation] = useState(null);
  
  // Bookmarks (saved camera positions)
  const [bookmarks, setBookmarks] = useState([]);
  
  // Undo/Redo stacks
  const undoStack = useRef([]);
  const redoStack = useRef([]);
  
  // Update camera position
  const updateCamera = useCallback((position, target) => {
    setCameraPosition(position);
    setCameraTarget(target);
  }, []);
  
  // Select object
  const selectObject = useCallback((object) => {
    if (object) {
      setSelectionHistory(prev => [...prev, object]);
    }
    setSelectedObject(object);
  }, []);
  
  // Clear selection
  const clearSelection = useCallback(() => {
    setSelectedObject(null);
  }, []);
  
  // Hover object
  const hoverObject = useCallback((object) => {
    setHoveredObject(object);
  }, []);
  
  // Add annotation
  const addAnnotation = useCallback((annotation) => {
    const newAnnotation = {
      id: Date.now().toString(),
      createdAt: new Date().toISOString(),
      ...annotation
    };
    
    setAnnotations(prev => [...prev, newAnnotation]);
    undoStack.current.push({ type: 'ADD_ANNOTATION', data: newAnnotation });
    redoStack.current = [];
    
    return newAnnotation.id;
  }, []);
  
  // Update annotation
  const updateAnnotation = useCallback((id, updates) => {
    setAnnotations(prev => 
      prev.map(ann => 
        ann.id === id ? { ...ann, ...updates, updatedAt: new Date().toISOString() } : ann
      )
    );
  }, []);
  
  // Delete annotation
  const deleteAnnotation = useCallback((id) => {
    const annotation = annotations.find(a => a.id === id);
    if (annotation) {
      undoStack.current.push({ type: 'DELETE_ANNOTATION', data: annotation });
      redoStack.current = [];
    }
    
    setAnnotations(prev => prev.filter(a => a.id !== id));
  }, [annotations]);
  
  // Add bookmark
  const addBookmark = useCallback((name, position, target) => {
    const bookmark = {
      id: Date.now().toString(),
      name,
      position: position || cameraPosition,
      target: target || cameraTarget,
      createdAt: new Date().toISOString()
    };
    
    setBookmarks(prev => [...prev, bookmark]);
    return bookmark.id;
  }, [cameraPosition, cameraTarget]);
  
  // Delete bookmark
  const deleteBookmark = useCallback((id) => {
    setBookmarks(prev => prev.filter(b => b.id !== id));
  }, []);
  
  // Go to bookmark
  const goToBookmark = useCallback((id) => {
    const bookmark = bookmarks.find(b => b.id === id);
    if (bookmark) {
      updateCamera(bookmark.position, bookmark.target);
    }
  }, [bookmarks, updateCamera]);
  
  // Undo last action
  const undo = useCallback(() => {
    const action = undoStack.current.pop();
    if (!action) return;
    
    switch (action.type) {
      case 'ADD_ANNOTATION':
        setAnnotations(prev => prev.filter(a => a.id !== action.data.id));
        break;
      case 'DELETE_ANNOTATION':
        setAnnotations(prev => [...prev, action.data]);
        break;
      default:
        break;
    }
    
    redoStack.current.push(action);
  }, []);
  
  // Redo last undone action
  const redo = useCallback(() => {
    const action = redoStack.current.pop();
    if (!action) return;
    
    switch (action.type) {
      case 'ADD_ANNOTATION':
        setAnnotations(prev => [...prev, action.data]);
        break;
      case 'DELETE_ANNOTATION':
        setAnnotations(prev => prev.filter(a => a.id !== action.data.id));
        break;
      default:
        break;
    }
    
    undoStack.current.push(action);
  }, []);
  
  // Get annotations for a specific object
  const getObjectAnnotations = useCallback((objectId) => {
    return annotations.filter(a => a.attachedTo === objectId);
  }, [annotations]);
  
  // Clear all annotations
  const clearAnnotations = useCallback(() => {
    undoStack.current.push({ type: 'CLEAR_ANNOTATIONS', data: [...annotations] });
    redoStack.current = [];
    setAnnotations([]);
  }, [annotations]);
  
  // Focus camera on object
  const focusOnObject = useCallback((object) => {
    if (!object || !object.position) return;
    
    const pos = object.position;
    const distance = 5; // Default focus distance
    
    const newPosition = [
      pos[0] + distance,
      pos[1] + distance,
      pos[2] + distance
    ];
    
    updateCamera(newPosition, pos);
  }, [updateCamera]);
  
  // Export state (for saving/loading)
  const exportState = useCallback(() => {
    return {
      annotations,
      bookmarks,
      selectionHistory,
      cameraPosition,
      cameraTarget
    };
  }, [annotations, bookmarks, selectionHistory, cameraPosition, cameraTarget]);
  
  // Import state
  const importState = useCallback((state) => {
    if (state.annotations) setAnnotations(state.annotations);
    if (state.bookmarks) setBookmarks(state.bookmarks);
    if (state.cameraPosition) setCameraPosition(state.cameraPosition);
    if (state.cameraTarget) setCameraTarget(state.cameraTarget);
  }, []);
  
  return {
    // Camera
    cameraPosition,
    cameraTarget,
    updateCamera,
    
    // Selection
    selectedObject,
    hoveredObject,
    selectionHistory,
    selectObject,
    clearSelection,
    hoverObject,
    focusOnObject,
    
    // Annotations
    annotations,
    activeAnnotation,
    setActiveAnnotation,
    addAnnotation,
    updateAnnotation,
    deleteAnnotation,
    getObjectAnnotations,
    clearAnnotations,
    
    // Bookmarks
    bookmarks,
    addBookmark,
    deleteBookmark,
    goToBookmark,
    
    // Undo/Redo
    undo,
    redo,
    canUndo: undoStack.current.length > 0,
    canRedo: redoStack.current.length > 0,
    
    // Import/Export
    exportState,
    importState
  };
}

export default useSpatialState;
