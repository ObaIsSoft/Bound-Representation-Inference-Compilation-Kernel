import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, FileText, Image as ImageIcon, FileSpreadsheet, FileCode, Box, AlertCircle, Check, Loader2 } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface FileUploadZoneProps {
    files: File[];
    onFilesChange: (files: File[]) => void;
    maxFiles?: number;
}

interface FileWithPreview extends File {
    preview?: string;
    uploading?: boolean;
    error?: string;
}

// File category configurations
const FILE_CATEGORIES = {
    '3d': {
        extensions: ['.stl', '.step', '.stp', '.obj', '.fbx', '.gltf', '.glb', '.3mf', '.ply'],
        maxSize: 100 * 1024 * 1024, // 100MB
        icon: Box,
        color: '#8B5CF6',
        label: '3D Model'
    },
    'pdf': {
        extensions: ['.pdf'],
        maxSize: 50 * 1024 * 1024, // 50MB
        icon: FileText,
        color: '#EF4444',
        label: 'PDF'
    },
    'image': {
        extensions: ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'],
        maxSize: 20 * 1024 * 1024, // 20MB
        icon: ImageIcon,
        color: '#10B981',
        label: 'Image'
    },
    'spreadsheet': {
        extensions: ['.xlsx', '.xls', '.csv'],
        maxSize: 20 * 1024 * 1024, // 20MB
        icon: FileSpreadsheet,
        color: '#22C55E',
        label: 'Spreadsheet'
    },
    'document': {
        extensions: ['.docx', '.pptx'],
        maxSize: 20 * 1024 * 1024, // 20MB
        icon: FileText,
        color: '#3B82F6',
        label: 'Document'
    },
    'code': {
        extensions: ['.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.py', '.js', '.ts', '.jsx', '.tsx', '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.swift', '.kt', '.cs'],
        maxSize: 10 * 1024 * 1024, // 10MB
        icon: FileCode,
        color: '#6B7280',
        label: 'Text/Code'
    }
};

function getFileCategory(filename: string): keyof typeof FILE_CATEGORIES | 'unknown' {
    const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'));
    for (const [category, config] of Object.entries(FILE_CATEGORIES)) {
        if (config.extensions.includes(ext)) {
            return category as keyof typeof FILE_CATEGORIES;
        }
    }
    return 'unknown';
}

function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export default function FileUploadZone({ files, onFilesChange, maxFiles = 6 }: FileUploadZoneProps) {
    const { theme } = useTheme();
    const [errors, setErrors] = useState<string[]>([]);

    const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
        // Clear previous errors
        setErrors([]);

        // Check if adding these files would exceed max
        if (files.length + acceptedFiles.length > maxFiles) {
            setErrors([`Maximum ${maxFiles} files allowed`]);
            return;
        }

        // Handle rejected files
        const newErrors: string[] = [];
        rejectedFiles.forEach((rejection) => {
            const error = rejection.errors[0];
            newErrors.push(`${rejection.file.name}: ${error.message}`);
        });

        // Validate file sizes
        acceptedFiles.forEach((file) => {
            const category = getFileCategory(file.name);
            if (category !== 'unknown') {
                const maxSize = FILE_CATEGORIES[category].maxSize;
                if (file.size > maxSize) {
                    newErrors.push(`${file.name}: Exceeds ${formatFileSize(maxSize)} limit for ${FILE_CATEGORIES[category].label}`);
                }
            }
        });

        if (newErrors.length > 0) {
            setErrors(newErrors);
        }

        // Add previews for images
        const filesWithPreviews = acceptedFiles.map(file => {
            if (file.type.startsWith('image/')) {
                return Object.assign(file, { preview: URL.createObjectURL(file) }) as FileWithPreview;
            }
            return file as FileWithPreview;
        });

        onFilesChange([...files, ...filesWithPreviews]);
    }, [files, onFilesChange, maxFiles]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        maxFiles: maxFiles - files.length,
        accept: {
            'application/pdf': ['.pdf'],
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
            'application/vnd.ms-excel': ['.xls'],
            'text/csv': ['.csv'],
            'text/plain': ['.txt'],
            'text/markdown': ['.md'],
            'application/json': ['.json'],
            'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
            'model/stl': ['.stl'],
            'model/obj': ['.obj'],
            'model/gltf+json': ['.gltf'],
            'model/gltf-binary': ['.glb'],
        }
    });

    const removeFile = (index: number) => {
        const file = files[index];
        if ((file as FileWithPreview).preview) {
            URL.revokeObjectURL((file as FileWithPreview).preview!);
        }
        const newFiles = files.filter((_, i) => i !== index);
        onFilesChange(newFiles);
        setErrors([]);
    };

    const clearAll = () => {
        files.forEach(file => {
            if ((file as FileWithPreview).preview) {
                URL.revokeObjectURL((file as FileWithPreview).preview!);
            }
        });
        onFilesChange([]);
        setErrors([]);
    };

    const getFileIcon = (filename: string) => {
        const category = getFileCategory(filename);
        if (category === 'unknown') return FileText;
        return FILE_CATEGORIES[category].icon;
    };

    const getFileColor = (filename: string) => {
        const category = getFileCategory(filename);
        if (category === 'unknown') return theme.colors.text.muted;
        return FILE_CATEGORIES[category].color;
    };

    const getFileLabel = (filename: string) => {
        const category = getFileCategory(filename);
        if (category === 'unknown') return 'File';
        return FILE_CATEGORIES[category].label;
    };

    return (
        <div className="w-full">
            {/* Drop Zone */}
            {files.length < maxFiles && (
                <div
                    {...getRootProps()}
                    className={`
                        border-2 border-dashed rounded-xl p-6 cursor-pointer transition-all duration-200
                        ${isDragActive 
                            ? 'border-primary bg-primary/10 scale-[1.02]' 
                            : 'border-gray-600 hover:border-gray-500 hover:bg-gray-800/50'
                        }
                    `}
                    style={{
                        borderColor: isDragActive ? theme.colors.accent.primary : theme.colors.border.primary,
                        backgroundColor: isDragActive ? `${theme.colors.accent.primary}15` : 'transparent'
                    }}
                >
                    <input {...getInputProps()} />
                    <div className="flex flex-col items-center gap-3 text-center">
                        <Upload 
                            size={32} 
                            style={{ 
                                color: isDragActive 
                                    ? theme.colors.accent.primary 
                                    : theme.colors.text.muted 
                            }} 
                        />
                        <div>
                            <p 
                                className="font-medium"
                                style={{ color: theme.colors.text.primary }}
                            >
                                {isDragActive ? 'Drop files here' : 'Drop files here or click to browse'}
                            </p>
                            <p 
                                className="text-sm mt-1"
                                style={{ color: theme.colors.text.muted }}
                            >
                                Up to {maxFiles} files • 3D models up to 100MB
                            </p>
                        </div>
                        
                        {/* File type icons */}
                        <div className="flex items-center gap-2 mt-2 flex-wrap justify-center">
                            {Object.entries(FILE_CATEGORIES).map(([cat, config]) => {
                                const Icon = config.icon;
                                return (
                                    <div 
                                        key={cat}
                                        className="flex items-center gap-1 px-2 py-1 rounded text-xs"
                                        style={{ 
                                            backgroundColor: `${config.color}20`,
                                            color: config.color,
                                            border: `1px solid ${config.color}40`
                                        }}
                                        title={`${config.label}: up to ${formatFileSize(config.maxSize)}`}
                                    >
                                        <Icon size={12} />
                                        <span>{config.label}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* Errors */}
            {errors.length > 0 && (
                <div 
                    className="mt-3 p-3 rounded-lg border"
                    style={{ 
                        backgroundColor: '#EF444415',
                        borderColor: '#EF444440',
                        color: '#EF4444'
                    }}
                >
                    <div className="flex items-center gap-2 mb-2">
                        <AlertCircle size={16} />
                        <span className="font-medium">File upload errors:</span>
                    </div>
                    <ul className="text-sm space-y-1 ml-6">
                        {errors.map((error, i) => (
                            <li key={i}>{error}</li>
                        ))}
                    </ul>
                </div>
            )}

            {/* File List */}
            {files.length > 0 && (
                <div className="mt-4 space-y-2">
                    <div className="flex items-center justify-between">
                        <span 
                            className="text-sm font-medium"
                            style={{ color: theme.colors.text.secondary }}
                        >
                            {files.length} file{files.length !== 1 ? 's' : ''} selected
                        </span>
                        <button
                            onClick={clearAll}
                            className="text-sm flex items-center gap-1 px-2 py-1 rounded transition-colors hover:bg-red-500/20"
                            style={{ color: '#EF4444' }}
                        >
                            <X size={14} />
                            Clear all
                        </button>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {files.map((file, index) => {
                            const Icon = getFileIcon(file.name);
                            const color = getFileColor(file.name);
                            const label = getFileLabel(file.name);
                            const preview = (file as FileWithPreview).preview;

                            return (
                                <div
                                    key={index}
                                    className="flex items-center gap-3 p-3 rounded-lg border transition-all hover:border-opacity-80"
                                    style={{
                                        backgroundColor: theme.colors.bg.secondary,
                                        borderColor: `${color}40`,
                                        borderLeftWidth: '3px',
                                        borderLeftColor: color
                                    }}
                                >
                                    {/* Icon or Preview */}
                                    {preview ? (
                                        <img
                                            src={preview}
                                            alt={file.name}
                                            className="w-12 h-12 object-cover rounded"
                                        />
                                    ) : (
                                        <Icon 
                                            size={24} 
                                            style={{ color }}
                                        />
                                    )}

                                    {/* File Info */}
                                    <div className="flex-1 min-w-0">
                                        <p 
                                            className="font-medium text-sm truncate"
                                            style={{ color: theme.colors.text.primary }}
                                            title={file.name}
                                        >
                                            {file.name}
                                        </p>
                                        <div className="flex items-center gap-2 text-xs" style={{ color: theme.colors.text.muted }}>
                                            <span>{label}</span>
                                            <span>•</span>
                                            <span>{formatFileSize(file.size)}</span>
                                        </div>
                                    </div>

                                    {/* Remove Button */}
                                    <button
                                        onClick={() => removeFile(index)}
                                        className="p-1.5 rounded-lg transition-colors hover:bg-red-500/20"
                                        style={{ color: theme.colors.text.muted }}
                                        title="Remove file"
                                    >
                                        <X size={16} />
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Max files reached message */}
            {files.length >= maxFiles && (
                <div 
                    className="mt-3 p-3 rounded-lg text-center text-sm"
                    style={{ 
                        backgroundColor: `${theme.colors.accent.primary}15`,
                        color: theme.colors.accent.primary
                    }}
                >
                    <Check size={16} className="inline mr-2" />
                    Maximum {maxFiles} files selected
                </div>
            )}
        </div>
    );
}
