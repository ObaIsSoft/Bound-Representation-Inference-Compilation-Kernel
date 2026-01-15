import React from 'react';
import { Check, X, RotateCcw } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const ProposalCard = ({ proposal, onResolve }) => {
    const { theme } = useTheme();

    const getCardStyle = () => {
        if (proposal.status === 'pending') {
            return {
                backgroundColor: theme.colors.accent.primary + '1A',
                borderColor: theme.colors.accent.primary + '66'
            };
        } else if (proposal.status === 'approved') {
            return {
                backgroundColor: theme.colors.status.success + '1A',
                borderColor: theme.colors.status.success + '4D',
                opacity: 0.75
            };
        } else {
            return {
                backgroundColor: theme.colors.bg.primary + '80',
                borderColor: theme.colors.border.primary,
                opacity: 0.5
            };
        }
    };

    return (
        <div
            className="p-3 rounded border font-mono transition-all relative overflow-hidden"
            style={getCardStyle()}
        >
            <div className="flex justify-between items-center mb-2">
                <span className="text-[9px] font-bold tracking-widest" style={{ color: theme.colors.accent.primary }}>
                    {proposal.agent}
                </span>
                <div className="flex items-center gap-1.5">
                    {proposal.status === 'pending' ? (
                        <>
                            <button
                                onClick={() => onResolve(proposal.id, 'approved')}
                                className="p-1 rounded"
                                style={{ color: theme.colors.status.success }}
                                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = theme.colors.status.success + '1A'}
                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                            >
                                <Check size={12} />
                            </button>
                            <button
                                onClick={() => onResolve(proposal.id, 'denied')}
                                className="p-1 rounded"
                                style={{ color: theme.colors.status.error }}
                                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = theme.colors.status.error + '1A'}
                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                            >
                                <X size={12} />
                            </button>
                        </>
                    ) : (
                        <button
                            onClick={() => onResolve(proposal.id, 'pending')}
                            className="p-1 rounded"
                            style={{ color: theme.colors.text.tertiary }}
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = theme.colors.bg.tertiary}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                        >
                            <RotateCcw size={10} />
                        </button>
                    )}
                </div>
            </div>
            <div
                className={`text-[10px] leading-relaxed ${proposal.status === 'denied' ? 'line-through' : ''}`}
                style={{ color: theme.colors.text.primary }}
            >
                {proposal.action}
            </div>
        </div>
    );
};

export default ProposalCard;
