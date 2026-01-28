import React from 'react';
import { User, Shield, CreditCard, LogOut, Mail, Award } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';

const AccountPage = () => {
    const { theme } = useTheme();

    return (
        <div
            className="flex-1 h-full flex flex-col overflow-hidden"
            style={{ backgroundColor: theme.colors.bg.primary }}
        >
            <PanelHeader title="User Account" icon={User} />

            <div className="flex-1 overflow-y-auto p-6">
                <div className="max-w-2xl mx-auto space-y-8">

                    {/* Profile Header */}
                    <div className="flex items-center gap-6 p-6 rounded-lg" style={{ backgroundColor: theme.colors.bg.secondary, border: `1px solid ${theme.colors.border.primary}` }}>
                        <div className="w-20 h-20 rounded-full flex items-center justify-center text-3xl font-bold"
                            style={{ backgroundColor: theme.colors.accent.primary, color: theme.colors.bg.primary }}>
                            OB
                        </div>
                        <div className="space-y-1">
                            <h1 className="text-2xl font-bold font-mono" style={{ color: theme.colors.text.primary }}>Obafemi</h1>
                            <div className="flex items-center gap-2 text-sm" style={{ color: theme.colors.text.secondary }}>
                                <Mail size={14} />
                                <span>user@example.com</span>
                            </div>
                            <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider mt-2"
                                style={{ backgroundColor: theme.colors.accent.primary + '20', color: theme.colors.accent.primary }}>
                                <Award size={12} />
                                <span>BRICK Architect</span>
                            </div>
                        </div>
                    </div>

                    {/* Account Type */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 font-mono" style={{ color: theme.colors.text.secondary }}>
                            <Shield size={16} style={{ color: theme.colors.accent.primary }} />
                            <h2 className="text-sm font-bold uppercase tracking-wider">Plan & Security</h2>
                        </div>

                        <div className="rounded-lg border overflow-hidden" style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.secondary }}>
                            <div className="p-4 border-b flex justify-between items-center" style={{ borderColor: theme.colors.border.primary }}>
                                <div>
                                    <div className="font-bold text-sm" style={{ color: theme.colors.text.primary }}>BRICK Pro License</div>
                                    <div className="text-xs" style={{ color: theme.colors.text.secondary }}>Active until Dec 2026</div>
                                </div>
                                <button className="text-xs px-3 py-1.5 rounded" style={{ backgroundColor: theme.colors.bg.tertiary, color: theme.colors.text.primary }}>Manage</button>
                            </div>
                            <div className="p-4 flex justify-between items-center">
                                <div>
                                    <div className="font-bold text-sm" style={{ color: theme.colors.text.primary }}>Two-Factor Authentication</div>
                                    <div className="text-xs" style={{ color: theme.colors.text.secondary }}>Enabled via GitHub</div>
                                </div>
                                <div className="text-xs font-bold" style={{ color: theme.colors.status.success }}>ACTIVE</div>
                            </div>
                        </div>
                    </div>

                    {/* Preferences Stub */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 font-mono" style={{ color: theme.colors.text.secondary }}>
                            <CreditCard size={16} style={{ color: theme.colors.accent.primary }} />
                            <h2 className="text-sm font-bold uppercase tracking-wider">Billing</h2>
                        </div>
                        <div className="p-4 rounded-lg text-sm text-center" style={{ backgroundColor: theme.colors.bg.secondary, color: theme.colors.text.muted, border: `1px dashed ${theme.colors.border.primary}` }}>
                            No active invoices.
                        </div>
                    </div>

                    {/* Logout */}
                    <div className="pt-8 border-t" style={{ borderColor: theme.colors.border.primary }}>
                        <button className="flex items-center gap-2 text-sm font-bold px-4 py-2 rounded hover:opacity-80 transition-opacity"
                            style={{ backgroundColor: theme.colors.status.error + '20', color: theme.colors.status.error }}>
                            <LogOut size={16} />
                            Sign Out
                        </button>
                    </div>

                </div>
            </div>
        </div>
    );
};

export default AccountPage;
