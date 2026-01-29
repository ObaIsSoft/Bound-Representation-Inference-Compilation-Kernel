import React, { useState, useEffect } from 'react';
import { User, Shield, CreditCard, LogOut, Mail, Award, Save, Edit2, Loader } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';

const AccountPage = () => {
    const { theme } = useTheme();

    // State
    const [profile, setProfile] = useState(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState(null);

    // Edit Mode State
    const [isEditing, setIsEditing] = useState(false);
    const [editForm, setEditForm] = useState({});

    // Fetch Profile on Mount
    useEffect(() => {
        fetchProfile();
    }, []);

    const fetchProfile = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/user/profile');
            if (res.ok) {
                const data = await res.json();
                setProfile(data);
                setEditForm(data);
            } else {
                throw new Error('Failed to load profile');
            }
        } catch (err) {
            console.error(err);
            setError("Could not load user profile.");
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            const res = await fetch('http://localhost:8000/api/user/profile', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: editForm.name,
                    email: editForm.email
                })
            });

            if (res.ok) {
                const updated = await res.json();
                setProfile(updated);
                setIsEditing(false);
            } else {
                setError("Failed to save changes.");
            }
        } catch (err) {
            setError("Save failed. Check backend connection.");
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full" style={{ backgroundColor: theme.colors.bg.primary }}>
                <Loader className="animate-spin" size={24} style={{ color: theme.colors.text.muted }} />
            </div>
        );
    }

    return (
        <div
            className="flex-1 h-full flex flex-col overflow-hidden"
            style={{ backgroundColor: theme.colors.bg.primary }}
        >
            <PanelHeader title="User Account" icon={User} />

            <div className="flex-1 overflow-y-auto p-4 md:p-6">
                <div className="max-w-4xl mx-auto space-y-6">

                    {/* ERROR ALERT */}
                    {error && (
                        <div className="p-3 rounded text-sm font-bold border"
                            style={{
                                backgroundColor: theme.colors.status.error + '20',
                                color: theme.colors.status.error,
                                borderColor: theme.colors.status.error
                            }}>
                            {error}
                        </div>
                    )}

                    {/* MAIN GRID LAYOUT */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

                        {/* LEFT COLUMN: IDENTITY */}
                        <div className="md:col-span-2 space-y-6">

                            {/* Profile Header Card */}
                            <div className="p-6 rounded-lg relative overflow-hidden"
                                style={{
                                    backgroundColor: theme.colors.bg.secondary,
                                    border: `1px solid ${theme.colors.border.primary}`
                                }}>

                                <div className="absolute top-4 right-4">
                                    {!isEditing ? (
                                        <button onClick={() => setIsEditing(true)}
                                            className="p-2 rounded hover:brightness-110 transition-all flex items-center gap-2 text-xs font-bold uppercase"
                                            style={{ backgroundColor: theme.colors.bg.tertiary, color: theme.colors.text.secondary }}>
                                            <Edit2 size={14} /> Edit
                                        </button>
                                    ) : (
                                        <div className="flex items-center gap-2">
                                            <button onClick={() => { setIsEditing(false); setEditForm(profile); }}
                                                className="px-3 py-1.5 rounded text-xs font-bold transition-all"
                                                style={{ color: theme.colors.text.muted }}>
                                                Cancel
                                            </button>
                                            <button onClick={handleSave} disabled={saving}
                                                className="px-3 py-1.5 rounded text-xs font-bold transition-all flex items-center gap-1 shadow-lg"
                                                style={{ backgroundColor: theme.colors.accent.primary, color: theme.colors.bg.primary }}>
                                                {saving ? <Loader size={12} className="animate-spin" /> : <Save size={12} />}
                                                Save
                                            </button>
                                        </div>
                                    )}
                                </div>

                                <div className="flex flex-col sm:flex-row items-center gap-6">
                                    {/* Avatar */}
                                    <div className="w-24 h-24 rounded-full flex items-center justify-center text-3xl font-bold shadow-inner"
                                        style={{ backgroundColor: theme.colors.accent.primary, color: theme.colors.bg.primary }}>
                                        {profile.avatar_initials || '??'}
                                    </div>

                                    {/* Identity Details */}
                                    <div className="space-y-2 text-center sm:text-left flex-1 w-full">
                                        {isEditing ? (
                                            <div className="space-y-3 max-w-xs mx-auto sm:mx-0">
                                                <input
                                                    value={editForm.name}
                                                    onChange={e => setEditForm({ ...editForm, name: e.target.value })}
                                                    className="w-full px-3 py-2 rounded text-lg font-bold outline-none border focus:border-opacity-100"
                                                    style={{
                                                        backgroundColor: theme.colors.bg.tertiary,
                                                        color: theme.colors.text.primary,
                                                        borderColor: theme.colors.accent.primary
                                                    }}
                                                    placeholder="Full Name"
                                                />
                                                <input
                                                    value={editForm.email}
                                                    onChange={e => setEditForm({ ...editForm, email: e.target.value })}
                                                    className="w-full px-3 py-2 rounded text-sm font-mono outline-none border focus:border-opacity-100"
                                                    style={{
                                                        backgroundColor: theme.colors.bg.tertiary,
                                                        color: theme.colors.text.secondary,
                                                        borderColor: theme.colors.border.secondary
                                                    }}
                                                    placeholder="Email Address"
                                                />
                                            </div>
                                        ) : (
                                            <>
                                                <h1 className="text-3xl font-bold font-mono tracking-tight" style={{ color: theme.colors.text.primary }}>
                                                    {profile.name}
                                                </h1>
                                                <div className="flex items-center justify-center sm:justify-start gap-2 text-sm opacity-80" style={{ color: theme.colors.text.secondary }}>
                                                    <Mail size={14} />
                                                    <span>{profile.email}</span>
                                                </div>
                                            </>
                                        )}

                                        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider mt-2 border"
                                            style={{
                                                backgroundColor: theme.colors.accent.primary + '10',
                                                color: theme.colors.accent.primary,
                                                borderColor: theme.colors.accent.primary + '30'
                                            }}>
                                            <Award size={12} />
                                            <span>{profile.role || 'User'}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Stats / Activity Stub - Responsive Grid */}
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-4 rounded-lg border flex flex-col items-center justify-center py-8"
                                    style={{ backgroundColor: theme.colors.bg.secondary, borderColor: theme.colors.border.primary }}>
                                    <div className="text-2xl font-bold font-mono" style={{ color: theme.colors.text.primary }}>12</div>
                                    <div className="text-[10px] uppercase tracking-widest font-bold mt-1" style={{ color: theme.colors.text.muted }}>Projects</div>
                                </div>
                                <div className="p-4 rounded-lg border flex flex-col items-center justify-center py-8"
                                    style={{ backgroundColor: theme.colors.bg.secondary, borderColor: theme.colors.border.primary }}>
                                    <div className="text-2xl font-bold font-mono" style={{ color: theme.colors.text.primary }}>48h</div>
                                    <div className="text-[10px] uppercase tracking-widest font-bold mt-1" style={{ color: theme.colors.text.muted }}>Sim Time</div>
                                </div>
                            </div>
                        </div>

                        {/* RIGHT COLUMN: SETTINGS */}
                        <div className="space-y-6">

                            {/* Account Type */}
                            <div className="rounded-lg border overflow-hidden"
                                style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.secondary }}>
                                <div className="p-3 border-b flex items-center gap-2"
                                    style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.tertiary }}>
                                    <Shield size={14} style={{ color: theme.colors.accent.primary }} />
                                    <span className="text-xs font-bold uppercase tracking-wider" style={{ color: theme.colors.text.secondary }}>Subscription</span>
                                </div>
                                <div className="p-4 space-y-3">
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <div className="font-bold text-sm" style={{ color: theme.colors.text.primary }}>{profile.plan || 'Free Tier'}</div>
                                            <div className="text-xs mt-0.5 opacity-70" style={{ color: theme.colors.text.secondary }}>Renews Dec 2026</div>
                                        </div>
                                        <div className="text-[10px] font-bold px-2 py-0.5 rounded bg-green-500/20 text-green-500 border border-green-500/30">
                                            ACTIVE
                                        </div>
                                    </div>
                                    <button className="w-full text-xs font-bold py-2 rounded transition-colors hover:brightness-110"
                                        style={{ backgroundColor: theme.colors.bg.tertiary, color: theme.colors.text.primary }}>
                                        Manage Plan
                                    </button>
                                </div>
                            </div>

                            {/* Preferences */}
                            <div className="rounded-lg border overflow-hidden"
                                style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.secondary }}>
                                <div className="p-3 border-b flex items-center gap-2"
                                    style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.tertiary }}>
                                    <CreditCard size={14} style={{ color: theme.colors.accent.primary }} />
                                    <span className="text-xs font-bold uppercase tracking-wider" style={{ color: theme.colors.text.secondary }}>Billing</span>
                                </div>
                                <div className="p-4 text-center">
                                    <div className="text-sm italic opacity-50" style={{ color: theme.colors.text.muted }}>
                                        No active invoices
                                    </div>
                                </div>
                            </div>

                            {/* Logout */}
                            <div className="pt-4">
                                <button className="w-full flex items-center justify-center gap-2 text-sm font-bold px-4 py-3 rounded border hover:bg-red-500/10 transition-colors"
                                    style={{
                                        backgroundColor: 'transparent',
                                        color: theme.colors.status.error,
                                        borderColor: theme.colors.status.error + '40'
                                    }}>
                                    <LogOut size={16} />
                                    Sign Out
                                </button>
                            </div>

                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
};

export default AccountPage;
