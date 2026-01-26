import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Box as BoxIcon, Radar, Navigation, Ruler, Zap } from 'lucide-react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, PerspectiveCamera, Environment, ContactShadows, Edges, Text, Html, Stars } from '@react-three/drei';
import { EffectComposer, Bloom, ChromaticAberration, Scanline } from '@react-three/postprocessing';
import * as THREE from 'three';

const PHASES = {
    SENSOR_FUSION: 0,
    MULTI_MODEL: 1,
    FLIGHT_PATH: 2,
    IGNITION: 3,
    COMPLETE: 4
};

const EQUATIONS = [
    "F = ma", "E = mc²", "σ = Eε", "∇ × E", "∂u/∂t", "∫ P dV",
    "Re = ρvL/μ", "M = v/a", "Cl = 2L/ρv²S"
];

const BootSequence = ({ onComplete }) => {
    const [phase, setPhase] = useState(PHASES.SENSOR_FUSION);
    const [logs, setLogs] = useState([]);
    const [modelIndex, setModelIndex] = useState(0);

    // Skip logic
    useEffect(() => {
        // New key for V12 Stable
        if (sessionStorage.getItem('brick_boot_v12_stable')) {
            onComplete?.();
            setPhase(PHASES.COMPLETE);
        } else {
            runCinematicSequence();
        }
    }, []);

    // ESC Listener
    useEffect(() => {
        const handleKey = (e) => {
            if (e.key === 'Escape') {
                sessionStorage.setItem('brick_boot_v12_stable', 'true');
                setPhase(PHASES.COMPLETE);
                onComplete?.();
            }
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [onComplete]);

    // --- SFX SYSTEM ---
    const playSFX = (type) => {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.connect(gain);
            gain.connect(ctx.destination);

            const now = ctx.currentTime;

            if (type === 'startup') {
                // Low Rumble Riser
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(50, now);
                osc.frequency.exponentialRampToValueAtTime(200, now + 1.5);
                gain.gain.setValueAtTime(0, now);
                gain.gain.linearRampToValueAtTime(0.2, now + 0.5);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 2.0);
                osc.start(now);
                osc.stop(now + 2.0);
            } else if (type === 'ping') {
                // High Tech Ping
                osc.type = 'sine';
                osc.frequency.setValueAtTime(800, now);
                osc.frequency.exponentialRampToValueAtTime(1200, now + 0.1);
                gain.gain.setValueAtTime(0.1, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
                osc.start(now);
                osc.stop(now + 0.5);
            } else if (type === 'glitch') {
                // Data Glitch
                osc.type = 'square';
                osc.frequency.setValueAtTime(100, now);
                osc.frequency.linearRampToValueAtTime(500, now + 0.05);
                osc.frequency.linearRampToValueAtTime(50, now + 0.1);
                gain.gain.setValueAtTime(0.15, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
                osc.start(now);
                osc.stop(now + 0.2);
            } else if (type === 'reveal') {
                // Massive Bass Drop
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(100, now);
                osc.frequency.exponentialRampToValueAtTime(30, now + 1.0);
                gain.gain.setValueAtTime(0, now);
                gain.gain.linearRampToValueAtTime(0.5, now + 0.1);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 3.0);
                osc.start(now);
                osc.stop(now + 3.0);
            }
        } catch (e) {
            console.warn("Audio Context blocked or failed", e);
        }
    };

    const runCinematicSequence = async () => {
        // STAGE 1: SENSOR FUSION (2.5s)
        playSFX('startup');
        setLogs(["AGENTS WAKING UP...", "PHYSICS ENGINE: WARMING UP...", "GEOMETRY KERNEL: ACTIVE"]);
        await wait(2500);

        // STAGE 2: MULTI-MODEL DIMENSIONING (4.5s)
        setPhase(PHASES.MULTI_MODEL);
        setLogs(prev => []);

        // Cycle models
        playSFX('ping');
        setModelIndex(0); // Mavic 3 Pro
        await wait(2000);
        playSFX('ping');
        setModelIndex(1); // F-22
        await wait(1500);
        playSFX('ping');
        setModelIndex(2); // House
        await wait(1500);

        // STAGE 3: FLIGHT PATH (1.5s)
        setPhase(PHASES.FLIGHT_PATH);
        playSFX('glitch');
        await wait(1500);

        // STAGE 4: IGNITION (BRICK REVEAL) (5.0s)
        setPhase(PHASES.IGNITION);
        playSFX('reveal');
        await wait(5000); // Extended 3.5 -> 5.0

        // COMPLETE
        sessionStorage.setItem('brick_boot_v12_stable', 'true');
        setPhase(PHASES.COMPLETE);
        setTimeout(() => onComplete?.(), 1000);
    };

    const wait = (ms) => new Promise(r => setTimeout(r, ms));

    if (phase === PHASES.COMPLETE) return null;

    return (
        <AnimatePresence>
            <motion.div
                className="fixed inset-0 z-[9999] bg-[#050510] flex flex-col items-center justify-center overflow-hidden font-mono text-amber-500"
                initial={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 1.0 }}
            >
                {/* STAGE 1: SENSOR FUSION */}
                {phase === PHASES.SENSOR_FUSION && (
                    <motion.div className="relative w-full h-full flex items-center justify-center">
                        <motion.div
                            className="absolute w-[600px] h-[600px] rounded-full border border-amber-500/20"
                            style={{ background: 'conic-gradient(from 0deg, transparent 0deg, rgba(245, 158, 11, 0.1) 360deg)' }}
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        />
                        <div className='relative z-10 p-8 rounded-full border-2 border-amber-500/50 bg-black/40 backdrop-blur-md'>
                            <Radar size={64} className="text-amber-500 animate-pulse" />
                        </div>

                        {EQUATIONS.map((eq, i) => (
                            <motion.div
                                key={i}
                                className="absolute text-sm text-amber-300/80 font-bold tracking-widest"
                                initial={{
                                    x: (Math.random() - 0.5) * window.innerWidth * 0.6,
                                    y: (Math.random() - 0.5) * window.innerHeight * 0.6,
                                    opacity: 0, scale: 0.5, filter: 'blur(4px)'
                                }}
                                animate={{ opacity: [0, 1, 0], scale: 1.5, filter: 'blur(0px)' }}
                                transition={{ duration: 2.5, delay: i * 0.2, repeat: Infinity }}
                            >
                                {eq}
                            </motion.div>
                        ))}

                        <div className="absolute bottom-24 text-center w-full">
                            {logs.map((log, i) => (
                                <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="text-amber-500 font-bold tracking-[0.2em] text-sm mb-2 uppercase drop-shadow-[0_0_8px_rgba(245,158,11,0.8)]">
                                    {`> ${log}`}
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                )}

                {/* STAGE 2: MULTI-MODEL DIMENSIONING (3D CANVAS HIGH FIDELITY) */}
                {phase === PHASES.MULTI_MODEL && (
                    <motion.div className="flex flex-col items-center justify-center w-full h-full relative">
                        <div className="w-full h-[70vh] relative">
                            <Canvas gl={{ antialias: false, toneMapping: THREE.ReinhardToneMapping, toneMappingExposure: 1.5 }}>
                                {/* Centered Camera: Y=0 to look directly at target */}
                                <PerspectiveCamera makeDefault position={[0, 0, 10]} fov={40} />
                                <color attach="background" args={['#050510']} />

                                {/* Lighting */}
                                <ambientLight intensity={0.2} />
                                <pointLight position={[10, 10, 10]} intensity={2} color="#F59E0B" />
                                <spotLight position={[-10, 0, 5]} intensity={5} color="#EF4444" angle={0.5} penumbra={1} />
                                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

                                <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.2}>
                                    <group rotation={[0, Math.PI / 4, 0]}>
                                        {modelIndex === 0 && <Mavic3Pro />}
                                        {modelIndex === 1 && <F22Raptor />}
                                        {modelIndex === 2 && <ResidentialHouse />}
                                    </group>
                                </Float>
                            </Canvas>

                            {/* Overlaid Dimensions */}
                            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                                <AnimatePresence mode='wait'>
                                    {modelIndex === 0 && <Dimensions key="d" d1="RANGE: 28KM" d2="TIME: 43MIN" label="MAVIC 3 PRO" />}
                                    {modelIndex === 1 && <Dimensions key="p" d1="DRAG: 0.04" d2="WING: 13.5m" label="RAPTOR (F-22)" />}
                                    {modelIndex === 2 && <Dimensions key="h" d1="SQFT: 2.5k" d2="LOAD: SAFE" label="RESIDENTIAL" />}
                                </AnimatePresence>
                            </div>
                        </div>

                        <div className="mt-8 text-center">
                            <div className="text-xs text-amber-500/80 tracking-[0.5em] mb-2 drop-shadow-lg">COMPUTING TOPOLOGY...</div>
                            <div className="w-64 h-1 bg-amber-900 mx-auto rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-amber-500"
                                    initial={{ width: "0%" }}
                                    animate={{ width: "100%" }}
                                    transition={{ duration: 1.5, repeat: Infinity }}
                                />
                            </div>
                        </div>
                    </motion.div>
                )}

                {/* STAGE 3: FLIGHT PATH */}
                {phase === PHASES.FLIGHT_PATH && (
                    <motion.div className="relative w-full h-full flex items-center justify-center">
                        <Navigation size={64} className="text-amber-500 mb-8 drop-shadow-[0_0_15px_rgba(245,158,11,1)]" />
                        <svg className="absolute inset-0 w-full h-full pointer-events-none">
                            <defs>
                                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" style={{ stopColor: 'transparent' }} />
                                    <stop offset="50%" style={{ stopColor: '#F59E0B' }} />
                                    <stop offset="100%" style={{ stopColor: 'transparent' }} />
                                </linearGradient>
                            </defs>
                            <motion.path
                                d="M 0 500 C 400 100, 800 100, 1200 500"
                                fill="none"
                                stroke="url(#grad1)"
                                strokeWidth="4"
                                initial={{ pathLength: 0, opacity: 0 }}
                                animate={{ pathLength: 1, opacity: 1 }}
                                transition={{ duration: 1.2, ease: "easeInOut" }}
                            />
                        </svg>
                        <h2 className="text-3xl font-black tracking-[0.5em] text-white drop-shadow-[0_0_10px_rgba(245,158,11,0.8)]">TRAJECTORY LOCKED</h2>
                    </motion.div>
                )}

                {/* STAGE 4: REVEAL (The Massive BRICK) */}
                {phase === PHASES.IGNITION && (
                    <>
                        {/* Background Falling Bricks Canvas */}
                        <div className="absolute inset-0 z-0">
                            <Canvas camera={{ position: [0, 0, 20], fov: 50 }}>
                                <ambientLight intensity={0.5} />
                                <pointLight position={[10, 10, 10]} intensity={1} color="#F59E0B" />
                                <FallingBricks />
                            </Canvas>
                        </div>

                        <motion.div
                            className="flex flex-col items-center justify-center relative z-20"
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ type: "spring", bounce: 0.5 }}
                        >
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                                className="mb-8"
                            >
                                <BoxIcon size={120} className="text-amber-500 drop-shadow-[0_0_30px_rgba(245,158,11,0.8)]" strokeWidth={1.5} />
                            </motion.div>

                            <AnimatedTitle text="BRICK" />

                            <div className="flex items-center gap-6 text-amber-500 tracking-[0.6em] text-lg font-bold">
                                <span className="w-16 h-0.5 bg-amber-500 shadow-[0_0_10px_orange]"></span>
                                SYSTEM STANDBY
                                <span className="w-16 h-0.5 bg-amber-500 shadow-[0_0_10px_orange]"></span>
                            </div>
                        </motion.div>
                    </>
                )}

                {/* Persistent Footer */}
                <div className="absolute bottom-6 flex flex-col items-center gap-2 opacity-50 z-30">
                    <span className="text-[10px] text-amber-700 tracking-[0.5em]">BOUND REPRESENTATION INFERENCE COMPILATION KERNEL v1.0</span>
                </div>
            </motion.div>
        </AnimatePresence>
    );
};

const FallingBricks = () => {
    const count = 40;
    const meshRef = useRef();
    const dummy = useMemo(() => new THREE.Object3D(), []);

    // Initial random positions above screen
    const particles = useMemo(() => {
        const temp = [];
        for (let i = 0; i < count; i++) {
            const x = (Math.random() - 0.5) * 20; // Spread horizontally
            const y = 15 + Math.random() * 20;    // Start high up
            const z = (Math.random() - 0.5) * 10; // Depth
            const rot = [Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI];
            const speed = 0.1 + Math.random() * 0.2;
            // Pile target: x roughly same, y accumulates at bottom
            const targetY = -8 + (Math.random() * 3); // Floor level with some pile height
            temp.push({ x, y, z, rot, speed, targetY, landed: false });
        }
        return temp;
    }, []);

    useFrame(() => {
        if (!meshRef.current) return;

        particles.forEach((p, i) => {
            if (!p.landed) {
                p.y -= p.speed; // Fall
                p.rot[0] += 0.02; // Spin tumble
                p.rot[1] += 0.02;

                if (p.y <= p.targetY) {
                    p.y = p.targetY;
                    p.landed = true;
                    // Snap rotation to flat-ish? 
                    p.rot[0] = 0;
                    p.rot[2] = 0;
                }
            }

            dummy.position.set(p.x, p.y, p.z);
            dummy.rotation.set(p.rot[0], p.rot[1], p.rot[2]);
            dummy.scale.set(1, 1, 1);
            dummy.updateMatrix();
            meshRef.current.setMatrixAt(i, dummy.matrix);
        });
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={meshRef} args={[null, null, count]}>
            <boxGeometry args={[1.5, 0.8, 0.8]} /> {/* Brick dimensions */}
            <meshStandardMaterial color="#F59E0B" roughness={0.6} metalness={0.1} />
        </instancedMesh>
    );
};

const AnimatedTitle = ({ text }) => {
    // "Majestic Reveal" Animation - Slow, Smooth, Geometric
    const container = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.2, // Slower stagger
                delayChildren: 0.5
            }
        }
    };

    const child = {
        hidden: {
            opacity: 0,
            y: 40,
            scale: 1.2,
            filter: "blur(8px)"
        },
        visible: {
            opacity: 1,
            y: 0,
            scale: 1,
            filter: "blur(0px)",
            transition: {
                type: "spring",
                damping: 20, // Higher damping for less jitter
                stiffness: 100, // Softer spring
                mass: 1.5, // Heavier feel
                duration: 1.2 // Explicit duration fallback
            }
        }
    };

    return (
        <motion.div
            variants={container}
            initial="hidden"
            animate="visible"
            className="flex overflow-visible mb-6"
        >
            {text.split("").map((letter, index) => (
                <motion.span
                    key={index}
                    variants={child}
                    className="text-9xl font-black text-white font-sans tracking-wide mx-6 inline-block" // Increased margin to mx-6 for wider spacing
                    style={{
                        fontFamily: "'League Spartan', sans-serif",
                    }}
                >
                    {letter}
                </motion.span>
            ))}
        </motion.div>
    );
};

// --- HOLOGRAPHIC MODELS (PREMIUM) ---

const HolographicMaterial = ({ color = "#F59E0B", opacity = 0.3 }) => (
    <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        toneMapped={false}
    />
);

const WireframeMaterial = ({ color = "#F59E0B" }) => (
    <meshBasicMaterial color={color} wireframe />
);

// --- REALISTIC MATERIALS ---

const MetallicMaterial = ({ color = "#888888", roughness = 0.4, metalness = 0.8 }) => (
    <meshStandardMaterial
        color={color}
        roughness={roughness}
        metalness={metalness}
        envMapIntensity={1}
    />
);

const GlassMaterial = ({ color = "#00aaff", opacity = 0.3 }) => (
    <meshPhysicalMaterial
        color={color}
        transmission={0.6}
        opacity={opacity}
        transparent
        roughness={0}
        metalness={0.1}
        thickness={0.5}
        envMapIntensity={1}
    />
);

// --- DJI MAVIC 3 PRO IMPLEMENTATION ---
const Mavic3Pro = () => {
    const group = useRef();

    useFrame((state) => {
        group.current.rotation.y = state.clock.elapsedTime * 0.2;
        group.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    });

    const scale = 0.05;

    return (
        <group ref={group} scale={scale} rotation={[0, Math.PI, 0]}>
            {/* MAIN FUSELAGE */}
            <group>
                <mesh position={[0, 0, 0]} scale={[1, 1, 0.6]}>
                    <sphereGeometry args={[18, 32, 32]} />
                    <MetallicMaterial color="#333" roughness={0.5} />
                </mesh>
                {/* Rear Body */}
                <mesh position={[0, 0, 15]} scale={[0.9, 1, 0.6]}>
                    <sphereGeometry args={[16, 32, 16]} />
                    <MetallicMaterial color="#222" roughness={0.6} />
                </mesh>
                {/* Top Cover */}
                <mesh position={[0, 5, -5]} scale={[0.8, 0.5, 1.2]}>
                    <sphereGeometry args={[15, 32, 16]} />
                    <MetallicMaterial color="#111" roughness={0.4} />
                </mesh>
            </group>

            {/* GIMBAL CAMERA */}
            <group position={[0, -2, -22]}>
                <mesh castShadow receiveShadow>
                    <boxGeometry args={[25, 20, 25]} />
                    <MetallicMaterial color="#111" roughness={0.2} />
                </mesh>
                {/* Lenses */}
                <mesh position={[6, 0, -12]} rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[5, 5, 4]} />
                    <GlassMaterial color="#222" opacity={0.9} />
                </mesh>
                <mesh position={[-6, 4, -12]} rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[3, 3, 4]} />
                    <GlassMaterial color="#222" opacity={0.9} />
                </mesh>
                <mesh position={[-6, -4, -12]} rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[2, 2, 4]} />
                    <GlassMaterial color="#222" opacity={0.9} />
                </mesh>
            </group>

            {/* ARMS & MOTORS */}
            <DroneArm position={[-25, -5, -25]} rotation={[0, -0.5, 0]} label="FL" />
            <DroneArm position={[25, -5, -25]} rotation={[0, 0.5, 0]} label="FR" />
            <DroneArm position={[-25, 0, 25]} rotation={[0, 0.5, 0]} label="RL" />
            <DroneArm position={[25, 0, 25]} rotation={[0, -0.5, 0]} label="RR" />

            {/* SENSORS */}
            <mesh position={[0, 8, 0]}>
                <boxGeometry args={[10, 2, 10]} />
                <MetallicMaterial color="#orange" />
            </mesh>
        </group>
    );
};

const DroneArm = ({ position, rotation }) => {
    return (
        <group position={position} rotation={rotation}>
            {/* Arm Shaft */}
            <mesh position={[0, 0, 15]} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[2, 3, 40]} />
                <MetallicMaterial color="#444" />
            </mesh>

            {/* Motor Housing */}
            <group position={[0, 2, 35]}>
                <mesh>
                    <cylinderGeometry args={[8, 8, 12]} />
                    <MetallicMaterial color="#111" />
                </mesh>
                <Propeller />
            </group>

            {/* LED */}
            <mesh position={[0, -4, 35]}>
                <sphereGeometry args={[2]} />
                <meshBasicMaterial color="#00FF00" toneMapped={false} />
            </mesh>
        </group>
    )
}

const Propeller = () => {
    const ref = useRef();
    useFrame((state, delta) => {
        ref.current.rotation.y += delta * 20;
    });

    return (
        <group ref={ref} position={[0, 6, 0]}>
            <mesh>
                <cylinderGeometry args={[2, 2, 2]} />
                <MetallicMaterial color="#888" />
            </mesh>
            <mesh position={[18, 0, 0]} scale={[1, 0.1, 0.5]}>
                <sphereGeometry args={[18, 16, 16]} />
                <meshPhysicalMaterial color="#333" transparent opacity={0.8} transmission={0.2} blur={1} />
            </mesh>
            <mesh position={[-18, 0, 0]} scale={[1, 0.1, 0.5]}>
                <sphereGeometry args={[18, 16, 16]} />
                <meshPhysicalMaterial color="#333" transparent opacity={0.8} transmission={0.2} blur={1} />
            </mesh>
        </group>
    )
}


const F22Raptor = () => {
    const group = useRef();
    const scale = 0.00016;

    useFrame((state) => {
        group.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.3 + Math.PI;
        group.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
        group.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.05 + (Math.PI / 2);
    });

    // F-22 Geometry Construction (Approximated from Specs)
    // Use Shapes for flat components (Wings, Stabilizers)
    const wingShape = useMemo(() => {
        const shape = new THREE.Shape();
        shape.moveTo(0, 0);
        shape.lineTo(2000, 3000); // Leading Edge Sweep
        shape.lineTo(6780, -5600); // Tip
        shape.lineTo(6780, -6600); // Tip TE
        shape.lineTo(1000, -6000); // Root TE
        shape.lineTo(0, 0);
        return shape;
    }, []);

    const vStabShape = useMemo(() => {
        const shape = new THREE.Shape();
        shape.moveTo(0, 0);
        shape.lineTo(1200, -2000);
        shape.lineTo(2500, -2000);
        shape.lineTo(3500, 0);
        return shape;
    }, []);

    // Color Palette
    const stealthGray = "#7c858e";
    const darkNozzle = "#333333";
    const canopyGold = "#d4af37";

    return (
        <group ref={group} scale={scale}>
            <group rotation={[-Math.PI / 2, 0, 0]}>
                {/* 1. NOSE & FORWARD FUSELAGE */}
                {/* Diamond-like profile using scaled cylinder/cone */}
                <mesh position={[0, 3500, 0]} scale={[1, 0.6, 1]}>
                    <cylinderGeometry args={[1100, 200, 7000, 4]} /> {/* 4-sided for diamond profile */}
                    <MetallicMaterial color={stealthGray} roughness={0.6} />
                </mesh>

                {/* 2. COCKPIT CANOPY */}
                <mesh position={[0, 4800, 600]} scale={[0.7, 1.8, 0.7]}>
                    <sphereGeometry args={[800, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.5]} />
                    <GlassMaterial color={canopyGold} opacity={0.8} />
                </mesh>

                {/* 3. AIR INTAKES (Side Mounted) */}
                <group position={[1200, 7000, -400]}>
                    <mesh rotation={[0, 0, 0.3]}>
                        <boxGeometry args={[800, 3000, 800]} />
                        <MetallicMaterial color={stealthGray} />
                    </mesh>
                </group>
                <group position={[-1200, 7000, -400]}>
                    <mesh rotation={[0, 0, -0.3]}>
                        <boxGeometry args={[800, 3000, 800]} />
                        <MetallicMaterial color={stealthGray} />
                    </mesh>
                </group>

                {/* 4. MAIN BODY BLEND (Chines) */}
                <mesh position={[0, 10000, -200]} scale={[2.5, 1, 0.5]}>
                    <cylinderGeometry args={[1000, 1200, 8000, 32]} />
                    <MetallicMaterial color={stealthGray} />
                </mesh>

                {/* 5. AFT FUSELAGE / ENGINES HOUSING */}
                <mesh position={[0, 15000, -200]} scale={[2.2, 1, 0.6]}>
                    <boxGeometry args={[1200, 5000, 1000]} />
                    <MetallicMaterial color={stealthGray} />
                </mesh>
            </group>

            {/* WINGS */}
            <group position={[0, 0, 0]}>
                <mesh position={[800, 6000, -200]} rotation={[0, 0, 0]}>
                    <extrudeGeometry args={[wingShape, { depth: 150, bevelEnabled: true, bevelSize: 50, bevelThickness: 50 }]} />
                    <MetallicMaterial color={stealthGray} />
                </mesh>
                <mesh position={[-800, 6000, -200]} rotation={[0, Math.PI, 0]}>
                    <extrudeGeometry args={[wingShape, { depth: 150, bevelEnabled: true, bevelSize: 50, bevelThickness: 50 }]} />
                    <MetallicMaterial color={stealthGray} />
                </mesh>
            </group>

            {/* VERTICAL STABILIZERS (Canted) */}
            <group position={[0, 14000, 500]}>
                <mesh position={[1200, 0, 0]} rotation={[Math.PI / 2, -0.47, 0]}>
                    <extrudeGeometry args={[vStabShape, { depth: 100, bevelEnabled: true, bevelSize: 20 }]} />
                    <MetallicMaterial color={stealthGray} />
                </mesh>
                <mesh position={[-1200, 0, 0]} rotation={[Math.PI / 2, 0.47, 0]}>
                    <extrudeGeometry args={[vStabShape, { depth: 100, bevelEnabled: true, bevelSize: 20 }]} />
                    <MetallicMaterial color={stealthGray} />
                </mesh>
            </group>

            {/* HORIZONTAL STABILIZERS */}
            <group position={[0, 16000, 0]}>
                <mesh position={[1500, 0, 0]} rotation={[Math.PI / 2, 0, -0.2]}>
                    <boxGeometry args={[2000, 2000, 100]} />
                    <MetallicMaterial color={stealthGray} />
                </mesh>
                <mesh position={[-1500, 0, 0]} rotation={[Math.PI / 2, 0, 0.2]}>
                    <boxGeometry args={[2000, 2000, 100]} />
                    <MetallicMaterial color={stealthGray} />
                </mesh>
            </group>

            {/* ENGINE NOZZLES (2D Vectoring) */}
            <group>
                <mesh position={[500, 18200, -200]} rotation={[Math.PI / 2, 0, 0]}>
                    <boxGeometry args={[800, 400, 1200]} />
                    <MetallicMaterial color={darkNozzle} roughness={0.8} />
                </mesh>
                <mesh position={[-500, 18200, -200]} rotation={[Math.PI / 2, 0, 0]}>
                    <boxGeometry args={[800, 400, 1200]} />
                    <MetallicMaterial color={darkNozzle} roughness={0.8} />
                </mesh>
            </group>
        </group>
    );
}

const ResidentialHouse = () => {
    const group = useRef();

    // Scale factor: Optimized to be ~1.5x the Drone size
    // Real length 3050mm -> Target 3.0 units -> Scale 0.001
    const scale = 0.001;

    useFrame((state) => {
        group.current.rotation.y = state.clock.elapsedTime * 0.1;
    });

    const roofShape = useMemo(() => {
        // Gable roof profile
        // House Width ~2033mm. Pitch 6/12 (~26.5 deg).
        // Ridge height = (2033/2) * (6/12) = 508mm.
        const halfWidth = 1016;
        const height = 508;

        const shape = new THREE.Shape();
        shape.moveTo(-halfWidth - 300, 0); // Overhang left
        shape.lineTo(0, height + 50); // Ridge
        shape.lineTo(halfWidth + 300, 0); // Overhang right
        return shape;
    }, []);

    return (
        <group ref={group} scale={scale} position={[0, -1.35, 0]}>
            {/* FOUNDATION */}
            <mesh position={[0, -150, 0]}>
                <boxGeometry args={[2033 + 50, 305, 3050 + 50]} />
                <HolographicMaterial color="#555" opacity={0.6} />
                <Edges color="#777" />
            </mesh>

            {/* MAIN STRUCTURE (1st Floor) */}
            <mesh position={[0, 228, 0]}>
                <boxGeometry args={[2033, 457, 3050]} />
                <HolographicMaterial color="#F59E0B" opacity={0.2} />
                <Edges color="#F59E0B" />
            </mesh>

            {/* ROOF SYSTEM */}
            <group position={[0, 457, 0]}>
                <mesh position={[0, 0, -1825]}>
                    <extrudeGeometry args={[roofShape, { depth: 3650, bevelEnabled: false }]} />
                    <HolographicMaterial color="#444" opacity={0.5} />
                    <Edges color="#EF4444" />
                </mesh>
            </group>

            {/* GARAGE (Attached Side) */}
            <group position={[1016 + 457, 0, 800]}>
                <mesh position={[0, 228, 0]}>
                    <boxGeometry args={[914, 457, 1016]} />
                    <HolographicMaterial color="#888" opacity={0.3} />
                    <Edges color="#AAA" />
                </mesh>
            </group>

            {/* CHIMNEY */}
            <mesh position={[600, 800, -800]}>
                <boxGeometry args={[150, 1000, 150]} />
                <HolographicMaterial color="#EF4444" opacity={0.6} />
                <Edges color="#EF4444" />
            </mesh>

            {/* INTERIOR HIGHLIGHTS */}
            <group>
                <mesh position={[500, 100, 1000]}>
                    <boxGeometry args={[400, 200, 100]} />
                    <meshBasicMaterial color="#00FFFF" transparent opacity={0.3} />
                </mesh>
                <mesh position={[-500, 50, -1000]}>
                    <boxGeometry args={[300, 100, 400]} />
                    <meshBasicMaterial color="#00FF00" transparent opacity={0.3} />
                </mesh>
            </group>
        </group>
    )
}

// --- UI OVERLAYS ---

const Dimensions = ({ d1, d2, label }) => (
    <motion.div
        className="relative w-80 h-80 pointer-events-none"
        initial={{ opacity: 0, scale: 1.1 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 1.05, filter: 'blur(10px)' }}
        transition={{ duration: 0.5 }}
    >
        {/* Label */}
        <div className="absolute top-10 w-full text-center">
            <h2 className="text-3xl font-black text-amber-500 tracking-[0.2em] drop-shadow-[0_0_10px_rgba(245,158,11,1)]">{label}</h2>
        </div>

        {/* Targeting Reticle */}
        <div className="absolute inset-0 border border-amber-500/30 rounded-lg"></div>
        <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-amber-500"></div>
        <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-amber-500"></div>
        <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-amber-500"></div>
        <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-amber-500"></div>

        {/* Dim lines */}
        <div className="absolute -left-16 top-1/2 -translate-y-1/2 flex items-center">
            <div className="h-px w-12 bg-amber-500 shadow-[0_0_5px_orange]"></div>
            <span className="ml-2 text-xs text-amber-500 font-mono font-bold bg-black/60 px-2 py-1 border border-amber-500/50">{d1}</span>
        </div>
        <div className="absolute -right-16 top-1/2 -translate-y-1/2 flex items-center flex-row-reverse">
            <div className="h-px w-12 bg-amber-500 shadow-[0_0_5px_orange]"></div>
            <span className="mr-2 text-xs text-amber-500 font-mono font-bold bg-black/60 px-2 py-1 border border-amber-500/50">{d2}</span>
        </div>
    </motion.div>
);

export default BootSequence;
