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

    const runCinematicSequence = async () => {
        // STAGE 1: SENSOR FUSION (2.5s)
        setLogs(["AGENTS WAKING UP...", "PHYSICS ENGINE: WARMING UP...", "GEOMETRY KERNEL: ACTIVE"]);
        await wait(2500);

        // STAGE 2: MULTI-MODEL DIMENSIONING (4.5s)
        setPhase(PHASES.MULTI_MODEL);
        setLogs(prev => []);

        // Cycle models
        setModelIndex(0); // Mavic 3 Pro
        await wait(2000); // Extended for detail
        setModelIndex(1); // Plane
        await wait(1500);
        setModelIndex(2); // House
        await wait(1500);

        // STAGE 3: FLIGHT PATH (1.5s)
        setPhase(PHASES.FLIGHT_PATH);
        await wait(1500);

        // STAGE 4: IGNITION (BRICK REVEAL) (2.5s)
        setPhase(PHASES.IGNITION);
        await wait(2500);

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
                                        {modelIndex === 1 && <HoloPlane />}
                                        {modelIndex === 2 && <HoloHouse />}
                                    </group>
                                </Float>

                                {/* POST-PROCESSING REMOVED FOR STABILITY */}
                                {/* Glow is now handled via AdditiveBlending on materials */}
                            </Canvas>

                            {/* Overlaid Dimensions */}
                            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                                <AnimatePresence mode='wait'>
                                    {modelIndex === 0 && <Dimensions key="d" d1="RANGE: 28KM" d2="TIME: 43MIN" label="MAVIC 3 PRO" />}
                                    {modelIndex === 1 && <Dimensions key="p" d1="DRAG: 0.04" d2="WING: 14m" label="AERO AGENT" />}
                                    {modelIndex === 2 && <Dimensions key="h" d1="AREA: 2.4k" d2="LOAD: STAT" label="STRUCT AGENT" />}
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

                        <h1 className="text-9xl font-black tracking-tighter text-white mb-6 drop-shadow-[0_0_40px_rgba(245,158,11,0.6)]">
                            BRICK
                        </h1>

                        <div className="flex items-center gap-6 text-amber-500 tracking-[0.6em] text-lg font-bold">
                            <span className="w-16 h-0.5 bg-amber-500 shadow-[0_0_10px_orange]"></span>
                            SYSTEM STANDBY
                            <span className="w-16 h-0.5 bg-amber-500 shadow-[0_0_10px_orange]"></span>
                        </div>
                    </motion.div>
                )}

                {/* Persistent Footer */}
                <div className="absolute bottom-6 flex flex-col items-center gap-2 opacity-50">
                    <span className="text-[10px] text-amber-700 tracking-[0.5em]">ARES AEROSPACE KERNEL v10.0</span>
                </div>
            </motion.div>
        </AnimatePresence>
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

// --- DJI MAVIC 3 PRO IMPLEMENTATION ---
const Mavic3Pro = () => {
    const group = useRef();

    useFrame((state) => {
        group.current.rotation.y = state.clock.elapsedTime * 0.2;
        group.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    });

    // Scale down from mm to Three.js units (approx 1 unit = 50mm)
    const scale = 0.05;

    // Dimensions from spec (unfolded)
    // 347.5 x 290.8 x 107.7 mm

    return (
        <group ref={group} scale={scale} rotation={[0, Math.PI, 0]}>
            {/* MAIN FUSELAGE */}
            <group>
                <mesh position={[0, 0, 0]} scale={[1, 1, 0.6]}>
                    <sphereGeometry args={[18, 32, 32]} />
                    <HolographicMaterial color="#333" opacity={0.8} />
                    <Edges color="#F59E0B" threshold={15} />
                </mesh>
                {/* Rear Body */}
                <mesh position={[0, 0, 15]} scale={[0.9, 1, 0.6]}>
                    <sphereGeometry args={[16, 32, 16]} />
                    <HolographicMaterial color="#222" opacity={0.8} />
                    <Edges color="#555" />
                </mesh>
                {/* Top Cover */}
                <mesh position={[0, 5, -5]} scale={[0.8, 0.5, 1.2]}>
                    <sphereGeometry args={[15, 32, 16]} />
                    <HolographicMaterial color="#111" />
                    <Edges color="#F59E0B" />
                </mesh>
            </group>

            {/* GIMBAL CAMERA (The "Eye") */}
            <group position={[0, -2, -22]}>
                <mesh castShadow receiveShadow>
                    <boxGeometry args={[25, 20, 25]} />
                    <HolographicMaterial color="#000" />
                    <Edges color="#EF4444" />
                </mesh>
                {/* Lenses */}
                <mesh position={[6, 0, -12]} rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[5, 5, 4]} />
                    <meshBasicMaterial color="#00FFFF" />
                </mesh>
                <mesh position={[-6, 4, -12]} rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[3, 3, 4]} />
                    <meshBasicMaterial color="#00FFFF" />
                </mesh>
                <mesh position={[-6, -4, -12]} rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[2, 2, 4]} />
                    <meshBasicMaterial color="#00FFFF" />
                </mesh>
            </group>

            {/* ARMS & MOTORS */}
            <DroneArm position={[-25, -5, -25]} rotation={[0, -0.5, 0]} label="FL" />
            <DroneArm position={[25, -5, -25]} rotation={[0, 0.5, 0]} label="FR" />
            <DroneArm position={[-25, 0, 25]} rotation={[0, 0.5, 0]} label="RL" />
            <DroneArm position={[25, 0, 25]} rotation={[0, -0.5, 0]} label="RR" />

            {/* SENSORS & DETAILS */}
            <mesh position={[0, 8, 0]}>
                <boxGeometry args={[10, 2, 10]} />
                <meshBasicMaterial color="#F59E0B" /> {/* GPS module */}
            </mesh>
        </group>
    );
};

const DroneArm = ({ position, rotation, label }) => {
    return (
        <group position={position} rotation={rotation}>
            {/* Arm Shaft */}
            <mesh position={[0, 0, 15]} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[2, 3, 40]} />
                <HolographicMaterial color="#444" opacity={0.6} />
                <Edges color="#444" />
            </mesh>

            {/* Motor Housing */}
            <group position={[0, 2, 35]}>
                <mesh>
                    <cylinderGeometry args={[8, 8, 12]} />
                    <HolographicMaterial color="#222" />
                    <Edges color="#F59E0B" />
                </mesh>
                {/* Propeller - Spinning */}
                <Propeller />
            </group>

            {/* Landing Gear / LED */}
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
        ref.current.rotation.y += delta * 20; // Fast spin
    });

    return (
        <group ref={ref} position={[0, 6, 0]}>
            {/* Hub */}
            <mesh>
                <cylinderGeometry args={[2, 2, 2]} />
                <meshStandardMaterial color="#888" />
            </mesh>
            {/* Blade 1 */}
            <mesh position={[18, 0, 0]} scale={[1, 0.1, 0.5]}>
                <sphereGeometry args={[18, 16, 16]} />
                <HolographicMaterial color="#F59E0B" opacity={0.2} />
            </mesh>
            {/* Blade 2 */}
            <mesh position={[-18, 0, 0]} scale={[1, 0.1, 0.5]}>
                <sphereGeometry args={[18, 16, 16]} />
                <HolographicMaterial color="#F59E0B" opacity={0.2} />
            </mesh>
            {/* Motion Blur Disk */}
            <mesh rotation={[Math.PI / 2, 0, 0]}>
                <circleGeometry args={[36, 32]} />
                <meshBasicMaterial color="#F59E0B" transparent opacity={0.05} side={THREE.DoubleSide} />
            </mesh>
        </group>
    )
}


const HoloPlane = () => {
    const group = useRef();
    useFrame((state) => {
        group.current.rotation.z = Math.sin(state.clock.elapsedTime) * 0.2;
        group.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    });

    return (
        <group ref={group} scale={1.2}>
            {/* Sleek Delta Body */}
            <mesh rotation={[Math.PI / 2, 0, 0]}>
                <coneGeometry args={[0.5, 4, 4]} />
                <HolographicMaterial color="#111" />
                <Edges color="#EF4444" threshold={10} />
            </mesh>

            {/* Holographic Wings */}
            <mesh position={[0, 0, 0.5]} rotation={[Math.PI / 2, 0, 0]}>
                <bufferGeometry>
                    <float32BufferAttribute attach="attributes-position" count={3} array={new Float32Array([
                        -2, -1, 0,
                        2, -1, 0,
                        0, 1, 0
                    ])} itemSize={3} />
                </bufferGeometry>
                <meshBasicMaterial color="#F59E0B" side={THREE.DoubleSide} wireframe transparent opacity={0.3} />
            </mesh>

            {/* Engine Glow */}
            <mesh position={[0, 0, 2.1]} rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[0.3, 0.05, 16, 32]} />
                <meshBasicMaterial color="#00FFFF" toneMapped={false} />
            </mesh>
        </group>
    )
}

const HoloHouse = () => {
    const group = useRef();
    useFrame((state) => {
        group.current.rotation.y += 0.005;
    });

    return (
        <group ref={group} scale={0.8}>
            {/* Wireframe Modern Structure */}
            <mesh position={[0, 0, 0]}>
                <boxGeometry args={[2, 2, 2]} />
                <HolographicMaterial color="#3B82F6" />
                <Edges color="#3B82F6" />
            </mesh>

            {/* Internal Core */}
            <mesh>
                <boxGeometry args={[1.5, 1.5, 1.5]} />
                <HolographicMaterial color="#000" />
                <Edges color="#F59E0B" />
            </mesh>

            {/* Floating Roof Segments */}
            <mesh position={[0, 1.5, 0]}>
                <coneGeometry args={[1.8, 1, 4]} />
                <HolographicMaterial color="#333" opacity={0.2} />
                <Edges color="#EF4444" />
            </mesh>
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
