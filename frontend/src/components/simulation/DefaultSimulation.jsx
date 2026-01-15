import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows, Edges, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../../contexts/ThemeContext';
import { ASSET_REGISTRY } from '../../utils/assetRegistry';
import HyperRealismEffects from './HyperRealismEffects';
import ProceduralEngine from './ProceduralEngine';
import Thermometer3D from './Thermometer3D';
import AerodynamicsView from './AerodynamicsView';
import { useSettings } from '../../contexts/SettingsContext';
import { useSimulation } from '../../contexts/SimulationContext';
import { useDesign } from '../../contexts/DesignContext';
import { OpenSCADMesh } from './OpenSCADMesh';
import VMKRenderer from './VMKRenderer';

/**
 * The Symbolic Mesh component.
 * Instead of loading a file, it interprets the compiled B-Rep data.
 * Uses manual animation logic for maximum stability in the reconciler.
 */
const PartRenderer = ({ asset, viewMode = 'realistic' }) => {
    if (!asset) return null;

    const renderMaterial = (originalMaterial) => {
        const baseProps = {
            color: viewMode === 'solid' ? '#888888' : originalMaterial?.color || '#cccccc',
            transparent: viewMode === 'xray',
            opacity: viewMode === 'xray' ? 0.3 : 1,
            side: viewMode === 'interior' ? THREE.BackSide : THREE.DoubleSide
        };

        switch (viewMode) {
            // --- Standard Modes ---
            case 'realistic':
                // Standard PBR
                return <meshStandardMaterial
                    {...baseProps}
                    roughness={originalMaterial?.roughness ?? 0.5}
                    metalness={originalMaterial?.metalness ?? 0.5}
                />;
            case 'matte':
                return <meshStandardMaterial {...baseProps} roughness={1} metalness={0} />;
            case 'wireframe':
                return <meshBasicMaterial {...baseProps} wireframe color="#00ff88" />;
            case 'heatmap':
                // In simulation, this should react to temp. For now, visual placeholder.
                // Using Normal material as base for colorful "heat" look
                return <meshNormalMaterial />;
            case 'hyperrealism':
                return <meshPhysicalMaterial {...baseProps} roughness={0} metalness={1} clearcoat={1} clearcoatRoughness={0} />;
            case 'solid':
                return <meshStandardMaterial {...baseProps} color="#888888" />;
            case 'interior':
                return <meshStandardMaterial {...baseProps} side={THREE.BackSide} color="#ffaa00" />;
            case 'shaded':
                return <meshPhongMaterial {...baseProps} shininess={100} />;
            case 'xray':
                return <meshStandardMaterial {...baseProps} transparent opacity={0.3} depthWrite={false} />;
            case 'hidden_line':
                return <meshBasicMaterial color="#1e1e1e" polygonOffset polygonOffsetFactor={1} />;

            // --- Simulation Modes ---
            case 'stress':
                // Visualizing stress points - simplified as red/orange gradient
                return <meshStandardMaterial color="#ef4444" wireframe={true} emissive="#7f1d1d" />;
            case 'flow':
                // Visualizing aerodynamics - translucent with wireframe overlay
                return <meshPhysicalMaterial color="#3b82f6" transmission={0.6} active={true} clearcoat={1} />;

            default:
                return <meshStandardMaterial {...baseProps} />;
        }
    };

    const renderGeometry = (type, args) => {
        switch (type) {
            case 'box': return <boxGeometry args={args} />;
            case 'sphere': return <sphereGeometry args={args} />;
            case 'cylinder': return <cylinderGeometry args={args} />;
            case 'cone': return <coneGeometry args={args} />;
            case 'torus': return <torusGeometry args={args} />;
            default: return <boxGeometry args={[1, 1, 1]} />;
        }
    };

    if (asset.type === 'primitive') {
        return (
            <mesh castShadow receiveShadow>
                {renderGeometry(asset.geometry, asset.args)}
                {renderMaterial(asset.material)}
                {viewMode === 'hidden_line' && <Edges color="white" threshold={15} />}
            </mesh>
        );
    } else if (asset.type === 'composition') {
        return (
            <group>
                {asset.elements.map((el, idx) => (
                    <mesh
                        key={idx}
                        position={el.position || [0, 0, 0]}
                        rotation={el.rotation || [0, 0, 0]}
                        castShadow
                        receiveShadow
                    >
                        {renderGeometry(el.geometry, el.args)}
                        {renderMaterial({ color: el.color })}
                        {viewMode === 'hidden_line' && <Edges color="white" threshold={15} />}
                    </mesh>
                ))}
            </group>
        );
    }
    return null;
};

const PhysicalObject = ({ geometryData, isRunning = false, theme, kclSource, viewMode, activeAsset }) => {
    const meshRef = useRef();
    const groupRef = useRef();
    const { physState } = useSimulation();

    // Physics-driven motion using real state data
    useFrame((state) => {
        if (groupRef.current) {
            if (isRunning && physState) {
                // Use real position from physics engine
                if (physState.position) {
                    groupRef.current.position.x = THREE.MathUtils.lerp(
                        groupRef.current.position.x,
                        physState.position.x,
                        0.1
                    );
                    groupRef.current.position.y = THREE.MathUtils.lerp(
                        groupRef.current.position.y,
                        physState.position.y + 0.5, // Offset for visual center
                        0.1
                    );
                    groupRef.current.position.z = THREE.MathUtils.lerp(
                        groupRef.current.position.z,
                        physState.position.z,
                        0.1
                    );
                }

                // Use real orientation from physics engine
                if (physState.orientation) {
                    groupRef.current.rotation.x = THREE.MathUtils.lerp(
                        groupRef.current.rotation.x,
                        physState.orientation.pitch || 0,
                        0.1
                    );
                    groupRef.current.rotation.y = THREE.MathUtils.lerp(
                        groupRef.current.rotation.y,
                        physState.orientation.yaw || 0,
                        0.1
                    );
                    groupRef.current.rotation.z = THREE.MathUtils.lerp(
                        groupRef.current.rotation.z,
                        physState.orientation.roll || 0,
                        0.1
                    );
                }
            } else {
                // Smoothly return to origin when idle
                groupRef.current.position.x = THREE.MathUtils.lerp(groupRef.current.position.x, 0, 0.05);
                groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, 0.5, 0.05);
                groupRef.current.position.z = THREE.MathUtils.lerp(groupRef.current.position.z, 0, 0.05);
                groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, 0, 0.05);
                groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, 0, 0.05);
                groupRef.current.rotation.z = THREE.MathUtils.lerp(groupRef.current.rotation.z, 0, 0.05);
            }
        }
    });

    // Detect if source is OpenSCAD code
    const isOpenSCAD = kclSource && (
        kclSource.includes('union()') ||
        kclSource.includes('difference()') ||
        kclSource.includes('cylinder(') ||
        kclSource.includes('cube(') ||
        kclSource.includes('sphere(') ||
        kclSource.includes('$fn')
    );

    console.log('PhysicalObject - isOpenSCAD:', isOpenSCAD, 'kclSource length:', kclSource?.length);

    return (
        <group ref={groupRef}>
            <group position={[0, 0.5, 0]}>
                {geometryData ? (
                    <primitive object={geometryData} />
                ) : isOpenSCAD ? (
                    <OpenSCADMesh scadCode={kclSource} viewMode={viewMode} theme={theme} />
                ) : activeAsset ? (
                    activeAsset.type === 'procedural' ? (
                        <ProceduralEngine viewMode={viewMode} config={activeAsset.config} />
                    ) : (
                        <PartRenderer asset={activeAsset} viewMode={viewMode} />
                    )
                ) : (
                    // Legacy Fallback - Now routed through PartRenderer for consistent View Modes
                    <PartRenderer
                        asset={{
                            type: 'primitive',
                            geometry: kclSource && kclSource.includes('boxGeometry') ? 'box' :
                                kclSource && kclSource.includes('sphereGeometry') ? 'sphere' : 'cylinder',
                            args: kclSource && kclSource.includes('boxGeometry') ? [2, 2, 2] :
                                kclSource && kclSource.includes('sphereGeometry') ? [1.5, 32, 32] : [1.2, 1.2, 2.8, 64],
                            material: {
                                color: theme.colors.text.muted,
                                roughness: 0.2,
                                metalness: 0.8
                            }
                        }}
                        viewMode={viewMode}
                    />
                )}
            </group>

            {/* Visual indicator for rotation/physics center */}
            <mesh position={[0, -0.9, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[1.4, 1.5, 64]} />
                <meshBasicMaterial
                    color={theme.colors.accent.primary}
                    transparent
                    opacity={0.3}
                    side={THREE.DoubleSide}
                />
            </mesh>
        </group>
    );
};

// --- Generic Device Assembly Renderer ---
const DeviceAssembly = ({ isExploded, viewMode, theme, parts = [] }) => {
    return (
        <group>
            {parts.map((part, i) => (
                <ExplodingPart
                    key={part.id || i}
                    part={part}
                    isExploded={isExploded}
                    viewMode={viewMode}
                    theme={theme}
                />
            ))}
        </group>
    );
};

const ExplodingPart = ({ part, isExploded, viewMode, theme }) => {
    const meshRef = useRef();
    // Use springs for translation would be ideal, but for now raw lerp for control

    useFrame((state, delta) => {
        if (!meshRef.current) return;

        // Target Position
        const targetX = isExploded ? part.vector[0] : 0;
        const targetY = isExploded ? part.vector[1] : (part.id === 'chassis_btm' ? 0 : 0.1 * (state.clock.elapsedTime % 1)); // Stacked when assembled
        const targetZ = isExploded ? part.vector[2] : 0;

        // Simple stack logic for assembled state to avoid Z-fighting
        const stackY = {
            'chassis_btm': 0, 'battery': 0.25, 'mainboard': 0.45,
            'fan_left': 0.6, 'fan_right': 0.6, 'keyboard': 0.8, 'screen': 0.95
        }[part.id] || 0;

        const finalY = isExploded ? part.vector[1] : stackY;

        // Smooth Translation
        meshRef.current.position.x = THREE.MathUtils.lerp(meshRef.current.position.x, isExploded ? part.vector[0] : 0, 0.1);
        meshRef.current.position.y = THREE.MathUtils.lerp(meshRef.current.position.y, finalY, 0.1);
        meshRef.current.position.z = THREE.MathUtils.lerp(meshRef.current.position.z, isExploded ? part.vector[2] : 0, 0.1);

        // Rotation Logic: Spin on local axis when exploded
        if (isExploded) {
            meshRef.current.rotation.y += 0.01; // Slow spin
            meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime) * 0.05; // Gentle tilt
        } else {
            // Reset rotation
            meshRef.current.rotation.set(0, 0, 0);
        }
    });

    return (
        <group>
            {/* The Part */}
            <mesh ref={meshRef} castShadow receiveShadow>
                {part.type === 'box' ? <boxGeometry args={part.args} /> : <cylinderGeometry args={part.args} />}
                {/* Re-use generic renderMaterial logic ideally, but mocking here for speed */}
                <meshStandardMaterial
                    color={part.color}
                    roughness={0.3}
                    metalness={0.8}
                    wireframe={viewMode === 'wireframe'}
                />
                <Edges color="#444" threshold={15} />
            </mesh>

            {/* Guide Line (Only visible when exploded and moved) */}
            {isExploded && (part.vector[0] !== 0 || part.vector[1] !== 0 || part.vector[2] !== 0) && (
                <line>
                    <bufferGeometry>
                        <float32BufferAttribute
                            attach="attributes-position"
                            array={new Float32Array([0, 0, 0, part.vector[0], part.vector[1], part.vector[2]])}
                            count={2}
                            itemSize={3}
                        />
                    </bufferGeometry>
                    <lineBasicMaterial color={theme.colors.accent.primary} transparent opacity={0.3} />
                </line>
            )}
        </group>
    );
};

/**
 * Visualizes Physics Oracle Results (Orbits, Fields, etc.)
 */
const OrbitVisualizer = ({ data, theme }) => {
    const { domain, method } = data;

    // ASTROPHYSICS: Orbit & Transfer Visualization
    if (domain === 'ASTROPHYSICS') {
        const isTransfer = method.includes('Hohmann');

        // Scale: 1 AU = 10 World Units (if Transfer)
        // Scale: 1 Earth Radius = 2 World Units (if Orbit)
        const scale = isTransfer ? 10 : 2;

        if (isTransfer) {
            const r1 = (data.origin_au || 1.0) * scale;
            const r2 = (data.destination_au || 1.5) * scale;

            // Transfer Ellipse (Semi-major axis a = (r1+r2)/2)
            // Center is at focus... wait. Sun is focus.
            // Perihelion at r1. Aphelion at r2.
            // Major Axis = r1 + r2. a = (r1+r2)/2.
            // Center of ellipse is shifted from Sun by c = a - r1.
            const a = (r1 + r2) / 2;
            const c = a - r1;

            // Ellipse Curve
            const curve = new THREE.EllipseCurve(
                -c, 0,            // ax, aY (Center)
                a, a * 0.8,       // xRadius, yRadius (approx minor axis)
                0, Math.PI,       // StartAngle, EndAngle (Half orbit)
                false,            // Clockwise
                0                 // Rotation
            );
            const points = curve.getPoints(50);

            return (
                <group>
                    {/* Sun */}
                    <mesh>
                        <sphereGeometry args={[1, 32, 32]} />
                        <meshBasicMaterial color="#fbbf24" />
                    </mesh>

                    {/* Origin Orbit (Green) */}
                    <mesh rotation={[Math.PI / 2, 0, 0]}>
                        <ringGeometry args={[r1 - 0.05, r1 + 0.05, 64]} />
                        <meshBasicMaterial color="#4ade80" side={THREE.DoubleSide} transparent opacity={0.3} />
                    </mesh>

                    {/* Destination Orbit (Red) */}
                    <mesh rotation={[Math.PI / 2, 0, 0]}>
                        <ringGeometry args={[r2 - 0.05, r2 + 0.05, 64]} />
                        <meshBasicMaterial color="#f87171" side={THREE.DoubleSide} transparent opacity={0.3} />
                    </mesh>

                    {/* Transfer Trajectory (Yellow Line) */}
                    <Line
                        points={points}
                        color="#fbbf24"
                        lineWidth={2}
                        dashed
                        dashScale={1}
                        rotation={[Math.PI / 2, 0, 0]}
                    />

                    {/* Text Label (billboard) */}
                    <group position={[r2, 0, 0]}>
                        <mesh>
                            <sphereGeometry args={[0.2]} />
                            <meshBasicMaterial color="#f87171" />
                        </mesh>
                    </group>
                </group>
            );
        } else {
            // Simple Orbit (GEO etc)
            const r = 5; // Fixed visual radius for single orbit
            return (
                <group>
                    <mesh>
                        <sphereGeometry args={[2, 32, 32]} />
                        <meshStandardMaterial color="#3b82f6" roughness={0.6} />
                    </mesh>
                    <Line
                        points={new THREE.EllipseCurve(0, 0, r, r, 0, 2 * Math.PI, false, 0).getPoints(64)}
                        color="#ffffff"
                        lineWidth={1}
                        rotation={[Math.PI / 2, 0, 0]}
                    />
                </group>
            );
        }
    }

    // THERMODYNAMICS: Heat Map (Simple)
    if (domain === 'THERMODYNAMICS') {
        // Show a glowing radiator panel
        const area = data.required_area_m2 || 100;
        const side = Math.sqrt(area);
        const scale = Math.min(side / 10, 5); // visually clamp

        return (
            <group>
                <mesh>
                    <boxGeometry args={[scale, 0.1, scale]} />
                    <meshStandardMaterial
                        color="#ef4444"
                        emissive="#ef4444"
                        emissiveIntensity={2}
                        toneMapped={false}
                    />
                </mesh>
            </group>
        );
    }

    return null;
};

/**
 * Motion Trail Visualization
 * Shows the path traveled by the locomotive
 */
const MotionTrail = ({ trail, theme }) => {
    const points = useMemo(() => {
        if (!trail || trail.length < 2) return null;

        return trail.map(p => new THREE.Vector3(p.x, p.y + 0.5, p.z));
    }, [trail]);

    if (!points || points.length < 2) return null;

    const curve = useMemo(() => new THREE.CatmullRomCurve3(points), [points]);

    return (
        <mesh>
            <tubeGeometry args={[curve, 64, 0.02, 8, false]} />
            <meshBasicMaterial
                color={theme.colors.accent.primary}
                opacity={0.4}
                transparent
            />
        </mesh>
    );
};

/**
 * High-Fidelity 3D Simulation Bay.
 * Implements a formal B-Rep buffer with physics-aligned lighting.
 * Built with a simplified reconciler path to ensure stability across environments.
 */
export default function DefaultSimulation({ isRunning = false, kclSource = "", viewMode = "realistic", isExploded = false, physicsResult, showGrid = true }) {
    const { theme } = useTheme();
    const { show3DThermometer } = useSettings();
    const { testParams, physState, motionTrail } = useSimulation();
    const { activeTab } = useDesign();

    const isUntitled = activeTab?.name?.startsWith('Untitled Design');
    // Show 3D thermometer if setting is enabled OR if a thermal test is running
    const showTemp = show3DThermometer || (activeTab?.name === 'Warp Core Prototype.brick');

    // Use prop source or fall back to active tab code
    const actualSource = kclSource || activeTab?.content || "";

    // Validate if source contains meaningful code
    const hasValidCode = React.useMemo(() => {
        if (!actualSource || actualSource.trim().length < 10) return false;

        // Check for valid code patterns
        const validPatterns = [
            /\{.*\}/s,                    // JSON objects
            /<Part\s+id=/,                // Legacy Part syntax
            /union\(\)|difference\(\)/,   // OpenSCAD boolean ops
            /module\s+\w+/,               // OpenSCAD modules
            /cylinder\(|cube\(|sphere\(/,  // OpenSCAD primitives
            /\$fn\s*=/,                   // OpenSCAD settings
            /function\s+\w+/,             // JavaScript/KCL functions
            /const\s+\w+\s*=/,            // Variable declarations
        ];

        return validPatterns.some(pattern => pattern.test(actualSource));
    }, [actualSource]);

    console.log('DefaultSimulation - hasValidCode:', hasValidCode, 'actualSource length:', actualSource?.length);

    // Parse Part ID or JSON Definition (Memoized)
    const activeAsset = React.useMemo(() => {
        if (!actualSource || !hasValidCode) return null;
        try {
            const parsed = JSON.parse(actualSource);
            // Allow 'primitive', 'procedural' and 'assembly' types
            // Note: 'primitive' is the base type for simple geometry like cylinders
            if (parsed && (parsed.type === 'primitive' || parsed.type === 'composition' || parsed.type === 'procedural' || parsed.type === 'assembly')) {
                return parsed;
            }
        } catch (e) {
            // Not JSON, try Regex (Legacy Flow)
            const partMatch = actualSource.match(/<Part id="([^"]+)"/);
            if (partMatch && partMatch[1]) {
                return ASSET_REGISTRY.find(a => a.id === partMatch[1]);
            }
        }
        return null;
    }, [actualSource, hasValidCode]);

    return (
        <div className="flex-1 relative overflow-hidden w-full h-full" style={{ backgroundColor: theme.colors.bg.primary }}>
            {/* Conditional Rendering: Full Aerodynamics View for 'flow' mode */}
            {viewMode === 'flow' ? (
                <AerodynamicsView geometryData={null} viewMode={viewMode} />
            ) : (
                <>
                    {/* Standard 3D View */}
                    <Canvas
                        shadows
                        camera={{ position: [6, 4, 6], fov: 35 }}
                        gl={{
                            antialias: true,
                            powerPreference: "high-performance",
                            preserveDrawingBuffer: true
                        }}
                        onCreated={({ gl }) => {
                            gl.domElement.addEventListener('webglcontextlost', (event) => {
                                event.preventDefault();
                                console.warn('WebGL Context Lost - Attempting Recovery');
                            }, false);
                            gl.domElement.addEventListener('webglcontextrestored', () => {
                                console.log('WebGL Context Restored');
                            }, false);
                        }}
                    >
                        <color attach="background" args={[theme.colors.bg.primary]} />

                        {isExploded ? (
                            <DeviceAssembly
                                isExploded={isExploded}
                                viewMode={viewMode}
                                theme={theme}
                                parts={activeAsset?.parts || []}
                            />
                        ) : viewMode === 'micro' ? (
                            <VMKRenderer viewMode={viewMode} />
                        ) : (
                            hasValidCode && (activeAsset || actualSource) ? (
                                <PhysicalObject isRunning={isRunning} theme={theme} kclSource={actualSource} viewMode={viewMode} activeAsset={activeAsset} />
                            ) : null
                        )}

                        {viewMode === 'hyperrealism' && <HyperRealismEffects />}

                        {/* Motion Trail - Shows path traveled */}
                        {isRunning && motionTrail && motionTrail.length > 1 && (
                            <MotionTrail trail={motionTrail} theme={theme} />
                        )}

                        {/* PHYSICS ORACLE VISUALIZATION */}
                        {physicsResult && (
                            <OrbitVisualizer data={physicsResult} theme={theme} />
                        )}

                        <ContactShadows position={[0, -1.01, 0]} opacity={0.4} scale={10} blur={1.5} far={0.8} color="#000000" />

                        {/* Lighting Setup */}
                        <Environment preset="studio" />
                        <ambientLight intensity={0.5} />
                        <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
                        <directionalLight position={[-10, 10, -5]} intensity={0.5} />

                        {/* Grid Helper */}
                        {showGrid && (
                            <gridHelper
                                args={[20, 20, theme.colors.accent.primary, theme.colors.border.secondary]}
                                position={[0, -0.01, 0]}
                            />
                        )}

                        <OrbitControls
                            enableDamping
                            dampingFactor={0.1}
                        />
                    </Canvas>

                    {/* Coordinate System Gizmo Overlay */}
                    <div className="absolute top-4 right-4 text-[7px] font-mono flex flex-col items-end gap-0.5 opacity-40" style={{ color: theme.colors.text.muted }}>
                        <div className="flex items-center gap-2"><span>X-AXIS</span> <div className="w-8 h-px bg-rose-500" /></div>
                        <div className="flex items-center gap-2"><span>Y-AXIS</span> <div className="w-8 h-px bg-emerald-500" /></div>
                        <div className="flex items-center gap-2"><span>Z-AXIS</span> <div className="w-8 h-px bg-blue-500" /></div>
                    </div>
                </>
            )}
        </div>
    );
}
