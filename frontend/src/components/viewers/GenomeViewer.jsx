import React, { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import {
    OrbitControls,
    PerspectiveCamera,
    Environment,
    ContactShadows,
    Float,
    MeshDistortMaterial,
    useGLTF,
    Center,
    Text,
    Grid
} from '@react-three/drei';
import { useTheme } from '../../contexts/ThemeContext';
import { Box as BoxIcon, Loader2, Sparkles, Move, ZoomIn, Layers } from 'lucide-react';

const ModelLoader = ({ modelId }) => {
    // URL for the new geometry model endpoint
    const url = modelId ? `http://localhost:8000/api/geometry/model/${modelId}` : null;

    // This will eventually load the real GLB from the backend
    // For now, if no url is provided, we show a professional placeholder
    if (!url) {
        return (
            <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
                <Center>
                    <mesh castShadow receiveShadow>
                        <boxGeometry args={[2, 2, 2]} />
                        <MeshDistortMaterial
                            color="#f97316"
                            speed={3}
                            distort={0.4}
                            radius={1}
                        />
                    </mesh>
                </Center>
            </Float>
        );
    }

    try {
        const { scene } = useGLTF(url);
        return <primitive object={scene} />;
    } catch (e) {
        return (
            <Center>
                <Text color="white" fontSize={0.2} anchorX="center" anchorY="middle">
                    Mesh Encoding...
                </Text>
            </Center>
        );
    }
};

const GenomeViewer = ({ path, fileName, modelId }) => {
    const { theme } = useTheme();

    return (
        <div className="h-full w-full relative overflow-hidden flex flex-col">
            {/* Header Overlay */}
            <div className="absolute top-6 left-6 z-10 flex flex-col gap-2">
                <div className="flex items-center gap-4 p-4 rounded-3xl bg-black/40 border border-white/10 backdrop-blur-2xl shadow-2xl">
                    <div className="p-2.5 rounded-xl bg-gradient-to-br from-orange-400 to-orange-600 shadow-lg shadow-orange-500/20">
                        <BoxIcon size={20} className="text-white" />
                    </div>
                    <div>
                        <div className="text-[10px] font-black uppercase tracking-[0.2em] opacity-40 leading-none mb-1.5">Reactive Genesis Engine</div>
                        <div className="text-sm font-black leading-none tracking-tight">{fileName || 'Untitled Genome'}</div>
                    </div>
                </div>

                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/5 backdrop-blur-md w-fit">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-[9px] font-bold uppercase opacity-60">Live Environment Active</span>
                </div>
            </div>

            {/* Main 3D Scene */}
            <div className="flex-1 cursor-grab active:cursor-grabbing">
                <Canvas shadows dpr={[1, 2]} camera={{ position: [5, 5, 5], fov: 35 }}>
                    <Suspense fallback={<Loader2 className="animate-spin text-orange-500" />}>
                        <Center top>
                            <ModelLoader modelId={modelId} />
                        </Center>

                        {/* Scenery & Environment */}
                        <Environment preset="city" />
                        <ContactShadows
                            position={[0, -1.5, 0]}
                            opacity={0.4}
                            scale={10}
                            blur={2.5}
                            far={4}
                        />

                        <Grid
                            infiniteGrid
                            fadeDistance={30}
                            fadeStrength={5}
                            cellSize={0.5}
                            sectionSize={2.5}
                            sectionThickness={1}
                            cellThickness={0.5}
                            sectionColor="#ffffff"
                            cellColor="#ffffff"
                            opacity={0.05}
                        />
                    </Suspense>

                    <OrbitControls
                        makeDefault
                        enableDamping
                        dampingFactor={0.05}
                        minPolarAngle={0}
                        maxPolarAngle={Math.PI / 1.75}
                    />

                    {/* Lighting */}
                    <ambientLight intensity={0.5} />
                    <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
                    <pointLight position={[-10, -10, -10]} intensity={0.5} />
                </Canvas>
            </div>

            {/* Pro HUD Controls */}
            <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10 flex items-center gap-6 px-8 py-4 rounded-3xl bg-black/60 border border-white/10 backdrop-blur-3xl shadow-3xl">
                <div className="flex items-center gap-5">
                    <button className="flex flex-col items-center gap-1 group">
                        <div className="p-2.5 rounded-xl bg-white/5 group-hover:bg-white/10 transition-colors border border-white/5"><ZoomIn size={18} /></div>
                        <span className="text-[8px] font-black uppercase opacity-40">Zoom</span>
                    </button>
                    <button className="flex flex-col items-center gap-1 group">
                        <div className="p-2.5 rounded-xl bg-white/5 group-hover:bg-white/10 transition-colors border border-white/5"><Move size={18} /></div>
                        <span className="text-[8px] font-black uppercase opacity-40">Orbit</span>
                    </button>
                    <button className="flex flex-col items-center gap-1 group">
                        <div className="p-2.5 rounded-xl bg-white/5 group-hover:bg-white/10 transition-colors border border-white/5"><Layers size={18} /></div>
                        <span className="text-[8px] font-black uppercase opacity-40">Layers</span>
                    </button>
                </div>

                <div className="w-[1px] h-10 bg-white/10" />

                <button className="group relative flex items-center gap-3 px-6 py-3 rounded-2xl bg-orange-500 hover:bg-orange-400 transition-all shadow-lg shadow-orange-500/20 active:scale-95">
                    <Sparkles size={16} className="text-white group-hover:rotate-12 transition-transform" />
                    <span className="text-xs font-black uppercase tracking-tight text-white">Refine Geometry</span>
                </button>
            </div>

            {/* Engineering Metrics Sidebar */}
            <div className="absolute top-24 right-8 flex flex-col gap-4 z-10">
                {[
                    { label: 'Complexity', value: 'Level 7', color: 'text-orange-400' },
                    { label: 'DNA Stability', value: '98.4%', color: 'text-green-400' },
                    { label: 'Volume Est.', value: '1.42 m³', color: 'text-blue-400' }
                ].map((stat, i) => (
                    <div key={i} className="group p-4 pr-6 rounded-2xl bg-black/40 border-l-2 border-white/5 hover:border-white/20 backdrop-blur-xl transition-all text-right min-w-[140px]">
                        <div className="text-[10px] font-black uppercase tracking-widest opacity-30 mb-1">{stat.label}</div>
                        <div className={`text-sm font-black ${stat.color} tabular-nums`}>{stat.value}</div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default GenomeViewer;
