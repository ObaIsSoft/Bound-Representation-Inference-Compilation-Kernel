import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

const Thermometer3D = ({ temperature = 20, visible = false }) => {
    if (!visible) return null;

    // Clamp temperature for visual representation (-50 to 150)
    const normalizedTemp = Math.min(150, Math.max(-50, temperature));
    // Map -50..150 to height 0.2..1.8 (approx)
    // Range = 200 degrees
    // Height Range = 1.6 units
    const mercuryHeight = 0.2 + ((normalizedTemp + 50) / 200) * 1.6;

    // Color interpolation: Blue (-50) -> Green (20) -> Red (100+)
    const mercuryColor = useMemo(() => {
        if (temperature < 0) return '#3b82f6'; // Blue
        if (temperature < 40) return '#10b981'; // Green
        return '#ef4444'; // Red
    }, [temperature]);

    return (
        <group position={[3.5, 0, 0]}>
            {/* Glass Tube Body */}
            <mesh position={[0, 1, 0]}>
                <cylinderGeometry args={[0.15, 0.15, 2.2, 16]} />
                <meshPhysicalMaterial
                    color="#ffffff"
                    transmission={0.9}
                    opacity={0.3}
                    transparent
                    roughness={0}
                    metalness={0.1}
                    thickness={0.1}
                />
            </mesh>

            {/* Mercury Bulb */}
            <mesh position={[0, 0, 0]}>
                <sphereGeometry args={[0.25, 32, 32]} />
                <meshStandardMaterial color={mercuryColor} roughness={0.2} metalness={0.5} />
            </mesh>

            {/* Mercury Column (Dynamic Height) */}
            <mesh position={[0, mercuryHeight / 2, 0]}>
                <cylinderGeometry args={[0.08, 0.08, mercuryHeight, 16]} />
                <meshStandardMaterial color={mercuryColor} emissive={mercuryColor} emissiveIntensity={0.5} />
            </mesh>

            {/* Backplate / Scale */}
            <mesh position={[0, 1, -0.16]}>
                <boxGeometry args={[0.5, 2.3, 0.02]} />
                <meshStandardMaterial color="#333333" metalness={0.8} roughness={0.2} />
            </mesh>

            {/* Ticks and Labels */}
            {[-40, -20, 0, 20, 40, 60, 80, 100, 120, 140].map((tick) => {
                const tickH = 0.2 + ((tick + 50) / 200) * 1.6;
                return (
                    <group key={tick} position={[0.15, tickH, -0.14]}>
                        <mesh position={[0.05, 0, 0]}>
                            <boxGeometry args={[0.2, 0.02, 0.01]} />
                            <meshStandardMaterial color="#888888" />
                        </mesh>
                        <Text
                            position={[0.35, 0, 0]}
                            fontSize={0.12}
                            color="white"
                            anchorX="left"
                            anchorY="middle"
                        >
                            {tick}°
                        </Text>
                    </group>
                );
            })}

            {/* Current Value Display (Floating) */}
            <group position={[0, 2.4, 0]}>
                <Text
                    fontSize={0.3}
                    font="https://fonts.gstatic.com/s/roboto/v18/KFOmCnqEu92Fr1Mu4mxM.woff"
                    color={mercuryColor}
                    outlineWidth={0.02}
                    outlineColor="#000000"
                >
                    {temperature}°C
                </Text>
            </group>

            {/* Base Stand */}
            <mesh position={[0, -0.1, 0]}>
                <cylinderGeometry args={[0.4, 0.5, 0.2, 32]} />
                <meshStandardMaterial color="#222222" metalness={0.6} roughness={0.4} />
            </mesh>
        </group>
    );
};

export default Thermometer3D;
