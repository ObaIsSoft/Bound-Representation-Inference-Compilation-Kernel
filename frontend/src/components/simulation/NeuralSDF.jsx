import React, { useMemo, useRef, useEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

// Vertex Shader: Fullscreen Quad
const vertexShader = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}
`;

// Fragment Shader
const fragmentShader = `
    precision highp float;
    varying vec2 vUv;
    uniform float uTime;
    uniform vec2 uResolution;
    uniform vec3 uCameraPos;
    
    // Neural Weights Texture (Flattened)
    uniform sampler2D uWeights;
    
    // Fetch weight value from flattened index
    float getW(int index) {
        int y = index / 64;
        int x = index - (y * 64);
        return texelFetch(uWeights, ivec2(x, y), 0).r;
    }

    // SIREN Activation
    float sine(float x) {
        return sin(30.0 * x);
    }

    // Inference Function (Hardcoded 3->32->32->32->1)
    float map_neural(vec3 p) {
        float h1[32];
        float h2[32];
        float h3[32];
        
        int idx = 0;
        
        // Layer 1 (3 -> 32)
        for(int i=0; i<32; i++) {
            float val = 0.0;
            val += p.x * getW(idx++);
            val += p.y * getW(idx++);
            val += p.z * getW(idx++);
            h1[i] = sine(val + getW(96 + i));
        }
        idx += 32;

        // Layer 2 (32 -> 32)
        int biasStart = 128 + 1024;
        
        for(int i=0; i<32; i++) {
            float val = 0.0;
            for(int j=0; j<32; j++) {
                val += h1[j] * getW(idx++);
            }
            h2[i] = sine(val + getW(biasStart + i));
        }
        idx += 32;

        // Layer 3 (32 -> 32)
        biasStart = idx + 1024;
        
        for(int i=0; i<32; i++) {
            float val = 0.0;
            for(int j=0; j<32; j++) {
                val += h2[j] * getW(idx++);
            }
            h3[i] = sine(val + getW(biasStart + i));
        }
        idx += 32;

        // Output Layer (32 -> 1)
        float dist = 0.0;
        for(int j=0; j<32; j++) {
            dist += h3[j] * getW(idx++);
        }
        dist += getW(idx);
        
        return dist;
    }

    // Raymarching - Optimized for Demand Mode
    const int MAX_STEPS = 64;
    const float MAX_DIST = 4.0;
    const float SURF_DIST = 0.005;

    void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
        vec3 ro = uCameraPos;
        vec3 rd = normalize(vec3(uv, -1.0));
        
        float d0 = 0.0;
        vec3 p;
        
        for(int i=0; i<MAX_STEPS; i++) {
            p = ro + rd * d0;
            if(length(p) > 2.0) { d0 = MAX_DIST + 1.0; break; }

            float ds = map_neural(p);
            d0 += ds;
            if(d0 > MAX_DIST || abs(ds) < SURF_DIST) break;
        }
        
        if(d0 < MAX_DIST) {
            // Normals
            vec2 e = vec2(0.02, 0); 
            vec3 n = normalize(vec3(
                map_neural(p + e.xyy) - map_neural(p - e.xyy),
                map_neural(p + e.yxy) - map_neural(p - e.yxy),
                map_neural(p + e.yyx) - map_neural(p - e.yyx)
            ));
            
            // Simple lighting
            vec3 light = normalize(vec3(1, 1, 1));
            float diff = max(dot(n, light), 0.2);
            
            vec3 col = vec3(0.4, 0.6, 0.9) * diff;
            gl_FragColor = vec4(col, 1.0);
        } else {
            gl_FragColor = vec4(0.1, 0.1, 0.1, 1.0);
        }
    }
`;

export const NeuralSDF = ({ design, region }) => {
    const meshRef = useRef();
    const materialRef = useRef();
    const { size, camera } = useThree();
    const [texture, setTexture] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Uniforms
    const uniforms = useMemo(() => ({
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2() },
        uCameraPos: { value: new THREE.Vector3() },
        uWeights: { value: null }
    }), []);

    // Train on design change
    useEffect(() => {
        if (!design) return;

        const trainNetwork = async () => {
            setLoading(true);
            setError(null);

            try {
                console.log('[NeuralSDF] Training network for design...');

                const response = await fetch('http://localhost:8000/api/neural_sdf/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ design, region })
                });

                if (!response.ok) {
                    throw new Error(`Training failed: ${response.statusText}`);
                }

                const result = await response.json();
                console.log('[NeuralSDF] Training complete:', result.metadata);

                // Pack weights into texture
                const packedData = new Float32Array(4096).fill(0);
                let ptr = 0;

                const append = (arr) => {
                    for (let i = 0; i < arr.length; i++) packedData[ptr++] = arr[i];
                };

                const weights = result.weights;
                if (weights && weights.length >= 4) {
                    append(weights[0].weight);
                    append(weights[0].bias);
                    append(weights[1].weight);
                    append(weights[1].bias);
                    append(weights[2].weight);
                    append(weights[2].bias);
                    append(weights[3].weight);
                    append(weights[3].bias);
                }

                // Create Data Texture
                const tex = new THREE.DataTexture(
                    packedData,
                    64,
                    64,
                    THREE.RedFormat,
                    THREE.FloatType
                );
                tex.needsUpdate = true;
                tex.minFilter = THREE.NearestFilter;
                tex.magFilter = THREE.NearestFilter;

                setTexture(tex);
                if (materialRef.current) {
                    materialRef.current.uniforms.uWeights.value = tex;
                }

            } catch (err) {
                console.error('[NeuralSDF] Training error:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        trainNetwork();
    }, [design?.id, region]);

    useFrame((state) => {
        if (materialRef.current) {
            materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
            materialRef.current.uniforms.uResolution.value.set(size.width, size.height);
            materialRef.current.uniforms.uCameraPos.value.copy(camera.position);
        }
    });

    // Loading Overlay
    if (loading) {
        return (
            <group>
                <mesh position={[0, 0, 0]}>
                    <planeGeometry args={[3, 1]} />
                    <meshBasicMaterial color="#000000" opacity={0.8} transparent />
                </mesh>
            </group>
        );
    }

    // Error State
    if (error) {
        console.error('[NeuralSDF] Render error:', error);
        return null;
    }

    if (!texture) return null;

    return (
        <mesh ref={meshRef}>
            <planeGeometry args={[2, 2]} />
            <shaderMaterial
                ref={materialRef}
                vertexShader={vertexShader}
                fragmentShader={fragmentShader}
                uniforms={uniforms}
            />
        </mesh>
    );
};
