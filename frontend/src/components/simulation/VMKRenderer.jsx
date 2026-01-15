import React, { useMemo, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const VMK_VERTEX_SHADER = `
varying vec2 vUv;
varying vec3 vPos;
void main() {
    vUv = uv;
    vPos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const VMK_FRAGMENT_SHADER = `
uniform vec3 stockDims;
uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform float time;

// Toolpath Data (Max 50 for MVP)
struct ToolPath {
    vec3 start;
    vec3 end;
    float radius;
};
uniform ToolPath toolPaths[50];
uniform int numPaths;

varying vec2 vUv;
varying vec3 vPos;

// --- SDF Primitives ---

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// --- Scene Map ---

float map(vec3 p) {
    // 1. Stock (The Block)
    float d_stock = sdBox(p, stockDims * 0.5);

    // 2. Subtractions (The Cuts)
    float d_cuts = 10000.0; // Init far away

    for(int i = 0; i < 50; i++) {
        if (i >= numPaths) break;
        float d_path = sdCapsule(p, toolPaths[i].start, toolPaths[i].end, toolPaths[i].radius);
        d_cuts = min(d_cuts, d_path);
    }

    // Boolean Subtraction: max(A, -B)
    return max(d_stock, -d_cuts);
}

// --- Raymarching ---

vec3 calcNormal(vec3 p) {
    const float h = 0.0001; // Precision
    const vec2 k = vec2(1, -1);
    return normalize(
        k.xyy * map(p + k.xyy * h) +
        k.yyx * map(p + k.yyx * h) +
        k.yxy * map(p + k.yxy * h) +
        k.xxx * map(p + k.xxx * h)
    );
}

void main() {
    // --- Raymarching ---
    
    // Ray origin and direction in World Space
    vec3 ro = cameraPosition; 
    vec3 rd = normalize(vPos - cameraPosition);
    
    float t = 0.0;
    float tMax = 20.0;
    
    vec3 p = vPos; // Start at surface
    float d = map(p);
    
    if (d > 0.001) {
        // We are in "Air" (cut part of the box). March forward!
        float t_march = 0.0;
        for(int i=0; i<64; i++) {
            p += rd * d; // Step
            d = map(p);
            if(d < 0.001 || t_march > 5.0) break;
            t_march += d;
        }
        
        if (d > 0.01) discard; // Punched through to other side?
    }
    
    // Lighting
    vec3 n = calcNormal(p);
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(n, lightDir), 0.0);
    vec3 col = vec3(0.0, 1.0, 0.6) * (0.2 + 0.8 * diff); // Tech Green
    
    // Inner Glow
    float rim = 1.0 - max(dot(n, -rd), 0.0);
    col += vec3(0.4, 0.8, 1.0) * pow(rim, 3.0);

    gl_FragColor = vec4(col, 1.0);
}
`;

const VMKRenderer = ({ viewMode }) => {
    // 1. Fetch State
    const [vmkState, setVmkState] = useState(null);

    useEffect(() => {
        if (viewMode === 'micro') {
            fetch('http://localhost:8000/api/vmk/history')
                .then(res => res.json())
                .then(data => setVmkState(data))
                .catch(err => console.error("VMK Fetch Error", err));
        }
    }, [viewMode]);

    // 2. Uniforms
    const uniforms = useMemo(() => {
        if (!vmkState) return {
            stockDims: { value: new THREE.Vector3(10, 10, 5) },
            numPaths: { value: 0 },
            toolPaths: { value: [] },
            time: { value: 0 }
        };

        const paths = vmkState.history.slice(0, 50).map(op => ({
            start: new THREE.Vector3(...op.path[0]),
            end: new THREE.Vector3(...op.path[1]),
            radius: 0.1 // Hardcoded for MVP, should come from tool
        }));

        // Pad array if needed? GLSL needs fixed size? 
        // Three.js handles value arrays well if struct matches.

        return {
            stockDims: { value: new THREE.Vector3(...vmkState.stock_dims) },
            numPaths: { value: paths.length },
            toolPaths: { value: paths }, // Three.js might need flattening or special handling for struct arrays
            time: { value: 0 }
        };
    }, [vmkState]);

    // 3. Render
    if (viewMode !== 'micro') return null;

    return (
        <mesh>
            <boxGeometry args={vmkState ? vmkState.stock_dims : [10, 10, 5]} />
            <shaderMaterial
                vertexShader={VMK_VERTEX_SHADER}
                fragmentShader={VMK_FRAGMENT_SHADER}
                uniforms={uniforms}
                transparent
            />
        </mesh>
    );
};

export default VMKRenderer;
