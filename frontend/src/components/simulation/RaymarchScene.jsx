import React, { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';

/**
 * BRICK OS: High-Fidelity Physical Synthesis Viewport (GLSL Kernel)
 * Implements "Subtractive Machining" logic via Raymarching.
 */

const RaymarchKernel = ({ isaState }) => {
    const meshRef = useRef();
    const { size } = useThree();

    const materialDef = useMemo(() => {
        return {
            uniforms: {
                uTime: { value: 0 },
                uResolution: { value: new THREE.Vector2(size.width, size.height) },
                uCameraPos: { value: new THREE.Vector3() },
                uCameraDir: { value: new THREE.Vector3() },

                // Base Shape Control
                uBaseShape: { value: 1 }, // 0=Empty, 1=Box, 2=Sphere
                uBaseDims: { value: new THREE.Vector3(1, 1, 1) },

                // ISA Component Arrays - each element must be unique
                uCompPos: { value: Array.from({ length: 16 }, () => new THREE.Vector3()) },
                uCompRadius: { value: new Float32Array(16) },
                uCompOp: { value: new Int32Array(16) },
                uCount: { value: 0 }
            },
            vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1.0);
        }
      `,
            fragmentShader: `
        uniform float uTime;
        uniform vec2 uResolution;
        uniform vec3 uCameraPos;
        // uniform vec3 uCameraDir; // Unused but reserved
        
        uniform int uBaseShape;
        uniform vec3 uBaseDims;
        
        uniform vec3 uCompPos[16];
        uniform float uCompRadius[16];
        uniform int uCompOp[16];
        uniform int uCount;

        varying vec2 vUv;

        // --- SDF Primitives ---

        float sdSphere(vec3 p, float s) {
          return length(p) - s;
        }

        float sdBox(vec3 p, vec3 b) {
          vec3 q = abs(p) - b;
          return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
        }

        float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
          vec3 pa = p - a, ba = b - a;
          float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
          return length(pa - ba * h) - r;
        }

        // --- Smooth Operations ---

        float smin(float a, float b, float k) {
          float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
          return mix(b, a, h) - k * h * (1.0 - h);
        }

        // --- The Map Function (Physical State Resolver) ---

        float map(vec3 p) {
          // 1. Initial State - Dynamic Base Geometry
          float d = 100.0; // Default to empty/far away
          
          if (uBaseShape == 1) {
             // Box (dims are full Width/Height/Depth in args, so half for extents)
             d = sdBox(p, uBaseDims * 0.5); 
          } else if (uBaseShape == 2) {
             // Sphere (dims.x is radius)
             d = sdSphere(p, uBaseDims.x);
          } else {
             return 100.0; // Empty
          }

          // 2. Subtractive Machining & Hierarchical Detailing
          for(int i = 0; i < 16; i++) {
            if(i >= uCount) break;

            vec3 pos = uCompPos[i];
            float rad = uCompRadius[i];
            int op = uCompOp[i];

            if (op == 1) {
              // Loop/Sweep (Capsule) - Subtractive
              float d_comp = sdCapsule(p, pos - vec3(1.0, 0.0, 0.0), pos + vec3(1.0, 0.0, 0.0), rad);
              d = max(d, -d_comp);
            } else if (op == 2) {
              // Smooth Additive
              float d_comp = length(p - pos) - rad;
              d = smin(d, d_comp, 0.2);
            } else {
              // Standard Union
              float d_comp = length(p - pos) - rad;
              d = min(d, d_comp);
            }
          }
          return d;
        }

        // --- Lighting & Rendering ---

        vec3 calcNormal(vec3 p) {
          vec2 e = vec2(0.001, 0.0);
          return normalize(vec3(
            map(p + e.xyy) - map(p - e.xyy),
            map(p + e.yxy) - map(p - e.yxy),
            map(p + e.yyx) - map(p - e.yyx)
          ));
        }

        void main() {
          vec2 uv = vUv * 2.0 - 1.0;
          uv.x *= uResolution.x / uResolution.y;

          vec3 ro = uCameraPos;
          vec3 forward = normalize(-uCameraPos); // Look at zero
          vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
          vec3 up = cross(forward, right);
          vec3 rd = normalize(forward + uv.x * right + uv.y * up);

          float t = 0.0;
          float d = 0.0;
          for(int i = 0; i < 100; i++) {
            vec3 p = ro + rd * t;
            d = map(p);
            if(d < 0.001) {
              vec3 n = calcNormal(p);
              float diff = dot(n, normalize(vec3(1.0, 2.0, 3.0))) * 0.5 + 0.5;
              vec3 col = mix(vec3(0.1, 0.15, 0.2), vec3(0.4, 0.6, 0.8), diff);
              
              // Grid
              float grid = (sin(p.x * 20.0) * sin(p.z * 20.0)) * 0.02;
              col += grid;

              gl_FragColor = vec4(col, 1.0);
              return;
            }
            t += d;
            if(t > 20.0) break;
          }
          
          // Background
          gl_FragColor = vec4(mix(vec3(0.02, 0.03, 0.05), vec3(0.0, 0.0, 0.0), vUv.y * 0.5 + 0.5), 1.0);
        }
      `
        };
    }, [size]);

    useFrame((state) => {
        if (!meshRef.current) return;
        const { uniforms } = meshRef.current.material;

        uniforms.uTime.value = state.clock.getElapsedTime();
        uniforms.uCameraPos.value.copy(state.camera.position);
        uniforms.uResolution.value.set(size.width, size.height);

        if (isaState) {
            // Update Base Geometry
            uniforms.uBaseShape.value = isaState.baseShape;
            uniforms.uBaseDims.value.set(
                isaState.baseDims[0],
                isaState.baseDims[1],
                isaState.baseDims[2]
            );

            // Update Components
            if (isaState.components) {
                uniforms.uCount.value = Math.min(isaState.components.length, 16);
                isaState.components.forEach((comp, i) => {
                    if (i < 16) {
                        uniforms.uCompPos.value[i].set(comp.pos[0], comp.pos[1], comp.pos[2]);
                        uniforms.uCompRadius.value[i] = comp.radius;
                        uniforms.uCompOp.value[i] = comp.op_type;
                    }
                });
            }
        }
    });

    return (
        <mesh ref={meshRef}>
            <planeGeometry args={[2, 2]} />
            <shaderMaterial
                key="raymarch-shader"
                {...materialDef}
                onBeforeCompile={(shader) => {
                    console.log('[RaymarchScene] Shader compiled successfully');
                }}
            />
        </mesh>
    );
};

export default function RaymarchScene({ design, isaState }) {
    // Convert design data to VMK format
    const activeState = useMemo(() => {
        // Default Empty State
        const state = {
            baseShape: 0, // 0=Empty
            baseDims: [1, 1, 1],
            components: [],
            count: 0
        };

        // Priority 1: Use isaState if provided (from VMK backend)
        if (isaState) return { ...state, ...isaState };

        // Priority 2: Convert design data to VMK format
        if (design?.asset) {
            const asset = design.asset;

            // Extract geometry from asset
            if (asset.type === 'primitive') {
                const { geometry, args = [] } = asset;

                if (geometry === 'box') {
                    // Map Box to Base Shape 1
                    const [width = 1, height = 1, depth = 1] = args;
                    state.baseShape = 1;
                    state.baseDims = [width, height, depth];
                } else if (geometry === 'sphere') {
                    // Map Sphere to Base Shape 2
                    const [radius = 0.5] = args;
                    state.baseShape = 2;
                    state.baseDims = [radius, 0, 0];
                }
            }
        }

        return state;
    }, [isaState, design]);

    return (
        <Canvas>
            <PerspectiveCamera makeDefault position={[4, 3, 4]} fov={35} />
            <OrbitControls
                enableDamping
                dampingFactor={0.05}
                target={[0, 0, 0]}
                enableZoom={true}
                zoomSpeed={1.0}
                minDistance={2}
                maxDistance={20}
            />
            <RaymarchKernel isaState={activeState} />
        </Canvas>
    );
}
