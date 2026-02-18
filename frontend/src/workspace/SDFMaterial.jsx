/**
 * SDF Material - Shader-based Signed Distance Field Rendering
 * 
 * Renders SDF geometry using raymarching in a shader.
 * Supports various SDF primitives and operations.
 */

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// SDF Fragment Shader
const SDF_FRAGMENT_SHADER = `
  uniform float uTime;
  uniform vec3 uCamera;
  
  varying vec3 vPosition;
  varying vec3 vNormal;
  
  // SDF Primitives
  float sdSphere(vec3 p, float r) {
    return length(p) - r;
  }
  
  float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
  }
  
  float sdCylinder(vec3 p, float h, float r) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
  }
  
  // SDF Operations
  float opUnion(float d1, float d2) {
    return min(d1, d2);
  }
  
  float opSubtract(float d1, float d2) {
    return max(-d1, d2);
  }
  
  float opIntersect(float d1, float d2) {
    return max(d1, d2);
  }
  
  // Smooth blending
  float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
  }
  
  // Scene SDF - Modify this based on your geometry
  float map(vec3 p) {
    // Example: Two spheres with smooth blend
    float sphere1 = sdSphere(p - vec3(-0.5, 0.0, 0.0), 0.6);
    float sphere2 = sdSphere(p - vec3(0.5, 0.0, 0.0), 0.6);
    float box = sdBox(p - vec3(0.0, 0.8, 0.0), vec3(1.0, 0.2, 0.5));
    
    float blend = opSmoothUnion(sphere1, sphere2, 0.3);
    return opUnion(blend, box);
  }
  
  // Calculate normal using gradient
  vec3 calcNormal(vec3 p) {
    float eps = 0.001;
    vec2 h = vec2(eps, 0.0);
    return normalize(vec3(
      map(p + h.xyy) - map(p - h.xyy),
      map(p + h.yxy) - map(p - h.yxy),
      map(p + h.yyx) - map(p - h.yyx)
    ));
  }
  
  // Raymarch
  float raymarch(vec3 ro, vec3 rd) {
    float t = 0.0;
    for (int i = 0; i < 100; i++) {
      vec3 p = ro + t * rd;
      float d = map(p);
      if (d < 0.001) return t;
      if (t > 100.0) break;
      t += d;
    }
    return -1.0;
  }
  
  // Soft shadow
  float calcSoftShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 64; i++) {
      float h = map(ro + rd * t);
      if (h < 0.001) return 0.0;
      res = min(res, k * h / t);
      t += clamp(h, 0.01, 0.5);
      if (t > maxt) break;
    }
    return res;
  }
  
  // Ambient occlusion
  float calcAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
      float h = 0.01 + 0.12 * float(i) / 4.0;
      float d = map(p + h * n);
      occ += (h - d) * sca;
      sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
  }
  
  void main() {
    // Ray direction
    vec3 ro = uCamera;
    vec3 rd = normalize(vPosition - ro);
    
    // Raymarch
    float t = raymarch(ro, rd);
    
    if (t < 0.0) {
      discard;
    }
    
    // Hit point
    vec3 p = ro + t * rd;
    vec3 n = calcNormal(p);
    
    // Lighting
    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
    vec3 viewDir = -rd;
    
    // Diffuse
    float diff = max(dot(n, lightDir), 0.0);
    
    // Specular
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(n, halfDir), 0.0), 64.0);
    
    // Shadows
    float shadow = calcSoftShadow(p, lightDir, 0.02, 5.0, 8.0);
    
    // Ambient occlusion
    float ao = calcAO(p, n);
    
    // Color
    vec3 baseColor = vec3(0.4, 0.5, 0.6);
    vec3 ambient = vec3(0.1, 0.12, 0.15) * ao;
    vec3 diffuse = baseColor * diff * shadow;
    vec3 specular = vec3(0.8) * spec * shadow;
    
    vec3 color = ambient + diffuse + specular;
    
    // Fog
    float fog = exp(-0.02 * t);
    color = mix(vec3(0.05, 0.05, 0.1), color, fog);
    
    gl_FragColor = vec4(color, 1.0);
  }
`;

// SDF Vertex Shader
const SDF_VERTEX_SHADER = `
  varying vec3 vPosition;
  varying vec3 vNormal;
  
  void main() {
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/**
 * SDF Material Component
 */
export function SDFMaterial({ parameters = {} }) {
  const materialRef = useRef();
  
  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uCamera: { value: new THREE.Vector3(0, 0, 5) }
  }), []);
  
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      materialRef.current.uniforms.uCamera.value.copy(state.camera.position);
    }
  });
  
  return (
    <shaderMaterial
      ref={materialRef}
      uniforms={uniforms}
      vertexShader={SDF_VERTEX_SHADER}
      fragmentShader={SDF_FRAGMENT_SHADER}
      side={THREE.DoubleSide}
    />
  );
}

export default SDFMaterial;
