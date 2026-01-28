/**
 * BRICK OS: Unified SDF Kernel
 * 
 * Single shader supporting all view modes through uniform configuration.
 * Replaces: VMKRenderer, RaymarchScene, DefaultSimulation shaders
 * 
 * View Modes:
 * 0 = Realistic (PBR with environment)
 * 1 = Wireframe (edge detection)
 * 2 = X-Ray (rim transparency)
 * 3 = Matte (diffuse only)
 * 4 = Thermal (temperature heatmap: blue=cold, red=hot)
 * 5 = Cutaway (cross-section with interior)
 * 6 = Solid (flat unlit color)
 * 7 = Stress/Flow (physics overlay arrows)
 */

precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D; // [FIX] Required for uMeshSDFTexture

// ============================================
// UNIFORMS
// ============================================
uniform float uTime;
uniform vec2 uResolution;
uniform vec3 uCameraPos;
uniform int uViewMode;
uniform vec3 uClipPlane;
uniform float uClipOffset;
uniform float uClipEnabled;
uniform vec3 uBaseColor;
uniform float uMetalness;
uniform float uRoughness;

// Phase 10: Thermal Data
uniform float uThermalTemp; // Temperature in Celsius
uniform bool uThermalEnabled;


// Geometry State
uniform int uBaseShape;      // 0=Empty, 1=Box, 2=Sphere, 3=Cylinder
uniform vec3 uBaseDims;
uniform int uComponentCount;

// Physics Data
uniform float uPhysicsDataEnabled;
uniform sampler2D uPhysicsTexture;

// Mesh SDF Texture (Phase 8: Imported Meshes)
uniform bool uMeshSDFEnabled;
uniform sampler3D uMeshSDFTexture;
uniform vec3 uMeshBounds[2];      // [0]=min, [1]=max
uniform float uMeshSDF_min;
uniform float uMeshSDF_max;

// Theme & Background
uniform vec3 uBgColor1;
uniform vec3 uBgColor2;
uniform float uGridEnabled;
uniform vec3 uGridColor;

varying vec2 vUv;

// ============================================
// CONSTANTS
// ============================================
const vec3 LIGHT_DIR = normalize(vec3(1.0, 2.0, 3.0));
const vec3 LIGHT_DIR_2 = normalize(vec3(-2.0, 1.0, -1.0));
const vec3 AMBIENT = vec3(0.08, 0.09, 0.12);

// ============================================
// SDF PRIMITIVES
// ============================================

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdCylinder(vec3 p, float h, float r) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// ============================================
// MESH SDF SAMPLING (Phase 8: Texture Atlas)
// ============================================

// Component Definition (Matches Backend Manifest)
struct ComponentSDF {
    vec3 atlasOffset;
    vec3 atlasScale;
    vec3 localBounds[2];
    vec2 sdfRange;
    mat4 transform; // World to Local Matrix (Inverse of Mesh Transform)
};

#define MAX_COMPONENTS 16 // WebGL Loop Limit Safe
uniform ComponentSDF uComponents[MAX_COMPONENTS];
uniform int uAtlasComponentCount; // Number of active components

// Semantic Sketch Primitives
struct SDFPrimitive {
    int type; // 0=None, 1=Capsule
    vec3 p0;
    vec3 p1;
    float radius;
    float blendStrength;
};
#define MAX_PRIMITIVES 16
uniform SDFPrimitive uPrimitives[MAX_PRIMITIVES];
uniform int uPrimitiveCount; // Number of active primitives

// Light Pen Sketching (Subtractive)
uniform vec3 uSketchPoints[128]; // Increased from 32
uniform int uSketchCount;




// Helper for soft min (optional, but nice for intersections)
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float sampleMeshSDF(vec3 p) {
    if (!uMeshSDFEnabled) return 1e10;

    // Single Mesh Mode (Legacy / Simple)
    if (uAtlasComponentCount <= 1) {
        // Fallback to original logic if no complex manifest
        // ... (Original Logic Kept for safety or re-implemented here)
        // For now, let's assume if count <=1 we might still use the atlas logic 
        // OR the legacy uniforms. Let's use legacy uniforms for robust fallback.
        if (uAtlasComponentCount == 0) {
             vec3 uvw = (p - uMeshBounds[0]) / (uMeshBounds[1] - uMeshBounds[0]);
             if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return 1e10;
             float normalized = texture(uMeshSDFTexture, uvw).r;
             return normalized * (uMeshSDF_max - uMeshSDF_min) + uMeshSDF_min;
        }
    }

    // Atlas Mode
    float d_min = 1e10;
    
    // Unrolled Loop slightly better for some drivers, but standard for ok
    for(int i=0; i<MAX_COMPONENTS; i++) {
        if (i >= uAtlasComponentCount) break;

        // 1. World -> Local Component Space
        // Note: uComponents[i].transform MUST be the INVERSE World Matrix 
        // to take World P -> Local P
        vec4 localP4 = uComponents[i].transform * vec4(p, 1.0);
        vec3 localP = localP4.xyz; 

        // 2. AABB Check (Optimization)
        vec3 bMin = uComponents[i].localBounds[0];
        vec3 bMax = uComponents[i].localBounds[1];
        
        // Exact box distance for empty space skipping
        vec3 d_box_vec = max(abs(localP - (bMin+bMax)*0.5) - (bMax-bMin)*0.5, 0.0);
        float d_box = length(d_box_vec);
        
        // If inside box, d_box is 0. If outside far, it's > 0
        // We can just add this to the texture sample if we want exactness, 
        // or skip texture sample if > epsilon.
        // For accurate SDF, we need the value inside.
        
        bool insideBox = (d_box < 0.01);
        
        float dist = 1e10;

        if (insideBox) {
            // 3. Map to Atlas UV
            vec3 uv_local = (localP - bMin) / (bMax - bMin);
            vec3 uv_atlas = uComponents[i].atlasOffset + uv_local * uComponents[i].atlasScale;
            
            // 4. Sample
            float norm = texture(uMeshSDFTexture, uv_atlas).r;
            dist = norm * (uComponents[i].sdfRange.y - uComponents[i].sdfRange.x) + uComponents[i].sdfRange.x;
        } else {
             dist = d_box + 0.05; // Approximation outside
        }

        d_min = min(d_min, dist);
    }

    return d_min;
}

// ============================================
// SCENE MAP
// ============================================

// Analytic Primitives
float sdCapsule( vec3 p, vec3 a, vec3 b, float r ) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

float map(vec3 p) {
    float d = 100.0;
    
    // Native SDF primitives
    if (uBaseShape == 1) {
        d = sdBox(p, uBaseDims * 0.5);
    } else if (uBaseShape == 2) {
        d = sdSphere(p, uBaseDims.x);
    } else if (uBaseShape == 3) {
        d = sdCylinder(p, uBaseDims.y * 0.5, uBaseDims.x * 0.5);
    } else {
        d = 100.0;
    }
    
    // Combine with mesh SDF (union operation)
    if (uMeshSDFEnabled) {
        float d_mesh = sampleMeshSDF(p);
        d = min(d, d_mesh); // Union: take minimum distance
    }

    // Blend Semantic Sketches (Proposed)
    for(int i=0; i<MAX_PRIMITIVES; i++) {
        if (i >= uPrimitiveCount) break;
        if (uPrimitives[i].type == 1) { // Capsule
            float d_prim = sdCapsule(p, uPrimitives[i].p0, uPrimitives[i].p1, uPrimitives[i].radius);
            // Smooth Union
            d = smin(d, d_prim, uPrimitives[i].blendStrength > 0.0 ? uPrimitives[i].blendStrength : 0.1);
        }
    }
    
    // Light Pen Subtractive Sketching
    for(int i=0; i<127; i++) { // 128 points max
        if (i >= uSketchCount - 1) break;
        
        vec3 p0 = uSketchPoints[i];
        vec3 p1 = uSketchPoints[i+1];
        
        // 5cm cut radius
        float d_stroke = sdCapsule(p, p0, p1, 0.05);
        
        // Subtraction: max(d, -shape)
        d = max(d, -d_stroke);
    }
    
    // Clip Plane (cutaway or interior mode)
    if (uClipEnabled > 0.5) {
        float clipDist = dot(p, normalize(uClipPlane)) - uClipOffset;
        d = max(d, clipDist);
    }
    
    // Auto-clip for Interior mode (mode 10)
    if (uViewMode == 10) {
        vec3 autoClipDir = normalize(vec3(1.0, 1.0, 0.0));
        float autoClipDist = dot(p, autoClipDir) - 0.3;
        d = max(d, autoClipDist);
    }
    
    return d;
}


// ============================================
// RAYMARCHING UTILITIES
// ============================================

vec3 calcNormal(vec3 p) {
    const float h = 0.0001;
    const vec2 k = vec2(1, -1);
    return normalize(
        k.xyy * map(p + k.xyy * h) +
        k.yyx * map(p + k.yyx * h) +
        k.yxy * map(p + k.yxy * h) +
        k.xxx * map(p + k.xxx * h)
    );
}

float calcAO(vec3 pos, vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.01 + 0.12 * float(i) / 4.0;
        float d = map(pos + h * nor);
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

float calcSoftShadow(vec3 ro, vec3 rd, float mint, float maxt) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 16; i++) {
        float h = map(ro + rd * t);
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.02, 0.10);
        if (res < 0.001 || t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

// ============================================
// NOISE FUNCTIONS
// ============================================

vec3 hash(vec3 p) {
	p = vec3(dot(p,vec3(127.1,311.7, 74.7)),
			 dot(p,vec3(269.5,183.3,246.1)),
			 dot(p,vec3(113.5,271.9,124.6)));
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

// 3D Gradient Noise
float noise(in vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
	vec3 u = f*f*(3.0-2.0*f);
    return mix(mix(mix(dot(hash(i + vec3(0.0,0.0,0.0)), f - vec3(0.0,0.0,0.0)), 
                       dot(hash(i + vec3(1.0,0.0,0.0)), f - vec3(1.0,0.0,0.0)), u.x),
                   mix(dot(hash(i + vec3(0.0,1.0,0.0)), f - vec3(0.0,1.0,0.0)), 
                       dot(hash(i + vec3(1.0,1.0,0.0)), f - vec3(1.0,1.0,0.0)), u.x), u.y),
               mix(mix(dot(hash(i + vec3(0.0,0.0,1.0)), f - vec3(0.0,0.0,1.0)), 
                       dot(hash(i + vec3(1.0,0.0,1.0)), f - vec3(1.0,0.0,1.0)), u.x),
                   mix(dot(hash(i + vec3(0.0,1.0,1.0)), f - vec3(0.0,1.0,1.0)), 
                       dot(hash(i + vec3(1.0,1.0,1.0)), f - vec3(1.0,1.0,1.0)), u.x), u.y), u.z);
}

// Background Function with Grid
vec3 getBackground(vec3 rd) {
    // Gradient
    vec3 bg = mix(uBgColor2, uBgColor1, rd.y * 0.5 + 0.5);
    
    // Floor Grid
    if (uGridEnabled > 0.5 && rd.y < -0.01) {
        float t = -10.0 / rd.y; // Intersection with y = -10 (floor plane below object)
        if (t > 0.0) {
            vec3 pos = vec3(0.0) + rd * t; // ro assumed 0 for direction, but need offset relative to cam
            // Better: intersection with plane y = -4.0 relative to camera lookat
            // For now simple infinite grid on XZ plane at y = -5.0
            
            // Re-calculate p based on actual camera
            // p = ro + rd * t
            // We want y = -5.0
            // ro.y + rd.y * t = -5.0 => t = (-5.0 - ro.y) / rd.y
            
            // Note: ro is uCameraPos. 
            // Since we render object at 0,0,0, floor should be below uBaseDims.y
            
            // Fixed Grid at Y = 0.0 (Origin)
            // This ensures (0,0,0) is always on the grid pattern.
            float floorY = -0.01; 
            float tFloor = (floorY - uCameraPos.y) / rd.y;
            
            if (tFloor > 0.0 && tFloor < 100.0) {
                vec3 pFloor = uCameraPos + rd * tFloor;
                
                // Grid Pattern
                float gridSize = 2.0; // Restore original 2m Grid
                vec2 gridUV = abs(fract(pFloor.xz / gridSize) - 0.5);
                
                // Original thinner lines
                float line = smoothstep(0.45, 0.48, max(gridUV.x, gridUV.y));
                
                // Fade out at distance - Extended Horizon (User Request: "Plentier")
                // Reduced falloff from 0.05 to 0.02 for ~2.5x view distance
                float alpha = exp(-length(pFloor.xz) * 0.02); 
                
                // Original subtle theme-based color
                // This respects the 'Show Grid' color from Settings (uGridColor)
                bg = mix(bg, uGridColor, line * alpha * 0.2); // Slightly boosted from 0.15 for visibility
            }
        }
    }
    return bg;
}

// ============================================
// THERMAL VIEW HELPER (Phase 10)
// ============================================

vec3 getThermalColor(float temp) {
    // Temperature to color mapping (Celsius)
    // < 30°C  = Blue (Cool/Safe)
    // 50°C    = Cyan
    // 100°C   = Green → Yellow
    // 200°C   = Orange
    // 500°C+  = Red → White (Critical)
    
    if (temp < 30.0) {
        // Blue zone (0-30°C)
        float t = temp / 30.0;
        return mix(vec3(0.0, 0.0, 0.8), vec3(0.0, 0.5, 1.0), t);
    } else if (temp < 60.0) {
        // Cyan to Green (30-60°C)
        float t = (temp - 30.0) / 30.0;
        return mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 1.0, 0.5), t);
    } else if (temp < 100.0) {
        // Green to Yellow (60-100°C)
        float t = (temp - 60.0) / 40.0;
        return mix(vec3(0.0, 1.0, 0.5), vec3(1.0, 1.0, 0.0), t);
    } else if (temp < 200.0) {
        // Yellow to Orange (100-200°C)
        float t = (temp - 100.0) / 100.0;
        return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), t);
    } else if (temp < 500.0) {
        // Orange to Red (200-500°C)
        float t = (temp - 200.0) / 300.0;
        return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), t);
    } else {
        // Red to White (500°C+)  - Extreme heat
        float t = clamp((temp - 500.0) / 500.0, 0.0, 1.0);
        return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), t);
    }
}

// ============================================
// LIGHTING MODELS
// ============================================

// Mode 0: Realistic PBR-like lighting
vec3 pbrLighting(vec3 pos, vec3 normal, vec3 viewDir) {
    // Diffuse
    float diff1 = max(dot(normal, LIGHT_DIR), 0.0);
    float diff2 = max(dot(normal, LIGHT_DIR_2), 0.0) * 0.3;
    
    // Specular (Blinn-Phong approximation)
    vec3 halfDir = normalize(LIGHT_DIR + viewDir);
    float specPower = mix(8.0, 128.0, 1.0 - uRoughness);
    float spec = pow(max(dot(normal, halfDir), 0.0), specPower);
    
    // Fresnel
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
    float fresnelStrength = mix(0.04, 0.5, uMetalness);
    
    // Environment reflection (fake)
    vec3 reflectDir = reflect(-viewDir, normal);
    vec3 envColor = mix(vec3(0.1, 0.15, 0.2), vec3(0.4, 0.5, 0.6), reflectDir.y * 0.5 + 0.5);
    
    // Ambient occlusion
    float ao = calcAO(pos, normal);
    
    // Soft shadow
    float shadow = calcSoftShadow(pos + normal * 0.01, LIGHT_DIR, 0.02, 2.5);
    
    // Combine
    vec3 diffuseColor = uBaseColor * (1.0 - uMetalness);
    vec3 specColor = mix(vec3(0.04), uBaseColor, uMetalness);
    
    vec3 color = AMBIENT * uBaseColor * ao;
    color += diffuseColor * (diff1 * shadow + diff2) * vec3(1.0, 0.95, 0.9);
    color += specColor * spec * shadow;
    color += envColor * fresnel * fresnelStrength;
    
    return color;
}

// Mode 1: Wireframe (Blueprint Style - Transparent faces)
// Mode 1: Wireframe (Technical Edge Detection)
// Mode 1: Wireframe (Technical Edge Detection)
// ============================================
// ANALYTIC EDGES (Replaces unstable fwidth)
// ============================================

float getAnalyticEdge(vec3 p) {
    float edge = 0.0;
    float thickness = 0.015; // Tuned for "thin line" look

    if (uBaseShape == 1) { // Box
        vec3 d = abs(p) - uBaseDims * 0.5;
        // Edges are where 2 components are near max (d near 0)
        // Check if point is near edge of face
        vec3 onFace = step(vec3(-thickness), d); // 1 if on face
        // We are on an edge if at least 2 faces are "active"? 
        // Simpler: Check if we are within 'thickness' distance of the corner lines.
        // Box edges are along axes.
        
        vec3 a = abs(p);
        vec3 b = uBaseDims * 0.5;
        
        // Distance to edges
        float dx = length(vec2(max(a.y - b.y, 0.0), max(a.z - b.z, 0.0))); // This logic is for outside SDF.
        // Inside/Surface logic:
        // We are on surface. 
        // Edge if y is nearing b.y AND z is nearing b.z (for X axis edge)
        float ex = step(b.y - thickness, a.y) * step(b.z - thickness, a.z);
        float ey = step(b.x - thickness, a.x) * step(b.z - thickness, a.z);
        float ez = step(b.x - thickness, a.x) * step(b.y - thickness, a.y);
        
        edge = max(ex, max(ey, ez));
        
    } else if (uBaseShape == 2) { // Sphere
        // Only Silhouette (handled by rim) + maybe equator?
        // Let's stick to Rim only for sphere for clean look
        edge = 0.0;
        
    } else if (uBaseShape == 3) { // Cylinder
        float r = uBaseDims.x * 0.5;
        float h = uBaseDims.y * 0.5;
        
        // Rim caps
        float rDist = length(p.xz);
        float onCapRim = step(r - thickness, rDist) * step(h - thickness, abs(p.y));
        // Side vertical lines? Maybe not needed.
        edge = onCapRim;
    }
    
    return edge;
}

// Mode 1: Wireframe (Analytic + Rim)
vec4 wireframeShading(vec3 pos, vec3 normal, vec3 viewDir) {
    // 1. Analytic Geometric Edges (Crisp, Stable)
    float isEdge = getAnalyticEdge(pos);
    
    // 2. Rim / Silhouette (Always valid)
    float rim = pow(1.0 - abs(dot(normal, viewDir)), 3.0);
    float isRim = step(0.85, rim);

    float lines = max(isEdge, isRim);
    
    vec3 wireColor = vec3(0.0, 0.9, 1.0); // Cyan
    
    if (lines < 0.5) return vec4(0.0);
    return vec4(wireColor, 1.0);
}

// Mode 2: X-Ray (Ghost/Transparency style)
vec3 xrayShading(vec3 normal, vec3 viewDir) {
    // Opacity based on viewing angle (Fresnel)
    // Edges are opaque, center is transparent
    float opacity = pow(1.0 - abs(dot(normal, viewDir)), 3.0);
    
    vec3 rimColor = vec3(0.4, 0.7, 1.0); // Light Blue
    vec3 innerColor = vec3(0.0, 0.1, 0.2); // Dark Blueish (bones)
    
    // We simulate transparency by blending with the scene background
    // Assuming dark background in app
    return mix(innerColor, rimColor, opacity) * (0.3 + 0.7 * opacity);
}

// Mode 3: Matte diffuse
vec3 matteShading(vec3 pos, vec3 normal) {
    float diff = dot(normal, LIGHT_DIR) * 0.5 + 0.5;
    float ao = calcAO(pos, normal);
    return uBaseColor * diff * ao;
}

// Mode 4: Heatmap (blue-green-red gradient)
vec3 heatmapShading(vec3 pos, vec3 normal) {
    // Use Y position as heat proxy (or physics texture if available)
    // Use Y position as heat proxy (or physics texture if available)
    float heatValue;
    if (uPhysicsDataEnabled > 0.5) {
        vec2 uv = (pos.xz + 1.0) * 0.5;
        heatValue = 0.5; // Would sample from uPhysicsTexture
    } else {
        // Procedural Thermal Simulation (Noise + Gradient)
        // Independent of bounding box size
        float noiseVal = noise(pos * 3.0 + vec3(0.0, uTime * 0.5, 0.0));
        float vertical = smoothstep(-1.0, 1.0, pos.y);
        
        // Heat rises + Hotspots
        heatValue = mix(vertical, noiseVal * 0.5 + 0.5, 0.6);
    }
    heatValue = clamp(heatValue, 0.0, 1.0);
    
    // Scientific colormap (viridis-like)
    vec3 c0 = vec3(0.267, 0.004, 0.329); // Purple
    vec3 c1 = vec3(0.282, 0.140, 0.458);
    vec3 c2 = vec3(0.127, 0.566, 0.550); // Teal
    vec3 c3 = vec3(0.993, 0.906, 0.144); // Yellow
    
    vec3 color;
    if (heatValue < 0.33) {
        color = mix(c0, c1, heatValue * 3.0);
    } else if (heatValue < 0.66) {
        color = mix(c1, c2, (heatValue - 0.33) * 3.0);
    } else {
        color = mix(c2, c3, (heatValue - 0.66) * 3.0);
    }
    
    // Add subtle shading
    float diff = dot(normal, LIGHT_DIR) * 0.2 + 0.8;
    return color * diff;
}

// Mode 5: Cutaway with interior highlight
vec3 cutawayShading(vec3 pos, vec3 normal, vec3 viewDir) {
    vec3 baseColor = pbrLighting(pos, normal, viewDir);
    
    // Highlight cut surface
    if (uClipEnabled > 0.5) {
        vec3 clipNorm = normalize(uClipPlane);
        float cutFace = abs(dot(normal, clipNorm));
        if (cutFace > 0.9) {
            // This is the cut surface - show cross-section pattern
            float pattern = sin(pos.x * 30.0) * sin(pos.y * 30.0) * sin(pos.z * 30.0);
            vec3 crossColor = mix(vec3(0.8, 0.4, 0.1), vec3(1.0, 0.6, 0.2), pattern * 0.5 + 0.5);
            return crossColor;
        }
    }
    
    return baseColor;
}

// Mode 6: Solid flat color
vec3 solidShading() {
    return uBaseColor;
}

// Mode 7: Stress/Flow visualization
// Mode 7: Stress Analysis (Cantilever Model)
uniform float uStressLevel; // 0.0 = Safe, 1.0 = Yield
uniform float uMaxDeflection;

vec3 stressShading(vec3 pos, vec3 normal) {
     // Cantilever Physics:
     // Moment M = F * (L - x). Stress is max at anchor (x=0).
     // We assume anchor is at min X or 0.
     // Let's assume object is roughly centered or normalized.
     
     // Normalize X relative to bounding box
     float xNorm = clamp((pos.x + uBaseDims.x * 0.5) / uBaseDims.x, 0.0, 1.0);
     
     // Stress Profile: Linear decrease from root (0) to tip (1)
     float stressProfile = 1.0 - xNorm; 
     
     // Modulation by global stress level (Safety Factor inverted)
     // If uStressLevel is low (Safe), colors are cool.
     // If uStressLevel is high (Fail), colors get hot.
     
     float finalStress = stressProfile * uStressLevel;
     
     // Color Map: Blue (0) -> Green -> Yellow -> Red (1)
     vec3 cBlue = vec3(0.0, 0.2, 1.0);
     vec3 cGreen = vec3(0.0, 1.0, 0.2);
     vec3 cYellow = vec3(1.0, 1.0, 0.0);
     vec3 cRed = vec3(1.0, 0.0, 0.0);
     
     vec3 color;
     if (finalStress < 0.33) {
         color = mix(cBlue, cGreen, finalStress * 3.0);
     } else if (finalStress < 0.66) {
         color = mix(cGreen, cYellow, (finalStress - 0.33) * 3.0);
     } else {
         color = mix(cYellow, cRed, (finalStress - 0.66) * 3.0);
     }
     
     // Add oscillating warning if yielding
     if (uStressLevel > 0.95) {
         float flash = sin(uTime * 10.0) * 0.5 + 0.5;
         color = mix(color, vec3(1.0, 1.0, 1.0), flash * 0.5);
     }
     
     // Grid/Isolines to show deformation (fake)
     float iso = sin(pos.x * 20.0) * sin(pos.y * 20.0);
     color *= (0.8 + 0.2 * smoothstep(0.9, 0.95, iso));
     
     return color;
}

// Mode 11: Flow Dynamics (Streamlines)
vec3 flowShading(vec3 pos, vec3 normal) {
    // Animated streamlines along surface
    float flow = sin(pos.x * 4.0 + pos.y * 2.0 + uTime * 5.0);
    float lines = smoothstep(0.9, 0.95, flow);
    
    vec3 baseColor = vec3(0.2, 0.2, 0.2);
    vec3 flowColor = vec3(0.0, 0.8, 1.0); // Cyan flow
    
    return mix(baseColor, flowColor, lines);
}

// Mode 8: Hidden Line (technical drawing style)
// Mode 8: Hidden Line (Technical Drawing)
vec3 hiddenLineShading(vec3 pos, vec3 normal, vec3 viewDir) {
    vec3 paper = vec3(0.95, 0.95, 0.92); // Paper
    vec3 ink = vec3(0.1, 0.1, 0.15);     // Dark Ink
    
    // Edge detection (Screen space)
    vec3 nDelta = fwidth(normal);
    float edgeFactor = length(nDelta);
    float isEdge = smoothstep(0.05, 0.2, edgeFactor);
    
    // Rim
    float rim = pow(1.0 - abs(dot(normal, viewDir)), 3.0);
    float isRim = smoothstep(0.8, 1.0, rim);
    
    float lineWeight = max(isEdge, isRim);
    return mix(paper, ink, lineWeight);
}

// Mode 9: Shaded (simple 3-tone cel shading)
vec3 shadedShading(vec3 pos, vec3 normal) {
    float diff = dot(normal, LIGHT_DIR);
    
    // 3-tone quantization
    vec3 shadow = uBaseColor * 0.4;
    vec3 mid = uBaseColor * 0.7;
    vec3 lit = uBaseColor;
    
    vec3 color;
    if (diff < 0.0) {
        color = shadow;
    } else if (diff < 0.5) {
        color = mid;
    } else {
        color = lit;
    }
    
    // Subtle rim light
    float rim = pow(1.0 - abs(dot(normal, LIGHT_DIR)), 2.0) * 0.15;
    return color + rim;
}

// Mode 10: Interior (auto-clipped cross-section)
vec3 interiorShading(vec3 pos, vec3 normal, vec3 viewDir) {
    // Check if we're on the cut face
    vec3 clipNorm = normalize(vec3(1.0, 1.0, 0.0)); // Diagonal cut
    float onCutFace = abs(dot(normal, clipNorm));
    
    if (onCutFace > 0.8) {
        // Cross-hatching pattern for cut surface
        float hatch1 = smoothstep(0.9, 1.0, abs(sin((pos.x + pos.y) * 20.0)));
        float hatch2 = smoothstep(0.9, 1.0, abs(sin((pos.x - pos.y) * 20.0)));
        float hatch = max(hatch1, hatch2);
        
        vec3 cutColor = vec3(0.9, 0.7, 0.4); // Goldenrod interior
        vec3 hatchColor = vec3(0.6, 0.4, 0.2);
        return mix(cutColor, hatchColor, hatch);
    }
    
    // Regular PBR for exterior surfaces
    return pbrLighting(pos, normal, viewDir);
}

// ============================================
// MAIN
// ============================================

void main() {
    vec2 uv = vUv * 2.0 - 1.0;
    uv.x *= uResolution.x / uResolution.y;
    
    vec3 ro = uCameraPos;
    vec3 forward = normalize(-uCameraPos);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);
    
    // Calculate Background
    vec3 bgColor = getBackground(rd);
    float bgLum = dot(bgColor, vec3(0.299, 0.587, 0.114)); // Luminance for adaptive contrast
    
    vec3 accColor = vec3(0.0);
    float accAlpha = 0.0;
    
    float t = 0.0;
    
    // Accumulative Raymarching Loop (allows transparency/back-faces)
    // Increased steps to ensure we penetrate objects for inner lines
    for (int i = 0; i < 150; i++) {
        vec3 p = ro + rd * t;
        float rawD = map(p);
        float d = abs(rawD); // Use abs to traverse inside objects
        
        // Hit surface (either front or back face)
        if (d < 0.002) {
            vec3 n = calcNormal(p);
            // Flip normal if we are hitting back face (ray exiting or inside)
            if (rawD < 0.0) n = -n;
            
            vec3 viewDir = -rd;
            vec3 color = vec3(0.0);
            float alpha = 1.0;
            
            // View Mode Selection
            if (uViewMode == 0) {
                color = pbrLighting(p, n, viewDir);
            } else if (uViewMode == 1) {
                // Wireframe (Technical)
                vec4 res = wireframeShading(p, n, viewDir);
                color = res.rgb;
                alpha = res.a;
                
                // Adaptive Contrast: If background is light, force dark lines
                if (bgLum > 0.5) {
                     // Override Cyan with Dark Blue for visibility
                     if (length(color) > 0.5) color = vec3(0.0, 0.1, 0.35); 
                }
                
            } else if (uViewMode == 2) {
                color = xrayShading(n, viewDir);
                alpha = length(color) * 0.5 + 0.1; 
            } else if (uViewMode == 3) {
                color = matteShading(p, n);
            } else if (uViewMode == 4) {
                // Thermal View (Phase 10)
                if (uThermalEnabled) {
                    color = getThermalColor(uThermalTemp);
                } else {
                    // Fallback to generic gradient if no data
                    color = heatmapShading(p, n);
                }
            } else if (uViewMode == 5) {
                color = cutawayShading(p, n, viewDir);
            } else if (uViewMode == 6) {
                color = solidShading();
            } else if (uViewMode == 7) {
                color = stressShading(p, n);
            } else if (uViewMode == 11) {
                color = flowShading(p, n);
            } else if (uViewMode == 8) {
                color = hiddenLineShading(p, n, viewDir);
            } else {
                color = matteShading(p, n);
            }
            
            // Gamma
            color = pow(color, vec3(0.4545));
            
            // Noise Filter for Wireframe: Ignore faint artifacts
            if (uViewMode == 1 && alpha < 0.1) {
                t += 0.05;
                continue;
            }
            
            // Accumulate
            accColor += (1.0 - accAlpha) * color * alpha;
            accAlpha += (1.0 - accAlpha) * alpha;
            
            // Early exit if opaque
            if (accAlpha >= 0.95) break;
            
            // Push ray through surface to avoid self-intersection loops
            t += 0.05; 
            continue;
        }
        
        // Step forward
        // Ensure minimum step to avoid stagnation near 0
        t += max(d * 0.8, 0.002); 
        
        if (t > 20.0) break;
    }
    
    // Blend with background
    vec3 finalColor = accColor + bgColor * (1.0 - accAlpha);
    gl_FragColor = vec4(finalColor, 1.0);
}
    

